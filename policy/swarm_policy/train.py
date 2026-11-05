"""In-house Double DQN + VDN learner over C++ vector environments."""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import swarmsim
import torch
import yaml
from torch import Tensor

from .model import GQN, tensors
from .replay import NStep, Replay, pack, split


@dataclass
class TrainConfig:
    env: str = "scenarios/v0/env.yaml"
    seed: int = 42
    steps: int = 5_000_000
    envs: int = 16
    threads: int = 4
    hidden: int = 32
    heads: int = 4
    message_dim: int = 16
    lr: float = 0.0005
    gamma: float = 0.99
    n_step: int = 5
    batch: int = 64
    replay_capacity: int = 1_000_000
    replay_compression: bool = True
    warmup: int = 2000
    update_every: int = 4
    tau: float = 0.005
    grad_clip: float = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.3
    checkpoint_every: int = 250_000
    eval_every: int = 100_000
    eval_episodes: int = 200
    eval_seed: int = 10_000_000
    red_count: int = 3
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "offline"
    device: str = "cpu"
    shaping: float = 0.0
    shaping_mode: str = "potential"
    team_loss_penalty: float = 0.0
    curriculum: list[dict] = field(
        default_factory=lambda: [
            {"weights": [1, 0, 0], "threshold": 0.7, "budget": 2_000_000},
            {"weights": [1, 1, 0], "threshold": 0.7, "budget": 2_000_000},
            {"weights": [1, 1, 1], "threshold": 0.7, "budget": 1_000_000},
        ]
    )


def team_sum(values: Tensor, owners: Tensor, count: int) -> Tensor:
    """VDN sum over agents present at this step; an empty team has value zero."""
    return values.new_zeros(count).index_add(0, owners, values)


def double_dqn_target(
    online_q: Tensor, target_q: Tensor, owners: Tensor, rewards: Tensor, discounts: Tensor
) -> Tensor:
    """Select online actions, evaluate target weights, and sum only present next agents."""
    best = online_q.argmax(-1)
    future = target_q.gather(1, best[:, None]).flatten()
    return rewards + discounts * team_sum(future, owners, len(rewards))


def update(model: GQN, target: GQN, optimiser, samples, cfg: TrainConfig) -> float:
    observation = pack([x.observation for x in samples])
    next_observation = pack([x.next_observation for x in samples])
    actions = torch.as_tensor(np.concatenate([x.actions for x in samples]), device=cfg.device)
    owners = torch.as_tensor(observation["agent_env"], device=cfg.device)
    next_owners = torch.as_tensor(next_observation["agent_env"], device=cfg.device)
    q = model(*tensors(observation, cfg.device))[0]
    predicted = team_sum(q.gather(1, actions[:, None]).flatten(), owners, len(samples))
    with torch.no_grad():
        inputs = tensors(next_observation, cfg.device)
        y = double_dqn_target(
            model(*inputs)[0],
            target(*inputs)[0],
            next_owners,
            torch.tensor([x.reward for x in samples], device=cfg.device),
            torch.tensor([x.discount for x in samples], device=cfg.device),
        )
    loss = torch.nn.functional.smooth_l1_loss(predicted, y)
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimiser.step()
    with torch.no_grad():
        for slow, fast in zip(target.parameters(), model.parameters(), strict=True):
            slow.lerp_(fast, cfg.tau)
    return float(loss.detach())


def load(path: str | Path, device: str = "cpu") -> GQN:
    """Restore a model using the architecture stored in its checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = GQN(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def potential(graph: dict, initial_team_size: int) -> float:
    """Optional shared potential: negative nearest fused-threat distance per alive defender."""
    distances = [1.0] * (initial_team_size - len(graph["self"]))
    for agent in range(len(graph["self"])):
        nodes = graph["global_threat"][graph["global_owner"] == agent]
        if len(nodes):
            distances.append(min(1.0, float(nodes[:, 6].min())))
        else:
            distances.append(1.0)
    return -sum(distances)


def closing_reward(graph: dict) -> float:
    """One shared team score for moving toward the nearest currently fused threat."""
    progress = []
    for agent in range(len(graph["self"])):
        nodes = graph["global_threat"][graph["global_owner"] == agent]
        if len(nodes):
            relative = nodes[np.argmin(nodes[:, 6]), :3]
            length = np.linalg.norm(relative)
            if length > 1e-9:
                progress.append(float(np.dot(graph["self"][agent, :3], relative / length)))
    return float(np.mean(progress)) if progress else 0.0


def train(cfg: TrainConfig, out: Path) -> dict:
    """Train on real env transitions, checkpoint and evaluate with disjoint fixed seeds."""
    from swarm_eval.evaluate import evaluate_policy
    from swarm_eval.tracking import Run

    if cfg.seed + cfg.steps + cfg.envs >= cfg.eval_seed:
        raise ValueError("training seed stream could overlap reserved evaluation seeds")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = GQN(cfg.hidden, cfg.heads, cfg.message_dim).to(cfg.device)
    target = copy.deepcopy(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    replay = Replay(cfg.replay_capacity, rng, cfg.replay_compression)
    accumulators = [NStep(cfg.n_step, cfg.gamma) for _ in range(cfg.envs)]
    env_config = swarmsim.EnvConfig.load(cfg.env)
    if cfg.red_count:
        env_config.red_min = env_config.red_max = cfg.red_count
    stage = 0
    stage_start = 0
    env_config.behaviour_weights = cfg.curriculum[stage]["weights"]
    env = swarmsim.VecEnv(env_config, cfg.envs, cfg.threads)
    observation = env.reset(cfg.seed)
    run = Run(out, asdict(cfg), cfg.wandb_mode)
    start = time.perf_counter()
    losses, evals = [], []
    next_diagnosis = 10000
    action_histogram = np.zeros(27, dtype=np.int64)
    greedy_histogram = np.zeros(27, dtype=np.int64)
    actor_intercepts = 0
    actor_episodes = 0
    next_update, next_eval, next_checkpoint = cfg.warmup, cfg.eval_every, cfg.checkpoint_every
    best_score = (-1.0, -1.0)
    last_eval = None

    def save(name: str, steps: int):
        path = out / name
        torch.save(
            {
                "model": model.state_dict(),
                "target": target.state_dict(),
                "optimiser": optimiser.state_dict(),
                "model_config": model.settings(),
                "train_config": asdict(cfg),
                "steps": steps,
                "stage": stage,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            },
            path,
        )
        return path

    for steps in range(cfg.envs, cfg.steps + cfg.envs, cfg.envs):
        epsilon = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * min(
            1.0, steps / max(1, cfg.steps * cfg.exploration_fraction)
        )
        with torch.no_grad():
            actor_q = model(*tensors(observation, cfg.device))[0]
            actions = actor_q.argmax(-1).cpu().numpy()
        greedy_histogram += np.bincount(actions, minlength=27)
        explore = rng.random(len(actions)) < epsilon
        actions[explore] = rng.integers(27, size=int(explore.sum()))
        action_histogram += np.bincount(actions, minlength=27)
        result = env.step(actions.tolist())
        for i, transition in enumerate(result["transitions"]):
            actor_intercepts += transition["intercepted"]
            actor_episodes += int(transition["terminated"] or transition["truncated"])
            obs = split(observation, i)
            following = split(transition["observation"], i)
            reward = transition["reward"]
            if cfg.shaping:
                if cfg.shaping_mode == "closing_speed":
                    reward += cfg.shaping * closing_reward(following)
                elif cfg.shaping_mode == "potential":
                    next_potential = (
                        0.0
                        if transition["terminated"]
                        else potential(following, env_config.blue_count)
                    )
                    reward += cfg.shaping * (
                        cfg.gamma * next_potential - potential(obs, env_config.blue_count)
                    )
                else:
                    raise ValueError("unknown shared shaping mode")
            if len(following["self"]) < len(obs["self"]):
                reward -= cfg.team_loss_penalty
            for item in accumulators[i].append(
                obs,
                actions[observation["agent_env"] == i],
                reward,
                following,
                transition["terminated"],
                transition["truncated"],
            ):
                replay.add(item)
        observation = result["observation"]
        # A decision step is one environment step, not one batched call.
        while steps >= next_update and len(replay) >= cfg.batch:
            loss = update(model, target, optimiser, replay.sample(cfg.batch), cfg)
            losses.append(loss)
            next_update += cfg.update_every
        if steps >= next_diagnosis:
            run.log(
                {
                    "actor_intercepts": actor_intercepts,
                    "actor_episodes": actor_episodes,
                    "action_histogram": action_histogram.tolist(),
                    "greedy_action_histogram": greedy_histogram.tolist(),
                    "q_min": float(actor_q.min()) if actor_q.numel() else 0,
                    "q_max": float(actor_q.max()) if actor_q.numel() else 0,
                    "epsilon": epsilon,
                    "decisions_per_second": steps / (time.perf_counter() - start),
                },
                steps,
            )
            action_histogram[:] = 0
            greedy_histogram[:] = 0
            next_diagnosis += 10000
        if steps >= next_eval:
            score = evaluate_policy(
                model,
                cfg.env,
                cfg.eval_seed,
                cfg.eval_episodes,
                cfg.red_count,
                cfg.curriculum[stage]["weights"],
            )
            last_eval = score
            evals.append({"steps": steps, "stage": stage, **score})
            run.log(
                {
                    "loss": float(np.mean(losses[-100:])),
                    "epsilon": epsilon,
                    "survival": score["survival"],
                    "stage": stage,
                },
                steps,
            )
            selection = (score["survival"], score["intercept_fraction"])
            if selection > best_score:
                best_score = selection
                run.checkpoint(save("best.pt", steps))
            (out / "evaluations.json").write_text(json.dumps(evals, indent=2))
            next_eval += cfg.eval_every
        advance = steps - stage_start >= cfg.curriculum[stage]["budget"] or (
            last_eval is not None and last_eval["survival"] >= cfg.curriculum[stage]["threshold"]
        )
        if advance and stage + 1 < len(cfg.curriculum):
            stage += 1
            stage_start, last_eval = steps, None
            env_config.behaviour_weights = cfg.curriculum[stage]["weights"]
            env = swarmsim.VecEnv(env_config, cfg.envs, cfg.threads)
            observation = env.reset(cfg.seed + steps)
            accumulators = [NStep(cfg.n_step, cfg.gamma) for _ in range(cfg.envs)]
        if steps >= next_checkpoint:
            run.checkpoint(save(f"checkpoint-{steps}.pt", steps))
            next_checkpoint += cfg.checkpoint_every
    path = save("last.pt", steps)
    run.checkpoint(path)
    metrics = {
        "steps": steps,
        "updates": len(losses),
        "loss_first": float(np.mean(losses[:100])) if losses else None,
        "loss_last": float(np.mean(losses[-100:])) if losses else None,
        "duration_s": time.perf_counter() - start,
        "evaluations": evals,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run.finish()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scenarios/v0/train.yaml")
    parser.add_argument("--out", type=Path, default=Path("experiments/train"))
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.config).read_text())
    if args.steps:
        values["steps"] = args.steps
    print(json.dumps(train(TrainConfig(**values), args.out), indent=2))


if __name__ == "__main__":
    main()
