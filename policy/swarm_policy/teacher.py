"""Explicitly separate PN teacher sanity check and optional behaviour-cloning warm start."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import swarmsim
import torch
from swarm_eval.evaluate import evaluate_policy
from swarm_eval.tracking import Run

from .model import GQN, tensors
from .replay import pack


@dataclass
class TeacherConfig:
    env: str = "scenarios/v0/env.yaml"
    seed: int = 7000000
    decisions: int = 50000
    updates: int = 10000
    batch: int = 32
    hidden: int = 32
    lr: float = 0.001
    eval_seed: int = 10000000
    eval_episodes: int = 200
    dagger_rounds: int = 0
    dagger_decisions: int = 5000
    dagger_updates: int = 3000


def sanity(cfg: TeacherConfig, out: Path) -> dict:
    """Verify expert reward and learn its discrete decisions; this is not a sparse-RL result."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env_cfg = swarmsim.EnvConfig.load(cfg.env)
    env_cfg.red_min = env_cfg.red_max = 3
    env_cfg.behaviour_weights = [1, 0, 0]
    env = swarmsim.TeamEnv(env_cfg)
    graph = env.reset(cfg.seed)
    run = Run(out, asdict(cfg), "offline", "teacher-sanity")
    start = time.perf_counter()
    graphs = []
    labels = []
    episodes = []
    reward = 0
    intercepts = 0
    seed = cfg.seed
    for step in range(cfg.decisions):
        actions = np.asarray(
            env.teacher_actions(swarmsim.Guidance.PN, swarmsim.Assignment.GREEDY), dtype=np.int64
        )
        if len(actions):
            graphs.append(graph)
            labels.append(actions)
        result = env.step(actions.tolist())
        reward += result["reward"]
        intercepts += result["intercepted"]
        graph = result["observation"]
        if result["terminated"] or result["truncated"]:
            hit = any(e[1] == 4 for e in env.world.events())
            episodes.append(
                {
                    "seed": seed,
                    "reward": reward,
                    "intercepts": intercepts,
                    "survived": result["terminated"] and not hit,
                    "truncated": result["truncated"],
                }
            )
            seed += 1
            reward = 0
            intercepts = 0
            graph = env.reset(seed)
    model = GQN(hidden=cfg.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    counts = np.bincount(np.concatenate(labels), minlength=27)
    weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    weights /= weights.mean()
    losses = []
    for step in range(cfg.updates):
        indices = rng.integers(len(graphs), size=cfg.batch)
        batch = pack([graphs[i] for i in indices])
        action_targets = torch.as_tensor(np.concatenate([labels[i] for i in indices]))
        q = model(*tensors(batch))[0]
        loss = torch.nn.functional.cross_entropy(q, action_targets, weight=weights)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimiser.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 1000 == 0:
            run.log({"cross_entropy": float(np.mean(losses[-1000:])), "updates": step + 1})
    # Dataset aggregation queries the expert on the learner's visited states.
    # It is an explicitly teacher-assisted diagnostic, independent of sparse RL.
    aggregation = []
    for round_index in range(cfg.dagger_rounds):
        beta = max(0.0, 1.0 - (round_index + 1) / cfg.dagger_rounds)
        graph = env.reset(seed + 1)
        seed += 1
        for _ in range(cfg.dagger_decisions):
            expert = np.asarray(
                env.teacher_actions(swarmsim.Guidance.PN, swarmsim.Assignment.GREEDY),
                dtype=np.int64,
            )
            if len(expert):
                graphs.append(graph)
                labels.append(expert)
            with torch.no_grad():
                action = model(*tensors(graph))[0].argmax(-1).numpy()
            if rng.random() < beta:
                action = expert
            result = env.step(action.tolist())
            graph = result["observation"]
            if result["terminated"] or result["truncated"]:
                seed += 1
                graph = env.reset(seed)
        for _ in range(cfg.dagger_updates):
            indices = rng.integers(len(graphs), size=cfg.batch)
            batch = pack([graphs[i] for i in indices])
            targets = torch.as_tensor(np.concatenate([labels[i] for i in indices]))
            q = model(*tensors(batch))[0]
            loss = torch.nn.functional.cross_entropy(q, targets)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimiser.step()
        score = evaluate_policy(model, cfg.env, cfg.eval_seed, cfg.eval_episodes, 3, [1, 0, 0])
        row = {"round": round_index + 1, "expert_probability": beta, **score}
        aggregation.append(row)
        (out / "aggregation.json").write_text(json.dumps(aggregation, indent=2))
        run.log(
            {
                "aggregation_round": round_index + 1,
                "survival": score["survival"],
                "intercept_fraction": score["intercept_fraction"],
            }
        )
    score = evaluate_policy(model, cfg.env, cfg.eval_seed, cfg.eval_episodes, 3, [1, 0, 0])
    metrics = {
        "experiment": "teacher-assisted diagnostic, not standalone RL acceptance",
        "aggregation": aggregation,
        "teacher_episodes": episodes,
        "teacher_intercepts": sum(x["intercepts"] for x in episodes),
        "teacher_survival": float(np.mean([x["survived"] for x in episodes])),
        "action_histogram": counts.tolist(),
        "loss_first": float(np.mean(losses[:100])),
        "loss_last": float(np.mean(losses[-100:])),
        "evaluation": score,
        "duration_s": time.perf_counter() - start,
    }
    path = out / "teacher.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.settings(),
            "teacher_config": asdict(cfg),
        },
        path,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run.log({"survival": score["survival"], "intercept_fraction": score["intercept_fraction"]})
    run.checkpoint(path)
    run.finish()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/2026-11-04_m4-learner/teacher")
    )
    parser.add_argument("--decisions", type=int, default=50000)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            sanity(
                TeacherConfig(
                    decisions=args.decisions, updates=args.updates, dagger_rounds=args.dagger_rounds
                ),
                args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
