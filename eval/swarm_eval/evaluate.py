"""Fixed-seed evaluation and recording for shared policies, C++ runtime and classical baselines."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import swarmsim
import torch
import yaml
from swarm_policy.model import tensors
from swarm_policy.train import load


def summarise(rows: list[dict]) -> dict:
    """Means and 95% intervals; missing intercept/clear times stay missing."""
    result = {"episodes": len(rows), "truncated": sum(x["truncated"] for x in rows)}
    for key in (
        "survival",
        "intercept_fraction",
        "defender_survival",
        "first_intercept_s",
        "clear_s",
        "episode_s",
    ):
        values = np.array([x[key] for x in rows if x[key] is not None], dtype=float)
        if len(values):
            mean = float(values.mean())
            if key == "survival":
                # Wilson interval remains meaningful even for zero successes.
                n = len(values)
                z = 1.96
                denom = 1 + z * z / n
                center = (mean + z * z / (2 * n)) / denom
                half = z * np.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / denom
                interval = [max(0, center - half), min(1, center + half)]
            else:
                half = (
                    1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0
                )
                interval = [mean - half, mean + half]
            result[key] = mean
            result[key + "_ci95"] = interval
        else:
            result[key] = None
            result[key + "_ci95"] = None
    latency = np.concatenate([x["latency_ms"] for x in rows])
    result["latency_ms"] = {
        "p50": float(np.percentile(latency, 50)),
        "p99": float(np.percentile(latency, 99)),
    }
    return result


@torch.no_grad()
def decision(model, graph: dict, explain: bool = False) -> tuple[np.ndarray, dict]:
    """Policy decision plus optional same-step message and global interventions."""
    device = next(model.parameters()).device
    inputs = list(tensors(graph, device))
    outputs = model(*inputs)
    q = outputs[0]
    actions = q.argmax(-1)
    if not explain:
        return actions.cpu().numpy(), {}
    details = {
        "q": q.cpu().numpy(),
        "actions": actions.cpu().numpy(),
        "margin": (q.topk(2, dim=-1).values[:, 0] - q.topk(2, dim=-1).values[:, 1]).cpu().numpy(),
        "attention_local": outputs[1].cpu().numpy(),
        "attention_global": outputs[2].cpu().numpy(),
        "attention_teammate": outputs[3].cpu().numpy(),
        "messages": outputs[4].cpu().numpy(),
    }
    own = inputs[0].clone()
    own[:, 8] = 0
    ablated = model(own, *inputs[1:])[0]
    details["global_delta"] = (q - ablated).cpu().numpy()
    influence = np.zeros(len(graph["teammate_src"]), np.float32)
    flips = np.zeros(len(influence), np.int64)
    for source in range(len(q)):
        mask = torch.ones(len(q), device=device)
        mask[source] = 0
        altered = model(*inputs[:-1], mask)[0]
        keep = graph["teammate_src"] == source
        owner = graph["teammate_owner"][keep]
        influence[keep] = (q - altered).norm(dim=-1).cpu().numpy()[owner]
        flips[keep] = (actions != altered.argmax(-1)).cpu().numpy()[owner]
    details["message_influence"] = influence
    details["message_argmax_flip"] = flips
    return actions.cpu().numpy(), details


@torch.no_grad()
def episode(policy, cfg, seed: int, record: Path | None = None) -> dict:
    """Run a real TeamEnv episode; optionally store every state, graph and decision as NPZ."""
    env = swarmsim.TeamEnv(cfg)
    graph = env.reset(seed)
    rng = np.random.default_rng(seed)
    states = []
    statuses = []
    times = []
    logs = []
    latencies = []
    intercepted = 0
    while True:
        world = env.world
        state, status = world.snapshot()
        if record:
            states.append(state)
            statuses.append(status)
            times.append(env.time)
        before = time.perf_counter()
        details: dict = {}
        if isinstance(policy, str):
            if policy in ("idle", "random"):
                actions = (
                    [swarmsim.NOOP] * len(graph["self"])
                    if policy == "idle"
                    else rng.integers(27, size=len(graph["self"])).tolist()
                )
                result = env.step(actions)
            else:
                guidance = swarmsim.Guidance.PN if policy.startswith("pn") else swarmsim.Guidance.PP
                assignment = (
                    swarmsim.Assignment.HUNGARIAN
                    if policy.endswith("hungarian")
                    else swarmsim.Assignment.GREEDY
                )
                result = env.step_baseline(guidance, assignment)
        else:
            actions, details = decision(policy, graph, bool(record))
            latency = (time.perf_counter() - before) * 1000 / max(1, len(actions))
            if record:
                # Plain JSON strings avoid object arrays and allow_pickle when loading NPZ.
                logs.append(
                    json.dumps(
                        {
                            "graph": {k: v.tolist() for k, v in graph.items()},
                            "headings": env.headings,
                            "decision": {k: v.tolist() for k, v in details.items()},
                        }
                    )
                )
            result = env.step(actions.tolist())
        if isinstance(policy, str):
            # Baseline timings include the environment; labelled separately in reporting.
            latency = (time.perf_counter() - before) * 1000 / max(1, len(graph["self"]))
        latencies.append(latency)
        intercepted += result["intercepted"]
        graph = result["observation"]
        if result["terminated"] or result["truncated"]:
            break
    events = world.events()
    spawn = [e[0] for e in events if e[1] == 0]
    first = [e[0] for e in events if e[1] == 2]
    asset_hit = any(e[1] == 4 for e in events)
    survival = result["terminated"] and not asset_hit
    final, status = world.snapshot()
    if record:
        states.append(final)
        statuses.append(status)
        times.append(env.time)
        scenario = world.scenario()
        record.parent.mkdir(parents=True, exist_ok=True)
        routes = world.waypoints()
        # Ragged routes use a numeric flat array plus offsets, never pickled objects.
        offsets = np.cumsum([0] + [len(x) for x in routes])
        np.savez_compressed(
            record,
            seed=seed,
            n_blue=world.n_blue,
            n_red=world.n_red,
            t=times,
            states=states,
            status=statuses,
            events=np.asarray(events).reshape(-1, 5),
            asset=scenario["asset"],
            buildings=np.asarray(scenario["buildings"]).reshape(-1, 6),
            red=scenario["red"],
            route_points=np.asarray([p for route in routes for p in route]).reshape(-1, 3),
            route_offsets=offsets,
            asset_hit=asset_hit,
            truncated=result["truncated"],
            intercepted=intercepted,
            decisions=np.asarray(logs),
        )
    return {
        "seed": seed,
        "survival": survival,
        "truncated": result["truncated"],
        "intercept_fraction": intercepted / world.n_red,
        "defender_survival": float(np.mean(status[: world.n_blue] == 1)) if world.n_blue else 0,
        "first_intercept_s": min(first) - min(spawn) if first else None,
        "clear_s": env.time - min(spawn) if survival else None,
        "episode_s": env.time,
        "latency_ms": latencies,
    }


def evaluate_policy(
    model, env_path: str, seed: int, count: int, red_count: int = 3, weights: list | None = None
) -> dict:
    cfg = swarmsim.EnvConfig.load(env_path)
    if red_count:
        cfg.red_min = cfg.red_max = red_count
    if weights is not None:
        cfg.behaviour_weights = weights
    return summarise([episode(model, cfg, seed + i) for i in range(count)])


def main():
    from .tracking import Run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", help="local checkpoint/directory, W&B run URL, or model artifact reference"
    )
    parser.add_argument("--suite", choices=["v0"], default="v0")
    parser.add_argument("--config", help="Override the selected suite configuration")
    parser.add_argument("--out", type=Path, default=Path("experiments/eval"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--cpp", action="store_true")
    args = parser.parse_args()
    settings_path = (
        Path(args.config) if args.config else Path("scenarios") / args.suite / "eval.yaml"
    )
    settings = yaml.safe_load(settings_path.read_text())
    if args.episodes:
        settings["episodes"] = args.episodes
    torch.set_num_threads(1)
    policy = None
    if args.run:
        from .tracking import resolve_checkpoint

        checkpoint = resolve_checkpoint(args.run, args.out / "models")
        policy = load(checkpoint)
    run = Run(args.out, settings, settings.get("wandb_mode", "offline"), "eval")
    model_path = ""
    if args.cpp and policy is not None:
        from swarm_policy.export import export

        onnx_path = checkpoint.with_suffix(".onnx")
        if not onnx_path.exists() or onnx_path.stat().st_mtime < checkpoint.stat().st_mtime:
            sample_env = swarmsim.TeamEnv(swarmsim.EnvConfig.load(settings["env"]))
            export(policy, sample_env.reset(settings["seed"]), onnx_path)
        model_path = str(onnx_path)
    results = []
    for condition in settings["conditions"]:
        cfg = swarmsim.EnvConfig.load(settings["env"])
        cfg.red_min = cfg.red_max = condition["red"]
        behaviour = condition["behaviour"]
        cfg.behaviour_weights = [
            1 if behaviour == "mixed" or b == behaviour else 0 for b in swarmsim.BEHAVIOURS
        ]
        names = settings["policies"] + (["gqn"] if policy is not None else [])
        for name in names:
            rows = []
            for i in range(settings["episodes"]):
                seed = settings["seed"] + i
                path = (
                    args.out / "episodes" / f"{name}-{condition['red']}-{behaviour}-{i}.npz"
                    if i < args.record
                    else None
                )
                if args.cpp:
                    # C++ path includes world, perception, baseline/runtime decision and metrics.
                    rows.append(swarmsim.evaluate_cpp(cfg, seed, name, model_path))
                    if path:
                        episode(policy if name == "gqn" else name, cfg, seed, path)
                else:
                    rows.append(episode(policy if name == "gqn" else name, cfg, seed, path))
            summary = {"policy": name, **condition, **summarise(rows)}
            results.append(summary)
            run.log({f"{name}/{condition['red']}/{behaviour}/survival": summary["survival"]})
            print(json.dumps(summary), flush=True)
    (args.out / "metrics.json").write_text(json.dumps({"evaluation": results}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
