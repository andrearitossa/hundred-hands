"""CPU environment throughput and native-runtime latency/size sweeps."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import swarmsim
import torch
import yaml
from swarm_policy.export import export
from swarm_policy.model import GQN


def benchmark(settings: dict, out: Path, what: str) -> dict:
    """Measure actual C++ graph/env/runtime paths; all durations are wall-clock milliseconds."""
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if what == "env":
        for count in settings["env_counts"]:
            for threads in settings["threads"]:
                cfg = swarmsim.EnvConfig.load(settings["env"])
                env = swarmsim.VecEnv(cfg, count, threads)
                graph = env.reset(settings["seed"])
                before = time.perf_counter()
                for _ in range(settings["iterations"]):
                    graph = env.step([13] * len(graph["self"]))["observation"]
                elapsed = time.perf_counter() - before
                rows.append(
                    {
                        "envs": count,
                        "threads": threads,
                        "seconds": elapsed,
                        "decisions_per_second": count * settings["iterations"] / elapsed,
                    }
                )
    else:
        for hidden in settings["hidden_sizes"]:
            for threats in settings["threat_counts"]:
                for mates in settings["teammate_counts"]:
                    cfg = swarmsim.EnvConfig.load(settings["env"])
                    cfg.blue_count = mates + 1
                    cfg.red_min = cfg.red_max = threats
                    cfg.onboard_range = cfg.ground_range = 5000
                    env = swarmsim.TeamEnv(cfg)
                    graph = env.reset(settings["seed"])
                    for _ in range(110):
                        graph = env.step([13] * len(graph["self"]))["observation"]
                    model = GQN(hidden=hidden)
                    path = out / f"h{hidden}-t{threats}-m{mates}.onnx"
                    export(model, graph, path)
                    runtime = swarmsim.Runtime(str(path))
                    for _ in range(10):
                        runtime.forward(graph)
                    samples = []
                    tracking = []
                    for _ in range(settings["iterations"]):
                        before = time.perf_counter()
                        runtime.forward(graph)
                        inference = (time.perf_counter() - before) * 1000
                        # Includes graph construction, excludes physics and tracker work.
                        n = max(1, len(graph["self"]))
                        samples.append((inference + env.timings["graph_ms"]) / n)
                        tracking.append(env.timings["tracking_ms"] / n)
                    rows.append(
                        {
                            "hidden": hidden,
                            "threats": threats,
                            "teammates": mates,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "bytes": path.stat().st_size,
                            "p50_ms": float(np.percentile(samples, 50)),
                            "p99_ms": float(np.percentile(samples, 99)),
                            "tracking_ms": float(np.mean(tracking)),
                        }
                    )
    metrics: dict = {"what": what, "platform": platform.platform(), "measurements": rows}
    if what == "latency":
        passing = [
            h
            for h in settings["hidden_sizes"]
            if all(
                r["p99_ms"] < settings["target_ms"] / settings["headroom"]
                for r in rows
                if r["hidden"] == h
            )
        ]
        metrics["chosen_hidden"] = max(passing) if passing else None
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump(settings))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["env", "latency"], required=True)
    parser.add_argument("--config", default="scenarios/v0/bench.yaml")
    parser.add_argument("--out", type=Path, default=Path("experiments/bench"))
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(yaml.safe_load(Path(args.config).read_text()), args.out, args.what), indent=2
        )
    )


if __name__ == "__main__":
    main()
