"""Small engineering baseline check and VecEnv throughput sweep; run from repo root."""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
import swarmsim
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
config = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
config.red_min = config.red_max = 3
config.behaviour_weights = [1, 0, 0]
resolved = yaml.safe_load(Path("scenarios/v0/env.yaml").read_text())
resolved["limits"] = yaml.safe_load(Path("scenarios/assets/x500/limits.yaml").read_text())
resolved["red"]["count_min"] = resolved["red"]["count_max"] = 3
resolved["red"]["behaviour_weights"] = {"straight": 1, "waypoint": 0, "evasive": 0}
(OUT / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
results = {}
start = time.perf_counter()
for name in ["idle", "random", "pp_greedy", "pn_greedy", "pp_hungarian", "pn_hungarian"]:
    rows = []
    for seed in range(20):
        env = swarmsim.TeamEnv(config)
        graph = env.reset(10000 + seed)
        rng = np.random.default_rng(seed)
        intercepted = 0
        while True:
            if name in ("idle", "random"):
                actions = (
                    [swarmsim.NOOP] * len(graph["self"])
                    if name == "idle"
                    else rng.integers(0, 27, len(graph["self"])).tolist()
                )
                r = env.step(actions)
            else:
                r = env.step_baseline(
                    swarmsim.Guidance.PP if name.startswith("pp") else swarmsim.Guidance.PN,
                    swarmsim.Assignment.GREEDY
                    if name.endswith("greedy")
                    else swarmsim.Assignment.HUNGARIAN,
                )
            graph = r["observation"]
            intercepted += r["intercepted"]
            if r["terminated"] or r["truncated"]:
                rows.append(
                    {
                        "seed": 10000 + seed,
                        "survived": r["reward"] > 0 and r["terminated"],
                        "truncated": r["truncated"],
                        "intercepted": intercepted,
                        "seconds": env.time,
                    }
                )
                break
    results[name] = {
        "survival": np.mean([r["survived"] for r in rows]),
        "intercept_fraction": np.mean([r["intercepted"] / 3 for r in rows]),
        "truncated": sum(r["truncated"] for r in rows),
        "episodes": rows,
    }
    print(name, {k: v for k, v in results[name].items() if k != "episodes"}, flush=True)
bench = []
for n in (1, 4, 16, 32):
    e = swarmsim.VecEnv(config, n, n)
    g = e.reset(20000)
    before = time.perf_counter()
    for _ in range(100):
        g = e.step([swarmsim.NOOP] * len(g["self"]))["observation"]
    elapsed = time.perf_counter() - before
    bench.append(
        {"envs": n, "threads": n, "seconds": elapsed, "env_decisions_per_second": n * 100 / elapsed}
    )
    print(bench[-1], flush=True)
(OUT / "metrics.json").write_text(json.dumps({"baselines": results, "throughput": bench}, indent=2))
(OUT / "meta.json").write_text(
    json.dumps(
        {
            "platform": platform.platform(),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "cpu_count": os.cpu_count(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "duration_s": time.perf_counter() - start,
            "wandb": None,
            "purpose": "local engineering diagnosis",
        },
        indent=2,
    )
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(list(results), [v["survival"] for v in results.values()])
ax.set(
    ylabel="Asset survival fraction",
    ylim=(0, 1),
    title="3 vs 3 straight; 20 seeds per policy (engineering check)",
)
fig.tight_layout()
fig.savefig(OUT / "baselines.png", dpi=140)
plt.close(fig)
