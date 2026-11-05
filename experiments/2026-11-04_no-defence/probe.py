"""No-defence single-attacker diagnosis; run from the repository root."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import swarmsim

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="after")
parser.add_argument("--episodes", type=int, default=1000)
args = parser.parse_args()
start = time.perf_counter()
rows = []
for behaviour in swarmsim.BEHAVIOURS:
    c = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    c.blue_count = 0
    c.red_min = c.red_max = 1
    c.behaviour_weights = [float(b == behaviour) for b in swarmsim.BEHAVIOURS]
    env = swarmsim.Env(c)
    failures = []
    for seed in range(args.episodes):
        env.reset(seed)
        while True:
            r = env.step(np.empty((0, 3)))
            if r["done"]:
                break
        if not r["asset_hit"] or r["truncated"]:
            failures.append({"seed": seed, "result": r, "events": env.events()})
    row = {"behaviour": behaviour, "episodes": args.episodes, "failures": failures}
    rows.append(row)
    print(behaviour, "failures", len(failures), [x["seed"] for x in failures], flush=True)
Path(f"experiments/2026-11-04_no-defence/{args.output}.json").write_text(
    json.dumps(
        {
            "results": rows,
            "duration_s": time.perf_counter() - start,
            "platform": platform.platform(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree": "uncommitted no-defence corridor fix",
        },
        indent=2,
    )
)
