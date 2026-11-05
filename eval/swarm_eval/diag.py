"""Diagnostics: run systems in isolation and write metrics + artefacts to an output directory."""

import argparse
import collections
import json
import time
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import swarmsim

from swarm_eval.episode import run_episode
from swarm_eval.replay import write_html


def _route_clearance(ep: dict, drone: int, pos: np.ndarray) -> float:
    """Horizontal distance (m) from a crash to the attacker's planned route."""
    red = drone - ep["n_blue"]
    pts = [
        ep["states"][0, drone, :2],
        *[np.array(p)[:2] for p in ep["routes"][red]],
        ep["asset"][:2],
    ]
    best = np.inf
    for a, b in pairwise(pts):
        u = np.clip(np.dot(pos - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(pos - (a + u * (b - a)))))
    return round(best, 1)


def crash_gallery(env: swarmsim.Env, crashes: list[dict], path: Path, n: int = 6) -> None:
    """Zoomed top-down plots of attacker crashes: path flown, planned route, buildings."""
    path.unlink(missing_ok=True)
    if not crashes:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, crashes[:n]):
        ep = run_episode(env, c["seed"])
        d = ep["n_blue"] + c["red"]
        ax.axis("on")
        for b in ep["buildings"]:
            ax.add_patch(plt.Rectangle(b[:2] - b[3:5], *(2 * b[3:5]), color="0.75"))
            ax.text(b[0], b[1], f"h={2 * b[5]:.0f}", ha="center", fontsize=8)
        alive = ep["status"][:, d] == 1
        xy = ep["states"][alive, d, :2]
        ax.plot(*xy.T, color="tab:red", label="flown")
        route = np.array(
            [
                ep["states"][0, d, :2],
                *[np.array(p)[:2] for p in ep["routes"][c["red"]]],
                ep["asset"][:2],
            ]
        )
        ax.plot(*route.T, "k--o", ms=3, lw=1, label="planned")
        ax.plot(*c["pos"][:2], "kx", ms=10, mew=2)
        ax.set(
            xlim=(c["pos"][0] - 90, c["pos"][0] + 90),
            ylim=(c["pos"][1] - 90, c["pos"][1] + 90),
            aspect="equal",
        )
        ax.set_title(
            f"seed {c['seed']} R{c['red']} z={c['pos'][2]} route_len={c['route_len']}", fontsize=10
        )
    axes.flat[0].legend(fontsize=8)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def diag_sim(cfg_path: str, out: Path, episodes: int, behaviour: str | None) -> dict:
    """Idle-defender rollouts: outcomes, resolution causes, timing; replays of notable episodes."""
    cfg = swarmsim.EnvConfig.load(cfg_path)
    if behaviour:
        cfg.behaviour_weights = [float(b == behaviour) for b in swarmsim.BEHAVIOURS]
    env = swarmsim.Env(cfg)
    causes: collections.Counter[str] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    hit_seeds: list[int] = []
    miss_seeds: list[int] = []
    crash_seeds: list[int] = []
    lengths: list[float] = []
    crashes: list[dict] = []
    start = time.perf_counter()
    steps = 0
    for seed in range(episodes):
        ep = run_episode(env, seed)
        steps += len(ep["t"]) - 1
        lengths.append(ep["t"][-1])
        (hit_seeds if ep["asset_hit"] else miss_seeds).append(seed)
        for e in ep["events"]:
            kinds[swarmsim.EVENT_KINDS[int(e[1])]] += 1
            if swarmsim.EVENT_KINDS[int(e[1])] == "lost":
                side = "blue" if e[2] < ep["n_blue"] else "red"
                causes[f"{side}:{swarmsim.LOSS_CAUSES[int(e[4])]}"] += 1
                if side == "red":
                    k = int(np.abs(ep["t"] - e[0]).argmin())
                    pos = ep["states"][k, int(e[2]), :3]
                    crash_seeds.append(seed)
                    crashes.append(
                        {
                            "seed": seed,
                            "red": int(e[2]) - ep["n_blue"],
                            "t": round(float(e[0]), 2),
                            "pos": pos.round(1).tolist(),
                            "cause": swarmsim.LOSS_CAUSES[int(e[4])],
                            "route_len": len(ep["routes"][int(e[2]) - ep["n_blue"]]),
                            "route_clearance_m": _route_clearance(ep, int(e[2]), pos[:2]),
                        }
                    )
    wall = time.perf_counter() - start
    report = {
        "behaviour": behaviour or "mix",
        "episodes": episodes,
        "asset_hit_rate": len(hit_seeds) / episodes,
        "not_hit_seeds": miss_seeds[:20],
        "episode_s": dict(
            zip(("min", "p50", "max"), np.percentile(lengths, [0, 50, 100]).round(1).tolist())
        ),
        "events": dict(kinds),
        "losses": dict(causes),
        "red_crashes_off_route": sum(c["route_clearance_m"] > 20 for c in crashes),
        "decisions_per_s_single_thread": round(steps / wall),
    }
    with (out / f"crashes_{report['behaviour']}.jsonl").open("w") as f:
        f.writelines(json.dumps(c) + "\n" for c in crashes)
    crash_gallery(env, crashes, out / f"crashes_{report['behaviour']}.png")
    picks = list(dict.fromkeys(miss_seeds[:4] + crash_seeds[:6] + hit_seeds[:2]))[:10]
    write_html(
        [run_episode(env, s) for s in picks],
        out / f"replay_{report['behaviour']}.html",
        f"Idle defenders, {report['behaviour']}",
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system", choices=["sim"])
    ap.add_argument("--config", default="scenarios/v0/env.yaml")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("experiments/2026-11-04_m1-sim"))
    ap.add_argument("--behaviour", choices=swarmsim.BEHAVIOURS, action="append")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    reports = [diag_sim(a.config, a.out, a.episodes, b) for b in a.behaviour or swarmsim.BEHAVIOURS]
    if not a.behaviour:
        (a.out / "sim.json").write_text(json.dumps(reports, indent=1))
    for r in reports:
        print(json.dumps(r))
