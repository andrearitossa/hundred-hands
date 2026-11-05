"""Render real TeamEnv episodes to short 3D perspective GIFs for the v0 review."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import swarmsim
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from swarm_policy.model import tensors
from swarm_policy.train import load


def run_episode(cfg, seed: int, policy: str, model=None) -> dict:
    """Run one real TeamEnv episode and return world snapshots plus event log."""
    env = swarmsim.TeamEnv(cfg)
    graph = env.reset(seed)
    scenario = env.world.scenario()
    states, statuses, times = [], [], []
    while True:
        state, status = env.world.snapshot()
        states.append(state.copy())
        statuses.append(status.copy())
        times.append(env.time)
        if policy == "idle":
            result = env.step([swarmsim.NOOP] * len(graph["self"]))
        elif policy == "pn_greedy":
            result = env.step_baseline(
                swarmsim.Guidance.PN, swarmsim.Assignment.GREEDY
            )
        elif policy == "random":
            rng = np.random.default_rng(seed + 7)
            actions = rng.integers(27, size=len(graph["self"])).tolist()
            result = env.step(actions)
        elif policy == "gqn":
            assert model is not None
            with torch.no_grad():
                actions = model(*tensors(graph))[0].argmax(-1).cpu().numpy()
            result = env.step(actions.tolist())
        else:
            raise ValueError(policy)
        graph = result["observation"]
        if result["terminated"] or result["truncated"]:
            break
    state, status = env.world.snapshot()
    states.append(state.copy())
    statuses.append(status.copy())
    times.append(env.time)
    events = [tuple(e) for e in env.world.events()]
    status_array = np.asarray(statuses)
    return {
        "seed": seed,
        "policy": policy,
        "scenario": scenario,
        "states": np.asarray(states),
        "status": status_array,
        "times": np.asarray(times),
        "events": events,
        "asset_hit": any(int(e[1]) == 4 for e in events),
        "truncated": bool(result["truncated"]),
        "defender_survival": float(np.mean(status_array[-1, :3] == 1)),
    }


def choose(cfg, policy: str, model=None, seeds=range(30)) -> dict:
    """Pick the clearest demonstration: idle loses; a baseline succeeds with survivors."""
    candidates = []
    fallback = None
    for seed in seeds:
        episode = run_episode(cfg, seed, policy, model)
        if fallback is None:
            fallback = episode
        if policy == "idle" and episode["asset_hit"] and not episode["truncated"]:
            candidates.append(episode)
        if policy != "idle" and not episode["asset_hit"] and not episode["truncated"]:
            candidates.append(episode)
    if not candidates:
        return fallback
    if policy == "idle":
        return min(candidates, key=lambda e: len(e["times"]))
    return max(candidates, key=lambda e: (e["defender_survival"], -len(e["times"])))


def add_box(ax, center, half) -> None:
    """Add an axis-aligned 3D building box."""
    cx, cy, cz = center
    hx, hy, hz = half
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    mesh = Poly3DCollection(
        [vertices[index] for index in faces],
        facecolor="#bfc0b8",
        edgecolor="#8e8f88",
        linewidth=0.55,
        alpha=0.9,
    )
    ax.add_collection3d(mesh)


def add_ground(ax) -> None:
    """Add a pale square ground plane at z=0."""
    plane = np.array([[-500, -500, 0], [500, -500, 0], [500, 500, 0], [-500, 500, 0]])
    ground = Poly3DCollection(
        [plane], facecolor="#e9e9df", edgecolor="#d3d3c8", linewidth=0.5, alpha=0.55
    )
    ax.add_collection3d(ground)


def render(episode: dict, path: Path, title: str, max_frames: int = 80) -> None:
    """Draw selected frames in 3D with buildings, altitude, drones and trails."""
    s = episode["states"]
    status = episode["status"]
    times = episode["times"]
    buildings = episode["scenario"]["buildings"]
    asset = episode["scenario"]["asset"]
    n = s.shape[1]
    # Include the initial state and the final event; subsample long episodes.
    picks = np.linspace(0, len(times) - 1, min(len(times), max_frames)).round().astype(int)
    picks = np.unique(picks)
    frames = []
    for frame_index, index in enumerate(picks):
        fig = plt.figure(figsize=(6.8, 6.4), dpi=90)
        ax = fig.add_axes([0.02, 0.02, 0.94, 0.88], projection="3d")
        ax.set_xlim(-520, 520)
        ax.set_ylim(-520, 520)
        ax.set_zlim(0, 160)
        ax.set_box_aspect((1, 1, 0.44))
        ax.view_init(elev=24, azim=-60 + 6 * frame_index / max(1, len(picks) - 1))
        ax.set_facecolor("#f4f4ef")
        ax.set_xlabel("x (m)", labelpad=4)
        ax.set_ylabel("y (m)", labelpad=4)
        ax.set_zlabel("z (m)", labelpad=4)
        ax.grid(color="#d8d8cf", linewidth=0.5, alpha=0.65)
        add_ground(ax)
        for box in buildings:
            add_box(ax, box[:3], box[3:])
        # The asset is at ground level; use a marker plus a vertical stem to show its position.
        ax.scatter(
            asset[0], asset[1], asset[2], s=150, color="#2a8a57",
            edgecolor="#175c38", linewidth=0.8, depthshade=False, zorder=10
        )
        ax.plot(
            [asset[0], asset[0]], [asset[1], asset[1]], [0, 25],
            color="#2a8a57", alpha=0.45, linewidth=1.2
        )
        for drone in range(n):
            colour = "#2166c4" if drone < 3 else "#c8372d"
            halo = "#6aa5f2" if drone < 3 else "#ef6f64"
            alive = np.flatnonzero(status[: index + 1, drone] == 1)
            if len(alive):
                points = s[alive, drone]
                ax.plot(
                    points[:, 0], points[:, 1], points[:, 2],
                    color=halo, alpha=0.5, linewidth=1.3
                )
            if status[index, drone] == 1:
                x, y, z = s[index, drone, :3]
                ax.plot([x, x], [y, y], [0, z], color=colour, alpha=0.18, linewidth=0.8)
                ax.scatter(
                    x, y, z, s=55 if drone < 3 else 42, color=colour,
                    edgecolor="white", linewidth=0.7, depthshade=False, zorder=10
                )
                label = f"B{drone}" if drone < 3 else f"R{drone - 3}"
                ax.text(x, y, z + 7, label, fontsize=7, color=colour, zorder=11)
        visible_events = [
            e
            for e in episode["events"]
            if float(e[0]) <= times[index] and int(e[1]) in (2, 3, 4)
        ]
        if visible_events:
            latest = visible_events[-1]
            ax.set_title(
                f"{title}\nt={times[index]:.1f}s · last event: "
                f"{swarmsim.EVENT_KINDS[int(latest[1])]}",
                fontsize=11,
            )
        else:
            ax.set_title(f"{title}\nt={times[index]:.1f}s", fontsize=11)
        fig.canvas.draw()
        image = Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")
        frames.append(image)
        plt.close(fig)
    # Hold the final result for a moment so the outcome is readable.
    frames.extend([frames[-1]] * 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    print(path, f"{len(frames)} frames", f"{path.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/2026-11-04_v0-review/gifs")
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    cfg.red_min = cfg.red_max = 3
    cfg.behaviour_weights = [1, 0, 0]
    idle = choose(cfg, "idle", seeds=[args.seed] if args.seed is not None else range(30))
    render(idle, args.out / "idle_asset_hit.gif", "Idle defenders · 3 straight attackers")
    pn = choose(cfg, "pn_greedy")
    render(pn, args.out / "pn_greedy_defence.gif", "PN + greedy baseline · 3 straight attackers")
    # Current learner diagnostic, not a success claim: use the M7 integration checkpoint.
    model = load("experiments/2026-11-04_m7-end-to-end/train/last.pt")
    gqn = run_episode(cfg, 10_000_000, "gqn", model)
    render(gqn, args.out / "gqn_current_policy.gif", "Current GQN policy · integration checkpoint")
    for episode in (idle, pn, gqn):
        print(
            f"{episode['policy']} seed={episode['seed']} "
            f"asset_hit={episode['asset_hit']} truncated={episode['truncated']} "
            f"defender_survival={episode.get('defender_survival', float('nan')):.2f} "
            f"duration={episode['times'][-1]:.1f}s"
        )


if __name__ == "__main__":
    main()
