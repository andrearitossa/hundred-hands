"""One short real training → export → C++ evaluation → decision replay → report chain."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import swarmsim
from swarm_policy.export import check, export
from swarm_policy.train import TrainConfig, load, train

from .evaluate import episode, summarise
from .replay import write_html
from .report import report


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/2026-11-04_m7-end-to-end"))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = TrainConfig(
        steps=args.steps,
        envs=8,
        warmup=256,
        batch=32,
        update_every=16,
        eval_every=args.steps,
        eval_episodes=args.episodes,
        checkpoint_every=args.steps,
        replay_capacity=20000,
    )
    metrics = train(cfg, args.out / "train")
    model = load(args.out / "train" / "last.pt")
    env_cfg = swarmsim.EnvConfig.load(cfg.env)
    env_cfg.red_min = env_cfg.red_max = 3
    env_cfg.behaviour_weights = [1, 0, 0]
    env = swarmsim.TeamEnv(env_cfg)
    graph = env.reset(cfg.eval_seed)
    model_path = args.out / "train" / "last.onnx"
    export(model, graph, model_path)
    graphs = []
    seed = cfg.eval_seed
    for _ in range(1000):
        graphs.append(graph)
        result = env.step([13] * len(graph["self"]))
        graph = result["observation"]
        if result["terminated"] or result["truncated"]:
            seed += 1
            graph = env.reset(seed)
    parity = check(model, model_path, graphs)
    np.savez_compressed(
        args.out / "parity-observations.npz",
        graphs=np.array([json.dumps({k: v.tolist() for k, v in g.items()}) for g in graphs]),
    )
    (args.out / "parity.json").write_text(json.dumps(parity, indent=2))
    from .tracking import Run

    eval_run = Run(
        args.out / "evaluation",
        {"seed": cfg.eval_seed, "episodes": args.episodes},
        cfg.wandb_mode,
        "eval",
    )
    results = []
    for name in ("idle", "random", "pp_greedy", "pn_greedy", "pp_hungarian", "pn_hungarian", "gqn"):
        rows = [
            swarmsim.evaluate_cpp(env_cfg, cfg.eval_seed + i, name, str(model_path))
            for i in range(args.episodes)
        ]
        results.append({"policy": name, "red": 3, "behaviour": "straight", **summarise(rows)})
    eval_run.log({row["policy"] + "/survival": row["survival"] for row in results})
    eval_run.finish()
    episode_path = args.out / "episodes" / "gqn.npz"
    episode(model, env_cfg, cfg.eval_seed, episode_path)
    write_html(
        [dict(np.load(episode_path))], args.out / "replay.html", "v0 end-to-end decision replay"
    )
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "evaluation": results,
                "training": metrics,
                "parity": parity,
                "duration_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )
    meta = json.loads((args.out / "train" / "meta.json").read_text())
    meta.update(
        duration_s=time.perf_counter() - started,
        finished_at=datetime.now(UTC).isoformat(),
        command=[
            "swarm-walkthrough",
            "--out",
            str(args.out),
            "--steps",
            str(args.steps),
            "--episodes",
            str(args.episodes),
        ],
        eval_seed=cfg.eval_seed,
        parity_observations=1000,
        scope="integration only; does not establish M4 learner performance",
    )
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    for name in ("config.yaml", "source-manifest.json"):
        shutil.copy2(args.out / "train" / name, args.out / name)
    report(args.out)
    (args.out / "README.md").write_text(
        "# v0 end-to-end run\n\nReal VecEnv training, checkpoint, ONNX export, 1,000-observation C++ parity, native evaluation against six baselines, full decision NPZ and HTML replay.\n\nSee report.md, metrics.json, train/config.yaml and train/meta.json. This small run verifies integration, not learner performance.\n"
    )
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
