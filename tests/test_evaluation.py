"""Evaluation uses the same real physics and preserves all decision records."""

import json
from pathlib import Path

import numpy as np
import swarmsim
import torch
from swarm_eval.evaluate import episode, summarise
from swarm_eval.replay import write_html
from swarm_policy.model import GQN


def test_native_baseline_matches_python_driver():
    c = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    c.red_min = c.red_max = 3
    c.behaviour_weights = [1, 0, 0]
    for seed in (10000000, 10000001):
        python = episode("pn_greedy", c, seed)
        cpp = swarmsim.evaluate_cpp(c, seed, "pn_greedy")
        for key in python:
            if key != "latency_ms":
                assert python[key] == cpp[key], key


def test_record_and_replay_without_pickle(tmp_path: Path):
    torch.set_num_threads(1)
    c = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    c.max_decisions = 3
    path = tmp_path / "episode.npz"
    row = episode(GQN(), c, 123, path)
    assert row["truncated"]
    with np.load(path, allow_pickle=False) as data:
        assert len(data["states"]) == 4
        assert len(data["decisions"]) == 3
        first = json.loads(data["decisions"][0])
        for key in (
            "q",
            "actions",
            "margin",
            "attention_local",
            "attention_global",
            "attention_teammate",
            "messages",
            "global_delta",
            "message_influence",
            "message_argmax_flip",
        ):
            assert key in first["decision"]
        assert np.asarray(first["decision"]["q"]).shape == (3, 27)
        html = tmp_path / "replay.html"
        write_html([dict(data)], html)
        assert "__DATA__" not in html.read_text()
        assert "global_delta" in html.read_text()
    stats = summarise([row])
    assert stats["truncated"] == 1 and stats["survival"] == 0
    assert stats["survival_ci95"][1] > 0


def test_baseline_recording_has_valid_empty_decisions(tmp_path: Path):
    c = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    c.max_decisions = 2
    path = tmp_path / "idle.npz"
    episode("idle", c, 9, path)
    with np.load(path, allow_pickle=False) as data:
        write_html([dict(data)], tmp_path / "idle.html")
        assert '"decisions":[]' in (tmp_path / "idle.html").read_text()


def test_local_checkpoint_resolution(tmp_path: Path):
    from swarm_eval.tracking import resolve_checkpoint

    (tmp_path / "last.pt").write_bytes(b"last")
    assert resolve_checkpoint(str(tmp_path), tmp_path) == tmp_path / "last.pt"
    (tmp_path / "best.pt").write_bytes(b"best")
    assert resolve_checkpoint(str(tmp_path), tmp_path) == tmp_path / "best.pt"


def test_wandb_run_resolution_prefers_best_model(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    import wandb
    from swarm_eval.tracking import resolve_checkpoint

    (tmp_path / "best.pt").write_bytes(b"model")
    requested = []

    def artifact(version, aliases, kind="model"):
        return SimpleNamespace(
            version=version,
            aliases=aliases,
            type=kind,
            download=lambda root: str(tmp_path),
        )

    best = artifact("v2", ["best"])
    latest = artifact("v3", ["latest"])

    def run(identifier):
        requested.append(identifier)
        return SimpleNamespace(logged_artifacts=lambda: [best, latest, artifact("v9", [], "data")])

    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=run))
    assert (
        resolve_checkpoint("https://wandb.ai/team/project/runs/abc", tmp_path)
        == tmp_path / "best.pt"
    )
    assert requested == ["team/project/abc"]
