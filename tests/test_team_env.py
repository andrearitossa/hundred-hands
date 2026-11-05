"""Packed graph semantics and lifetime across the Python boundary."""

from pathlib import Path

import numpy as np
import pytest
import swarmsim

ROOT = Path(__file__).resolve().parents[1]


def config():
    return swarmsim.EnvConfig.load(str(ROOT / "scenarios/v0/env.yaml"))


def test_graph_exact_self_and_teammates():
    c = config()
    raw, team = swarmsim.Env(c), swarmsim.TeamEnv(c)
    raw.reset(123)
    graph = team.reset(123)
    states, _ = raw.snapshot()
    asset = raw.scenario()["asset"]
    expected = []
    for state in states[: raw.n_blue]:
        relative = asset - state[:3]
        expected.append(
            [*state[3:] / 15, state[2] / 1000, *relative / 1000, np.linalg.norm(relative) / 1000, 1]
        )
    np.testing.assert_array_equal(graph["self"], np.asarray(expected, dtype=np.float32))
    for node, owner, source in zip(
        graph["teammate"], graph["teammate_owner"], graph["teammate_src"], strict=True
    ):
        delta = states[source, :3] - states[owner, :3]
        np.testing.assert_array_equal(node[:3], np.asarray(delta / 1000, dtype=np.float32))
    for array in graph.values():
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_auto_reset_and_storage_lifetime():
    c = config()
    c.max_decisions = 1
    env = swarmsim.VecEnv(c, 3, 2)
    graph = env.reset(40)
    saved = graph["self"].copy()
    out = env.step([swarmsim.NOOP] * len(saved))
    assert all(t["truncated"] and not t["terminated"] for t in out["transitions"])
    assert all(t["reward"] == 0 for t in out["transitions"])
    np.testing.assert_array_equal(graph["self"], saved)
    packed = out["observation"]
    for owner, source in zip(packed["teammate_owner"], packed["teammate_src"], strict=True):
        assert packed["agent_env"][owner] == packed["agent_env"][source]
    held = graph["self"]
    del env, graph, out
    np.testing.assert_array_equal(held, saved)


def test_invalid_action_does_not_advance_any_world():
    c = config()
    a, b = swarmsim.VecEnv(c, 2, 2), swarmsim.VecEnv(c, 2, 1)
    graph = a.reset(8)
    b.reset(8)
    actions = [swarmsim.NOOP] * len(graph["self"])
    actions[-1] = 27
    with pytest.raises(ValueError):
        a.step(actions)
    actions[-1] = swarmsim.NOOP
    np.testing.assert_array_equal(
        a.step(actions)["observation"]["self"], b.step(actions)["observation"]["self"]
    )


def test_generated_layout_threat_features_at_nonzero_heading():
    """Decode every threat column against independently transformed tracker input."""
    from swarm_policy.obs_layout import DIMS, LAYOUT, POS_NORM, TIME_NORM

    c = config()
    env = swarmsim.TeamEnv(c)
    graph = env.reset(93)
    for _ in range(110):
        graph = env.step([23] * len(graph["self"]))["observation"]
        if len(graph["self"]) == 0:
            pytest.fail("fixture lost every defender")
    states, _ = env.world.snapshot()
    asset = env.world.scenario()["asset"]
    for kind, tracks, owner_key in (
        ("local_threat", env.local_tracks, "local_owner"),
        ("global_threat", [env.fused_tracks] * len(graph["self"]), "global_owner"),
    ):
        assert graph[kind].shape[1] == DIMS[kind]
        seen = 0
        for agent, drone in enumerate(graph["agent_drone"]):
            pos, vel = states[drone, :3], states[drone, 3:]
            psi = env.headings[agent]

            def rotate(v, psi=psi):
                return np.array(
                    [
                        np.cos(psi) * v[0] + np.sin(psi) * v[1],
                        -np.sin(psi) * v[0] + np.cos(psi) * v[1],
                        v[2],
                    ]
                )

            rows = graph[kind][graph[owner_key] == agent]
            assert len(rows) == len(tracks[agent])
            for row, track in zip(rows, tracks[agent], strict=True):
                rel = track.pos - pos
                rv = track.vel - vel
                to_asset = asset - track.pos
                dist = np.linalg.norm(rel)
                ad = np.linalg.norm(to_asset)
                values = {
                    "rel_pos": rotate(rel) / POS_NORM,
                    "rel_vel": rotate(rv) / 15,
                    "dist": dist / POS_NORM,
                    "closing": -np.dot(rv, rel) / dist / 15,
                    "asset_dist": ad / POS_NORM,
                    "asset_speed": np.dot(track.vel, to_asset) / ad / 15,
                    "age": track.age / TIME_NORM,
                    "since_update": track.since_update / TIME_NORM,
                }
                for name, (begin, end, _) in LAYOUT[kind].items():
                    np.testing.assert_array_equal(
                        row[begin:end], np.asarray(values[name], dtype=np.float32).reshape(-1)
                    )
                seen += 1
        assert seen > 0
