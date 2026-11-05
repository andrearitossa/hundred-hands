"""Spec 12: with idle defenders every behaviour hits the asset, without truncation."""

import pytest
import swarmsim
from swarm_eval.episode import run_episode


@pytest.mark.parametrize("behaviour", swarmsim.BEHAVIOURS)
def test_idle_defenders_always_lose(behaviour: str) -> None:
    cfg = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    cfg.behaviour_weights = [float(b == behaviour) for b in swarmsim.BEHAVIOURS]
    env = swarmsim.Env(cfg)
    for seed in range(200):
        ep = run_episode(env, seed)
        assert ep["asset_hit"] and not ep["truncated"], f"seed {seed}"


@pytest.mark.parametrize("behaviour", swarmsim.BEHAVIOURS)
def test_single_attacker_without_defenders_hits_asset(behaviour: str) -> None:
    """A swarm must not hide an individual script's inability to reach the asset."""
    cfg = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
    cfg.blue_count = 0
    cfg.red_min = cfg.red_max = 1
    cfg.behaviour_weights = [float(b == behaviour) for b in swarmsim.BEHAVIOURS]
    env = swarmsim.TeamEnv(cfg)
    for seed in range(200):
        obs = env.reset(seed)
        assert len(obs["self"]) == 0
        for _ in range(cfg.max_decisions):
            result = env.step([])
            if result["terminated"] or result["truncated"]:
                break
        assert result["terminated"] and not result["truncated"], (behaviour, seed, result)
        assert result["reward"] == -1, (behaviour, seed, result)
        assert result["intercepted"] == 0
