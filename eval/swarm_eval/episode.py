"""Run one episode through a Python controller and record it as a replayable dict of arrays."""

from collections.abc import Callable

import numpy as np
import swarmsim

Controller = Callable[[swarmsim.Env], np.ndarray]


def idle(env: swarmsim.Env) -> np.ndarray:
    """Hover: zero velocity setpoint for every defender."""
    return np.zeros((env.n_blue, 3))


def run_episode(env: swarmsim.Env, seed: int, controller: Controller = idle) -> dict:
    """Play `seed` to completion; returns states [T, n, 6], statuses [T, n], events and scenario."""
    env.reset(seed)
    states, statuses, times = [], [], []
    truncated = asset_hit = False
    intercepted = 0
    while True:
        s, st = env.snapshot()
        states.append(s)
        statuses.append(st)
        times.append(env.time)
        r = env.step(controller(env))
        intercepted += r["intercepted"]
        asset_hit |= r["asset_hit"]
        truncated |= r["truncated"]
        if r["done"]:
            break
    s, st = env.snapshot()
    states.append(s)
    statuses.append(st)
    times.append(env.time)
    sc = env.scenario()
    return {
        "seed": seed,
        "n_blue": env.n_blue,
        "n_red": env.n_red,
        "t": np.array(times),
        "states": np.stack(states),
        "status": np.stack(statuses),
        "events": np.array(env.events(), dtype=float).reshape(-1, 5),
        "asset": np.asarray(sc["asset"]).ravel(),
        "buildings": np.array(sc["buildings"]).reshape(-1, 6),
        "red": np.array(sc["red"]).reshape(-1, 3),
        "routes": env.waypoints(),
        "asset_hit": asset_hit,
        "truncated": truncated,
        "intercepted": intercepted,
    }
