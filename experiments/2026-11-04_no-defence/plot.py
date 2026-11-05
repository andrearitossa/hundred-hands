"""Inspect a fixed sampled approach after the no-defence correction."""

from pathlib import Path

import matplotlib
import swarmsim

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

c = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
c.blue_count = 0
c.red_min = c.red_max = 1
c.behaviour_weights = [1, 0, 0]
e = swarmsim.Env(c)
e.reset(13)
s = e.scenario()
states, _ = e.snapshot()
asset = s["asset"]
spawn = states[0, :3]
fig, ax = plt.subplots(figsize=(6, 6))
for x, y, _, hx, hy, _ in s["buildings"]:
    ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, color="gray"))
ax.plot([spawn[0], asset[0]], [spawn[1], asset[1]], "r--", label="clear direct approach")
ax.scatter(*spawn[:2], marker="^", color="red", label="attacker spawn")
ax.scatter(*asset[:2], marker="*", s=150, color="blue", label="asset")
ax.set(
    xlim=(-500, 500),
    ylim=(-500, 500),
    xlabel="x (m)",
    ylabel="y (m)",
    title="Seed 13 after correction: zero defenders, one straight attacker",
)
ax.set_aspect("equal")
ax.legend()
fig.tight_layout()
fig.savefig(Path(__file__).with_name("approach.png"), dpi=140)
