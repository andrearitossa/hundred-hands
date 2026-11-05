# SPEC — Hundred Hands, v2: modular extensions

Status: draft for review · Builds on: [`SPEC_v1.md`](SPEC_v1.md) · Audience: engineers and Claude Code

v2 makes the project cooler and closer to real, where **real means ROS 2**. It is not a linear
release: it is a **catalogue of independent modules**, each one a self-contained experiment.

Real hardware flights remain out of scope.

---

## 1. Rules for every module

1. **Opt-in:** off by default behind a config flag. With the flag off, v1 results are unchanged.
2. **One question:** it states what it wants to learn or show.
3. **Own experiment:** its own `experiments/` folder, with results compared with the v1 policy.
4. **Own tests:** in the module, plus the v1 suite still passing.
5. **Demo where possible.**
6. **Declared dependencies:** only those in the catalogue. Pick modules in any order that respects them.

If the observation schema changes, it uses a new `schema_version` and codegen, and earlier versions stay loadable.

Curriculum axes, tests and detailed parameters for each module are in the design references:
[`realism.md`](v2_modules/realism.md) (**R**) and [`resilience.md`](v2_modules/resilience.md) (**S**).

---

## 2. Catalogue

### Sim-to-real (ROS 2)

| Module | Question | Needs | Done when | Design |
| --- | --- | --- | --- | --- |
| `ros2-stack` | Does the same policy work as separate decentralised nodes? | — | One agent process per drone; survival matches in-process within CI; Q-values within 1e-4 | R §11 |
| `rigid-body` | How large is the gap from point-mass to a real quad model? | — | Controller tuned; zero-shot drop and fine-tuned recovery recorded | R §4 |
| `px4-sitl` | Does it still work with a real autopilot in the loop? | `ros2-stack`, `rigid-body` | PX4 lockstep; parity with the internal controller, or an explained gap | S §10 |
| `sim-calibration` | Can ROS 2 logs close the gap? | `px4-sitl` | Gap before and after calibration, with shadow gate and rollback | S §9.2 |
| `netem-demo` | Can we show real packet loss between containers? | `ros2-stack`, `mesh-comms` | Multi-container demo recorded | S §10.1 |

### World and threats

| Module | Question | Needs | Done when | Design |
| --- | --- | --- | --- | --- |
| `osm-cities` | Does the policy transfer to real city layouts? | — | Tile library and held-out cities; results on held-out tiles | R §5.1 |
| `composite-asset` | Does the policy protect a building and its pipeline sensibly? | — | Part-level survival and attention | R §5.2 |
| `asset-networks` | Does it protect what matters most in a linked system? | `composite-asset` | Served-value results; attention vs criticality | S §8 |
| `wind` | How does performance degrade with wind? | — | Robustness curves, policy vs baseline | R §5.3 |
| `coordinated-red` | Do time-on-target attacks break the swarm? | — | Coordinated vs independent results | R §7.2 |
| `fixed-wing` | Can it handle targets that can't hover? | — | With and without fixed-wing | R §7.3 |
| `red-league` | What does the swarm do against an opponent that learns? | — | League and Elo; scripted guardrail | S §6 |
| `scenario-families` | Does it generalise to scenario types never seen? | at least two of the above | Leave-one-family-out matrix | S §8.3–8.4 |

### Perception and comms

| Module | Question | Needs | Done when | Design |
| --- | --- | --- | --- | --- |
| `noisy-sensing` | What survives noise, misses, false alarms and occlusion? | — | Degradation curves; false-alarm attention | R §6 |
| `mesh-comms` | What survives range limits, loss, delay and outages? | — | Resilience curves; per-drone `g_i` emerges from links | S §4 |
| `distributed-fusion` | What happens when drones disagree about the picture? | `mesh-comms` | Duplicate and false-track rates; effect on survival | S §4.3 |
| `learned-association` | Does a learned scorer beat classical association? | `noisy-sensing` | Learned vs classical ablation | S §7 |

### Policy and learning

| Module | Question | Needs | Done when | Design |
| --- | --- | --- | --- | --- |
| `recurrent-gqn` | Does memory help, and what does it store? | — | Recurrent vs feed-forward; state probes | S §5.2 |
| `hierarchical-rejoin` | Does a manager over readable goals improve rejoin? | v1 dropout | Flat vs hierarchical; keep or drop decision | S §5.3 |
| `context-adaptation` | Can a drone infer hidden conditions in flight? | `wind` or `rigid-body` | Gain on shifted conditions; decoded context | S §9.1 |
| `curriculum-1vN` | How far does an automatic 1v1 → N ladder scale? | — | `N_max` from measured compute; advancement record | R §9 |
| `distillation` | Can a small student match a large teacher? | a teacher that beats the deployable size | Teacher, student and scratch compared | S §5.4 |
| `imitation-teacher` | Does imitating the classical expert help or change the mechanism? Inspired by arXiv 2301.09689 | — | Imitation-only and imitation-then-RL vs v1 policy, using the v1 toolkit | New |
| `safety-layer` | What do masks and a filter cost and prevent? | — | Mask and filter tests; with vs without | R §8 |
| `mappo-comparison` | Are the v1 mechanisms specific to the value-based design? | — | MAPPO with the same encoder, in-house and small; v1 toolkit results vs GQN; shapes with empty node sets, permutation invariance, and learning on 3 vs 3 `straight` tested | New |
| `reinforcements` | How does the swarm absorb a new drone that enters mid-episode with an empty tracker? | v1 dropout | Reinforce event through the API and viewer, from a configured entry point; paired branches with and without it | New |

### Cool ablations and experiments

Each is small, and runs on existing checkpoints unless noted.

| Module | Question |
| --- | --- |
| `swarm-scaling` | At what team size does the policy trained at 3 stop coordinating? Evaluate at 2, 4, 5 and beyond |
| `behaviour-holdout` | Does a policy trained without `evasive` cope with `evasive`? (retrains) |
| `failure-prediction` | Can attention entropy, Q-margin and message influence predict asset loss before outcomes show it? Held-out seeds; AUC against a kinematics-only predictor |
| `message-surgery` | Can we make a drone change target by editing a teammate's message? |
| `decapitation` | Does destroying the currently most-attended drone hurt more than a random one? |
| `partition` | What happens when the swarm splits into two groups that can't talk? |
| `liar-drone` | How far does one drone's wrong message spread through the team? |
| `failure-early-warning` | Can the `failure-prediction` predictor trigger a useful real-time response? (needs `failure-prediction`) |
| `reward-archaeology` | Which shaping terms change the mechanism, not only the score? (retrains) |

### Demos

| Module | Shows |
| --- | --- |
| `live-viewer` | Interactive browser replay: attention, messages, Q-values, branch comparison |
| `foxglove-demo` | ROS 2 run with tracks, attention and overrides (`ros2-stack`) |
| `kill-a-drone` | Interactive: click to destroy or disconnect a drone in a replay branch and watch the swarm respond |

---

## 3. Suggested paths

Suggestions only. Any order respecting dependencies is valid.

- **Closest to real:** `ros2-stack` → `rigid-body` → `px4-sitl` → `sim-calibration`.
- **Paper follow-up:** `noisy-sensing` → `mesh-comms` → `recurrent-gqn` → `hierarchical-rejoin`.
- **Quick wins after v1:** the ablations table and `kill-a-drone`.

---

## 4. Non-goals

Real hardware flights · terrain elevation · electronic warfare, jamming, or deliberate comms attacks ·
comms security · human operator in the loop · high-fidelity fixed-wing models · on-board gradient
updates in flight · GPU physics · multiple defended sites per episode.
