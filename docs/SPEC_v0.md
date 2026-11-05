# SPEC — Hundred Hands, v0: build the pieces

Status: v0 scope frozen, refactored Nov 2026 · Next: [`SPEC_v1.md`](SPEC_v1.md) (the paper stack), [`SPEC_v2.md`](SPEC_v2.md) (modular extensions) · Audience: engineers and Claude Code

v0 **constructs and verifies every piece** that v1 needs. It does not produce scientific results.
Build only what is in this document. v0 interfaces must not make v1 or v2 hard to add (§1.2).

---

## 1. Goal

Build the defender swarm's parts: simulation, perception, control and baselines, one shared policy
that each drone runs, the learner, the runtime, and the instrumentation that shows what they do.

v0 is done when every piece below passes its acceptance check (§13), and together they run end to end:

1. **Simulation** obeys the rules, and idle defenders lose.
2. **Perception and graph** produce identical observations in C++ and Python.
3. **Classical baselines** run, with a first baseline table recorded.
4. **The learner works**: on 3 vs 3 `straight`, the GQN clearly beats random and idle. Beating PN + greedy is **not** a v0 goal; that is v1.
5. **Export and runtime**: ONNX parity, C++ evaluation path, latency recorded.
6. **Instrumentation**: every decision's interpretability data (§9) is logged and viewable in a replay.
7. **End-to-end skeleton, documented**: one command path runs the whole chain, and its documentation is complete. **v1 does not start until this is done** (§13.1).

### 1.1 Working method

- Every system has an **observable IN/OUT**: it can be run alone, with inspectable input and output.
- Implement, look at what it does, then iterate from the diagnosis.
- Each milestone ends with a **journal entry** in `experiments/`: what was tested, what was seen, what
  was fixed, assumptions, and a visual demo where possible.

### 1.2 Interfaces v1 relies on

v0 builds these seams, without the v1 features behind them:

- Global-view availability `g` is **per drone**, even though it is always 1 in v0.
- Drones can be absent from any step's graph; no code assumes a constant team size, including `Q_team`.
- All simulation state and randomness are owned by the env instance, so v1 can save and restore it.
- Per-drone state in the tracker and decoder is explicit, not hidden in closures or statics.

---

## 2. Design summary

| Topic | v0 decision |
| --- | --- |
| Learning side | Blue (defenders) learns. Red (attackers) is scripted. |
| Defenders | 3 interceptor quads + one abstract ground sensor at the asset |
| Attackers | 3–6 quads per episode (uniform), scripted: straight, waypoint, evasive |
| Asset | One sphere on the ground, random position each reset |
| World | 1 km × 1 km, flat ground, 5–10 random boxes (buildings) |
| Physics | MuJoCo, **point-mass** drones, limits derived from an X500-class model |
| Disturbances | None (no wind, no noise, no agent failures, ideal comms) |
| Sensing | Detector + Kalman tracker per drone; clean measurements, range-limited |
| Policy | One shared **Graph Q-Network (GQN)**, ego-centric, **no agent IDs** |
| Policy views | **Local branch** (own tracks) + **global branch** (fused swarm tracks + teammate messages) |
| Messages | Learned node embeddings, one synchronous round per decision |
| Value mixing | **VDN**: team Q = sum of per-drone Q |
| Reward | Global team reward only (no per-drone shaping) |
| Actions | 27 discrete ego-frame velocity primitives → cascaded velocity controller |
| Memory | In the tracker only; the GQN is feed-forward |
| Safety layer | None in v0 |
| Rates | Physics 500 Hz · controller 100 Hz · policy 10 Hz · messages 10 Hz |
| Episode end | All threats resolved, or asset hit. No timeout (bug-guard cap only). |
| Languages | C++ for env, sensing, tracking, control, baselines, inference. Python (PyTorch + PyTorch Geometric) for the learner. pybind11 between them. |
| Parallelism | C++ thread pool of CPU MuJoCo envs |
| Learner | Small in-house learner (double DQN + VDN), no RL framework |
| Target hardware | Companion ARM CPU, no GPU. Latency target < 10 ms per decision, set by benchmark. |
| Compute | Laptop for development; Colab Pro or an AWS/GCP GPU instance for training |
| Tracking | Weights & Biases + committed reports in `experiments/` |
| Determinism | Not bit-exact across machines in v0. Seeds are logged. Same-machine reproducible branching is added in v1. |

Reference for the policy design: Bouton et al., *Multi-agent Reinforcement Learning with Graph
Q-Networks for Antenna Tuning* (Ericsson), https://arxiv.org/abs/2302.01199 — shared graph
network, per-agent Q-values, additive (VDN-style) decomposition trained from one global reward,
generalises to more agents than seen in training.

---

## 3. Repository layout

Split by function. Each C++ directory is a separate CMake package that exposes **only a public
interface library** (`include/<pkg>/…`); implementation stays private.

```
/
├── docs/                    # SPEC_v0, SPEC_v1, SPEC_v2, design references
├── CLAUDE.md                # working instructions for Claude Code
├── README.md                # quick start, demo GIF, results table
├── LICENSE                  # Apache-2.0
├── CMakeLists.txt           # top-level superbuild (adds the C++ packages)
├── CMakePresets.json        # debug / release / relwithdebinfo
├── pyproject.toml           # uv workspace; scikit-build-core builds the extension
├── core/                    # C++: shared types (Vec3, EntityState, Track, Config), YAML loading
├── sim/                     # C++: MuJoCo world, entities, point-mass dynamics, red scripts,
│                            #      episode logic, VecEnv thread pool, state logger
├── perception/              # C++: sensor models, detector, Kalman tracker, track fusion,
│                            #      observation-graph builder (uses generated layout)
├── gnc/                     # C++: velocity controller, action-primitive decoder,
│                            #      baselines (pure pursuit, PN, greedy, Hungarian)
├── policy/
│   ├── swarm_policy/        # Python: GQN model, replay, learner, export to ONNX
│   └── runtime/             # C++: ONNX Runtime inference of the GQN
├── bindings/                # pybind11 module `swarmsim` (VecEnv + baselines + runtime)
├── scenarios/
│   ├── schema/obs_schema.yaml   # single source of truth for graph features
│   ├── v0/*.yaml                # env, curriculum, train and eval configs
│   └── assets/x500/             # vendored drone model + derived limits
├── eval/                    # Python: evaluation runner, metrics, interpretability, reports
├── experiments/             # one folder per experiment: config, metrics, report, plots
├── tools/
│   ├── codegen/             # obs_schema.yaml → C++ header + Python module
│   ├── docker/              # Dockerfile for Colab / cloud training
│   └── bench/               # throughput and latency benchmarks
└── ros/                     # reserved for v1 (README only, no code in v0)
```

### Dependency rules (enforced by CMake target visibility and a CI check)

```
core ← sim ← perception ← gnc
            ↖     ↑       ↑
             bindings → policy/runtime
Python: swarm_policy → swarmsim (bindings);  eval → swarm_policy, swarmsim
```

- `core` depends on nothing in the repo.
- Nothing depends on `eval/`, `experiments/`, `tools/` or `ros/`.
- No package includes another package's `src/`. Only `include/<pkg>/` is public.
- Python never re-implements env logic. Observations come from C++.

### Build and tooling

- **C++**: C++20, CMake ≥ 3.24, presets for debug/release. Each package is colcon-compatible
  (own `CMakeLists.txt`) so v1 can add ROS 2 packages beside them.
- **Python**: `uv` workspace, Python 3.11+. The `swarmsim` extension is built by
  `scikit-build-core` from the top-level CMake project, so `uv sync` builds everything.
- **External dependencies** (pin the latest stable at project start; record in `uv.lock` and
  CMake `FetchContent` tags): MuJoCo (C API), Eigen, yaml-cpp, pybind11, ONNX Runtime,
  GoogleTest; PyTorch, PyTorch Geometric, NumPy, onnx, wandb, PyYAML, matplotlib, pytest.
- **Vendored**: the drone model under `scenarios/assets/x500/` only.
- **Licences**: permissive only (Apache/MIT/BSD) in anything built into `swarmsim` or the
  runtime; copyleft is allowed in dev tooling only.
- **Formatting/lint**: clang-format, clang-tidy (core checks), ruff, mypy (policy, eval).
- **CI**: build, unit tests, schema/codegen consistency, C++↔Python tensor equality,
  ONNX parity, dependency-rule check, short smoke training run (≤ 2 min).

### Commands (must work on a clean machine)

```bash
uv sync                                                   # build C++ + install Python
uv run pytest && ctest --preset release                   # all tests
uv run swarm-train  --config scenarios/v0/train.yaml      # train
uv run swarm-eval   --run <wandb-or-path> --suite v0      # evaluate vs baselines
uv run swarm-report experiments/<id>                      # build report.md + plots
uv run swarm-replay experiments/<id>/episodes/<n>.npz     # visual replay
uv run swarm-bench  --what env|latency                    # benchmarks
```

---

## 4. Simulation (`sim/`)

### 4.1 World

| Item | v0 value (configurable in `scenarios/v0/env.yaml`) |
| --- | --- |
| Area | x, y ∈ [−500, 500] m; z ∈ [0, 150] m |
| Ground | Flat plane at z = 0 |
| Buildings | 5–10 axis-aligned boxes; footprint 10–40 m per side; height 10–60 m; no overlap with each other, the asset zone (r < 40 m) or the spawn ring |
| Asset | Sphere, radius 10 m, centred at ground level; position uniform in x, y ∈ [−300, 300] m, clear of buildings |
| Ground sensor | Co-located with the asset |

Everything is randomised at every reset.

### 4.2 Drones (point-mass in MuJoCo)

- Each drone is a MuJoCo body with three slide joints (x, y, z), a small sphere geom
  (radius 0.4 m), and three force actuators. Gravity is compensated by the controller, so the
  controller commands acceleration directly.
- MuJoCo timestep 0.002 s (500 Hz).
- **Limits come from the vendored X500-class model** (derivation in `scenarios/assets/x500/README.md`,
  values in `limits.yaml`): `a_max = 8 m/s²` and `az_max = 4 m/s²` are derived from thrust.
  `v_max = 15 m/s` and `vz_max = 5 m/s` are assumed, because the model has no drag or speed limit.
  An open X500 model may need converting from PX4's Gazebo SDF to MJCF, or adapting an existing
  open quad MJCF to X500 mass, inertia and thrust.
- Attackers use the same model with cruise speed sampled per drone in [6, 12] m/s, always below
  defender `v_max`, so interception is feasible.

### 4.3 Contacts and losses

| Event | Result |
| --- | --- |
| Drone hits ground, building, or leaves the area | Drone removed (lost) |
| Defender–defender contact (distance < 0.8 m) | Both removed |
| Defender–attacker | No physical contact (collision masks); handled by the intercept rule |
| Attacker within asset radius + 2 m | **Asset hit** → episode ends |
| Attacker hits a building | Attacker removed; counts as resolved, not as intercepted |

### 4.4 Intercept rule (abstract, no effector physics)

- Capture radius `r_c = 3 m` (config).
- When a defender **enters** `r_c` of an attacker, draw one Bernoulli trial with
  `p_n = 0.7` (config). On success, the attacker is removed (intercepted). The defender continues.
- After a failed trial, that defender–attacker pair gets no new trial until their distance
  exceeds `2 r_c`. This keeps the rule independent of the step rate.
- Checked at controller rate (100 Hz).

### 4.5 Scripted attackers (red)

All behaviours **always make progress toward the asset**, so defenders that do nothing lose every
episode. This is a hard invariant with a test (§12).

| Behaviour | Description |
| --- | --- |
| `straight` | Fly to the asset at cruise speed and cruise altitude, then dive onto it. |
| `waypoint` | 1–3 random waypoints between spawn and asset, then the asset. Segments are checked against buildings with `mj_ray`; resample until clear. |
| `evasive` | `straight` or `waypoint`, plus: when a defender is within 30 m and closing, apply a lateral or vertical jink (random side, 1 s, ≤ 60° off course), then resume. Jinks have a cooldown of 2 s. |

- Spawn: random points on the area perimeter, altitude 20–60 m, spawn times staggered
  uniformly in [0, 10] s.
- Behaviour mix is a config (`red.behaviour_weights`) so behaviours can be introduced
  iteratively: start with `straight`, then add `waypoint`, then `evasive`.
- Attackers use the same velocity controller as defenders (§6.1).

### 4.6 Defenders (blue)

- 3 defenders. Start hovering at random positions 50–150 m from the asset, altitude 20–50 m.
- Every 0.1 s, each alive defender receives one discrete action (§6.2) from the policy or a
  baseline.

### 4.7 Episode logic

- **Terminates** when (a) every attacker is resolved (intercepted or crashed), or (b) the asset
  is hit.
- **No timeout.** A bug-guard cap of 3000 decision steps (300 s) truncates the episode, logs a
  warning with the seed, and is counted in metrics as `truncated`. It is never part of the
  learning signal design.
- If all defenders are lost, the episode continues until the attackers resolve it (asset hit).

### 4.8 Vectorised environment

- `VecEnv` owns `N` independent environments and a fixed thread pool. `step(actions)` advances all
  envs by one decision step (50 physics steps, 10 controller steps) in parallel and auto-resets
  finished envs.
- Returns packed observation graphs (§5.6) as contiguous arrays exposed to Python **without
  copies** (pybind11 buffers).
- Typical `N`: 16–32 on a laptop or Colab, 64–256 on a cloud instance. Measured by
  `swarm-bench --what env` (decision steps per second per core).

### 4.9 Logging and replay

- Optional per-episode state log (positions, velocities, events, actions, and the policy
  interpretability data from §9) saved as `.npz`.
- `swarm-replay` renders a log in the MuJoCo viewer, with a 2D top-down matplotlib/HTML fallback
  for headless machines and Colab.

---

## 5. Perception (`perception/`)

v0 sensing is **clean** (no noise, no misses, no false alarms) but **range-limited**, and the policy
never sees ground truth directly. Everything goes through detector → tracker → graph.

### 5.1 Sensors

| Sensor | Range | Field of view | Occlusion by buildings |
| --- | --- | --- | --- |
| Defender onboard | 300 m | Full sphere | Config flag, **off** in v0 |
| Ground sensor at asset | 500 m | Hemisphere above ground | Config flag, **off** in v0 |

The sensors detect **non-cooperative** objects only (attackers). Teammates are known through
comms, not through sensors. Sensing runs at 10 Hz.

### 5.2 Detector

v0: a detection is the true position of each attacker within range, at the sensing time.
The interface already carries a covariance and a confidence (`p_hostile = 1.0` in v0) so v1 noise
and classification drop in without interface changes.

### 5.3 Tracker (memory lives here)

- One tracker per defender, plus one for the ground sensor.
- Constant-velocity Kalman filter per track; gated nearest-neighbour association.
- A track **coasts for 2 s** after its last update, then is dropped.
- Track state: position, velocity, covariance, age, time since last update.
- Track IDs are internal to the tracker. They are **never** fed to the policy.

### 5.4 Fusion (global view)

- Ideal comms in v0: the global picture is the union of all alive defenders' tracks and the
  ground sensor's tracks.
- De-duplicate by proximity: merge tracks within 5 m and 3 m/s of each other (average the state,
  keep the freshest update time).
- Every alive defender receives the same fused picture each decision step.

### 5.5 Observation graph (per defender, ego-centric)

All positions and velocities are expressed in the defender's **ego frame**: translated to the
defender's position and rotated by its current heading ψ (yaw of the velocity setpoint; z stays
up). Normalisation: positions ÷ 1000 m, velocities ÷ `v_max`, times ÷ 10 s.

**Self features**

| Feature | Dim |
| --- | --- |
| Own velocity (ego frame) | 3 |
| Altitude | 1 |
| Vector to asset (ego frame), distance to asset | 4 |
| Global view available flag `g` (always 1 in v0) | 1 |

**Local branch nodes**

| Node kind | Source | Features |
| --- | --- | --- |
| `local_threat` | Own tracker | rel. position (3), rel. velocity (3), distance (1), closing speed (1), threat→asset distance (1), threat speed toward asset (1), track age (1), time since update (1) |
| `asset` | Mission data | rel. position (3), distance (1) |

**Global branch nodes**

| Node kind | Source | Features |
| --- | --- | --- |
| `global_threat` | Fused picture | Same features as `local_threat` |
| `teammate` | Comms | rel. position (3), rel. velocity (3), distance (1), distance to asset (1), plus the teammate's message embedding (`d_m = 16`) |

- **Uncapped**: every node in range or connected is included. Variable counts are handled with
  segment indices, not padding to a fixed maximum.
- **No IDs and no type tokens.** The node kind is given by which attention block the node goes
  into (relation-typed attention).
- Dead or removed entities are simply absent.

### 5.6 Schema and code generation

- `scenarios/schema/obs_schema.yaml` defines every feature: name, kind, dimension, normalisation,
  and order.
- `tools/codegen` generates `perception/include/perception/obs_layout.hpp` and
  `policy/swarm_policy/obs_layout.py`. Generated files are committed and CI fails if they are
  stale.
- **Packed batch format** (per decision step, across all envs and agents): one float array per
  node kind (`[num_nodes_of_kind, feat_dim]`), plus an `int64` array mapping each node to its
  owning agent, an agent → env index, and alive masks. Teammate nodes reference the teammate's
  agent index, so messages can be gathered in the same step.
- CI test: for fixed seeds, the C++ graph builder's arrays decode identically through the
  generated Python layout (exact equality).

---

## 6. Guidance, navigation and control (`gnc/`)

### 6.1 Velocity controller (100 Hz)

- Input: a velocity setpoint `v_cmd` (world frame).
- Output: acceleration `a = K_v (v_cmd − v)`, clipped to `a_max` (horizontal) and the vertical
  limit; gravity compensation added; applied as actuator force.
- `K_v` default 2.0 s⁻¹ (config). Same controller for defenders and attackers.

### 6.2 Action primitives (policy output → setpoint)

27 discrete actions = 3 turn × 3 climb × 3 speed, relative to the defender's current setpoint:

| Axis | Options (per 0.1 s decision) |
| --- | --- |
| Turn (heading ψ) | −20°, 0, +20° |
| Climb (vertical speed) | −3, 0, +3 m/s |
| Speed (horizontal) | −3, 0, +3 m/s, clipped to [0, `v_max`] |

The decoder keeps per-defender state (ψ, horizontal speed) so heading stays defined at zero
speed. Magnitudes are config values, to be tuned once in M3.

### 6.3 Baselines (nice to have, cut any that cost too much)

Baselines act on the **same fused track picture** the policy's global branch sees, and output a
continuous velocity setpoint directly into the controller.

| Baseline | Rule |
| --- | --- |
| Pure pursuit (PP) | `v_cmd = v_max · unit(track_pos − own_pos)` |
| Proportional navigation (PN) | Lateral acceleration `N · V_c · λ̇` with `N = 3`, integrated into `v_cmd`, speed held at `v_max` |
| Greedy assignment | Each defender takes the unassigned threat with the lowest estimated time-to-intercept; repeat until defenders or threats run out |
| Hungarian assignment | Optimal one-to-one assignment on time-to-intercept cost (in-house implementation, about 100 lines) |

- Re-assign at 1 Hz or when a threat appears or disappears.
- Unassigned defenders guard: hold a point 60 m from the asset on the bearing of the nearest
  threat.
- Combinations evaluated: {PP, PN} × {greedy, Hungarian}. The **primary comparison is PN +
  greedy**.
- Also report a **random policy** and an **idle policy** (defenders hover) as sanity floors.

---

## 7. Policy: Graph Q-Network (`policy/`)

### 7.1 Architecture (per defender, shared weights)

```
self features ─────────────── MLP ─▶ h_self (32)

LOCAL BRANCH
  local_threat nodes ─ MLP ─▶ keys/values ─┐
                                           ├─ MHA(query=h_self, 4 heads) ─▶ z_lt (32)
  asset node ──────── MLP ─────────────────────────────────────────────▶ z_as (32)
  local = MLP([h_self, z_lt, z_as]) ─▶ (32)

MESSAGE  m_i = MLP(local) ─▶ (16)          # broadcast to teammates (phase 1)

GLOBAL BRANCH (phase 2)
  global_threat nodes ─ MLP ─▶ MHA(query=h_self, 4 heads) ─▶ z_gt (32)
  teammate nodes [features ⊕ m_j] ─ MLP ─▶ MHA(query=h_self, 4 heads) ─▶ z_tm (32)
  global = g · MLP([z_gt, z_tm]) ─▶ (32)    # g = availability flag (1 in v0)

Q-HEAD  Q_i = MLP([local, global, g]) ─▶ 27
```

- **Relation-typed attention**: one attention block per node kind (local threats, global
  threats, teammates). Empty node sets output a learned "empty" vector.
- **Messages**: one synchronous round per decision. Phase 1 computes and shares `m_i`;
  phase 2 runs the global branch. In training, the whole team is processed in one forward
  pass, so gradients flow through messages (as in the GQN paper).
- **Feed-forward** only. No recurrence in v0.
- Hidden size 32, 4 heads, ReLU. Expected size: tens of thousands of parameters. The final size
  is set by the latency benchmark (§10), not by this number.
- Implemented with PyTorch Geometric scatter/segment ops (or plain `torch` index ops), so that
  uncapped variable node counts batch without padding.

### 7.2 Value decomposition and objective

- Team value: `Q_team(s, a) = Σ_{i ∈ alive} Q_i(o_i, a_i)` (VDN).
- Each drone acting greedily on its own `Q_i` maximises `Q_team`, so execution is decentralised.
- **Double DQN** target with **n-step returns**, using the online network for argmax and the
  target network for value, per agent:
  `y = Σ_{k<n} γ^k r_{t+k} + γ^n Σ_{i ∈ alive(t+n)} Q_i^target(o'_i, argmax_a Q_i^online(o'_i, a))`
- Agents that die mid-episode drop out of the sum from that step on. Termination zeroes the
  bootstrap; the bug-guard truncation bootstraps.
- Huber loss on `Q_team − y`.

### 7.3 Reward (global, team-level only)

```
r_t = (1 / N_red) · (# attackers intercepted at t)
      − 1.0 · [asset hit at t]
      + 1.0 · [episode ends at t with the asset intact]
```

- No per-drone terms and no distance shaping. If learning stalls, the documented fallback is
  shared (still team-level) shaping, tried as a separate, logged experiment.
- Attackers that crash into buildings give no reward.

### 7.4 Learner (in-house, `policy/swarm_policy/`)

Keep it small and readable (target ≤ 1,500 lines for model + replay + learner + export).

| Component | v0 choice (config `scenarios/v0/train.yaml`) |
| --- | --- |
| Actors | `VecEnv` from `swarmsim`; batched forward on the learner device; ε-greedy per agent |
| Exploration | ε from 1.0 to 0.05 over the first 30 % of steps |
| Replay | Uniform, capacity 1M team-transitions (packed graphs stored as arrays); prioritised replay is an optional flag |
| Optimiser | Adam, lr 5e-4, grad-norm clip 10 |
| Batch | 64 team-transitions |
| Discount | γ = 0.99, n = 5 |
| Target network | Soft update τ = 0.005 |
| Updates | 1 gradient step per 4 decision steps (config) |
| Budget | Set per experiment; start with 5M decision steps |
| Checkpoints | Every 250k steps, plus best by eval survival rate |
| Eval during training | Every 100k steps: 200 episodes, greedy policy, fixed eval seeds |

### 7.5 Curriculum (built in v0, run to completion in v1)

Stages advance when greedy-policy survival exceeds a threshold (config, default 0.7) over 200
eval episodes, or after a step budget:

1. 3 vs 3–6, `straight` only
2. + `waypoint`
3. + `evasive` (final v0 mix: equal weights)

The full 1v1 → N curriculum is v1.

### 7.6 Export and runtime

- Export the trained GQN to **ONNX** (dynamic node counts).
- `policy/runtime/` runs it in C++ with ONNX Runtime from the same packed graph format.
- Parity test: C++ ONNX vs PyTorch Q-values, max absolute difference < 1e-4, on 1,000 logged
  observations.
- `swarm-eval` can run evaluation entirely in C++ (runtime + baselines) for speed.

---

## 8. Evaluation and metrics (`eval/`)

v0 builds the runner and metrics, and uses them for the baseline table (M3) and learner check (M4).
The full suite against a fully trained policy is a v1 experiment.

### 8.1 Metrics (per episode, aggregated with mean and 95 % CI over ≥ 1,000 episodes)

| Metric | Definition |
| --- | --- |
| **Asset survival rate** (primary) | Share of episodes ending with the asset intact |
| Threats neutralised before the asset | Intercepted attackers ÷ total attackers |
| Defender survival | Defenders alive at the end ÷ defenders at start |
| Time to first intercept | Seconds from first attacker spawn |
| Time to clear | Seconds until all attackers are resolved (survived episodes) |
| Episode length | Seconds |
| Truncated episodes | Count (should be 0) |
| Decision latency | Policy forward time per drone (p50, p99) |

### 8.2 Evaluation suite `v0`

- Fixed list of eval seeds (`scenarios/v0/eval.yaml`), never used in training.
- Conditions: red count 3, 4, 5, 6; behaviour `straight`, `waypoint`, `evasive`, mixed.
- Policies: GQN, PP/PN × greedy/Hungarian, random, idle.
- Output: one table (policy × condition × metrics), plus plots.

### 8.3 Ablations

Moved to v1 (`SPEC_v1.md` §6), as causal interventions. v0 only makes them possible: `g` and message zeroing are inputs to the model.

---

## 9. Interpretability

Logged for every decision of every defender in evaluation episodes, saved into the episode `.npz`:

| Item | Content |
| --- | --- |
| Q-values | All 27 values, chosen action, margin to the second-best action |
| Attention | Weights per head for each block: local threats, global threats, teammates |
| Message influence | For each teammate j: change in `Q_i` when `m_j` is zeroed (L2 norm and change of argmax) |
| Global-view influence | Change in `Q_i` when `g = 0` |

Episode-level analysis (built in v1; v0 only has to log the data above and show it in a replay):

- **Implied assignment** over time: which threat each defender attends to most, compared with
  the Hungarian assignment at the same moment.
- Coverage: threats with no defender attending, and duration.
- Coordination events: two defenders attending the same threat, and how it resolved.
- Timeline plot per episode: distances to threats, chosen primitives, Q-margins, intercept
  events.
- **Replays** with overlays: attention lines from each defender to threats, colour-coded by
  weight (MuJoCo viewer and top-down 2D/HTML).

---

## 10. Latency and size budget

- Target: **< 10 ms per decision** per drone on a companion ARM CPU (no GPU). This includes
  graph building and GQN inference; tracking is reported separately.
- Benchmark host: Raspberry Pi 5 (Cortex-A76) if available, otherwise an AWS Graviton instance
  pinned to one core as a proxy.
- `swarm-bench --what latency` sweeps hidden size {16, 32, 64} × node counts up to 2× the v0
  maximum (12 threats, 4 teammates), and reports p50/p99.
- The chosen size is the largest that meets the target at p99 with 2× headroom on the v0 worst
  case. Record the result in `experiments/`.

---

## 11. Experiments, tracking and reports

```
experiments/
└── 2026-11-20_gqn-v0-baseline/
    ├── README.md        # question, hypothesis, outcome (3–10 lines)
    ├── config.yaml      # full resolved config
    ├── meta.json        # git SHA, W&B run URL, hardware, seeds, duration
    ├── metrics.json     # eval-suite results
    ├── report.md        # generated by swarm-report: tables, plots, interpretation
    ├── plots/
    └── episodes/        # a few small .npz replays for the demo (large artefacts stay in W&B)
```

- Every training or eval run logs to **Weights & Biases**. Models are registered and tagged with
  the git SHA.
- Every result cited anywhere (README, demo, report) links to its `experiments/` folder.
- The README results table is regenerated from `experiments/*/metrics.json`.

---

## 12. Tests (minimum)

| Area | Test |
| --- | --- |
| Invariant | With idle defenders, **every** attacker behaviour hits the asset in 100 % of 200 seeded episodes, with no truncation |
| Sim | Point-mass respects `v_max`/`a_max`; contact and loss rules; intercept rule (statistical, p ≈ `p_n`); spawn and asset placement constraints |
| Red scripts | Waypoint segments clear of buildings; evasive cooldown; progress toward the asset over time |
| Perception | Tracker reproduces constant-velocity motion exactly on clean data; coast and drop timing; fusion de-duplication |
| Graph | Ego-frame transform; uncapped counts; codegen layout matches; C++ ↔ Python exact equality |
| GNC | Controller step response; primitive decoder; PN on a non-manoeuvring target intercepts; Hungarian matches brute force on small cases |
| Policy | Shapes with empty node sets; VDN target on a toy batch; permutation invariance (shuffling nodes doesn't change Q) |
| Export | ONNX parity < 1e-4 |
| Smoke | 2-minute training run finishes and the loss decreases |

---

## 13. Milestones

Each milestone ends with its acceptance check passing in CI (or recorded in `experiments/` for
results).

| # | Milestone | Acceptance check |
| --- | --- | --- |
| M0 | Repo skeleton | `uv sync`, tests and CI green on Linux (macOS if available); codegen round-trip works; dependency check passes |
| M1 | Simulation core | World, drones, asset, buildings, red behaviours, episode logic, replay; **idle-defender invariant passes** |
| M2 | Perception and graph | Sensors, tracker, fusion, graph builder, packed batches, VecEnv; C++ ↔ Python equality; env throughput recorded |
| M3 | GNC and baselines | Controller, 27 primitives (magnitudes tuned), PP/PN/greedy/Hungarian; first baseline table recorded |
| M4 | Learner | GQN, VDN, replay, training loop, curriculum code; on 3 vs 3 `straight`, clearly beats random and idle |
| M5 | Export and runtime | ONNX parity; C++ eval path; latency sweep recorded (x86 here; ARM when hardware is available) |
| M6 | Instrumentation | Interpretability logging; replay overlays; every piece reachable through `swarm-*` commands; repo clean |
| M7 | End-to-end skeleton | §13.1 walkthrough passes from a clean checkout, and its documentation is complete |

**v0 is done** when M7 is complete. Each milestone has a journal entry in `experiments/`. **This is the gate to v1.**

### 13.1 End-to-end skeleton

One short, scripted run goes through every piece, using the real code path with no stubs:

`uv sync` → tests → train a small policy → checkpoint → ONNX export → C++ runtime evaluation against baselines → interpretability-logged episodes → report and replay.

It proves the pieces connect, not that the policy is good. Run sizes are small enough for this laptop.

Documentation required before v1:

| Document | Content |
| --- | --- |
| `README.md` | What the project is, quick start, and the walkthrough command |
| `docs/ARCHITECTURE.md` | Data flow through the chain: what goes in and out of each piece, its command, and how to inspect it |
| `experiments/<date>_m7-end-to-end/` | Journal entry from the actual run: commands, outputs, replay, duration, hardware, git SHA |
| Known limits | What is assumed, untested or deferred, including deviations from this spec |

Acceptance: following only the README on a clean checkout reproduces the walkthrough. CI runs a shortened version.

---

## 14. Non-goals for v0

Everything in `SPEC_v1.md` and `SPEC_v2.md`: multi-seed scientific results, dropout, snapshot-and-branch,
probes, MAPPO, rigid-body dynamics, noise, wind, ROS 2, and the rest of the v2 catalogue.

---

## 15. Code style

- Small, readable modules; no speculative abstractions; no frameworks where 100 lines will do.
- C++: RAII, no raw `new`/`delete`, `const` by default, Eigen for maths, no exceptions across
  the pybind11 boundary without translation.
- Python: typed, dataclasses for configs, no global state; one config object passed down.
- All numbers in this spec live in YAML configs, never hard-coded.
- Public functions carry a one-line docstring or comment saying what they do and their units.
- Units: SI everywhere (m, s, m/s, rad internally; degrees only in configs and reports).

## 16. Open items to resolve during v0

| Item | Resolved in |
| --- | --- |
| X500-class limits | Resolved in M1 for acceleration; `v_max`, `vz_max` remain assumptions |
| Where `VecEnv` lives: it composes `sim`, `perception` and `gnc`, so it cannot sit in `sim` under the dependency rules | M2 |
| Primitive magnitudes (turn, climb, speed steps) | M3 |
| `p_n` and `r_c` giving a meaningful but non-trivial baseline survival (aim: PN + greedy in the 40–80 % range) | M3 |
| Number of parallel envs per platform | M2 |
| Learner hyperparameters for the M4 check | M4 |
| Candidate network size | M5 (final choice in v1) |
