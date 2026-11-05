> **Design reference for [v2 modules](../SPEC_v2.md).** This is the former "v1" spec, kept verbatim for its technical detail.
> Its goals, version labels, milestones and fixed ordering are superseded by `SPEC_v2.md`. Forced dropout, snapshot-and-branch and global-view dropout are now in `SPEC_v1.md`.

# SPEC — Hundred Hands, v1

Status: v1 scope (draft for review) · Builds on: `SPEC.md` (v0) · Audience: engineers and Claude Code

v1 makes **one drone and its world realistic**. v2 then makes the swarm resilient and adaptive.

This document only describes what **changes or is added** relative to v0. Everything in `SPEC.md`
still holds unless a section here replaces it. Do not start v1 work until v0 is done (v0 M7).
Items marked **v2** or listed under *Non-goals* are out of scope.

---

## 1. Goal

Starting from the v0 Graph Q-Network, train and evaluate a defender swarm under realistic
conditions:

- rigid-body X500-class quadrotors with a cascaded flight controller;
- real city geometry from OpenStreetMap, with buildings blocking sensors;
- noisy, imperfect sensing (noise, missed detections, false alarms);
- wind;
- a composite asset (a building plus the pipeline feeding it);
- coordinated attacks and, last, fixed-wing threats;
- a curriculum built into the environment, scaling from 1 vs 1 up to N vs N;
- safety masks and a safety filter;
- the whole agent stack running as ROS 2 nodes around a MuJoCo simulation node.

v1 is done when:

1. The policy trained through the full v1 curriculum is evaluated against the baselines on the
   v1 evaluation suite, with results recorded (no fixed margin, as in v0).
2. A **robustness report** shows how performance degrades with wind and sensor noise, for the
   policy and the baselines.
3. The same policy runs as **decentralised ROS 2 agent nodes** against the MuJoCo ROS 2 node, and
   its results match the in-process evaluation within statistical noise.
4. The repo stays clean and releasable; every result is in `experiments/`.

---

## 2. What changes from v0

| Area | v0 | v1 |
| --- | --- | --- |
| Drone dynamics | Point-mass | **Rigid-body X500-class**, 4 rotors, motor lag, drag; point-mass kept as a fast mode |
| Control | Velocity P-controller | **Cascaded**: velocity → attitude → body rate → mixer |
| World | Flat + random boxes | **OpenStreetMap city tiles** (flat ground); boxes kept for early curriculum |
| Sensor occlusion | Off | **On** (line of sight through buildings) |
| Sensing quality | Clean | **Noise, missed detections, false alarms**, range-dependent |
| Tracker | CV Kalman, clean | + track confirmation, Mahalanobis gating, uncertainty in features |
| Disturbances | None | **Constant random wind** per episode |
| Asset | One sphere | **Composite**: one building + a pipeline of segments |
| Attackers | Straight, waypoint, evasive | + **coordinated** attacks; + **fixed-wing** threats (last) |
| Team sizes | 3 vs 3–6 | **Curriculum built into the env**: 1 vs 1 → all mixes up to N vs N; N set by measured compute |
| Safety | None | **Action masks + safety filter** |
| Observation schema | v1 of schema | **Schema v2**: uncertainty, hostility score, multi-node asset |
| ROS 2 | None | **MuJoCo sim node + one agent node per drone** |
| Comms | Ideal | Still ideal (limits are v2) |
| Policy | Feed-forward GQN | Same architecture; inputs extended; size re-benchmarked |

Unchanged: shared GQN with local and global branches, no IDs, relation-typed attention,
message passing (one synchronous round), VDN, global team reward, 27 primitives, 10 Hz decisions,
C++ env and inference with a Python learner, in-house learner, W&B + `experiments/`.

---

## 3. Repository changes

Same monorepo (`hundred-hands`, under the `hundred-hands` GitHub organisation). New or extended
directories only:

```
sim/
  dynamics/          # point_mass and rigid_body models behind one interface
  curriculum/        # curriculum engine + compute-based N scaling
  threats/           # red behaviours: straight, waypoint, evasive, coordinated, fixed_wing
  world/             # map tiles, composite asset, wind
perception/          # noisy sensor models, track confirmation, gating, fusion v2
gnc/
  control/           # cascaded controller (velocity, attitude, rate, mixer)
  safety/            # action masks + safety filter
scenarios/
  maps/              # processed city tiles (generated; large files via Git LFS or a release asset)
  assets/x500/       # full MJCF (rotors, inertia, motor constants) + limits
  v1/*.yaml          # env, curriculum, train, eval, robustness configs
tools/
  osm/               # OSM → tile pipeline (Python; pyosmium, shapely, pyproj)
ros/                 # ROS 2 packages (colcon), see §11
```

Dependency rules from v0 still apply. New rules:

- `ros/*` may depend on any C++ library package; **nothing depends on `ros/`**.
- `tools/osm` produces files in `scenarios/maps/`; runtime code never imports `tools/`.
- OSM processing dependencies live in `tools/` only (dev dependency group), not in `swarmsim`.

ROS 2: use the current ROS 2 LTS distribution at the start of v1, pinned in `ros/README.md` and
the Docker image.

---

## 4. Dynamics and control

### 4.1 Rigid-body X500-class quadrotor

- Vendored MJCF in `scenarios/assets/x500/`: body mass and inertia, arm geometry, four rotor sites.
  Values come from the X500-class reference (PX4's Gazebo model or the manufacturer's data),
  with the derivation documented in `scenarios/assets/x500/README.md`.
- Each rotor: thrust `T = k_f ω²` applied at the rotor site along body z; yaw torque
  `τ = ±k_m ω²`; first-order motor lag (time constant `τ_m`, default 0.03 s, config).
- Linear aerodynamic drag on the body, relative to wind: `F_d = −C_d (v − w)` (wind in §6.3).
- MuJoCo timestep stays 0.002 s (500 Hz).
- `dynamics: point_mass | rigid_body` is an env config. Point-mass stays for early curriculum
  stages and fast experiments; it uses the same drag-with-wind term.

### 4.2 Cascaded controller (`gnc/control/`)

| Loop | Rate | Input → output |
| --- | --- | --- |
| Velocity | 100 Hz | velocity setpoint → desired acceleration → desired thrust vector |
| Attitude | 250 Hz | desired thrust direction + yaw → body-rate setpoint (geometric SO(3) or quaternion P) |
| Body rate | 500 Hz | rate setpoint → torque (PID) |
| Mixer | 500 Hz | collective thrust + torques → 4 rotor commands, clipped to motor limits |

- Yaw tracks the velocity heading ψ from the primitive decoder.
- Gains in `scenarios/assets/x500/control.yaml`, tuned once with step-response tests.
- **The policy interface does not change**: the 27 primitives still produce a velocity setpoint.
  This is what allows v0 policies to transfer.
- Attackers that are quadrotors use the same model and controller.

### 4.3 Sim-to-sim check (gate before training on rigid bodies)

1. Evaluate the final v0 policy **zero-shot** on rigid-body dynamics (no wind, clean sensing,
   v0 world). Record the drop in survival rate.
2. Fine-tune from the v0 checkpoint on rigid-body dynamics.
3. Record both in `experiments/`. The drop and the recovery are a headline result.

---

## 5. World

### 5.1 OpenStreetMap city tiles (`tools/osm/`, `scenarios/maps/`)

- Input: OSM extracts (`.osm.pbf`) for a curated list of cities, downloaded manually or by a
  documented script and **not** committed.
- Pipeline output: 1 km × 1 km tiles in a local east-north-up frame, each tile a small MJCF file
  plus a JSON with metadata (source city, bounding box, OSM timestamp, licence).
- Buildings: footprint polygon → height from `height`, else `building:levels × 3 m`, else 10 m.
  MuJoCo collides with convex shapes, so non-convex footprints are **split into convex parts**
  (convex decomposition of the polygon) and each part is extruded as a convex mesh or an oriented
  box.
- Ground stays flat (no terrain elevation in v1).
- Tile library: at least 60 tiles from at least 6 cities with different layouts (dense centre,
  suburban, industrial). 10 tiles from 2 cities are **held out** for evaluation only.
- Attribution: tiles carry the ODbL notice; README credits OpenStreetMap contributors.
- Large files are stored with Git LFS or published as a release asset; the pipeline is
  reproducible from the extracts.
- Area bounds, asset clearance and spawn rules from v0 apply on top of the tile.

### 5.2 Composite asset

One asset per episode, made of:

| Part | Geometry | Placement |
| --- | --- | --- |
| Building | An existing building in the tile (OSM tiles) or a generated box (box worlds), footprint 15–50 m | Random, ≥ 200 m from the area edge |
| Pipeline | Chain of 4–10 capsule segments on the ground, radius 0.5 m, total length 100–300 m | From a random source point to the building, routed around other buildings |

- **Destroyed** when the building is hit, or when `pipe_hits_to_cut` pipeline segments are hit
  (default 1: one hit cuts the supply). Both are config values.
- Hit rule (same as v0): an attacker within its target part's surface + 2 m.
- **Episode ends** when the asset is destroyed or all threats are resolved. No timeout; bug-guard
  cap kept.
- Each attacker is assigned an aim point: the building (default weight 0.5) or a random pipeline
  segment (0.5), in config.
- The ground sensor sits at the building.

### 5.3 Wind

- Constant horizontal wind per episode: direction uniform, speed uniform in
  `[0, wind_max]`; `wind_max` follows the curriculum (§9) up to 8 m/s.
- Applied through the drag term (§4.1) to every drone, defenders and attackers alike.
- **Not observed** by the policy in v1. The policy must be robust to it. Estimating and
  adapting to wind is v2 (online adaptation).
- Gusts: optional Ornstein–Uhlenbeck component, config flag, **off** by default in v1.

---

## 6. Perception

### 6.1 Sensor model (per scan, 10 Hz)

| Effect | Defender onboard | Ground sensor |
| --- | --- | --- |
| Max range | 300 m | 500 m |
| Occlusion | **On**: ray cast to the target; blocked if a building is in the way | **On** |
| Detection probability | `P_d(r) = P_d0 · (1 − (r / r_max)^4)`, `P_d0 = 0.95` | `P_d0 = 0.98` |
| Position noise | Isotropic Gaussian, `σ(r) = σ_0 + k_σ · r`; defaults `σ_0 = 0.5 m`, `k_σ = 0.01` | `σ_0 = 1.0 m`, `k_σ = 0.005` |
| False alarms | Poisson, mean `λ_fa = 0.2` per scan, uniform in the visible volume, slow random drift (clutter, e.g. birds) | `λ_fa = 0.5` |
| Hostility score `p_hostile` | Attacker: Beta(8, 2); false alarm: Beta(2, 6) | Same |

All parameters are config values that the curriculum ramps (§9). The detector interface is
unchanged from v0 (it already carried covariance and confidence).

### 6.2 Tracker

- Constant-velocity Kalman filter (as v0) with:
  - **Mahalanobis gating** and global nearest-neighbour association (Hungarian from `gnc/`);
  - **Confirmation**: a track is confirmed after 3 hits in its first 5 scans; only confirmed
    tracks enter the observation graph;
  - Coast 2 s, then drop (as v0);
  - Running mean of `p_hostile` per track.

### 6.3 Fusion

- Ideal comms still: all confirmed tracks from all defenders and the ground sensor are fused
  each step.
- Association by Mahalanobis gating; merged by covariance-weighted averaging (information form).
  Proximity-only merging from v0 is removed.

### 6.4 Observation schema v2

Changes to `scenarios/schema/obs_schema.yaml` (bump `schema_version: 2`; CI equality tests as v0):

| Node kind | Added features |
| --- | --- |
| `local_threat`, `global_threat` | Position uncertainty (√trace of position covariance), velocity uncertainty, mean `p_hostile`, number of sensors contributing (global only) |
| `asset` | **Now one node per asset part** (building + each pipe segment). Features: vector to the part's closest point (3), distance (1), part extent (half-length or half-size, 3), height (1) |
| Self | Own commanded vs actual speed difference (1), safety-filter-active flag (1) |

- Parts of the asset are told apart by geometry only (no type token), consistent with the v0
  "no IDs, no type tokens" rule.
- The asset block becomes a real attention block (query = self), like the threat blocks.
- Wind is not a feature (§5.3).

---

## 7. Threats (red)

### 7.1 Invariant (unchanged, extended)

Every behaviour, including coordinated and fixed-wing, must hit the asset in 100 % of 200
seeded episodes when defenders are idle. CI test per behaviour.

### 7.2 Coordinated attacks

- A **group** of 2–6 attackers shares one plan:
  - **Time on target**: spawn bearings spread at least 60° apart; each attacker's speed (within
    its limits) and route are chosen so all arrive within a 3 s window.
  - **Split targets** (optional, weight 0.5): the group divides between the building and pipeline
    segments.
- Members still use the individual behaviour rules (waypoint, evasive) for their own route.
- An episode can mix one coordinated group and independent attackers.

### 7.3 Fixed-wing threats (last item of v1)

- Simplified fixed-wing kinematics in MuJoCo (point-mass body with constraints):
  - speed in `[v_min, v_max_fw]`, default 15–25 m/s, **cannot hover**;
  - coordinated turns: turn rate `ω = g · tan(φ) / v`, bank angle `|φ| ≤ 45°`;
  - climb and descent rate limits ±5 m/s; terminal dive onto the target.
- Behaviours: `straight`, `waypoint` (turn-limited path), and loiter-then-dive (circle at
  60–120 m altitude for 5–20 s, then dive).
- Speeds and limits are tuned so that PN + greedy intercepts a non-zero share (head-on
  geometry matters because tail chases are infeasible at these speeds).
- No type token: fixed-wing and quad threats look the same to the policy except for their
  features (speed, turn behaviour).
- Higher-fidelity fixed-wing models (e.g. JSBSim) are not in v1.

---

## 8. Safety masks and safety filter (`gnc/safety/`)

Applied identically in training, in-process evaluation, baselines (filter only) and ROS 2.

### 8.1 Action masks (policy only)

- Before the argmax, each of the 27 primitives is forward-predicted for 1.0 s with the
  point-mass approximation of the resulting setpoint.
- A primitive is **masked** if the prediction leaves the area or altitude band
  [5, 140] m, comes within 3 m of a building or the ground, or within 2 m of a teammate's
  predicted path.
- Masked actions are excluded from **both** the greedy action and the double-DQN target argmax.
- If all actions are masked, the safety filter alone acts (§8.2).

### 8.2 Safety filter (controller level)

- At 100 Hz, if a collision with a building, the ground or a teammate is predicted within 0.5 s
  for the current velocity setpoint, override it with a minimum-change avoiding setpoint
  (brake, then climb).
- Every override is logged as an event (count and duration in metrics and in the
  interpretability report).
- Baselines use the same filter, so comparisons stay fair.

---

## 9. Curriculum engine (`sim/curriculum/`)

The environment owns the curriculum. The learner only reports evaluation results; the engine
decides what to sample next.

### 9.1 Team-size ladder

Stage `k` includes every team composition `(n_blue, n_red)` with `max(n_blue, n_red) ≤ k`:

| Stage | New compositions | Sampled set |
| --- | --- | --- |
| 1 | 1v1 | {1v1} |
| 2 | 2v1, 1v2, 2v2 | all with max ≤ 2 |
| 3 | 3v1, 3v2, 3v3, 1v3, 2v3 | all with max ≤ 3 |
| … | … | … |
| N | all new with max = N | all with max ≤ N |

- Sampling within a stage: 70 % new compositions, 30 % earlier ones (avoid forgetting); config.
- Advance when greedy-policy asset survival ≥ threshold (default 0.7) **for every composition**
  with `n_blue ≥ n_red` in the stage, over 100 eval episodes each, or when the stage step budget
  is spent (logged as "advanced on budget").

### 9.2 Compute-based N

`N_max` is decided online and recorded, not fixed in config:

- Before opening stage `k + 1`, the engine runs a short benchmark at that size:
  - **Env throughput**: decision steps per second on the current hardware must stay ≥
    `min_env_sps` (config, default 2,000 team steps/s);
  - **Inference latency**: per-drone forward pass p99 at the graph sizes of stage `k + 1` must
    stay < 10 ms on the ARM latency profile (measured value from v0 M6, scaled by node count
    from the benchmark curve).
- If either fails, `N_max = k`. The curriculum continues on the other axes only.
- The decision and the measurements are written to the experiment's `meta.json`.

### 9.3 Other curriculum axes

Each axis ramps from its v0 value to its full v1 range. Axes advance in this order, one at a
time, using the same threshold rule (applied to the current team-size set):

| Order | Axis | Start → full v1 |
| --- | --- | --- |
| 1 | Dynamics | point-mass → rigid-body |
| 2 | Sensing | clean → full noise, `P_d`, false alarms (3 sub-steps: 33 %, 66 %, 100 % of parameters) |
| 3 | Wind | 0 → 8 m/s (0, 3, 6, 8) |
| 4 | World | boxes → OSM tiles |
| 5 | Asset | sphere → composite |
| 6 | Threat behaviours | v0 mix → + coordinated |
| 7 | Fixed-wing threats | off → 30 % of attackers |

- The team-size ladder runs inside each axis stage (restarting at the current `N` reached, not
  at 1v1, once an axis has been passed at the full ladder).
- The axis order is a config list, so it can be reordered by experiment.
- The engine exposes its state (stage, axis, parameters, history) for logging and for
  reproducing any sampled task from a seed.

---

## 10. Policy and learning changes

- **Architecture**: the same GQN (SPEC.md §7.1), with schema v2 inputs and the asset block now a
  multi-node attention block. The global-branch availability flag stays at 1 (v2 makes it vary).
- **Warm start**: initialise from the v0 checkpoint; layers whose input size changed are
  re-initialised and listed in the run's `meta.json`.
- **Masks**: action masks applied in acting and in the target (§8.1).
- **Size**: re-run the latency sweep (v0 §10) with schema v2 and graph sizes up to `N_max`; pick
  the size meeting < 10 ms p99 with 2× headroom.
- **Reward**: same global team reward, with the asset term meaning "composite asset destroyed".
  Intercepts are still scored as `1 / N_red` each.
- **Memory**: the GQN stays feed-forward; memory remains in the tracker. Recurrent layers are
  v2.
- **ONNX export** in **two parts** for ROS 2 (§11): `local_and_message.onnx` (self + local
  branch → local embedding and message) and `global_and_q.onnx` (global branch + Q-head). The
  in-process runtime may keep the single graph; a parity test checks both paths give identical
  Q-values.

---

## 11. ROS 2 (`ros/`)

### 11.1 Packages

| Package | Type | Role |
| --- | --- | --- |
| `hh_interfaces` | msgs | `DroneState`, `Track`, `TrackArray`, `AssetStatus`, `Primitive`, `VelocitySetpoint`, `SwarmMessage` (embedding + sender pose), `Event`, `EpisodeInfo` |
| `hh_sim` | C++ node | Wraps `sim` (MuJoCo world, dynamics, red, intercept rule, episode logic). Publishes `/clock`, per-drone **sensor detections** (not ground truth), asset status, events. Subscribes to velocity setpoints. |
| `hh_agent` | C++ node | **One per defender.** Tracker + graph builder + two-part ONNX policy + action masks + primitive decoder + safety filter. Subscribes to its own detections and swarm messages. Publishes its velocity setpoint and its swarm message. |
| `hh_comms` | C++ node | Ideal all-to-all relay of swarm messages and fused-track inputs, with a per-tick barrier. (v2 adds range, loss and delay here.) |
| `hh_fusion` | C++ node | Fuses confirmed tracks from all agents + the ground sensor into the global picture and publishes it to connected agents. |
| `hh_bringup` | launch | Launches sim, comms, fusion and `n` agents from a scenario YAML; namespaces `/hh/blue/<k>/…` and `/hh/red/…` (namespaces are ROS plumbing; the policy never sees them) |
| `hh_viz` | node + config | Foxglove/RViz markers: drones, tracks with covariance, asset parts, attention lines (from agents' interpretability topic), safety overrides |

### 11.2 Tick protocol (10 Hz, lockstep on `/clock`)

1. `hh_sim` publishes detections for tick `t`.
2. Each `hh_agent` updates its tracker and runs `local_and_message.onnx`, then publishes its
   `SwarmMessage` and confirmed tracks.
3. `hh_comms` and `hh_fusion` wait for all live agents (barrier, timeout 50 ms of wall time,
   logged on expiry), then publish messages and the fused picture.
4. Each `hh_agent` runs `global_and_q.onnx`, applies masks, decodes the primitive, and publishes
   a velocity setpoint.
5. `hh_sim` advances 0.1 s (controller and physics inside the sim node), then tick `t + 1`.

The controller runs inside `hh_sim` in v1 (it stands in for the autopilot). PX4 SITL replacing it
is v2.

### 11.3 Parity requirement

- `hh_bringup` can run an evaluation scenario list headless and write the same `metrics.json`
  format as `swarm-eval`.
- Acceptance: on 500 episodes of the v1 eval suite, ROS 2 and in-process asset survival rates
  agree within their 95 % confidence intervals; per-decision Q-values agree within 1e-4 on
  logged ticks.
- Each agent runs in its own process (composition optional), which demonstrates that no agent
  reads another agent's sensors except through comms and fusion.

---

## 12. Evaluation

### 12.1 v1 evaluation suite

- Held-out OSM tiles only (§5.1), fixed seeds.
- Conditions: team sizes up to `N_max` (all compositions with `n_blue ≥ n_red`, plus 1 vs 2 and
  2 vs 3 as stress cases); dynamics rigid-body; sensing full v1; wind 0, 4, 8 m/s; behaviours
  independent, coordinated, with and without fixed-wing.
- Policies: v1 GQN, v0 GQN (zero-shot), PN/PP × greedy/Hungarian (on noisy fused tracks, with
  the safety filter), random, idle.

### 12.2 Metrics

All v0 metrics, plus:

| Metric | Definition |
| --- | --- |
| Building survival / pipeline survival | Separate survival rates for each asset part |
| Safety overrides | Count and total seconds per episode |
| Mask rate | Share of decisions with at least one masked primitive |
| False-alarm engagement | Share of defender-time spent attending a false-alarm track (from attention) |
| Curriculum record | Stage reached, `N_max`, steps per stage, stages advanced on budget |

### 12.3 Robustness report (`swarm-report --robustness`)

Degradation curves (asset survival with 95 % CI) for GQN and the best baseline:

- wind speed 0 → 10 m/s (beyond the training range);
- sensing scale 0 → 150 % of the v1 noise, miss and false-alarm parameters (beyond the training
  range);
- team-size stress: `N_max + 1` and `N_max + 2` attackers against `N_max` defenders.

### 12.4 Ablations

1. v0 policy zero-shot vs fine-tuned vs v1 curriculum (transfer value).
2. Without action masks (filter only).
3. Without the global branch (`g = 0`) under full v1 noise.
4. Curriculum order: default vs randomised axis order.
5. Point-mass-only training evaluated on rigid-body (sim-to-sim gap).

### 12.5 Interpretability additions

- Attention over asset parts (which part each defender is protecting).
- Attention given to false-alarm tracks over time.
- Safety override and mask events overlaid on episode timelines and replays.
- Sensitivity of Q-values to track uncertainty (Q change when uncertainty features are zeroed).

---

## 13. Tests (added)

| Area | Test |
| --- | --- |
| Rigid body | Hover equilibrium thrust; step responses meet rise-time targets per loop; mixer saturation |
| Point-mass ↔ rigid parity | Same setpoint sequence: positions agree within tolerance over 5 s without wind |
| Wind | Steady-state drift under constant wind matches drag model |
| OSM | Convex decomposition covers the footprint (area within 1 %); heights rule; tile bounds; held-out tiles never sampled in training |
| Occlusion | Known geometry blocks and unblocks line of sight correctly |
| Sensor model | Empirical `P_d(r)`, noise σ and false-alarm rate match config (statistical) |
| Tracker | Confirmation logic; gating rejects outliers; false alarms rarely confirmed at default λ |
| Composite asset | Pipeline routing avoids buildings; destruction rules; aim-point sampling |
| Threats | Idle-defender invariant for every behaviour; coordinated arrival window; fixed-wing turn and speed limits |
| Safety | Masks never allow a primitive whose prediction violates a rule; filter prevents scripted collision cases |
| Curriculum | Stage sets match the ladder definition; advancement rule; `N_max` decision on mocked benchmarks; state is reproducible from seed |
| Schema v2 | Codegen and C++ ↔ Python equality |
| ONNX split | Two-part and single-graph Q-values identical |
| ROS 2 | Launch test: one episode runs end to end; tick barrier; parity on a short scenario list |

---

## 14. Milestones (continuing from v0 M7)

| # | Milestone | Acceptance check |
| --- | --- | --- |
| M8 | Curriculum engine | Team-size ladder, axis framework, compute-based `N_max`; runs on v0 settings and reproduces v0 results |
| M9 | Rigid-body + cascaded control | X500 MJCF, controller tuned, tests pass; sim-to-sim check (§4.3) recorded |
| M10 | Noisy sensing + tracker v2 + schema v2 | Sensor/tracker tests; warm-started policy trains through the sensing axis |
| M11 | Wind | Wind axis passed; first robustness curve |
| M12 | OSM tiles + occlusion | Tile pipeline, library and held-out set; world axis passed |
| M13 | Composite asset | Asset axis passed; part-level metrics |
| M14 | Coordinated attacks | Behaviour axis passed; coordinated vs independent results |
| M15 | Safety masks + filter | Mask/filter tests; ablation 2 recorded |
| M16 | ROS 2 | Packages, tick protocol, two-part ONNX; parity requirement (§11.3) met |
| M17 | Fixed-wing threats | Fixed-wing axis passed; results with and without fixed-wing |
| M18 | v1 release | Full v1 eval suite, robustness report, ablations 1–5, interpretability additions, README results updated, demo replays (in-process and ROS 2 / Foxglove) |

Milestones M10–M14 and M17 each end by passing their curriculum axis (§9.3) or recording that the
axis advanced on budget.

---

## 15. Non-goals for v1 (v2 and later)

Communication limits (range, packet loss, delay) · global view randomly unavailable · link drops,
disconnection and rejoin · hierarchical RL · self-play or learning red team · recurrent layers ·
distillation · learned track fusion · online adaptation (including wind estimation) · PX4 or
ArduPilot SITL · real hardware flights · terrain elevation · asset networks (multiple linked
assets) · gusts on by default · JSBSim or other high-fidelity fixed-wing models · human operator ·
GPU physics · generalisation measured by scenario family (beyond held-out tiles).

---

## 16. Open items to resolve during v1

| Item | Resolved in |
| --- | --- |
| X500 physical parameters and controller gains | M9 |
| Motor lag and drag coefficients | M9 |
| Sensor parameters giving a meaningful baseline drop (aim: PN + greedy loses 10–30 points of survival at full noise vs clean) | M10 |
| `min_env_sps` threshold and resulting `N_max` on each training platform | M8 |
| `pipe_hits_to_cut` and building/pipe aim weights | M13 |
| Fixed-wing speed band that keeps interception feasible but hard | M17 |
| City list for the tile library and held-out split | M12 |
| Whether the GQN needs a larger hidden size for v1 inputs | M10 (re-checked at M18) |
