> **Design reference for [v2 modules](../SPEC_v2.md).** This is the former "v2" spec, kept verbatim for its technical detail.
> Its goals, version labels, milestones and fixed ordering are superseded by `SPEC_v2.md`. Forced dropout, snapshot-and-branch and global-view dropout are now in `SPEC_v1.md`.

# SPEC — Hundred Hands, v2

Status: v2 scope (draft for review) · Builds on: `SPEC.md` (v0) and `SPEC_v1.md` (v1) · Audience: engineers and Claude Code

v1 made one drone and its world realistic. **v2 makes the swarm resilient and adaptive**: drones keep
working when the network degrades or disappears, rejoin cleanly, face a learning opponent, and adapt
to the conditions they actually meet.

This document only describes what **changes or is added** relative to v1. Everything in `SPEC.md` and
`SPEC_v1.md` still holds unless a section here replaces it. Do not start v2 work until v1 is done
(v1 M18). Items under *Non-goals* are out of scope.

Several v2 topics were left open during scoping (generalisation method, online adaptation, rejoin).
For each, this spec proposes a default, marked **[proposed]**. Confirm or change those before the
relevant milestone starts.

---

## 1. Goal

Starting from the v1 policy, make the swarm work under realistic communication and against adaptive
opponents:

- **Two tiers, as intended from the start**: tier 1, a single drone with its own sensing, acting
  autonomously; tier 2, a swarm that pools sensing and coordinates when connected.
- Range-limited, lossy, delayed mesh communication with outages. A disconnected drone can neither
  report its perception nor receive coordination, and **the system must keep working**.
- Rejoining the swarm after a disconnect, explored with **hierarchical RL**.
- A **learning red team** (self-play league), alongside the scripted attackers.
- **Recurrent memory** in the policy.
- **Learned track association** for fusion.
- **Online adaptation**: a context-aware policy plus calibration of the MuJoCo model from ROS 2 data.
- **PX4 SITL** as the autopilot in ROS 2 runs.
- **Asset networks**: several linked assets whose value depends on each other.
- **Generalisation measured by scenario family**, not only by seed or held-out tiles.
- **Distillation** of a larger teacher into the deployable size, if needed.

v2 is done when:

1. The v2 policy is evaluated on the v2 suite against the baselines, the v1 policy, scripted red and
   league red, with results recorded.
2. A **resilience report** shows performance as communication degrades (range, loss, delay, outage
   rate) and as drones disconnect and rejoin, for the policy and the baselines.
3. A **generalisation report** shows results on held-out scenario families.
4. At least one **online adaptation** method shows a measured gain over the non-adaptive policy on
   shifted conditions (or a recorded negative result with analysis).
5. The policy runs in ROS 2 with **PX4 SITL** autopilots and degraded comms, with parity to
   in-process evaluation (§10.3).

---

## 2. What changes from v1

| Area | v1 | v2 |
| --- | --- | --- |
| Comms | Ideal all-to-all | **Range-limited mesh**, packet loss, delay, outages, building-blocked links |
| Global view | Always available (`g = 1`) | **Per-drone, varies**: `g_i = 0` when disconnected |
| Fusion | Central, ideal | **Distributed**: each drone fuses what it receives; ground station is one more node |
| Messages | Same-tick synchronous round | **Latest received**, with age; same-tick round only where delay is zero |
| Track association | Mahalanobis gating | **Learned association** scorer, classical fallback |
| Policy memory | Feed-forward | **Recurrent** (GRU), sequence replay |
| Policy structure | Flat GQN | Flat GQN + **hierarchical** variant (manager over pointer goals) for rejoin |
| Red team | Scripted | Scripted + **self-play league** |
| Asset | One composite asset | **Asset networks** (linked parts with service value) |
| Adaptation | None | **Context encoder** + **sim calibration** with shadow evaluation and rollback |
| Autopilot in ROS 2 | Controller inside `hh_sim` | **PX4 SITL** per defender (lockstep, offboard velocity) |
| Evaluation | Held-out tiles | **Held-out scenario families**, Elo vs league, resilience curves |
| Model size | Latency-bound | Optional **distillation** from a larger teacher |

Unchanged: GQN core (local + global branches, no IDs, relation-typed attention), 27 primitives,
VDN with a global team reward, 10 Hz decisions, safety masks and filter, C++ env and runtime with a
Python learner, in-house learner, monorepo, W&B + `experiments/`.

---

## 3. Repository changes

```
sim/
  comms/             # mesh comms model: range, loss, delay, outages, link occlusion
  red_league/        # hooks for learned red policies inside the C++ env
  world/assets_net/  # asset networks and service-value propagation
perception/
  association/       # learned association scorer (inference) + classical fallback
  fusion/            # distributed fusion (per node)
policy/swarm_policy/
  recurrent/         # GRU variant, sequence replay
  hierarchy/         # manager-worker variant (pointer goals)
  league/            # self-play league manager, Elo, opponent sampling
  adapt/             # context encoder, teacher-student training
  distill/           # teacher → student distillation
calibration/         # Python: MuJoCo parameter identification from logs
ros/
  hh_comms/          # extended with the mesh model and netem mode
  hh_px4_bridge/     # MuJoCo ↔ PX4 SITL sensor/actuator bridge (lockstep)
  hh_agent/          # distributed fusion, stale messages, PX4 offboard output
```

Dependency rules from v0/v1 still apply. New rules:

- `calibration/` reads logs and writes parameter files into `scenarios/`; nothing at runtime
  imports it.
- `ros/hh_px4_bridge` depends on PX4 message definitions (`px4_msgs`) and talks to PX4 SITL; PX4
  itself (BSD-3) is an external dependency built in the Docker image, not vendored.

---

## 4. Communication model (`sim/comms/`, mirrored in `ros/hh_comms`)

### 4.1 Link model

Nodes: every alive defender plus the **ground station** at the asset (the ground sensor's site).

| Effect | Model (all config, ramped by the curriculum) |
| --- | --- |
| Range | Link exists if distance ≤ `r_comm` (default 400 m; ground station 600 m) |
| Link occlusion | Optional: link blocked if the straight line crosses a building (flag, on in final v2 stage) |
| Packet loss | Per message, `p_loss(d) = p_0 + (1 − p_0) · (d / r_comm)^4`, `p_0` default 0.05 |
| Delay | Per message, integer ticks drawn from {0, 1, 2, 3} with configurable weights (0–300 ms) |
| Outages | Per node, two-state Markov chain: mean connected time 30 s, mean outage 5 s; during an outage the node neither sends nor receives |
| Relay | Messages forwarded up to `max_hops` (default 1; 2 as an experiment); each hop applies loss and delay |
| Bandwidth | Per node per tick: one swarm message (embedding) + up to `max_tracks_tx` tracks (default 16), lowest uncertainty first |

### 4.2 What a drone has at each tick

- Its own sensor tracks (always).
- Messages and tracks received from neighbours, each with an **age** in ticks.
- The ground station's tracks, if linked (directly or through relay).
- `g_i = 1` if it received anything from at least one other node within the last `T_conn`
  (default 1 s), otherwise `g_i = 0`.

A drone with `g_i = 0` acts on its local branch alone. This is the tier-1 behaviour; it must be
competent on its own.

### 4.3 Distributed fusion (`perception/fusion/`)

- Each drone runs its own fusion over its own confirmed tracks and the tracks it received.
- Received tracks are propagated forward to the current time with their age before fusing.
- Association uses the learned scorer (§7) or the classical fallback.
- Result: each drone's global picture can differ from its teammates'. This is intended.

### 4.4 Messages under delay

- Every message carries the sender's embedding computed at the sender's tick, plus its age.
- The receiver uses the **latest** message from each neighbour. Teammate node features gain
  `message_age` and `pose_age`; teammate pose is extrapolated to the current tick.
- Training: zero-delay messages within a tick keep the gradient path (as in v0/v1); delayed
  messages use the stored embedding with **stop-gradient**. The curriculum (§11) ramps the share of
  delayed messages, so learning starts from the differentiable case.

---

## 5. Policy changes

### 5.1 Schema v3

| Node kind | Added features |
| --- | --- |
| Self | `g_i` (now varies), time since last connected, number of neighbours, own outage flag |
| `global_threat` | Age of the newest contributing report, number of contributing nodes |
| `teammate` | `message_age`, `pose_age`, hop count |
| `asset` | Criticality (§8.2), served value, current part status |
| Self (adaptation) | Context vector `z` (§9.1), when the adaptation variant is enabled |

`schema_version: 3`; codegen and C++ ↔ Python equality tests as before.

### 5.2 Recurrent GQN

- A GRU (hidden 32) after the merge of local and global branches, before the Q-head.
- Hidden state carried across ticks per drone; reset at episode start and kept through
  disconnects (memory is what lets a drone keep acting sensibly after losing the swarm).
- Training: sequence replay in the style of R2D2 — stored hidden states, sequences of 40 ticks,
  burn-in of 10 ticks, VDN target per tick.
- Interpretability: log the hidden state; probe it with linear readouts (e.g. "is a threat behind
  me", "time since disconnect") and report probe accuracy.
- Ablation: recurrent vs feed-forward under each comms condition.

### 5.3 Hierarchical variant for rejoin and re-tasking [proposed]

Scoping left rejoin open and pointed to hierarchical RL. Proposed design, run as a research track
against the flat recurrent GQN:

- **Manager** (every 1 s, or immediately when `g_i` changes or a new threat is confirmed): chooses
  a **goal** by pointing at a node in the drone's own graph, with no IDs:
  - a threat node (intercept it),
  - an asset-part node (guard it),
  - a "reconnect" goal (move toward the last known position of the nearest teammate or the ground
    station),
  - "hold" (continue current goal).
  The pointer is an attention score over candidate nodes plus three fixed goal embeddings.
- **Worker**: the GQN, with the chosen goal's embedding added to the self features; acts at 10 Hz on
  the 27 primitives.
- **Training**: manager learned with SMDP Q-learning on the same team reward, VDN across drones;
  worker learned as before plus a small team-agnostic goal-progress term **used only to train the
  worker's goal-following**, never added to the team reward.
- **Rejoin behaviour**: when `g_i` flips 0 → 1, the manager re-decides immediately with the new
  global picture; teammates see the rejoined drone through messages and can re-decide too.
- Interpretability bonus: the manager's goal is a human-readable "what is this drone doing".
- Decision gate: keep the hierarchy only if it beats the flat recurrent GQN on the rejoin
  scenarios (§12.3) or matches it with clearly better interpretability.

### 5.4 Distillation (only if needed)

- Trigger: a larger teacher (hidden 64 or 128) beats the deployable-size student by more than the
  noise band on the v2 suite.
- Student: same architecture at deployable size (SPEC.md §10 latency rule).
- Loss: MSE on Q-values plus KL between softmax(Q/τ) distributions; optionally attention-map
  matching. Data: student's own rollouts with teacher labels (DAgger style).
- Record teacher, student and from-scratch student side by side.

---

## 6. Red team

### 6.1 Scripted red (kept)

All v1 behaviours stay and stay in the training mix (at least 30 % of episodes), so the defenders
never forget them. The idle-defender invariant still applies to scripted red.

### 6.2 Learned red (self-play league)

- Red quads and red fixed-wing are controlled by **red policies**: a GQN of the same family with its
  own weights per platform type (quad, fixed-wing), shared across red agents of that type.
- Red observes blue only through **its own sensors** (same sensor model, red-side parameters) and
  red-side comms (same comms model). No privileged information.
- Red actions: the 27 quad primitives; fixed-wing uses its own primitive set
  (turn ±bank, climb ±, speed ±) within its v1 limits.
- Red reward: the negative of the blue team reward (zero-sum).
- Red agents remain abstract (the same hit rule as scripted red); the point is to stress-test the
  defence, not to model specific systems.

### 6.3 League

- Pools: current blue, current red, frozen snapshots of both every 250k steps, and scripted red.
- Opponent sampling for blue: 30 % scripted red, 50 % league red with prioritised fictitious
  self-play (harder opponents sampled more), 20 % latest red.
- Red trains against current blue and frozen blue snapshots, the same way.
- Ratings: Elo (or TrueSkill) over all snapshots and scripted behaviours, from scheduled
  round-robin evaluation.
- Guardrail: if blue's survival against scripted red drops by more than the noise band, raise the
  scripted share automatically and log it.

---

## 7. Learned track association (`perception/association/`)

- A small network scores whether two tracks (from the same or different nodes) are the same object,
  from position and velocity difference normalised by their covariances, ages, `p_hostile`
  difference and sensor positions.
- Trained **supervised** from simulation ground truth (offline dataset from v1/v2 rollouts).
- Association: Hungarian assignment on learned scores with a minimum-score gate; fused state still
  by covariance-weighted averaging.
- Classical Mahalanobis association remains the fallback and the comparison.
- Reported: association precision/recall, track duplication rate, false-track rate, and the effect
  on asset survival.
- Runs in C++ via ONNX Runtime, inside each agent's fusion.

---

## 8. World and scenarios

### 8.1 Asset networks

- An episode can contain an **asset network**: 2–5 components linked by dependencies, for example
  a pumping station feeding a pipeline feeding two buildings, or a substation feeding power lines
  to several buildings.
- Each component has a demand value; the network's **served value** is the share of total demand
  still served, computed by propagating failures along dependencies.
- Episode ends when served value falls to 0 or all threats are resolved. Bug-guard cap kept.
- Attackers aim at components with configurable weights (uniform, by criticality, by scripted
  preference). Learned red chooses freely.

### 8.2 Reward with asset networks

Still one global team reward:

```
r_t = (1 / N_red) · (# attackers intercepted at t)
      − Δ(lost served value at t)            # in [0, 1] over the episode
      + (served value at episode end)
```

This reduces to the v1 reward when the network is a single composite asset.

Each asset node carries its **criticality** (served value lost if that part fails) as a feature, so
the policy can weigh parts without type tokens.

### 8.3 Scenario families

A family is a named region of the scenario space. Families for v2:

| Axis | Families |
| --- | --- |
| City layout | dense centre, suburban, industrial, open (few buildings) |
| Threat mix | quads only, fixed-wing heavy, coordinated waves, learned red |
| Conditions | calm, windy (6–10 m/s), high sensor noise, degraded comms |
| Asset | single composite, utility network, multi-building network |

A scenario is one family per axis. Families are defined in `scenarios/v2/families/*.yaml`.

### 8.4 Generalisation protocol [proposed]

Scoping deferred this (Q7.4). Proposed:

- **Leave-one-family-out** per axis: train on all families except one, evaluate on the held-out
  family (one run per held-out family per axis, 3 seeds each).
- Plus one fully held-out **combination** (e.g. industrial × coordinated waves × degraded comms ×
  utility network) never seen as a combination in training.
- Report a matrix: rows = held-out family, columns = metrics, with in-distribution results beside
  each for comparison.

---

## 9. Online adaptation [proposed]

Scoping deferred the details (Q7.5–7.8). Proposed two-part design, evaluated separately and
together:

### 9.1 Part A — context-aware policy (in flight, no weight updates)

- In simulation, each episode randomises hidden parameters: wind, mass, motor strength, drag, motor
  lag, sensor noise scale, comms quality.
- **Teacher**: GQN that also receives the true parameters (privileged).
- **Adaptation encoder**: a small network (1D conv or GRU) over the drone's last 2 s of own states,
  commanded setpoints and achieved accelerations, producing a context vector `z` (8 dims).
- **Student**: GQN with `z` as a self feature, trained to match the teacher (and then fine-tuned with
  RL), in the style of rapid motor adaptation.
- In flight, `z` updates every tick; no weights change. Shared to teammates in the swarm message
  when connected (Q7.8 default: share context, not weights).
- Interpretability: decode `z` back to parameter estimates and report accuracy (e.g. estimated vs
  true wind).

### 9.2 Part B — calibrate the simulator from ROS 2 data, then fine-tune offline

This is the "fine-tune the MuJoCo on ROS 2" loop from the original brief.

1. **Log** ROS 2 runs (PX4 SITL in v2; real flights later): states, setpoints, actuator outputs,
   sensor data, comms statistics.
2. **Identify** MuJoCo parameters (`calibration/`): drag, motor constants and lag, mass, controller
   response, sensor noise, comms loss and delay, by minimising multi-step trajectory and statistic
   errors (CMA-ES or gradient-free search; MuJoCo stays the model).
3. **Fine-tune** the policy offline in the calibrated simulation (from the current checkpoint).
4. **Shadow evaluation gate**: the candidate runs alongside the current policy on replayed and
   simulated scenarios; promoted only if it is not worse on any family beyond the noise band and
   better on the calibrated conditions.
5. **Deploy and rollback**: models are versioned; the agent node can roll back to the previous
   model on command or automatically if live metrics (safety overrides, mask rate) exceed limits.

### 9.3 Not in v2

On-board gradient updates during flight are not part of v2 (risky, hard to validate). A head-only
on-board fine-tuning experiment may run in simulation if time allows, flagged as exploratory.

---

## 10. ROS 2 and PX4 SITL

### 10.1 Comms in ROS 2

- `hh_comms` applies the §4 model to every message (range, loss, delay, outages, relay), using the
  simulator's positions and building geometry from `hh_sim`.
- Optional **netem mode** for demos: agents in separate containers, with real loss and delay
  injected by Linux `netem` and DDS/Zenoh carrying the traffic.
- `hh_fusion` is removed; fusion runs inside each `hh_agent` (§4.3). The ground station becomes an
  `hh_ground` node with the ground sensor's tracker and fusion.

### 10.2 PX4 SITL bridge (`ros/hh_px4_bridge`)

- One PX4 SITL instance per defender, in **lockstep** with MuJoCo through PX4's simulator
  interface: MuJoCo sends IMU, barometer, magnetometer and GNSS readings from the rigid-body state;
  PX4 returns actuator outputs, which drive the MuJoCo rotors.
- `hh_agent` commands PX4 in **offboard velocity mode** over the PX4 ROS 2 bridge (uXRCE-DDS):
  trajectory setpoints, offboard control mode heartbeat, arming and mode commands.
- The v1 cascaded controller is bypassed for PX4-flown drones; attackers keep the internal
  controller.
- Scale: PX4 SITL is for **evaluation and calibration data**, not training. Target ≥ 6 PX4 defenders
  in lockstep at ≥ 1× real time on the cloud instance; record the achieved rate.
- PX4 parameters for the X500-class airframe come from PX4's standard X500 configuration, with any
  changes recorded.

### 10.3 Parity requirements

- **Internal controller vs PX4**: with ideal comms and no wind, asset survival with PX4-flown
  defenders is within the 95 % CI of the internal-controller result, or the gap is explained and
  then reduced by calibration (§9.2).
- **ROS 2 vs in-process** (as v1): same comms model and seeds give results within CI.

---

## 11. Curriculum additions

New axes, appended to the v1 order (the v1 axes are already at full range when v2 starts):

| Order | Axis | Start → full v2 |
| --- | --- | --- |
| 8 | Comms range | ideal → `r_comm` 400 m (relay off) |
| 9 | Comms loss and delay | 0 → full `p_loss` and delay weights (33 / 66 / 100 %) |
| 10 | Outages | none → full Markov outage model |
| 11 | Link occlusion | off → on |
| 12 | Asset networks | composite → networks (2 → 5 components) |
| 13 | Red league | scripted only → league mix (§6.3) |

- The global-branch availability now varies naturally through axes 8–11; no artificial dropout is
  needed after axis 10.
- Team-size ladder and compute-based `N_max` work as in v1; `N_max` is re-measured with v2 costs.
- Scenario families (§8.3) are sampled within the training families once axis 13 is reached.

---

## 12. Evaluation

### 12.1 v2 evaluation suite

- Held-out tiles and held-out families (§8.4).
- Comms conditions: ideal, nominal v2, harsh (2× loss, 2× outage rate, half range).
- Opponents: scripted red (all behaviours), league snapshots, latest red.
- Policies: v2 flat recurrent GQN, v2 hierarchical variant, v1 GQN, baselines (on each drone's own
  fused picture under the same comms), random, idle.

### 12.2 Metrics (added)

| Metric | Definition |
| --- | --- |
| Served value at end | For asset networks; equals survival for single assets |
| Connectivity | Share of drone-time with `g_i = 1`; mean neighbours |
| Disconnected performance | Intercepts and coverage by drones while `g_i = 0` |
| Rejoin time | Seconds from reconnection to the drone taking a goal consistent with teammates (no two drones on one threat unless intended) |
| Duplicate engagements | Share of time two or more drones attend the same threat while others are uncovered |
| Comms load | Messages and tracks sent per drone per second |
| Elo | Blue and red ratings across the league |
| Association quality | Precision, recall, duplicate and false-track rates (§7) |
| Adaptation | Context-decoding accuracy; survival gain on shifted conditions |

### 12.3 Resilience report (`swarm-report --resilience`)

Curves with 95 % CI for the v2 policy, the v1 policy and the best baseline:

- comms range 100 → 800 m;
- packet loss 0 → 60 %;
- delay 0 → 500 ms;
- outage rate 0 → 3× nominal;
- **forced disconnect scenarios**: k of n defenders cut for 10 s at the moment threats appear, then
  reconnected — survival, disconnected performance and rejoin time;
- more attackers than trained: `N_max + 1 … N_max + 3`.

### 12.4 Ablations

1. Recurrent vs feed-forward (per comms condition).
2. Hierarchical vs flat (rejoin scenarios and overall).
3. Learned vs classical association.
4. Context encoder on vs off (shifted conditions).
5. Calibrated vs uncalibrated simulation (PX4 parity gap before and after).
6. Training with vs without the red league (against scripted and league opponents).
7. Stop-gradient on delayed messages vs dropping delayed messages entirely.

### 12.5 Interpretability additions

- Timeline of `g_i`, neighbours and message ages per drone, aligned with decisions.
- Attention shift between local and global branches as connectivity changes.
- Manager goals (hierarchical variant) as readable labels on replays.
- GRU probe results.
- Context `z` decoded to wind, mass and noise estimates over time.

---

## 13. Tests (added)

| Area | Test |
| --- | --- |
| Comms | Empirical loss, delay and outage statistics match config; range and link occlusion; relay hop counts; bandwidth caps |
| Availability | `g_i` matches the connectivity definition on scripted link schedules |
| Distributed fusion | Stale tracks are propagated by age; fused pictures converge when fully connected |
| Message staleness | Stop-gradient applied only to delayed messages; ages correct |
| Recurrent | Hidden-state reset rules; burn-in; equivalence with the feed-forward model when the GRU is bypassed |
| Hierarchy | Pointer selects only existing nodes; manager re-decides on `g_i` change |
| League | Opponent sampling ratios; snapshot freezing; Elo update on a toy round-robin |
| Learned red | Red sees only its own sensors; idle-red invariant: idle red never hits the asset |
| Association | Learned and classical agree on easy cases; ONNX parity |
| Asset networks | Served-value propagation on hand-made graphs; criticality features |
| Adaptation | Encoder training on synthetic data recovers known parameters; calibration recovers parameters of a sim with known perturbations |
| PX4 bridge | Lockstep with no dropped frames over 10 minutes; offboard arming and mode sequence; hover and step responses |
| Families | Train/eval split never leaks a held-out family or combination |

---

## 14. Milestones (continuing from v1 M18)

| # | Milestone | Acceptance check |
| --- | --- | --- |
| M19 | Comms model + distributed fusion | Comms tests; `g_i` live; v1 policy evaluated zero-shot under v2 comms (recorded) |
| M20 | Training under degraded comms | Schema v3, stale messages, comms axes 8–11 passed; first resilience curves |
| M21 | Recurrent GQN | Sequence replay; ablation 1 recorded |
| M22 | Hierarchical variant | Manager-worker trained; forced-disconnect scenarios; ablation 2; keep/drop decision recorded |
| M23 | Asset networks | Served value, reward, criticality features; axis 12 passed |
| M24 | Learned red + league | League running, Elo tracked, guardrail tested; axis 13 passed; ablation 6 |
| M25 | Learned association | Trained scorer, ONNX in agents; ablation 3 |
| M26 | Scenario families + generalisation | Families defined; leave-one-family-out runs; generalisation report |
| M27 | Adaptation A: context encoder | Teacher-student training; ablation 4; decoded-context accuracy |
| M28 | PX4 SITL | Bridge, lockstep, offboard control; parity check (§10.3) recorded |
| M29 | Adaptation B: calibration loop | Calibration from PX4 logs; shadow gate; rollback tested; ablation 5 |
| M30 | Distillation (conditional) | Trigger check recorded; if triggered, teacher/student/scratch comparison |
| M31 | v2 release | Full v2 suite, resilience and generalisation reports, ablations 1–7, interpretability additions, README results, demos (in-process, ROS 2 + PX4 + Foxglove, netem multi-container demo) |

---

## 15. Non-goals for v2 (later)

Real hardware flights · terrain elevation and outdoor weather beyond wind · electronic warfare,
jamming or deliberate comms attacks · comms security and authentication · human operator in the
loop · high-fidelity fixed-wing models (e.g. JSBSim) · on-board gradient updates in flight (beyond
the exploratory simulation experiment) · GPU physics · large-scale swarms beyond the measured
`N_max` · multiple defended sites per episode.

---

## 16. Open items to resolve during v2

| Item | Resolved in |
| --- | --- |
| Confirm or change the [proposed] designs: rejoin hierarchy (§5.3), generalisation protocol (§8.4), online adaptation (§9) | Before M22, M26, M27 respectively |
| Comms defaults (`r_comm`, loss, delay weights, outage rates) giving meaningful degradation | M19–M20 |
| `max_hops` 1 vs 2 | M20 |
| League sampling ratios and snapshot frequency | M24 |
| Reward weighting with asset networks (is `1 / N_red` per intercept still right?) | M23 |
| Achievable PX4 SITL count and real-time factor on cloud hardware | M28 |
| Whether distillation is needed | M30 |
| `N_max` under v2 costs | M20, re-checked at M31 |
