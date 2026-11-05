# Should we add a three-tier task suite?

Proposal under review:

1. 2D task “as it is”
2. 3D with point-mass dynamics
3. 3D with real-robot dynamics

Checked against the current implementation and `docs/SPEC_v0.md`,
`docs/SPEC_v1.md`, `docs/SPEC_v2.md` and `docs/v2_modules/realism.md` on
2026-11-04.

## Verdict

**The idea is good as a later experimental programme, but it does not make sense
to implement it now, and the “2D as it is” premise is wrong.**

- The current environment is **already 3D point mass**. `sim::Env` builds three
  slide joints (x, y, z), defenders and attackers have altitude, and the world is
  3D. The 2D view is only the top-down replay/plot projection.
- v0 explicitly scopes point-mass drones; v1 explicitly says its paper scope is
  **point-mass dynamics, clean sensing, ideal comms apart from dropout**.
- v2 already has a `rigid-body` module whose question is exactly the gap from
  point mass to a real quad model, with controller tuning, zero-shot transfer and
  fine-tuned recovery. A 3D rigid-body tier belongs there, not in v0/v1.
- The former v1 design preserved in `docs/v2_modules/realism.md` already specifies
  `dynamics: point_mass | rigid_body`, a cascaded X500 controller, point-mass kept
  as a fast mode, and a sim-to-sim check. That is the right blueprint.
- A custom 2D tier is **not in any current spec**. If the goal is simply an easier
  algorithm sanity check, use MPE2 `simple_tag` or another external 2D benchmark;
  do not build and maintain a second full environment inside this repo.
- “Real robot dynamics” here can only mean a **rigid-body X500 simulation**, not
  hardware. Hardware flights remain out of scope. PX4 SITL is the separate v2
  `px4-sitl` module and should not be bundled into this tier.
- v0 is still not complete (M4 learner, M3 primitive tuning, M7 evidence). v1 has
  an explicit entry gate: do not start it until v0 M7 is working and documented.
  Starting a three-tier suite now would delay the only gate that matters.

## What the current implementation actually supports

| Layer | Current state | Consequence for the proposal |
| --- | --- | --- |
| `sim::Env` | MuJoCo bodies with three slide joints, point-mass acceleration control; hard-coded 3D world | Already 3D point mass; adding 2D or rigid body requires a real dynamics abstraction, not a config line |
| `core::velocity_control` | Velocity P-controller producing acceleration | 3D rigid body needs a cascaded velocity → attitude → body-rate → mixer stack; policy interface can stay velocity-based |
| `gnc::Primitive` | 27 actions: 3 turn × 3 climb × 3 speed | 2D would need 9 actions or climb actions masked/no-op; a separate action/observation contract |
| `perception` / graph | Ego-frame 3D features; z, altitude and climb appear in the graph | 2D variant must decide whether to zero z features, retrain, and version the schema |
| `TeamEnv` / `VecEnv` | Independent of physics details | Good seam: a dynamics backend could slot under `TeamEnv` later |
| Evaluation | `swarm-eval --suite` is currently parsed but ignored; default config is what runs | A multi-tier suite needs the suite flag fixed first; otherwise “suite” is only a name |
| Baselines | PN/PP + greedy/Hungarian on fused tracks | Need per-tier baselines and checks; PN in 2D is not the same implementation as 3D PN |
| Docs | v0 = point mass; v1 = point mass paper stack; v2 = modular `rigid-body` | Current specs do not define a 2D tier; a 3-tier suite is a new scope decision |

Important correction: the current GIF is top-down because that is the rendering
view. The simulation is not 2D. Calling the current task “2D” would create a false
baseline and confuse the experiment journal.

## How the three tiers map to the documents

| Tier | Closest existing spec | Status |
| --- | --- | --- |
| 2D planar defence | No current module | New scope; not v0/v1; maybe a v2 module only if there is a specific research question |
| 3D point mass | v0 §4 / v1 scope, current implementation | Already the main task |
| 3D rigid-body X500 | `v2_modules/realism.md` §4 and `SPEC_v2` `rigid-body` | Planned v2 module; not v1 |
| Fidelity ladder / curriculum | Former v1 §9.3 “Dynamics: point-mass → rigid-body”; current v2 module rules | Could be one v2 experiment, not a new v0/v1 milestone |

The former v1 design is useful here because it already worked out:

- `dynamics: point_mass | rigid_body` as an env config;
- point mass retained as the fast mode;
- the policy interface unchanged: 27 primitives still produce a velocity setpoint;
- a cascaded controller in `gnc/control/`;
- a zero-shot and fine-tuned sim-to-sim check;
- rigid-body tests: hover equilibrium, step responses, mixer saturation,
  point-mass ↔ rigid-body position parity without wind.

Do not reintroduce that former v1 scope into the current v1 paper stack. The
current v1 is deliberately the minimal paper stack for the dropout/coordination
thesis; adding fidelity tiers would dilute it and consume compute needed for
multi-seed experiments.

## Does a three-tier suite have research value?

Yes, but for a different paper/phase:

- It measures **fidelity transfer**: how much of the coordination mechanism
  survives from planar point mass to 3D point mass to rigid-body control.
- It gives a clean **sim-to-sim gap** result: zero-shot drop and recovery after
  fine-tuning, without the confound of noise, wind or OSM changes.
- It supports **sample-efficiency and curriculum** questions: train on easier
  tiers, fine-tune on harder tiers; does the mechanism transfer?
- It gives reviewers a credible answer to “point-mass only?” without turning v1
  into a full realism project.

But it is a **fidelity ladder**, not three unrelated “tasks”. To be scientifically
useful, the tiers must share:

- the same asset-defence objective and reward definition;
- the same scenario seeds and attacker behaviours (or explicitly controlled
  variations);
- the same perception stack and graph semantics where possible;
- the same policy interface (velocity setpoints) or a documented mapping;
- the same evaluation metrics and baselines;
- a clear statement of which variables changed and which stayed fixed.

Otherwise a 2D/3D/rigid comparison confounds task difficulty, observation/action
space, sensor behaviour and controller dynamics, and cannot support a causal
claim.

## What “increasing difficulty” should mean

Do not assume the order is automatically harder:

- 2D removes altitude/climb and may make encirclement and interception easier,
  but it also changes the coordination problem and can make some threats
  trivially reachable.
- 3D point mass adds vertical manoeuvre, dive geometry and out-of-area failures.
- 3D rigid body adds attitude, thrust limits, motor lag and controller tuning;
  that is likely hardest, but it must be measured with the same baselines and
  seeds.

The experiment should report the **measured** ordering (survival, intercept
fraction, time-to-clear, sample efficiency, latency), not assert it. A 2D tier
that is easier for PN but harder for the GQN (or vice versa) is itself a result.

## Recommended implementation strategy

### Now

Do not implement any tier. Finish current v0:

1. primitive tuning and a green C++/Python test suite;
2. a real M4 learner attempt at spec scale;
3. M7 evidence and clean-checkout documentation.

The current task is already the “3D point mass” tier. Do not add a second
environment while the first one has not met its acceptance check.

### After v0 and within current v1

Keep v1 point-mass only, as written. If an external 2D sanity check is useful,
use **MPE2 `simple_tag` as a dev-only benchmark**, not a custom 2D defence env.
Fix `swarm-eval --suite` before claiming any “suite” support.

### After v1: v2 modules

Implement the fidelity ladder as one or two v2 modules, respecting v2 rules
(opt-in, own experiment, own tests, v1 suite still passing):

1. **`rigid-body`** — the existing planned module. Add the X500 MJCF, cascaded
   controller, motor lag, drag, and the former v1 §4.3 sim-to-sim check.
2. **`planar-2d`** — only if a paper needs the 2D rung or transfer question.
   Otherwise use MPE2 `simple_tag` externally. If implemented, make it a mode of
   the same environment, not a fork: lock z, use the 9-action planar subset or
   masked climb actions, version the observation schema, and reuse asset/red/
   reward/evaluation logic.
3. **`fidelity-transfer`** — optional umbrella experiment comparing frozen v1
   point-mass policy zero-shot on rigid body (and planar if present), then
   fine-tuned per tier. Report the gap and the mechanism metrics, not just
   survival.

A single `dynamics`/`motion` config with an internal backend interface is the
right design. `TeamEnv`, `VecEnv`, the graph builder, the learner and the
evaluation runner should not be forked per tier.

## Costs and risks

| Item | Cost/risk |
| --- | --- |
| 2D mode | New action/observation contract; red behaviour/dive logic; separate policy compatibility; test/eval duplication |
| 3D rigid body | New MJCF, cascaded controller, gain tuning, motor/drag models, parity tests; high effort and tuning risk |
| Evaluation | 3 tiers × policies × seeds × conditions; compute and report complexity multiply |
| Transfer | Checkpoints may not transfer because observation/action schemas differ; need schema versioning and warm-start rules |
| Focus | v1’s thesis is dropout/coordination, not fidelity; adding tiers risks not finishing the paper |
| Fairness | Baselines, reward, seeds and perception must be identical across tiers; otherwise the comparison is confounded |
| Maintenance | A second physics/action stack is a permanent burden; keep it behind a v2 flag |

## Final recommendation

- **Do not add the suite to v0 or the current v1.**
- **Keep the current 3D point-mass task as the v0/v1 main environment.**
- **Use MPE2 `simple_tag` if you need a cheap 2D algorithm sanity check now.**
- **Implement `rigid-body` as the existing v2 module after v1**, following
  `v2_modules/realism.md` §4 and the `SPEC_v2` `rigid-body` acceptance check.
- **Add the 2D tier only as an optional v2 `planar-2d` module if the research
  question is explicitly “how does coordination transfer across dimensionality
  and fidelity?”** Otherwise it is extra surface for little gain.
- **If you do it, call it a fidelity ladder, not three unrelated task suites**,
  and share one environment, one policy interface, one reward, one seed set and
  one metrics pipeline across tiers.
