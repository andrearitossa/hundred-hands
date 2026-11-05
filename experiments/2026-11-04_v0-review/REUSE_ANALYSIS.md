# Can we reuse code from existing environment suites?

Analysis date: 2026-11-04. This is an engineering/licensing analysis, not legal advice.
The Hundred Hands repository is **Apache-2.0** (`LICENSE`, `pyproject.toml`).

## Short answer

**Yes for ideas and small, attributed snippets; no for wholesale environment/backend
copying.** All the serious candidate suites I checked are permissively licensed
(MIT, BSD-3, Apache-2.0), so code can legally be reused inside this Apache-2.0
project if notices are preserved. But almost none of it fits the current C++/MuJoCo
architecture or the v0 task. Copying whole Python/PyBullet/Isaac Gym environments
would add a second physics stack, more dependencies and a larger maintenance surface
without solving the actual blocker (M4 learning).

Recommended stance:

- **Borrow task/reward/evaluation ideas freely** (ideas are not copyrightable).
- **Use the projects as optional dev/test dependencies**, not runtime replacements.
- **Copy only small utilities or adapters**, with source URL, commit, license and
  modification notes. Prefer reimplementation over copy for anything nontrivial.
- **Do not copy physics, simulators or core env loops** into `sim/`, `gnc/` or
  `perception/`. That would fight the existing C++/MuJoCo design and the v0 spec.
- Add a `THIRD_PARTY_NOTICES.md` and a license/attribution check if any code or
  assets are copied.

## Licence matrix (verified from repository LICENSE files)

| Project | Licence | Legal reuse in this Apache-2.0 repo | Practical fit |
| --- | --- | --- | --- |
| [MARSim](https://github.com/McFadden-S/MARSim) | MIT | Yes; keep MIT notice | Closest task concept, but grid/Gymnasium 0.28/PPO prototype; use as design reference, not backend |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | MIT | Yes; keep MIT notice | Best optional API adapter for external MARL tooling |
| [MPE2](https://github.com/Farama-Foundation/MPE2) | MIT | Yes; keep MIT notice | Excellent small external sanity benchmark; do not copy its particle sim |
| [MAgent2](https://github.com/Farama-Foundation/MAgent2) | MIT | Yes; keep MIT notice | Large-team adversarial reference; grid only |
| [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | MIT | Yes; keep MIT notice | Good Python/PyBullet drone reference; second physics stack, not v0 |
| [PyFlyt](https://github.com/jjshoots/PyFlyt) | MIT | Yes; keep MIT notice | Good 3D pursuit/dogfight/PettingZoo reference; PyBullet backend |
| [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) | BSD-3-Clause | Yes; keep copyright/licence notice | High-fidelity GPU drone reference; Isaac Gym dependency, not v0 |
| [SMAC](https://github.com/oxwhirl/smac) / [SMACv2](https://github.com/oxwhirl/smacv2) | MIT | Yes; keep MIT notice | Evaluation/protocol reference; StarCraft only |
| [JaxMARL](https://github.com/FLAIROx/JaxMARL) | Apache-2.0 | Yes; keep Apache notices/NOTICE and mark modifications | Only for a future JAX learner, not current PyTorch/C++ stack |
| [BenchMARL](https://github.com/facebookresearch/BenchMARL) | MIT | Yes; keep MIT notice | Benchmarking framework reference; heavy for v0 |
| [MARLlib](https://github.com/Replicable-MARL/MARLlib) | MIT | Yes; keep MIT notice | Algorithm/environment adapters; heavy dependency |
| [Melting Pot](https://github.com/google-deepmind/meltingpot) | Apache-2.0 | Yes; keep notices | Generalization benchmark only |
| [Neural MMO](https://github.com/NeuralMMO/environment) | MIT | Yes; keep MIT notice | Scale/generalization inspiration only |

Transitive dependencies need their own review. Examples: PyBullet (zlib-style),
JAX (Apache-2.0), Isaac Gym (NVIDIA licence), MuJoCo (Apache-2.0 in recent
versions), and any bundled drone meshes/models. A permissive top-level repository
does not automatically make every asset or dependency safe to vendor.

## What is worth reusing, by category

### 1. Interfaces and conventions (high value, low risk)

From **PettingZoo / Gymnasium**:

- The `ParallelEnv` shape: `reset() -> observations, infos`, `step(actions) ->
  observations, rewards, terminations, truncations, infos`, `agents`, `observation_spaces`,
  `action_spaces`, `possible_agents`.
- Agent-ID and alive/dead conventions, seed/reset semantics, and API-conformance tests.
- An optional `TeamEnv`/`VecEnv` adapter under `eval/` or `bindings/` would let external
  MARL libraries run the environment without touching the C++ core.

Why: this is mostly interface design, low code volume, and directly useful for
external comparisons. Use the library as a dependency or adapt the interface; do
not copy its implementation.

### 2. Task, reward and scenario ideas (high value, no code copying)

From **MARSim**:

- Roles and adversarial task taxonomy: friendly UAV/UGV, enemy UAV/UGV, escort,
  interception, resupply, kamikaze/collision modes.
- Reward categories: detection, defence proximity, destruction/objective bonuses.
- Procedural corridor/battlefield generation and scenario config structure.
- Evaluation metrics and saved JSON metrics.

Why: closest task concept. But MARSim is grid-based with an 11×11 local observation,
Gymnasium 0.28.1, Python 3.9+, and PPO; its assumptions conflict with the 3D
point-mass, limited-range tracker/fusion and GQN graph design. Use it as a design
checklist and related-work citation, not as a dependency.

From **MPE2 `simple_tag` / `simple_world_comm` / `simple_spread`**:

- Predator-prey/communication task semantics and fixed-seed evaluation patterns.
- A cheap external sanity check for the GQN/VDN learner: can the same algorithm
  learn a 2D cooperative pursuit task before debugging the 3D defence task?

Why: excellent algorithm test, but 2D particles and fixed small teams. Do not copy
the particle physics or replace the custom task.

From **SMAC / SMACv2**:

- Evaluation protocol: fixed scenario sets, win rate, episode length, seed handling,
  confidence intervals and reporting conventions.
- “Difficulty” and scenario variation ideas for future curricula.

Why: mature cooperative-MARL methodology; no drone dynamics or perception.

### 3. Drone-specific ideas (medium value, medium risk)

From **gym-pybullet-drones**:

- `MultiHoverAviary` / PettingZoo wrapper structure for per-drone observations and
  shared team rewards.
- Example packaging, training scripts, and the way they expose a multi-agent
  PettingZoo environment.
- Their per-drone observation/reward/termination conventions.

From **PyFlyt**:

- 3D waypoint, hover, gate and dogfight task structure.
- Reward/termination ideas for pursuit/evasion and fixed-wing dogfight.
- PettingZoo multi-agent environment patterns.

From **Aerial Gym Simulator**:

- Config-driven environment components, sensor models (LiDAR/camera), parallel
  thousands-of-drone training architecture.
- Useful for v2 high-fidelity sensing/scale, not v0.

Why: these are the right places to look if the project ever needs higher-fidelity
physics or a larger scale. For v0 they are references; swapping the backend would
invalidate the current MuJoCo point-mass assumptions, latency measurements and
C++ architecture.

### 4. Code that is small enough to adapt rather than copy

If you decide to add a PettingZoo adapter, the wrapper is small and mostly
mechanical. Reimplement it against the existing `TeamEnv`/`VecEnv` API and use
PettingZoo's `parallel_api_test` for conformance. That is safer than copying a
wrapper tied to PyBullet/AirSim observation spaces.

Good candidates for adaptation, not wholesale copy:

- PettingZoo `ParallelEnv` wrapper skeleton.
- API/conformance tests.
- Seed/reset and agent-alive bookkeeping helpers.
- Evaluation/report table structure.

## What not to copy into this repository

1. **A second physics engine/backend.** Do not import PyBullet, Isaac Gym or
   AirSim into the v0 runtime. The current C++/MuJoCo point-mass design is a
   deliberate v0 simplification; replacing it would break the spec, latency and
   test assumptions.
2. **Whole env loops from MARSim/MPE2/MAgent2/SMAC.** They encode grid, particle or
   StarCraft assumptions that do not match 3D drone defence, limited sensing or
   variable graph observations.
3. **Old or pinned framework code.** MARSim pins Gymnasium 0.28.1 and Python 3.9+;
   copying it would conflict with the current Python 3.12 and modern Gymnasium/
   PettingZoo APIs. Use current packages instead.
4. **Unlicensed/zero-activity prototypes.** A small repo can be useful as a read,
   but do not copy code/assets unless the licence is explicit and compatible.
5. **Assets/models without a separate licence check.** Drone meshes, textures,
   maps and pretrained checkpoints may have different terms from the code.
6. **Copied files without provenance.** If you copy even a small function, record
   source URL, commit, licence and modifications in `THIRD_PARTY_NOTICES.md`.

## Recommended concrete plan

### If the goal is to finish v0 quickly

Do **not** integrate an external env suite. Use only ideas:

1. Keep the custom `TeamEnv`/`VecEnv` as the single source of truth.
2. Fix primitive tuning and the learner first.
3. Optionally use `MPE2 simple_tag` as a **dev-only** external sanity check for the
   GQN/VDN implementation; do not make it a runtime dependency.
4. Add a `THIRD_PARTY_NOTICES.md` only if you copy code/assets; otherwise just
   document the references in the experiments/paper.

### If the goal is external credibility / publication

1. Add a thin, optional **PettingZoo `ParallelEnv` adapter** around `TeamEnv`
   (or `VecEnv`) and validate it with `pettingzoo.test.parallel_api_test`.
2. Run the GQN/VDN on **MPE2 `simple_tag`** as an external algorithmic sanity
   benchmark, clearly labelled as a 2D sanity check, not the target task.
3. Cite **MARSim**, **MPE2**, **gym-pybullet-drones/PyFlyt/Aerial Gym** and
   **SMAC** in related work, explaining why the custom environment is necessary.
4. Keep external suites in a `bench/` or dev dependency group, not in `swarmsim`
   or the C++ runtime.

### If the goal is higher fidelity or scale later

- **PyFlyt / gym-pybullet-drones** are the most practical physics/control references.
- **Aerial Gym** is the scale/high-fidelity reference, but Isaac Gym/GPU and
  licensing/deprecation need a separate decision.
- Do not mix them into v0; make it a v1/v2 experiment with a clear physics-backend
  interface.

## Concrete reuse checklist

- [ ] Copy only if the licence is MIT/BSD-3/Apache-2.0 and compatible.
- [ ] Preserve the original copyright/licence text.
- [ ] Add `THIRD_PARTY_NOTICES.md` with project, URL, commit, files, licence and
      modifications.
- [ ] Prefer a dependency or a clean reimplementation over copy.
- [ ] Check transitive dependency and asset licences separately.
- [ ] Keep copied code out of the public C++ package interfaces unless necessary.
- [ ] Add tests for any adapted code, especially API conformance and seed/reset
      behaviour.
- [ ] Do not copy code from papers, blog posts or unlicensed repos.
- [ ] Record why the code was reused, not just where it came from.

## Bottom line

The permissive licensing means **you can legally borrow small pieces**, but the
technical mismatch is the real constraint. The highest-value reuse is:

1. PettingZoo/Gymnasium **interface conventions** (optional adapter),
2. MPE2 **external sanity benchmark**,
3. MARSim **task/reward/scenario ideas**,
4. gym-pybullet-drones/PyFlyt/Aerial Gym **drone/physics references**,
5. SMAC **evaluation/reporting conventions**.

The lowest-value, highest-risk move is copying a whole Python/PyBullet/Isaac Gym
environment into this C++/MuJoCo stack. That would slow v0 down rather than speed
it up.
