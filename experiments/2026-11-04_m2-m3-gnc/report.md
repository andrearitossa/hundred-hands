# GNC, TeamEnv and VecEnv diagnosis

## Implementation and inspection

Replaced the truncated `gnc.hpp` with the primitive decoder, PP/PN guidance,
rectangular Hungarian and greedy assignment, TeamEnv, and a fixed-pool VecEnv.
The existing shared `core::velocity_control` remains the controller for both
sides. TeamEnv owns tracker, fused identity, decoder, guidance and assignment
state. VecEnv belongs in `gnc`, above perception and simulation, resolving the
M2 dependency-placement question without a cycle.

Python receives contiguous read-only graph buffers with capsule-owned storage.
Old buffers survive later steps and environment destruction. Alive agents are
packed without padding; all packed alive-mask entries are one. Agent owners and
teammate sources are offset when graphs are concatenated. Terminal/truncated
transition graphs remain distinct from automatically reset acting graphs.
Team rewards follow SPEC §7.3, and bug-guard truncation receives no success bonus.

## Commands and checks

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/hundred-hands-uv uv sync --offline
cmake --preset release
cmake --build --preset release -j 4
ctest --preset release
cmake -S . -B build/python -DSWARM_PYTHON=ON -DSWARM_TESTS=OFF \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/python -j 4
cmake --install build/python --prefix "$PWD/.venv/lib/python3.12/site-packages"
.venv/bin/python -m pytest -q
.venv/bin/python tools/codegen/generate.py --check
.venv/bin/ruff check tests/test_team_env.py experiments/2026-11-04_m2-m3-gnc/diagnose.py
MPLCONFIGDIR=/tmp/hundred-hands-mpl .venv/bin/python experiments/2026-11-04_m2-m3-gnc/diagnose.py
```

26/26 C++ tests pass ([log](ctest.log)). Coverage includes all primitive axis
combinations, controller convergence and limits, PN interception of a
constant-velocity target, Hungarian versus exhaustive search for 1–5 rows and
columns with negative costs, rectangular and unreachable assignments, defender
loss and empty teams, hit/intercept rewards, truncation, unique fused IDs,
parallel/serial equality over repeated auto-resets and teammate offsets.

6/6 Python tests pass ([log](pytest.log)). This includes the existing 600-episode
idle invariant (200 seeds for each attacker behaviour), exact float32 self and
teammate features across the binding, graph buffer layout and lifetime, final
observations on auto-reset, and rejection of invalid actions before advancing
any environment. Codegen and Ruff checks pass.

## Bugs found and fixes

- The starting GNC header ended mid-comment and had no implementation or build
  target. It is replaced and wired into CMake and pybind11.
- Tracker IDs are sensor-local. Two separated threats seen by two different
  sensors can both have ID zero. Baseline assignment initially confused them.
  TeamEnv now associates consecutive fused pictures by predicted proximity,
  assigning unique persistent internal identities. A separated-sensor regression
  test checks uniqueness and continuity. This changes no graph feature.
- PN needs an initial course; a zero velocity cannot define lateral steering.
  Initialise with pursuit and reacquire an opening target with pursuit, then
  integrate PN acceleration at the decision interval.
- Python initially imported the old installed extension even with a build-tree
  PYTHONPATH. Rebuilding alone was insufficient in the existing editable install;
  explicitly installing the rebuilt module into the local venv resolved it.
- The default uv cache is read-only in this workspace. The `/tmp` cache works;
  the already installed dependencies and fetched CMake sources permit offline
  builds. Matplotlib likewise uses a writable temporary cache.
- Removed a duplicate pre-existing Python `Env.waypoints` binding.

## First small baseline table

20 seeds per policy, seeds 10000–10019, three defenders and exactly three
straight attackers, default buildings and spawn staggering. Full configuration
is [config.yaml](config.yaml); per-episode records are in [metrics.json](metrics.json).

| Policy | Asset survival | Intercepted / initial attackers | Truncated |
| --- | ---: | ---: | ---: |
| Idle | 0% | 0% | 0 |
| Random primitives | 0% | 0% | 0 |
| PP + greedy | 25% | 48.3% | 0 |
| PN + greedy | 60% | 73.3% | 0 |
| PP + Hungarian | 20% | 46.7% | 0 |
| PN + Hungarian | 60% | 73.3% | 0 |

PN + greedy is inside the suggested 40–80% engineering range on this small
condition. Retain the default capture probability/radius and primitive steps;
this does not establish that the primitive magnitudes are optimal. Random uses
an explicitly seeded NumPy generator; all physical episode randomness remains
inside the env. Guard points and all guidance gains come from YAML.

## Short throughput sweep

Host: Intel i7-1165G7, WSL2 x86_64, four visible logical CPUs. 100 batched decisions
per point after reset; reported time includes Python dispatch, physics,
perception, packing and buffer wrapping, and excludes initial reset.
Measurements overlap the invariant test run, so this is a diagnostic throughput
record rather than a controlled benchmark.

| Envs / worker threads | Env decision steps / s | Per visible logical CPU / s |
| --- | ---: | ---: |
| 1 / 1 | 3588 | 897 |
| 4 / 4 | 7046 | 1761 |
| 16 / 16 | 9854 | 2463 |
| 32 / 32 | 9760 | 2440 |

16 environments are a useful starting point on this host; 32 bring no measured
benefit here. Thread count can be selected independently from environment count.
A longer isolated sweep should choose production defaults.

## Scope and remaining acceptance work

This entry completes the requested GNC and environment integration, not all v0
milestones. The baseline table uses 20 episodes per policy and one condition;
SPEC §8's ≥1000 episodes, confidence intervals and full behaviour/count suite
remain for the evaluation runner. Primitive tuning, controlled throughput runs,
learner, ONNX runtime and interpretability overlays remain later work. No W&B
run was created for this local engineering diagnosis (recorded as null in
metadata); that integration is still required for the evaluation runner.

All exposed graph arrays avoid a copy at the Python buffer boundary, but
internal graph building and packing still copy vectors. VecEnv's transition
list includes next graphs for every environment, including unfinished episodes.
Fused identity association is proximity-based and can swap near crossings;
IDs are for baseline continuity only and never reach the policy. PN uses
estimated tracker velocity, never ground truth. Baselines have no safety layer
or building avoidance, as specified for v0.
