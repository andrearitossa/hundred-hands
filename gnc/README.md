# GNC and team environments

`core::velocity_control` is the common controller used by red and blue in `sim`.
`gnc` composes `sim` and `perception` without introducing a reverse dependency into
`sim`. Public C++ entry points are in `include/gnc/gnc.hpp`.

```python
import swarmsim

cfg = swarmsim.EnvConfig.load("scenarios/v0/env.yaml")
env = swarmsim.VecEnv(cfg, count=16, threads=4)
obs = env.reset(123)
result = env.step([swarmsim.NOOP] * len(obs["self"]))
obs = result["observation"]
```

`step` takes exactly one action per packed alive agent, in `agent_env`,
`agent_drone` order. Removed agents are absent; `alive_mask` is one for each
packed agent. Action `9*turn + 3*climb + speed` uses indices 0/1/2 for negative,
zero, positive increments. No-op 13 keeps the current setpoint; from reset it
hovers. Decoder heading remains defined at zero speed.

Each graph contains contiguous read-only NumPy views of C++ arrays: `self`,
`asset`, `local_threat`, `global_threat`, `teammate`, the three `*_owner`
indices, `teammate_src`, `agent_env`, `agent_drone`, and `alive_mask`. Feature
columns follow the generated observation schema. Teammate sources index packed
agents across the entire batch. Storage belongs to a Python capsule, so arrays
remain valid after another step or destruction of the environment. No feature
array is copied while exposing its buffer to Python. Building/packing graphs
still allocates and copies internal C++ vectors.

`result["transitions"]` has one entry per environment: `reward`, `terminated`,
`truncated`, `intercepted`, and `observation`. This observation is the next graph
**before** auto-reset. `result["observation"]` instead contains reset graphs for
finished environments and next graphs for the others. Bootstrap truncations
from the transition observation; zero the bootstrap on termination. Reward is
intercepts / initial red count, minus asset hit, plus intact terminal success.

`TeamEnv` exposes `reset(seed)`, `step(actions)`, and
`step_baseline(Guidance.PN, Assignment.GREEDY)` for a single world, with no
auto-reset. Baselines use the same fused tracks as the graph. They reassign at
the YAML interval and on threat or defender membership changes. PN integrates
lateral acceleration and holds speed, using pursuit to initialise or reacquire
an opening course. Unassigned defenders move toward the configured guard point.

`VecEnv` has a fixed worker pool and releases the Python GIL during reset and
step. Results are assembled in environment order. Reset seed `s` starts env `i`
at `s+i`; auto-resets advance its seed by the environment count. Decoder,
tracker, fused identity, assignment, random generator and world state belong to
the individual environment. Calls on one VecEnv are serialised.

Run `cmake --build --preset release && ctest --preset release`, then `uv run
pytest`. The reproducible small baseline/throughput diagnosis lives in
`experiments/2026-11-04_m2-m3-gnc/diagnose.py`. Full suite evaluation, action tuning,
learner, ONNX runtime and policy interpretability are later work.
