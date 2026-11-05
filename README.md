# Hundred Hands

A C++ point-mass drone-defence environment with a shared graph Q-network learner
in Python. Three defenders protect one asset against scripted attackers. The
scope and acceptance checks are in [SPEC_v0](docs/SPEC_v0.md).

Use Linux x86_64, Python 3.12, CMake ≥3.24, Ninja, a C++20 compiler, and `uv`.
The first build downloads pinned MuJoCo and CPU ONNX Runtime distributions.

```bash
uv sync --locked
cmake --preset release
cmake --build --preset release -j 4
uv run pytest -q
ctest --preset release
uv run swarm-walkthrough
```

Or run the entire build, test and demonstration chain with one command:

```bash
bash tools/walkthrough.sh
```

The short walkthrough trains a small policy, saves a checkpoint, exports ONNX,
checks C++ parity on 1,000 observations, evaluates it and six baselines in C++,
records every policy decision into NPZ, and generates `report.md` and
`replay.html` under `experiments/2026-11-04_m7-end-to-end/`. Open the HTML file
in a browser. This short run proves integration; it does not prove useful
learning. Longer learner checks have their own experiment journals.

```bash
uv run swarm-train --config scenarios/v0/train.yaml --out experiments/my-training
uv run swarm-export experiments/my-training/best.pt
uv run swarm-eval --run experiments/my-training --suite v0 --out experiments/my-eval
uv run swarm-eval --run experiments/my-training --cpp --record 0 --out experiments/my-cpp-eval
uv run swarm-report experiments/my-eval
uv run swarm-replay experiments/my-eval/episodes/gqn-3-straight-0.npz -o replay.html
uv run swarm-replay experiments/my-eval/episodes/gqn-3-straight-0.npz --viewer
uv run swarm-bench --what env --out experiments/my-throughput
uv run swarm-bench --what latency --out experiments/my-latency
```

Training and evaluation default to **offline W&B**: metrics and model artifacts
are retained locally without requiring credentials. Set `wandb_mode: online` in
the run configuration to use a configured W&B account; existing offline runs
can be synced with `wandb sync <run-directory>`. Local logging requires permission
to open a local socket. Do not change to disabled logging to conceal failures.

See [architecture and inspection](docs/ARCHITECTURE.md),
[known limits](docs/KNOWN_LIMITS.md), and the [experiment index](experiments/README.md).

<!-- results:start -->
| Experiment | Policy | Red / behaviour | Episodes | Asset survival |
| --- | --- | --- | ---: | ---: |
| [2026-11-04_m3-baselines](experiments/2026-11-04_m3-baselines/report.md) | idle | 3 / straight | 1000 | 0.0% |
| [2026-11-04_m3-baselines](experiments/2026-11-04_m3-baselines/report.md) | random | 3 / straight | 1000 | 0.0% |
| [2026-11-04_m3-baselines](experiments/2026-11-04_m3-baselines/report.md) | pp_greedy | 3 / straight | 1000 | 41.0% |
| [2026-11-04_m3-baselines](experiments/2026-11-04_m3-baselines/report.md) | pn_greedy | 3 / straight | 1000 | 57.3% |
| [2026-11-04_m3-baselines](experiments/2026-11-04_m3-baselines/report.md) | pp_hungarian | 3 / straight | 1000 | 41.9% |
| [2026-11-04_m3-baselines](experiments/2026-11-04_m3-baselines/report.md) | pn_hungarian | 3 / straight | 1000 | 57.0% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | idle | 3 / straight | 3 | 0.0% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | random | 3 / straight | 3 | 0.0% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | pp_greedy | 3 / straight | 3 | 66.7% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | pn_greedy | 3 / straight | 3 | 100.0% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | pp_hungarian | 3 / straight | 3 | 66.7% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | pn_hungarian | 3 / straight | 3 | 100.0% |
| [2026-11-04_m7-end-to-end](experiments/2026-11-04_m7-end-to-end/report.md) | gqn | 3 / straight | 3 | 0.0% |
<!-- results:end -->
