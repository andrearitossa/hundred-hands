# v0 data flow and inspection

`core → sim → perception → gnc → runtime`; bindings expose these systems to
Python. No C++ package includes another package's implementation directory.
`python tools/check_dependencies.py` checks include boundaries. The shared
velocity controller lives in core so red scripts and defender control use the
same implementation without a dependency cycle.

| Piece | Input → output | Run / inspect |
| --- | --- | --- |
| Scenario and simulation | Env YAML + seed → MuJoCo world, states, events | `swarm-diag sim`; `Env.snapshot/scenario/events` |
| Perception | Clean range-limited detections → Kalman tracks → fused tracks | `TeamEnv.local_tracks/fused_tracks`; perception tests |
| Graph construction | Ego pose, tracks, asset, teammates → packed float32 graphs | `TeamEnv.reset/step`; generated schema and graph equality tests |
| Primitives/controller | 27 actions → remembered velocity setpoints → bounded accelerations/forces | GNC unit tests; actions stored in decision logs |
| Baselines | Same fused tracks → assignment → PP/PN setpoints | `swarm-eval --cpp`; config selects scenarios and fixed seeds |
| VecEnv | Packed actions → team transitions + auto-reset acting observations | `swarm-bench --what env`; `gnc/README.md` documents seed and ownership rules |
| GQN | Packed graphs → local embeddings → one message round → global branch → 27 Q-values | `swarm-train`; `model.py`, checkpoints, policy tests |
| Replay/learner | Team transitions → n-step returns → Double DQN/VDN loss → shared weights | `history.jsonl`, `metrics.json`, W&B; toy target and truncation tests |
| Export/runtime | Checkpoint + dynamic graph inputs → ONNX → C++ Q-values | `swarm-export`; saved observation NPZ and parity JSON |
| Evaluation | Policy/baseline + fixed suite seeds → episode metrics and confidence intervals | `swarm-eval`; `--cpp` runs each full episode in C++ |
| Decision logging | Graphs + network outputs + interventions → NPZ with JSON decision records | `numpy.load` with no pickle; every decision is retained |
| Report/replay | Metrics / recorded states and decisions → Markdown, plots, HTML/viewer | `swarm-report`, `swarm-replay` |

The observation schema is `scenarios/schema/obs_schema.yaml`. Regenerate with
`python tools/codegen/generate.py`; `--check` rejects stale C++/Python layouts.
No IDs enter policy feature tensors. Teammate source indices gather same-step
messages, so gradients pass through sending agents during training. Empty
relations use learned vectors; an isolated internal sentinel allows portable
ONNX empty-relation kernels without capping or padding the graph to a maximum.

Replay stores one whole team per transition. The n-step accumulator flushes at
both episode boundaries; only termination removes the bootstrap. Training uses
the pre-reset transition graph for truncations. Dead agents disappear from both
the graph and VDN sum. A stage change resets actor worlds and their n-step
accumulators. Checkpoints include model settings, weights, target network,
optimizer state, RNG state, configuration, step count and curriculum stage.

Evaluation seeds start at 10,000,000 and are reserved from actor seed streams;
export/benchmark seeds start at 20,000,000. The training code checks that the
requested budget cannot exhaust the reserved boundary. Default training is
3-vs-3 for the M4 check; use `red_count: 0` for the environment's 3–6 curriculum.
The curriculum has configurable weights, advancement thresholds and budgets.
An optional shared potential reward is disabled by default and must be logged
as a separate experiment if used.

NPZ contains numeric world state/events and ragged routes encoded with offsets.
The `decisions` string array contains one JSON record per decision, including
all packed graph arrays, ego headings, all Q-values/actions/margins, per-head
attention, message embeddings, per-teammate influence/argmax changes, and full
Q differences under global-view removal. JSON avoids pickled Python objects.
HTML shows the numeric decision record and local/global attention lines plus
teammate connections. `--viewer` opens the same states in MuJoCo and overlays
global threat attention. A display/OpenGL environment is needed for the viewer.

`swarm-walkthrough` composes the real systems with a small training/evaluation
budget; `tools/walkthrough.sh` precedes it with installation and all checks.
CI executes a shorter version. Scientific training and full evaluation suites
are separate from this integration demonstration.

`swarm-eval --run` resolves local checkpoints/directories, W&B run URLs, and model artifact references. Hosted resolution needs configured credentials; tests use a mocked API. Native evaluation exports a missing companion ONNX file automatically. Under `--cpp`, metrics come from native episodes while requested recordings are separate Python reruns of the same seeds.

C++ source packages are discoverable with `colcon list --base-paths core sim perception gnc policy/runtime`; use the same base paths for `colcon build`. Standalone entry points build the source dependency closure and install headers/static libraries. They do not provide installed CMake config exports for unrelated downstream projects.
