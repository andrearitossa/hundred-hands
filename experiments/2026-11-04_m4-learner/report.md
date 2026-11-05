# M4 learner diagnosis

M4 acceptance is **not met**. All completed short learner diagnostics have zero asset survival. These runs are below the spec's starting 5M-decision budget and are not substitutes for it.

| Experiment | Decisions | Final evaluation episodes | Survival | Intercept fraction |
| --- | ---: | ---: | ---: | ---: |
| smoke | 8,000 | 10 | 0.000 | 0.000 |
| sparse | 200,000 | 30 | 0.000 | 0.000 |
| shaped | 250,000 | 30 | 0.000 | 0.000 |
| closing | 500,000 | 30 | 0.000 | 0.011 |

See each child directory's config.yaml, meta.json, history.jsonl, metrics.json and model checkpoints. Seeds 42/43/44 identify the separate sparse/potential/closing controls; the files are authoritative for settings. Training and evaluation seeds are disjoint.

Sparse uses update_every=16, potential shaping uses 8; both are engineering diagnostics. Closing-speed shaping is separately labelled and does not alter the default sparse reward. Flat Q-values and dominant actions were observed in early checkpoints. New full-budget runs log Q range, action histograms, actor intercepts, episode counts and decision throughput every 10k decisions. Best checkpoint selection now breaks survival ties using intercept fraction, and future runs persist each evaluation immediately.

## Teacher sanity check

The quantised PN+greedy expert produced 189 intercepts across 78 completed episodes (67.95% asset survival). Therefore the primitive environment can supply positive intercept rewards. The separate behaviour-cloning diagnostic reduced cross entropy but its greedy policy scored 0/200 survival and zero intercepts. Teacher success is not learner success. Covariate shift and joint-action prediction remain possible causes; this result does not establish either one.

`teacher/shaping-check.json` records sparse, potential and closing-speed terms on another 20 expert episodes. Positive closing-speed terms confirm the intended direction on successful approaches; they do not prove that shaping solves exploration or preserves the sparse objective.

## Full-budget control

[The separate 5M run](../2026-11-04_m4-spec-scale/README.md) uses hidden width 64, one update per four decisions, 200 evaluation episodes and straight attackers. It retains the frozen 20°/3/3 primitives. Completion and success must be established from that run's metrics, not inferred from this journal.

CPU eager, traced and compiled update profiles are in that directory. Compilation did not improve the measured update time, so the control retains eager execution. Checkpoints and W&B are local/offline. No hosted run is claimed.

## Fixed 200-episode reevaluation

The completed sparse, potential, closing-speed and teacher checkpoints were reevaluated through the C++ ONNX runtime on seeds 10000000–10000199, 3-vs-3 straight. See each child's evaluation-200.json and evaluation.onnx. This supersedes the 30-episode diagnostic score for final checkpoint comparison.

| Checkpoint | Episodes | Asset survival | Intercept fraction |
| --- | ---: | ---: | ---: |
| sparse | 200 | 0.000 | 0.000 |
| shaped | 200 | 0.000 | 0.000 |
| closing | 200 | 0.000 | 0.005 |
| teacher | 200 | 0.000 | 0.000 |

All four remain below M4 acceptance. Closing-speed shaping yields a few intercepts without asset survival. Neither positive expert trajectories nor lower training loss establishes learned defence.

Dataset aggregation has now completed all six rounds. Its final 200-episode evaluation still has 0% asset survival and zero intercepts; see `dagger/metrics.json`. This failed teacher-assisted diagnostic does not satisfy M4.
