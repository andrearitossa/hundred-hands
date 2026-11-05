# Follow-up to the independent review

The review is retained unchanged as a snapshot. Its primary conclusion remains correct: M4 acceptance has not been demonstrated.

| Finding | Follow-up |
| --- | --- |
| Primitive sweep fails | Diagnosed a feedback error in the proxy: quantisation discarded continuous PN course corrections. Fixed the proxy, retained the capture assertion. All 12 combinations capture 3/3; retain 20°/3/3 because 30° offers negligible clear-time gain. Original failing output is archived. |
| Short learner runs are inadequate | Started a separate sparse 5M-decision control with update_every=4 and 200 fixed-seed evaluation episodes. Running, not accepted. |
| Exploration collapse | Added Q ranges, action histograms, actor intercept counts and throughput logging. Slow epsilon decay follows the full budget. No claim that this solves learning. |
| Closing-speed shaping | Completed the separate 500k diagnostic and fixed-seed 200-episode native reevaluation; survival remains zero, intercept fraction 0.005. Recorded shaping terms on 20 quantised-expert episodes. |
| Teacher sanity check | Positive expert rollouts verified: 189 intercepts/78 completed episodes. Behaviour clone still fails 200-episode evaluation; logged separately. |
| Missing journals/index/metadata | Added the M2/M3/M4/M5/M7 and validation entries, reports, figures and metadata. Historical missing source snapshots are explicitly disclosed rather than reconstructed. |
| README results placeholder | Regenerated from actual metrics. Three-episode walkthrough rows are integration evidence only. |
| Ignored --suite | Suite selects scenarios/<suite>/eval.yaml; only implemented v0 is accepted. |
| W&B references | Added run URL/artifact resolution, a mocked resolver regression and automatic ONNX export for native evaluation when needed. Live authenticated hosted resolution remains untested. |
| Empty baseline decision logs | Hardened JSON conversion and added a regression using an actual baseline recording. |
| Native metrics versus Python recordings | Documented separate reruns and latency definitions. |
| Python exact pin | Package now declares Python >=3.11; CI matrix covers 3.11 and 3.12. Local recorded validation uses 3.12. |
| CMake visibility | Dependency checker now inspects target links as well as includes. Added standalone CMake entry points and package manifests. |
| Viewer / ARM / full suite | Kept explicit limits: viewer display and ARM are unvalidated; the 16-condition suite is configured, the first M3 table covers one condition. |

Latest build/check logs are in ../2026-11-04_v0-validation/. The 5M control and complete learner outcomes remain the substantive acceptance work.
