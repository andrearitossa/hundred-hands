# Validation report

| Check | Result |
| --- | --- |
| Release C++ tests | 28/28 pass |
| Python tests | 23/23 pass; two legacy exporter warnings |
| Mypy / Ruff | Pass |
| Codegen / dependency boundaries | Pass |
| Standalone core configure/build/install | Pass |
| Colcon discovery and build/install | Five packages pass |
| Fresh source walkthrough | Complete chain passes; see clean-snapshot.json |
| M4 useful learned defence | Outstanding; no completed learner diagnostic meets acceptance |

Logs, machine metadata and the actual clean-checkout artifacts are retained beside this report. The clean source snapshot predates later diagnostic logging and report-layout refinements; it is identified by its own temporary git commit and source manifest. The user's original git state was not committed or cleaned.
