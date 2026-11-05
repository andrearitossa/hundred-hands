# M4 spec-scale sparse control

Status: **running; acceptance not established**. Configuration: `train.yaml`; command:

```bash
uv run swarm-train --config experiments/2026-11-04_m4-spec-scale/train.yaml --out experiments/2026-11-04_m4-spec-scale
```

5M environment decision steps, hidden 64, uniform replay capacity 1M, batch 64, update_every=4, five-step DDQN+VDN, 16 environments/4 worker threads, seed 45, 200 evaluation episodes at seed 10000000 every 100k decisions. Straight-only stage deliberately isolates the M4 check. Default sparse shared reward and 20°/3/3 primitives; no teacher assistance or shaping.

At both 100k and 200k decisions survival was 0/200; actor rollouts had produced one intercept by 200k. The run is expected to take hours at the measured CPU update throughput. `history.jsonl` is live evidence; `metrics.json` is written only at completion by the version launched here. Early run metadata has no source manifest; later code changes do not modify this running process. The launch settings are preserved in config.yaml and train.yaml. Do not treat a directory or checkpoint existing as a successful result.

`compile-profile.log` and `jit-profile.log` (where present) compare algorithm-preserving CPU execution alternatives. Compilation offered no useful measured speedup and was not enabled in this control.

Git stores `launch-meta.json` and `progress-snapshot.jsonl` as the pre-commit snapshot. Live history, logs, checkpoints and eventual final metrics remain local and ignored while training runs; committing final evidence is a separate step after completion.
