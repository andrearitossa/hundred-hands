# Current v0 limits and acceptance status

- Validation here is Linux x86_64 on WSL2. CMake currently fetches the Linux x64
  ONNX Runtime binary. ARM deployment and macOS are not validated; latency here
  is explicitly x86, as permitted by milestone M5.
- W&B runs are offline, including local model artifacts and git revision tags.
  No hosted URL is claimed. Sync requires the user's configured account.
- Simulation uses X500-derived point-mass acceleration limits; speed limits
  remain assumptions. There is no wind, sensor noise or safety layer.
- Sampled attacker approaches are conditioned on a clear building corridor so
  a lone attacker can reach the asset without defence. Hand-authored blocked
  routes still collide. This sampling correction invalidates older baseline
  tables as measurements of the current sampler.
- C++→Python graph buffers are views with owned lifetimes; graph construction,
  concatenation, replay storage and GPU transfer still copy data.
- The ONNX exporter uses the tested TorchScript export path. PyTorch emits a
  deprecation warning; a future exporter migration must retain dynamic/empty
  graph parity before replacing it.
- Baseline latency currently includes its environment step; learned-policy
  inference timings exclude that step. Use the dedicated latency sweep for
  graph-building + inference measurements, with tracking reported separately.
- The latency sweep reports CPU runtime plus graph-building time and Python
  binding overhead. It excludes physics and tracking. Measurements on a shared
  development machine are engineering evidence, not an embedded benchmark.
- Training is feed-forward and CPU by default. Saving checkpoints includes
  optimizer/RNG state, but exact environment snapshot/resume is v1 work.
- Full v0 evaluation configuration includes all red counts and behaviours.
  A short walkthrough evaluates only 3-vs-3 straight. A short run completing
  the chain must never be described as proving the M4 learner acceptance.
- The MuJoCo viewer requires a graphical session. Headless HTML replay is the
  default; visual viewer validation requires available display hardware.
- C++ packages support standalone builds and colcon via explicit base paths.
  They build their source dependency closure; installed CMake config exports for
  external consumers are not yet provided.
- This workspace began with most implementation files untracked. Implementation,
  documentation and experiment evidence are committed separately at the user’s
  request. Live outputs of the active 5M run remain local; its progress snapshot
  is versioned. Committing the repository does not establish M4 acceptance.

- M4 acceptance is outstanding: completed short sparse/shaped and teacher-cloning diagnostics do not beat idle/random on asset survival. A separate 5M-decision, 200-evaluation-episode control is running; its outcome is not yet claimed.
