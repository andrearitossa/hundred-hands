# M5 export and native runtime

Dynamic ONNX export and C++ CPU inference pass parity on 1,000 recorded observations. A separate empty-team/empty-relation regression also passes. The 16/32/64 width latency sweep meets the configured 2x margin on this x86 host; 64 is the largest passing candidate. ARM remains unmeasured. See report.md and gqn.observations.npz.
