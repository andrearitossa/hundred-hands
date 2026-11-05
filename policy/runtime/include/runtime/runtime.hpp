#pragma once
#include "gnc/gnc.hpp"
#include <memory>
#include <string>

namespace runtime {
struct Output {
  int64_t heads = 0, message_dim = 0;
  std::vector<float> q, local_attention, global_attention, teammate_attention, messages;
};
/// CPU ONNX inference over exactly the packed perception format, one thread per session.
class Runtime {
public:
  explicit Runtime(const std::string& path);
  ~Runtime();
  Output forward(const perception::ObsBatch& graph, std::span<const float> mask = {});
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
struct EpisodeMetrics {
  bool survived, truncated;
  double intercept_fraction, defender_survival, episode_s;
  double first_intercept_s = -1, clear_s = -1;
  std::vector<double> latency_ms;
};
/// Run an entire episode in C++ with runtime or baseline decisions; no Python callback.
EpisodeMetrics evaluate(core::EnvConfig cfg, uint64_t seed, const std::string& policy,
                        const std::string& model_path);
} // namespace runtime
