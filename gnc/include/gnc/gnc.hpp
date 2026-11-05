#pragma once
#include "core/velocity_control.hpp"
#include "perception/perception.hpp"
#include <memory>
#include <span>
#include <vector>

namespace gnc {
using core::Vec3;
inline constexpr int kNumActions = 27, kNoop = 13;
/// Explicit setpoint memory; action = 9*turn + 3*climb + speed.
struct Primitive {
  double psi = 0, speed = 0, vz = 0;
  void apply(int action, const core::EnvConfig &c);
  Vec3 setpoint() const;
};
/// Minimum total-cost rectangular one-to-one assignment; unmatched rows return
/// -1.
std::vector<int> hungarian(const std::vector<std::vector<double>> &cost);
/// Repeatedly choose the cheapest remaining pair.
std::vector<int> greedy_assign(const std::vector<std::vector<double>> &cost);
/// Constant-velocity interception time (s), infinity when unreachable.
double time_to_intercept(const Vec3 &r, const Vec3 &target_vel, double speed);
enum class Guidance { PurePursuit, PN };
enum class Assignment { Greedy, Hungarian };
/// Guidance velocity (m/s); PN integrates lateral acceleration over dt seconds.
Vec3 guide(const core::EntityState &own, const core::Track &target,
           const Vec3 &previous, const core::EnvConfig &c, Guidance mode,
           double dt);
struct TeamStep {
  perception::ObsBatch observation;
  double reward = 0;
  bool terminated = false, truncated = false;
  int intercepted = 0;
};
/// One world plus explicit perception, decoder and baseline state. No
/// auto-reset.
class TeamEnv {
public:
  explicit TeamEnv(core::EnvConfig c);
  perception::ObsBatch reset(uint64_t seed);
  perception::ObsBatch reset(sim::Scenario scenario, uint64_t seed);
  TeamStep step(std::span<const int> actions);
  /// Quantise a baseline's continuous command for teacher diagnostics; call once per decision.
  std::vector<int> teacher_actions(Guidance guidance, Assignment assignment);
  TeamStep step_baseline(Guidance guidance, Assignment assignment);
  /// Heading (rad) per packed alive agent, used to decode logged ego-frame nodes.
  std::vector<double> headings() const {
    std::vector<double> result;
    for (auto drone : obs_.agent_drone) result.push_back(primitives_[drone].psi);
    return result;
  }
  /// Wall-clock sensing and graph building durations (ms), across this team.
  std::pair<double,double> timings() const { return {tracking_ms_,graph_ms_}; }
  const sim::Env &world() const { return env_; }
  /// Local tracker estimates in packed alive-agent order, for diagnosis and parity tests.
  std::vector<std::vector<core::Track>> local_tracks() const {
    std::vector<std::vector<core::Track>> result;
    for (auto drone : obs_.agent_drone) result.push_back(trackers_[drone].tracks());
    return result;
  }
  const std::vector<core::Track> &fused() const { return fused_; }
  const perception::ObsBatch &observation() const { return obs_; }

private:
  std::vector<Vec3> baseline_commands(Guidance guidance, Assignment assignment);
  void initialise();
  void sense();
  TeamStep advance(const std::vector<Vec3> &commands);
  core::EnvConfig cfg_;
  sim::Env env_;
  std::vector<Primitive> primitives_;
  std::vector<perception::Tracker> trackers_;
  std::vector<core::Track> fused_;
  perception::ObsBatch obs_;
  std::vector<Vec3> previous_;
  std::vector<int> assigned_, threat_ids_, alive_ids_;
  double next_assignment_ = 0;
  int next_fused_id_ = 0;
  double tracking_ms_ = 0, graph_ms_ = 0;
};
struct VecStep {
  perception::ObsBatch observation;
  std::vector<TeamStep>
      transitions; ///< Final observations before auto-reset, one per env.
};
/// Independent worlds advanced by a fixed pool; actions follow packed agent
/// order.
class VecEnv {
public:
  VecEnv(core::EnvConfig c, int count, int threads = 0);
  ~VecEnv();
  VecEnv(const VecEnv &) = delete;
  VecEnv &operator=(const VecEnv &) = delete;
  perception::ObsBatch reset(uint64_t seed);
  VecStep step(std::span<const int> actions);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace gnc
