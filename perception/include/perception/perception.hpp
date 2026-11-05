#pragma once
#include <cstdint>
#include <span>
#include <vector>

#include "core/config.hpp"
#include "core/types.hpp"
#include "sim/env.hpp"

namespace perception {
using core::Detection;
using core::Track;
using core::Vec3;

/// True position of every active attacker within `range` (m) of `sensor`. Clean in v0.
std::vector<Detection> detect(const sim::Env& env, const Vec3& sensor, double range);

/// Constant-velocity Kalman tracks with gated nearest-neighbour association. Memory lives here.
class Tracker {
 public:
  explicit Tracker(const core::EnvConfig& c) : cfg_(&c) {}
  void update(double t, std::span<const Detection> detections);  ///< Scan at time t (s).
  const std::vector<Track>& tracks() const { return tracks_; }

 private:
  const core::EnvConfig* cfg_;
  std::vector<Track> tracks_;
  double last_t_ = 0;
  int next_id_ = 0;
};

/// Union of track lists, merging those within fus_pos (m) and fus_vel (m/s) of each other.
std::vector<Track> fuse(std::span<const std::vector<Track>* const> lists, const core::EnvConfig& c);

/// Packed observation graphs. One float array per node kind, nodes grouped by owning agent.
struct ObsBatch {
  std::vector<float> self, local_threat, asset, global_threat, teammate;
  std::vector<int64_t> local_owner, global_owner, teammate_owner, teammate_src;
  std::vector<int64_t> agent_env, agent_drone, alive_mask;  ///< Packed agents are alive; removed agents are absent.
  int64_t n_agents() const { return static_cast<int64_t>(agent_env.size()); }
  void clear();
  void append(const ObsBatch& o);  ///< Append, shifting agent indices.
};

/// What a defender knows of itself or a teammate. psi: heading of the velocity setpoint (rad).
struct Ego {
  Vec3 pos, vel;
  double psi;
};

/// Append one agent's ego-centric graph. `team` lists all present defenders, in agent order.
void build_agent(ObsBatch& b, int agent, int64_t env, int64_t drone, bool g, const std::vector<Track>& local,
                 const std::vector<Track>& fused, const Vec3& asset, std::span<const Ego> team,
                 const core::Limits& lim);
}  // namespace perception
