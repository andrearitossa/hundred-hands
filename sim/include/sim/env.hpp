#pragma once
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "core/config.hpp"
#include "core/types.hpp"

namespace sim {
using core::Vec3;

struct Box { Vec3 center, half; };
enum class Behaviour { Straight, Waypoint, Evasive };
struct RedSpawn { Vec3 pos; double time, cruise; Behaviour behaviour; };

/// Everything randomised at reset. Sampled, or hand-written by tests.
struct Scenario {
  Vec3 asset;
  std::vector<Box> buildings;
  std::vector<Vec3> blue;
  std::vector<RedSpawn> red;
};
Scenario sample_scenario(const core::EnvConfig& c, std::mt19937_64& rng);

enum class Status { Waiting, Active, Gone };
enum class EventKind { Spawn, TrialFail, Intercept, Lost, AssetHit, Jink, Truncated };
enum class LossCause { None, Ground, Building, OutOfArea, Teammate, Intercepted };

/// Drone indices are global: blue are [0, n_blue), red are [n_blue, n_blue + n_red).
struct Event { double t; EventKind kind; int drone; int other; LossCause cause; };

struct StepResult { int intercepted; bool asset_hit, done, truncated; };

/// One episode world: MuJoCo point masses, red scripts, contact and intercept rules.
class Env {
 public:
  explicit Env(core::EnvConfig cfg);
  ~Env();
  Env(const Env&) = delete;
  Env& operator=(const Env&) = delete;

  void reset(uint64_t seed);                       ///< Sample a scenario and start it.
  void reset(Scenario scenario, uint64_t seed);    ///< Start a given scenario.
  StepResult step(std::span<const Vec3> blue_vcmd);  ///< One decision; setpoints (m/s), world frame.

  const core::EnvConfig& config() const { return cfg_; }
  const Scenario& scenario() const { return sc_; }
  double time() const { return t_; }
  int n_blue() const { return static_cast<int>(sc_.blue.size()); }
  int n_red() const { return static_cast<int>(sc_.red.size()); }
  int n_drones() const { return n_blue() + n_red(); }
  core::EntityState state(int drone) const;
  Status status(int drone) const { return status_[drone]; }
  const std::vector<Event>& events() const { return events_; }
  const std::vector<std::vector<Vec3>>& waypoints() const { return waypoints_; }
  bool done() const { return done_; }

 private:
  struct Model;
  void build_world();
  void set_collidable(int drone, bool on);
  void remove(int drone, LossCause cause, int other = -1);
  void update_spawns();
  void check_contacts();
  void check_rules(StepResult& out);
  Vec3 red_setpoint(int red);
  bool segment_clear(const Vec3& a, const Vec3& b) const;

  core::EnvConfig cfg_;
  std::mt19937_64 rng_;
  std::unique_ptr<Model> m_;
  Scenario sc_;
  std::vector<Status> status_;
  std::vector<Event> events_;
  std::vector<char> armed_;  // n_blue x n_red: pair may draw a new intercept trial
  std::vector<std::vector<Vec3>> waypoints_;
  std::vector<size_t> wp_next_;
  std::vector<double> jink_until_, jink_ready_;
  std::vector<Vec3> jink_axis_;
  std::vector<double> jink_angle_;
  double t_ = 0;
  int decisions_ = 0;
  bool done_ = true;
};
}  // namespace sim
