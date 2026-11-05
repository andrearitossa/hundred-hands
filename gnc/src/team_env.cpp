#include "gnc/gnc.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace gnc {
TeamEnv::TeamEnv(core::EnvConfig c) : cfg_(c), env_(c) {}
perception::ObsBatch TeamEnv::reset(uint64_t seed) {
  env_.reset(seed);
  initialise();
  return obs_;
}
perception::ObsBatch TeamEnv::reset(sim::Scenario scenario, uint64_t seed) {
  env_.reset(std::move(scenario), seed);
  initialise();
  return obs_;
}
void TeamEnv::initialise() {
  primitives_.assign(env_.n_blue(), {});
  previous_.assign(env_.n_blue(), Vec3::Zero());
  assigned_.assign(env_.n_blue(), -1);
  threat_ids_.clear();
  alive_ids_.clear();
  next_assignment_ = 0;
  next_fused_id_ = 0;
  fused_.clear();
  trackers_.clear();
  for (int i = 0; i <= env_.n_blue(); ++i)
    trackers_.emplace_back(cfg_);
  sense();
}
void TeamEnv::sense() {
  const auto started=std::chrono::steady_clock::now();
  std::vector<const std::vector<core::Track> *> lists;
  std::vector<perception::Ego> team;
  std::vector<int> drones;
  for (int i = 0; i < env_.n_blue(); ++i)
    if (env_.status(i) == sim::Status::Active) {
      const auto s = env_.state(i);
      trackers_[i].update(env_.time(),
                          perception::detect(env_, s.pos, cfg_.onboard_range));
      lists.push_back(&trackers_[i].tracks());
      team.push_back({s.pos, s.vel, primitives_[i].psi});
      drones.push_back(i);
    }
  trackers_.back().update(
      env_.time(),
      perception::detect(env_, env_.scenario().asset, cfg_.ground_range));
  lists.push_back(&trackers_.back().tracks());
  auto fused = perception::fuse(lists, cfg_);
  // Tracker IDs are local to each sensor. Associate the fused picture
  // separately so two different sensors' track 0 cannot become the same
  // baseline assignment.
  std::vector<std::vector<double>> costs(
      fused.size(), std::vector<double>(fused_.size(), INFINITY));
  for (size_t i = 0; i < fused.size(); ++i)
    for (size_t j = 0; j < fused_.size(); ++j) {
      const double distance =
          (fused[i].pos - fused_[j].pos - cfg_.decision_dt() * fused_[j].vel)
              .norm();
      if (distance <= cfg_.trk_gate)
        costs[i][j] = distance;
    }
  const auto matched = greedy_assign(costs);
  for (size_t i = 0; i < fused.size(); ++i)
    fused[i].id = matched[i] < 0 ? next_fused_id_++ : fused_[matched[i]].id;
  fused_ = std::move(fused);
  obs_.clear();
  const auto graph_started=std::chrono::steady_clock::now();
  tracking_ms_=std::chrono::duration<double,std::milli>(graph_started-started).count();
  for (int i = 0; i < int(drones.size()); ++i)
    perception::build_agent(obs_, i, 0, drones[i], true,
                            trackers_[drones[i]].tracks(), fused_,
                            env_.scenario().asset, team, cfg_.limits);
  graph_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-graph_started).count();
}
TeamStep TeamEnv::advance(const std::vector<Vec3> &commands) {
  const auto r = env_.step(commands);
  sense();
  const double reward = double(r.intercepted) / std::max(1, env_.n_red()) -
                        double(r.asset_hit) +
                        double(r.done && !r.truncated && !r.asset_hit);
  return {obs_, reward, r.done && !r.truncated, r.truncated, r.intercepted};
}
TeamStep TeamEnv::step(std::span<const int> actions) {
  if (env_.done())
    throw std::logic_error("reset required before step");
  if (actions.size() != size_t(obs_.n_agents()))
    throw std::invalid_argument("one action per alive defender required");
  for (int a : actions)
    if (a < 0 || a >= kNumActions)
      throw std::invalid_argument("action outside [0,27)");
  std::vector<Vec3> commands(env_.n_blue(), Vec3::Zero());
  for (size_t j = 0; j < actions.size(); ++j) {
    int i = obs_.agent_drone[j];
    primitives_[i].apply(actions[j], cfg_);
    commands[i] = primitives_[i].setpoint();
  }
  return advance(commands);
}
std::vector<Vec3> TeamEnv::baseline_commands(Guidance guidance, Assignment assignment) {
  if (env_.done())
    throw std::logic_error("reset required before step");
  std::vector<int> ids, alive;
  for (const auto &t : fused_)
    ids.push_back(t.id);
  for (auto d : obs_.agent_drone)
    alive.push_back(d);
  if (env_.time() + 1e-9 >= next_assignment_ || ids != threat_ids_ ||
      alive != alive_ids_) {
    std::vector<std::vector<double>> cost(alive.size(),
                                          std::vector<double>(fused_.size()));
    for (size_t i = 0; i < alive.size(); ++i)
      for (size_t j = 0; j < fused_.size(); ++j)
        cost[i][j] = time_to_intercept(fused_[j].pos - env_.state(alive[i]).pos,
                                       fused_[j].vel, cfg_.limits.v_max);
    const auto result = assignment == Assignment::Greedy ? greedy_assign(cost)
                                                         : hungarian(cost);
    assigned_.assign(env_.n_blue(), -1);
    for (size_t i = 0; i < alive.size(); ++i)
      if (result[i] >= 0)
        assigned_[alive[i]] = fused_[result[i]].id;
    threat_ids_ = ids;
    alive_ids_ = alive;
    next_assignment_ = env_.time() + cfg_.reassign_s;
  }
  std::vector<Vec3> commands(env_.n_blue(), Vec3::Zero());
  for (int i : alive) {
    const auto own = env_.state(i);
    auto t = std::find_if(fused_.begin(), fused_.end(),
                          [&](const auto &x) { return x.id == assigned_[i]; });
    if (t != fused_.end())
      commands[i] =
          guide(own, *t, previous_[i], cfg_, guidance, cfg_.decision_dt());
    else {
      Vec3 guard = env_.scenario().asset;
      if (!fused_.empty()) {
        auto nearest = std::min_element(
            fused_.begin(), fused_.end(), [&](const auto &a, const auto &b) {
              return (a.pos - env_.scenario().asset).squaredNorm() <
                     (b.pos - env_.scenario().asset).squaredNorm();
            });
        Vec3 bearing = nearest->pos - guard;
        bearing.z() = 0;
        if (bearing.norm() > 1e-9)
          guard += cfg_.guard_dist * bearing.normalized();
      }
      guard.z() = cfg_.guard_alt;
      commands[i] =
          core::clip_velocity(cfg_.k_v * (guard - own.pos), cfg_.limits);
    }
    previous_[i] = commands[i];
  }
  return commands;
}
TeamStep TeamEnv::step_baseline(Guidance guidance, Assignment assignment) {
  auto commands=baseline_commands(guidance,assignment);
  for(auto drone:obs_.agent_drone) {
    primitives_[drone].speed=commands[drone].head<2>().norm();
    primitives_[drone].vz=commands[drone].z();
    if(primitives_[drone].speed>1e-9) primitives_[drone].psi=std::atan2(commands[drone].y(),commands[drone].x());
  }
  return advance(commands);
}
std::vector<int> TeamEnv::teacher_actions(Guidance guidance, Assignment assignment) {
  const auto commands=baseline_commands(guidance,assignment);
  std::vector<int> actions;
  for(auto drone:obs_.agent_drone) {
    double best=INFINITY;int chosen=kNoop;
    for(int action=0;action<kNumActions;++action) {
      auto candidate=primitives_[drone];candidate.apply(action,cfg_);
      const double cost=(candidate.setpoint()-commands[drone]).squaredNorm();
      if(cost<best) {best=cost;chosen=action;}
    }
    actions.push_back(chosen);
  }
  return actions;
}

} // namespace gnc
