#include "perception/perception.hpp"

#include <algorithm>
#include <cmath>

#include "perception/obs_layout.hpp"

namespace perception {
namespace {
using Mat6 = Eigen::Matrix<double, 6, 6>;

void put(float* dst, int offset, const Vec3& v, double norm) {
  for (int k = 0; k < 3; ++k) dst[offset + k] = static_cast<float>(v[k] / norm);
}
void put(float* dst, int offset, double v, double norm) { dst[offset] = static_cast<float>(v / norm); }

Vec3 to_ego(const Vec3& v, double psi) {
  const double c = std::cos(psi), s = std::sin(psi);
  return {c * v.x() + s * v.y(), -s * v.x() + c * v.y(), v.z()};
}

float* grow(std::vector<float>& v, int dim) {
  v.resize(v.size() + dim);
  return v.data() + v.size() - dim;
}

void put_threat(std::vector<float>& out, const Track& t, const Ego& me, const Vec3& asset, const core::Limits& l) {
  namespace o = obs::local_threat_;
  float* d = grow(out, o::kDim);
  const Vec3 rel = t.pos - me.pos, rel_v = t.vel - me.vel, to_asset = asset - t.pos;
  const double dist = rel.norm(), asset_dist = to_asset.norm();
  put(d, o::rel_pos, to_ego(rel, me.psi), obs::kPosNorm);
  put(d, o::rel_vel, to_ego(rel_v, me.psi), l.v_max);
  put(d, o::dist, dist, obs::kPosNorm);
  put(d, o::closing, dist > 1e-9 ? -rel_v.dot(rel) / dist : 0.0, l.v_max);
  put(d, o::asset_dist, asset_dist, obs::kPosNorm);
  put(d, o::asset_speed, asset_dist > 1e-9 ? t.vel.dot(to_asset) / asset_dist : 0.0, l.v_max);
  put(d, o::age, t.age, obs::kTimeNorm);
  put(d, o::since_update, t.since_update, obs::kTimeNorm);
}
}  // namespace

std::vector<Detection> detect(const sim::Env& env, const Vec3& sensor, double range) {
  std::vector<Detection> out;
  for (int r = env.n_blue(); r < env.n_drones(); ++r) {
    if (env.status(r) != sim::Status::Active) continue;
    const Vec3 p = env.state(r).pos;
    if ((p - sensor).norm() <= range) out.push_back({p, Eigen::Matrix3d::Identity(), 1.0});
  }
  return out;
}

void Tracker::update(double t, std::span<const Detection> dets) {
  const double dt = t - last_t_, dt2 = dt * dt, q = cfg_->trk_q;
  last_t_ = t;
  Mat6 F = Mat6::Identity();
  F.topRightCorner<3, 3>().diagonal().setConstant(dt);
  Mat6 Q = Mat6::Zero();
  Q.topLeftCorner<3, 3>().diagonal().setConstant(q * dt2 * dt2 / 4);
  Q.topRightCorner<3, 3>().diagonal().setConstant(q * dt2 * dt / 2);
  Q.bottomLeftCorner<3, 3>().diagonal().setConstant(q * dt2 * dt / 2);
  Q.bottomRightCorner<3, 3>().diagonal().setConstant(q * dt2);
  for (Track& tr : tracks_) {
    tr.pos += tr.vel * dt;
    tr.cov = (F * tr.cov * F.transpose() + Q).eval();
    tr.age += dt;
    tr.since_update += dt;
  }

  // Greedy global nearest neighbour inside the gate.
  std::vector<char> det_used(dets.size(), 0), trk_used(tracks_.size(), 0);
  for (size_t round = 0; round < std::min(dets.size(), tracks_.size()); ++round) {
    double best = cfg_->trk_gate;
    int bd = -1, bt = -1;
    for (size_t d = 0; d < dets.size(); ++d) {
      if (det_used[d]) continue;
      for (size_t k = 0; k < tracks_.size(); ++k) {
        if (trk_used[k]) continue;
        const double dist = (dets[d].pos - tracks_[k].pos).norm();
        if (dist < best) { best = dist; bd = int(d); bt = int(k); }
      }
    }
    if (bd < 0) break;
    det_used[bd] = trk_used[bt] = 1;
    Track& tr = tracks_[bt];
    const Eigen::Matrix3d S = tr.cov.topLeftCorner<3, 3>() + dets[bd].cov * cfg_->trk_meas_var;
    const Eigen::Matrix<double, 6, 3> K = tr.cov.leftCols<3>() * S.inverse();
    const Vec3 innov = dets[bd].pos - tr.pos;
    tr.pos += K.topRows<3>() * innov;
    tr.vel += K.bottomRows<3>() * innov;
    tr.cov = (tr.cov - K * tr.cov.topRows<3>()).eval();
    tr.since_update = 0;
  }
  std::vector<Track> kept;
  for (const Track& tr : tracks_) if (tr.since_update <= cfg_->trk_coast) kept.push_back(tr);
  tracks_ = std::move(kept);
  for (size_t d = 0; d < dets.size(); ++d) {
    if (det_used[d]) continue;
    Track tr;
    tr.id = next_id_++;
    tr.pos = dets[d].pos;
    tr.cov.setZero();
    tr.cov.topLeftCorner<3, 3>() = dets[d].cov * cfg_->trk_meas_var;
    tr.cov.bottomRightCorner<3, 3>().diagonal().setConstant(cfg_->trk_init_vel_var);
    tracks_.push_back(tr);
  }
}

std::vector<Track> fuse(std::span<const std::vector<Track>* const> lists, const core::EnvConfig& c) {
  std::vector<Track> out;
  std::vector<int> counts;
  for (const auto* list : lists) {
    for (const Track& t : *list) {
      size_t k = 0;
      for (; k < out.size(); ++k)
        if ((out[k].pos - t.pos).norm() < c.fus_pos && (out[k].vel - t.vel).norm() < c.fus_vel) break;
      if (k == out.size()) {
        out.push_back(t);
        counts.push_back(1);
      } else {
        const int n = counts[k]++;
        out[k].pos = (out[k].pos * n + t.pos) / (n + 1);
        out[k].vel = (out[k].vel * n + t.vel) / (n + 1);
        out[k].since_update = std::min(out[k].since_update, t.since_update);
        out[k].age = std::max(out[k].age, t.age);
      }
    }
  }
  return out;
}

void ObsBatch::clear() {
  for (auto* v : {&self, &local_threat, &asset, &global_threat, &teammate}) v->clear();
  for (auto* v : {&local_owner, &global_owner, &teammate_owner, &teammate_src, &agent_env, &agent_drone, &alive_mask}) v->clear();
}

void ObsBatch::append(const ObsBatch& o) {
  const int64_t shift = n_agents();
  const auto cat = [](auto& a, const auto& b) { a.insert(a.end(), b.begin(), b.end()); };
  const auto cat_shift = [&](auto& a, const auto& b) { for (int64_t x : b) a.push_back(x + shift); };
  cat(self, o.self); cat(local_threat, o.local_threat); cat(asset, o.asset);
  cat(global_threat, o.global_threat); cat(teammate, o.teammate);
  cat_shift(local_owner, o.local_owner); cat_shift(global_owner, o.global_owner);
  cat_shift(teammate_owner, o.teammate_owner); cat_shift(teammate_src, o.teammate_src);
  cat(agent_env, o.agent_env); cat(agent_drone, o.agent_drone); cat(alive_mask, o.alive_mask);
}

void build_agent(ObsBatch& b, int agent, int64_t env, int64_t drone, bool g, const std::vector<Track>& local,
                 const std::vector<Track>& fused, const Vec3& asset, std::span<const Ego> team,
                 const core::Limits& l) {
  const Ego& me = team[agent];
  const Vec3 to_asset = asset - me.pos;
  const double asset_dist = to_asset.norm();
  {
    namespace o = obs::self_;
    float* d = grow(b.self, o::kDim);
    put(d, o::vel, to_ego(me.vel, me.psi), l.v_max);
    put(d, o::altitude, me.pos.z(), obs::kPosNorm);
    put(d, o::asset_vec, to_ego(to_asset, me.psi), obs::kPosNorm);
    put(d, o::asset_dist, asset_dist, obs::kPosNorm);
    d[o::g] = g ? 1.0f : 0.0f;
  }
  {
    namespace o = obs::asset_;
    float* d = grow(b.asset, o::kDim);
    put(d, o::rel_pos, to_ego(to_asset, me.psi), obs::kPosNorm);
    put(d, o::dist, asset_dist, obs::kPosNorm);
  }
  for (const Track& t : local) {
    put_threat(b.local_threat, t, me, asset, l);
    b.local_owner.push_back(agent);
  }
  if (g) {
    for (const Track& t : fused) {
      put_threat(b.global_threat, t, me, asset, l);
      b.global_owner.push_back(agent);
    }
    namespace o = obs::teammate_;
    for (int j = 0; j < int(team.size()); ++j) {
      if (j == agent) continue;
      float* d = grow(b.teammate, o::kDim);
      const Vec3 rel = team[j].pos - me.pos;
      put(d, o::rel_pos, to_ego(rel, me.psi), obs::kPosNorm);
      put(d, o::rel_vel, to_ego(team[j].vel - me.vel, me.psi), l.v_max);
      put(d, o::dist, rel.norm(), obs::kPosNorm);
      put(d, o::asset_dist, (asset - team[j].pos).norm(), obs::kPosNorm);
      b.teammate_owner.push_back(agent);
      b.teammate_src.push_back(j);
    }
  }
  b.agent_env.push_back(env);
  b.agent_drone.push_back(drone);
  b.alive_mask.push_back(1);
}
}  // namespace perception
