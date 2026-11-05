#include "sim/env.hpp"

#include <mujoco/mujoco.h>

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>

#include "core/velocity_control.hpp"

namespace sim {

// Geoms are created in order: ground, buildings, asset, drones.
struct Env::Model {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  int first_drone_geom = 0;
  ~Model() { clear(); }
  void clear() {
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    d = nullptr;
    m = nullptr;
  }
};

namespace {
// Drones collide with the world only. Blue-blue and blue-red are distance rules (spec 4.3, 4.4).
constexpr int kDroneType = 1, kWorldType = 2;
constexpr mjtByte kBuildingsOnly[mjNGROUP] = {0, 1, 0, 0, 0, 0};
double uni(std::mt19937_64& g, double a, double b) { return std::uniform_real_distribution<>(a, b)(g); }
}  // namespace

Env::Env(core::EnvConfig cfg) : cfg_(cfg), m_(std::make_unique<Model>()) {}
Env::~Env() = default;

// MuJoCo's broadphase filters on body masks too, and those are fixed at compile: update both.
void Env::set_collidable(int drone, bool on) {
  const int g = m_->first_drone_geom + drone, b = m_->m->geom_bodyid[g];
  m_->m->geom_contype[g] = m_->m->body_contype[b] = on ? kDroneType : 0;
  m_->m->geom_conaffinity[g] = m_->m->body_conaffinity[b] = on ? kWorldType : 0;
}

void Env::build_world() {
  std::string x = std::format(
      "<mujoco><option timestep='{}' integrator='Euler'/><worldbody>"
      "<geom type='plane' size='0 0 1' contype='{}' conaffinity='{}' group='0'/>",
      cfg_.physics_dt, kWorldType, kDroneType);
  for (const Box& b : sc_.buildings)
    x += std::format("<geom type='box' pos='{} {} {}' size='{} {} {}' contype='{}' conaffinity='{}' group='1'/>",
                     b.center.x(), b.center.y(), b.center.z(), b.half.x(), b.half.y(), b.half.z(), kWorldType, kDroneType);
  x += std::format("<geom type='sphere' pos='{} {} 0' size='{}' contype='0' conaffinity='0' group='3'/>",
                   sc_.asset.x(), sc_.asset.y(), cfg_.asset_radius);
  std::string act;
  for (int i = 0; i < n_drones(); ++i) {
    x += std::format("<body name='d{0}'><joint name='d{0}x' type='slide' axis='1 0 0'/><joint name='d{0}y' type='slide' axis='0 1 0'/>"
                     "<joint name='d{0}z' type='slide' axis='0 0 1'/>"
                     "<geom type='sphere' size='{1}' mass='{2}' contype='0' conaffinity='0' group='2'/></body>",
                     i, cfg_.drone_radius, cfg_.limits.mass);
    for (char ax : {'x', 'y', 'z'}) act += std::format("<motor joint='d{}{}'/>", i, ax);
  }
  x += "</worldbody><actuator>" + act + "</actuator></mujoco>";

  char err[512] = "";
  mjSpec* spec = mj_parseXMLString(x.c_str(), nullptr, err, sizeof err);
  if (!spec) throw std::runtime_error(std::string("MJCF parse: ") + err);
  m_->clear();
  m_->m = mj_compile(spec, nullptr);
  if (!m_->m) {
    const std::string e = mjs_getError(spec);
    mj_deleteSpec(spec);
    throw std::runtime_error("MJCF compile: " + e);
  }
  mj_deleteSpec(spec);
  m_->d = mj_makeData(m_->m);
  m_->first_drone_geom = 1 + static_cast<int>(sc_.buildings.size()) + 1;
  for (int i = 0; i < n_drones(); ++i) {
    const Vec3& p = i < n_blue() ? sc_.blue[i] : sc_.red[i - n_blue()].pos;
    for (int k = 0; k < 3; ++k) m_->d->qpos[3 * i + k] = p[k];
  }
  mj_forward(m_->m, m_->d);
}

void Env::reset(uint64_t seed) {
  rng_.seed(seed);
  reset(sample_scenario(cfg_, rng_), seed + 0x9E3779B97F4A7C15ULL);
}

void Env::reset(Scenario scenario, uint64_t seed) {
  rng_.seed(seed);
  sc_ = std::move(scenario);
  build_world();
  t_ = 0;
  decisions_ = 0;
  done_ = false;
  events_.clear();
  status_.assign(n_drones(), Status::Waiting);
  armed_.assign(n_blue() * n_red(), 1);
  const size_t nr = n_red();
  waypoints_.assign(nr, {});
  wp_next_.assign(nr, 0);
  jink_until_.assign(nr, -1);
  jink_ready_.assign(nr, 0);
  jink_axis_.assign(nr, Vec3::UnitZ());
  jink_angle_.assign(nr, 0);

  for (int i = 0; i < n_blue(); ++i) {
    status_[i] = Status::Active;
    set_collidable(i, true);
  }
  // Waypoint routes: 1-3 points scattered around the spawn-to-asset line, resampled until clear.
  for (size_t r = 0; r < nr; ++r) {
    const RedSpawn& s = sc_.red[r];
    if (s.behaviour == Behaviour::Straight) continue;
    if (s.behaviour == Behaviour::Evasive && uni(rng_, 0, 1) < 0.5) continue;
    const Vec3 end(sc_.asset.x(), sc_.asset.y(), s.pos.z());
    const Vec3 dir = (end - s.pos).normalized(), side(-dir.y(), dir.x(), 0);
    const int n = std::uniform_int_distribution<>(cfg_.wp_min, cfg_.wp_max)(rng_);
    for (int attempt = 0; attempt < cfg_.wp_max_samples; ++attempt) {
      std::vector<Vec3> route;
      for (int k = 1; k <= n; ++k) {
        Vec3 p = s.pos + (end - s.pos) * (double(k) / (n + 1)) + side * uni(rng_, -cfg_.wp_lateral, cfg_.wp_lateral);
        p.head<2>() = p.head<2>().cwiseMax(-cfg_.half_extent + 1).cwiseMin(cfg_.half_extent - 1);
        route.push_back(p);
      }
      bool clear = true;
      Vec3 from = s.pos;
      for (const Vec3& p : route) { clear &= segment_clear(from, p); from = p; }
      clear &= segment_clear(from, end);
      // The dive leaves cruise altitude, so it needs its own check. It stops short of the asset's clear zone.
      const Vec3 to_end = end - from;
      const double dive = std::min(to_end.norm(), s.pos.z() * s.cruise / cfg_.limits.vz_max);
      clear &= segment_clear(end - to_end.normalized() * dive, Vec3(sc_.asset.x(), sc_.asset.y(), 0));
      if (clear) { waypoints_[r] = std::move(route); break; }
    }
  }
  update_spawns();
}

// A fan of parallel rays spaced under the smallest building side, so none can slip between them.
bool Env::segment_clear(const Vec3& a, const Vec3& b) const {
  const double clearance_z = 2 * cfg_.drone_radius;  // rays sweep below the body, so roofs can't be grazed
  const Vec3 v = b - a;
  const Vec3 side = Vec3(-v.y(), v.x(), 0).normalized();
  const int n = static_cast<int>(std::ceil(cfg_.wp_clearance / (cfg_.bld_side_min / 2)));
  for (int k = -n; k <= n; ++k) {
    const Vec3 p = a + side * (cfg_.wp_clearance * k / n) - Vec3::UnitZ() * clearance_z;
    int geom = -1;
    const mjtNum x = mj_ray(m_->m, m_->d, p.data(), v.data(), kBuildingsOnly, 1, -1, &geom, nullptr);
    if (x >= 0 && x <= 1) return false;
  }
  return true;
}

core::EntityState Env::state(int i) const {
  return {Eigen::Map<const Vec3>(m_->d->qpos + 3 * i), Eigen::Map<const Vec3>(m_->d->qvel + 3 * i)};
}

void Env::remove(int i, LossCause cause, int other) {
  if (status_[i] == Status::Gone) return;
  status_[i] = Status::Gone;
  set_collidable(i, false);
  events_.push_back({t_, cause == LossCause::Intercepted ? EventKind::Intercept : EventKind::Lost, i, other, cause});
}

void Env::update_spawns() {
  for (int r = 0; r < n_red(); ++r) {
    const int i = n_blue() + r;
    if (status_[i] == Status::Waiting && t_ >= sc_.red[r].time) {
      status_[i] = Status::Active;
      set_collidable(i, true);
      events_.push_back({t_, EventKind::Spawn, i, -1, LossCause::None});
    }
  }
}

Vec3 Env::red_setpoint(int r) {
  const RedSpawn& s = sc_.red[r];
  const int i = n_blue() + r;
  const core::EntityState me = state(i);
  Vec3 v;
  auto& route = waypoints_[r];
  while (wp_next_[r] < route.size() && (route[wp_next_[r]] - me.pos).head<2>().norm() < cfg_.wp_reach) ++wp_next_[r];
  if (wp_next_[r] < route.size()) {
    v = s.cruise * (route[wp_next_[r]] - me.pos).normalized();
  } else {
    // Hold cruise altitude, then dive along a straight line once the descent is within vz_max.
    const Vec3 to = sc_.asset - me.pos;
    const double h = to.head<2>().norm(), alt = me.pos.z();
    const Vec3 flat(to.x(), to.y(), 0);
    if (h * cfg_.limits.vz_max <= alt * s.cruise) {
      v = s.cruise * to.normalized();
      if (std::abs(v.z()) > cfg_.limits.vz_max) v *= cfg_.limits.vz_max / std::abs(v.z());
    } else {
      v = s.cruise * flat.normalized();
      v.z() = std::clamp(s.pos.z() - alt, -cfg_.limits.vz_max, cfg_.limits.vz_max);
    }
  }
  if (s.behaviour == Behaviour::Evasive) {
    if (t_ >= jink_ready_[r]) {
      for (int b = 0; b < n_blue(); ++b) {
        if (status_[b] != Status::Active) continue;
        const core::EntityState o = state(b);
        const Vec3 rel = o.pos - me.pos;
        if (rel.norm() < cfg_.ev_trigger && (o.vel - me.vel).dot(rel) < 0) {
          const bool vertical = uni(rng_, 0, 1) < 0.5, flip = uni(rng_, 0, 1) < 0.5;
          const Vec3 flat = Vec3(v.x(), v.y(), 0).normalized();
          jink_axis_[r] = vertical ? Vec3(flat.cross(Vec3::UnitZ())) : Vec3(Vec3::UnitZ());
          jink_angle_[r] = (flip ? -1 : 1) * uni(rng_, 0, cfg_.ev_max_angle);
          jink_until_[r] = t_ + cfg_.ev_duration;
          jink_ready_[r] = jink_until_[r] + cfg_.ev_cooldown;
          events_.push_back({t_, EventKind::Jink, i, b, LossCause::None});
          break;
        }
      }
    }
    if (t_ < jink_until_[r]) v = Eigen::AngleAxisd(jink_angle_[r], jink_axis_[r]) * v;
  }
  return v;
}

void Env::check_contacts() {
  const mjData* d = m_->d;
  for (int c = 0; c < d->ncon; ++c) {
    int g1 = d->contact[c].geom[0], g2 = d->contact[c].geom[1];
    if (g1 > g2) std::swap(g1, g2);  // world geoms come first
    const int drone = g2 - m_->first_drone_geom;
    if (drone >= 0) remove(drone, g1 == 0 ? LossCause::Ground : LossCause::Building);
  }
}

void Env::check_rules(StepResult& out) {
  const int nb = n_blue(), nr = n_red();
  for (int i = 0; i < n_drones(); ++i) {
    if (status_[i] != Status::Active) continue;
    const Vec3 p = state(i).pos;
    if (p.head<2>().cwiseAbs().maxCoeff() > cfg_.half_extent || p.z() > cfg_.z_max) remove(i, LossCause::OutOfArea);
  }
  for (int a = 0; a < nb; ++a)
    for (int b = a + 1; b < nb; ++b)
      if (status_[a] == Status::Active && status_[b] == Status::Active &&
          (state(a).pos - state(b).pos).norm() < cfg_.contact_distance) {
        remove(a, LossCause::Teammate, b);
        remove(b, LossCause::Teammate, a);
      }
  // Spec 4.4: one Bernoulli trial per entry into r_c; a failed pair re-arms beyond 2 r_c.
  for (int b = 0; b < nb; ++b) {
    if (status_[b] != Status::Active) continue;
    for (int r = 0; r < nr; ++r) {
      const int i = nb + r;
      if (status_[i] != Status::Active) continue;
      const double dist = (state(b).pos - state(i).pos).norm();
      char& armed = armed_[b * nr + r];
      if (dist > 2 * cfg_.capture_radius) armed = 1;
      if (dist < cfg_.capture_radius && armed) {
        armed = 0;
        if (uni(rng_, 0, 1) < cfg_.p_n) {
          remove(i, LossCause::Intercepted, b);
          ++out.intercepted;
        } else {
          events_.push_back({t_, EventKind::TrialFail, b, i, LossCause::None});
        }
      }
    }
  }
  bool unresolved = false;
  for (int r = 0; r < nr; ++r) {
    const int i = nb + r;
    if (status_[i] == Status::Gone) continue;
    unresolved = true;
    if (status_[i] == Status::Active &&
        (state(i).pos - sc_.asset).norm() < cfg_.asset_radius + cfg_.asset_hit_margin && !out.asset_hit) {
      out.asset_hit = true;
      events_.push_back({t_, EventKind::AssetHit, i, -1, LossCause::None});
    }
  }
  out.done |= out.asset_hit || !unresolved;
}

StepResult Env::step(std::span<const Vec3> blue_vcmd) {
  if (done_) throw std::logic_error("step() on a finished episode");
  if (static_cast<int>(blue_vcmd.size()) != n_blue()) throw std::invalid_argument("blue_vcmd size");
  StepResult out{};
  mjData* d = m_->d;
  const double g = -m_->m->opt.gravity[2];
  const auto& l = cfg_.limits;
  for (int ctl = 0; ctl < cfg_.control_per_decision && !out.done; ++ctl) {
    update_spawns();
    for (int i = 0; i < n_drones(); ++i) {
      Vec3 a = Vec3::Zero();
      if (status_[i] == Status::Active) {
        const Vec3 v_cmd = i < n_blue() ? blue_vcmd[i] : red_setpoint(i - n_blue());
        a = core::velocity_control(v_cmd, state(i).vel, cfg_.k_v, l);
      } else {
        for (int k = 0; k < 3; ++k) d->qvel[3 * i + k] = 0;  // parked: not in the world
      }
      for (int k = 0; k < 3; ++k) d->ctrl[3 * i + k] = l.mass * a[k];
      d->ctrl[3 * i + 2] += l.mass * g;
    }
    for (int p = 0; p < cfg_.physics_per_control; ++p) {
      mj_step(m_->m, d);
      check_contacts();
    }
    t_ += cfg_.control_dt();
    check_rules(out);
  }
  if (++decisions_ >= cfg_.max_decisions && !out.done) {
    out.done = out.truncated = true;
    events_.push_back({t_, EventKind::Truncated, -1, -1, LossCause::None});
  }
  done_ = out.done;
  return out;
}
}  // namespace sim
