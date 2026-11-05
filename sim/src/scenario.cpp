#include "sim/env.hpp"

#include <cmath>
#include <stdexcept>

namespace sim {
namespace {
double uni(std::mt19937_64& g, double a, double b) { return std::uniform_real_distribution<>(a, b)(g); }
int uni_int(std::mt19937_64& g, int a, int b) { return std::uniform_int_distribution<>(a, b)(g); }
bool overlaps_xy(const Box& a, const Box& b) {
  return ((a.center - b.center).head<2>().cwiseAbs().array() < (a.half + b.half).head<2>().array()).all();
}
double dist_xy(const Box& b, const Vec3& p) {
  return ((p - b.center).head<2>().cwiseAbs() - b.half.head<2>()).cwiseMax(0.0).norm();
}
// Reserve a horizontal flight corridor all the way to the asset. Checking its
// footprint at every altitude also covers the dive and controller descent lag.
bool clear_approach(const Vec3& from, const Vec3& to, const std::vector<Box>& boxes, double clearance) {
  for (const Box& b : boxes) {
    double enter = 0, leave = 1;
    bool intersects = true;
    for (int axis = 0; axis < 2; ++axis) {
      const double delta = to[axis] - from[axis];
      const double lo = b.center[axis] - b.half[axis] - clearance;
      const double hi = b.center[axis] + b.half[axis] + clearance;
      if (std::abs(delta) < 1e-12) {
        if (from[axis] < lo || from[axis] > hi) intersects = false;
      } else {
        double a = (lo - from[axis]) / delta, z = (hi - from[axis]) / delta;
        if (a > z) std::swap(a, z);
        enter = std::max(enter, a); leave = std::min(leave, z);
      }
    }
    if (intersects && enter <= leave) return false;
  }
  return true;
}
}  // namespace

Scenario sample_scenario(const core::EnvConfig& c, std::mt19937_64& g) {
  Scenario s;
  const double ar = c.asset_half_range;
  s.asset = {uni(g, -ar, ar), uni(g, -ar, ar), 0.0};

  const int n_bld = uni_int(g, c.bld_min, c.bld_max);
  while (static_cast<int>(s.buildings.size()) < n_bld) {
    const Vec3 half(uni(g, c.bld_side_min, c.bld_side_max) / 2, uni(g, c.bld_side_min, c.bld_side_max) / 2,
                    uni(g, c.bld_h_min, c.bld_h_max) / 2);
    const double r = c.half_extent - c.bld_spawn_clear;
    const Box b{{uni(g, -r + half.x(), r - half.x()), uni(g, -r + half.y(), r - half.y()), half.z()}, half};
    if (dist_xy(b, s.asset) < c.bld_asset_clear) continue;
    bool clash = false;
    for (const Box& o : s.buildings) clash |= overlaps_xy(b, o);
    if (!clash) s.buildings.push_back(b);
  }

  while (static_cast<int>(s.blue.size()) < c.blue_count) {
    const double d = uni(g, c.blue_dist_min, c.blue_dist_max), th = uni(g, -M_PI, M_PI);
    const Vec3 p = s.asset + Vec3(d * std::cos(th), d * std::sin(th), uni(g, c.blue_alt_min, c.blue_alt_max));
    bool inside = false;
    for (const Box& b : s.buildings) inside |= dist_xy(b, p) < 2 * c.drone_radius && p.z() < 2 * b.half.z() + 2 * c.drone_radius;
    for (const Vec3& o : s.blue) inside |= (o - p).norm() < 2 * c.contact_distance;
    if (!inside) s.blue.push_back(p);
  }

  const int n_red = uni_int(g, c.red_min, c.red_max);
  std::discrete_distribution<int> beh(c.behaviour_weights.begin(), c.behaviour_weights.end());
  for (int i = 0; i < n_red; ++i) {
    bool accepted = false;
    for (int attempt = 0; attempt < c.wp_max_samples; ++attempt) {
      const double h = c.half_extent, along = uni(g, -h, h), sign = uni_int(g, 0, 1) ? h : -h;
      const bool on_x = uni_int(g, 0, 1);
      const Vec3 pos(on_x ? sign : along, on_x ? along : sign, uni(g, c.red_alt_min, c.red_alt_max));
      if (!clear_approach(pos, s.asset, s.buildings, std::max(c.wp_clearance, c.drone_radius))) continue;
      s.red.push_back({pos, uni(g, 0.0, c.spawn_window), uni(g, c.cruise_min, c.cruise_max),
                       static_cast<Behaviour>(beh(g))});
      accepted = true;
      break;
    }
    if (!accepted) throw std::runtime_error("no clear attacker approach within red.waypoints.max_samples");
  }
  return s;
}
}  // namespace sim
