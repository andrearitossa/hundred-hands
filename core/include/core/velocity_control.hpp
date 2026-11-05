#pragma once
#include <algorithm>
#include "core/config.hpp"
#include "core/types.hpp"

namespace core {

/// Clip a velocity setpoint (m/s) to the speed limits.
inline Vec3 clip_velocity(Vec3 v, const Limits& l) {
  const double h = v.head<2>().norm();
  if (h > l.v_max) v.head<2>() *= l.v_max / h;
  v.z() = std::clamp(v.z(), -l.vz_max, l.vz_max);
  return v;
}

/// Acceleration (m/s^2, excluding gravity) tracking velocity setpoint: a = k_v (v_cmd - v), clipped.
inline Vec3 velocity_control(const Vec3& v_cmd, const Vec3& v, double k_v, const Limits& l) {
  Vec3 a = k_v * (clip_velocity(v_cmd, l) - v);
  const double h = a.head<2>().norm();
  if (h > l.a_max) a.head<2>() *= l.a_max / h;
  a.z() = std::clamp(a.z(), -l.az_max, l.az_max);
  return a;
}
}  // namespace core
