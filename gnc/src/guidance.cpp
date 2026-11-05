#include "gnc/gnc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace gnc {
void Primitive::apply(int a, const core::EnvConfig &c) {
  if (a < 0 || a >= kNumActions)
    throw std::invalid_argument("action outside [0,27)");
  psi = std::remainder(psi + (a / 9 - 1) * c.act_turn, 2 * std::numbers::pi);
  vz = std::clamp(vz + (a / 3 % 3 - 1) * c.act_climb, -c.limits.vz_max,
                  c.limits.vz_max);
  speed = std::clamp(speed + (a % 3 - 1) * c.act_speed, 0.0, c.limits.v_max);
}
Vec3 Primitive::setpoint() const {
  return {speed * std::cos(psi), speed * std::sin(psi), vz};
}
double time_to_intercept(const Vec3 &r, const Vec3 &v, double speed) {
  const double a = v.squaredNorm() - speed * speed, b = 2 * r.dot(v),
               c = r.squaredNorm();
  if (c < 1e-12)
    return 0;
  double t = std::numeric_limits<double>::infinity();
  if (std::abs(a) < 1e-12) {
    if (b < 0)
      t = -c / b;
  } else if (const double d = b * b - 4 * a * c; d >= 0) {
    for (double x :
         {(-b - std::sqrt(d)) / (2 * a), (-b + std::sqrt(d)) / (2 * a)})
      if (x >= 0)
        t = std::min(t, x);
  }
  return t;
}
static int columns(const std::vector<std::vector<double>> &c) {
  int m = c.empty() ? 0 : c[0].size();
  for (const auto &row : c) {
    if (int(row.size()) != m)
      throw std::invalid_argument("ragged cost matrix");
    for (double x : row)
      if (std::isnan(x) || x == -INFINITY)
        throw std::invalid_argument("invalid cost");
  }
  return m;
}
std::vector<int> greedy_assign(const std::vector<std::vector<double>> &c) {
  int m = columns(c);
  std::vector<int> out(c.size(), -1);
  std::vector<bool> used(m);
  for (;;) {
    double best = INFINITY;
    int r = -1, col = -1;
    for (int i = 0; i < int(c.size()); ++i)
      if (out[i] < 0)
        for (int j = 0; j < m; ++j)
          if (!used[j] && c[i][j] < best) {
            best = c[i][j];
            r = i;
            col = j;
          }
    if (r < 0)
      break;
    out[r] = col;
    used[col] = true;
  }
  return out;
}
std::vector<int> hungarian(const std::vector<std::vector<double>> &c) {
  const int n = c.size(), real = columns(c), m = real + n;
  std::vector<int> out(n, -1), p(m + 1), way(m + 1);
  if (!n || !real)
    return out;
  double scale = 1;
  for (const auto &row : c)
    for (double x : row)
      if (std::isfinite(x))
        scale = std::max(scale, std::abs(x));
  const double penalty = scale * (2 * n + 1);
  std::vector<double> u(n + 1), v(m + 1);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, INFINITY);
    std::vector<bool> used(m + 1);
    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      double delta = INFINITY;
      for (int j = 1; j <= m; ++j)
        if (!used[j]) {
          double cost = j <= real && std::isfinite(c[i0 - 1][j - 1])
                            ? c[i0 - 1][j - 1]
                            : penalty;
          double cur = cost - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      for (int j = 0; j <= m; ++j)
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else
          minv[j] -= delta;
      j0 = j1;
    } while (p[j0]);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  for (int j = 1; j <= real; ++j)
    if (p[j] && std::isfinite(c[p[j] - 1][j - 1]))
      out[p[j] - 1] = j - 1;
  return out;
}
Vec3 guide(const core::EntityState &own, const core::Track &target,
           const Vec3 &previous, const core::EnvConfig &c, Guidance mode,
           double dt) {
  const Vec3 r = target.pos - own.pos;
  if (r.norm() < 1e-9)
    return Vec3::Zero();
  Vec3 cmd = r.normalized() * c.limits.v_max;
  if (mode == Guidance::PN && previous.norm() > 1e-6) {
    const Vec3 relative = target.vel - own.vel;
    const double closing = std::max(0.0, -relative.dot(r.normalized()));
    const Vec3 omega = r.cross(relative) / r.squaredNorm();
    Vec3 acceleration =
        c.pn_gain * closing * omega.cross(previous.normalized());
    // Reacquire a receding target with pursuit rather than retaining an opening
    // course.
    if (closing > 0)
      cmd = (previous + dt * acceleration).normalized() * c.limits.v_max;
  }
  return core::clip_velocity(cmd, c.limits);
}
} // namespace gnc
