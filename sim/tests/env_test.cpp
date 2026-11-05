#include <gtest/gtest.h>

#include <random>

#include "core/velocity_control.hpp"
#include "sim/env.hpp"

using namespace sim;
namespace {
core::EnvConfig cfg() { return core::load_env_config(SWARM_ROOT "/scenarios/v0/env.yaml"); }

/// One blue, one far-away waiting red (spawns late) so the episode stays open.
Scenario lone_blue(Vec3 p) {
  Scenario s;
  s.asset = {300, 300, 0};
  s.blue = {p};
  s.red = {{Vec3(-490, 490, 40), 1000.0, 6.0, Behaviour::Straight}};
  return s;
}
LossCause fly_until_lost(Env& e, Vec3 v, int steps = 300) {
  const std::vector<Vec3> cmd(e.n_blue(), v);
  for (int i = 0; i < steps && e.status(0) == Status::Active; ++i) e.step(cmd);
  for (const Event& ev : e.events()) if (ev.drone == 0 && ev.kind == EventKind::Lost) return ev.cause;
  return LossCause::None;
}
}  // namespace

TEST(Limits, RespectsSpeedAndAccel) {
  auto c = cfg();
  Env e(c);
  e.reset(lone_blue({0, 0, 100}), 1);
  const std::vector<Vec3> cmd{Vec3(100, 50, 0)};
  Vec3 prev = Vec3::Zero();
  for (int i = 0; i < 100; ++i) {
    e.step(cmd);
    const Vec3 v = e.state(0).vel;
    EXPECT_LE(v.head<2>().norm(), c.limits.v_max + 1e-6);
    EXPECT_LE((v - prev).head<2>().norm() / c.decision_dt(), c.limits.a_max + 1e-6);
    prev = v;
  }
  EXPECT_NEAR(prev.head<2>().norm(), c.limits.v_max, 1e-2);
}

TEST(Limits, HoverHoldsAltitude) {
  Env e(cfg());
  e.reset(lone_blue({0, 0, 40}), 1);
  const std::vector<Vec3> cmd{Vec3::Zero()};
  for (int i = 0; i < 100; ++i) e.step(cmd);
  EXPECT_NEAR(e.state(0).pos.z(), 40, 0.05);
}

TEST(Contacts, Ground) {
  Env e(cfg());
  e.reset(lone_blue({0, 0, 5}), 1);
  EXPECT_EQ(fly_until_lost(e, {0, 0, -5}), LossCause::Ground);
}

TEST(Contacts, Building) {
  Env e(cfg());
  auto s = lone_blue({0, 0, 20});
  s.buildings.push_back({Vec3(30, 0, 30), Vec3(10, 10, 30)});
  e.reset(s, 1);
  EXPECT_EQ(fly_until_lost(e, {10, 0, 0}), LossCause::Building);
  EXPECT_NEAR(e.state(0).pos.x(), 19.6, 0.5);
}

TEST(Contacts, LeavesArea) {
  Env e(cfg());
  e.reset(lone_blue({495, 0, 40}), 1);
  EXPECT_EQ(fly_until_lost(e, {10, 0, 0}), LossCause::OutOfArea);
}

TEST(Contacts, TeammatesRemoveEachOther) {
  Env e(cfg());
  auto s = lone_blue({0, 0, 40});
  s.blue.push_back({10, 0, 40});
  e.reset(s, 1);
  const std::vector<Vec3> cmd{Vec3(5, 0, 0), Vec3(-5, 0, 0)};
  for (int i = 0; i < 50 && e.status(0) == Status::Active; ++i) e.step(cmd);
  EXPECT_EQ(e.status(0), Status::Gone);
  EXPECT_EQ(e.status(1), Status::Gone);
}

TEST(Contacts, RedCrashResolvesWithoutIntercept) {
  Env e(cfg());
  Scenario s;
  s.asset = {100, 0, 0};
  s.blue = {{-100, -100, 40}};
  s.buildings.push_back({Vec3(40, 0, 30), Vec3(10, 10, 30)});
  s.red = {{Vec3(0, 0, 30), 0.0, 10.0, Behaviour::Straight}};
  e.reset(s, 1);
  const std::vector<Vec3> cmd{Vec3::Zero()};
  StepResult r{};
  int intercepted = 0;
  while (!r.done) { r = e.step(cmd); intercepted += r.intercepted; }
  EXPECT_FALSE(r.asset_hit);
  EXPECT_EQ(intercepted, 0);
  EXPECT_EQ(e.status(1), Status::Gone);
}

TEST(Intercept, RateMatchesPnAndOnePerEntry) {
  auto c = cfg();
  Env e(c);
  int hits = 0, n = 3000;
  for (int k = 0; k < n; ++k) {
    Scenario s;  // red flies through a hovering blue: exactly one entry into r_c
    s.asset = {300, 0, 0};
    s.blue = {{0, 0, 40}};
    s.red = {{Vec3(-30, 0, 40), 0.0, 10.0, Behaviour::Straight}};
    e.reset(s, k);
    const std::vector<Vec3> cmd{Vec3::Zero()};
    int trials = 0;
    while (e.time() < 8 && !e.done()) e.step(cmd);
    for (const Event& ev : e.events()) trials += ev.kind == EventKind::Intercept || ev.kind == EventKind::TrialFail;
    ASSERT_EQ(trials, 1);
    hits += e.status(1) == Status::Gone;
  }
  EXPECT_NEAR(double(hits) / n, c.p_n, 0.03);
}

TEST(Asset, HitRadiusEndsEpisode) {
  auto c = cfg();
  Env e(c);
  Scenario s;
  s.asset = {60, 0, 0};
  s.blue = {{-100, -100, 40}};
  s.red = {{Vec3(0, 0, 30), 0.0, 10.0, Behaviour::Straight}};
  e.reset(s, 1);
  const std::vector<Vec3> cmd{Vec3::Zero()};
  StepResult r{};
  while (!r.done) r = e.step(cmd);
  EXPECT_TRUE(r.asset_hit);
  EXPECT_LE((e.state(1).pos - s.asset).norm(), c.asset_radius + c.asset_hit_margin);
}

TEST(Scenario, RespectsPlacementConstraints) {
  auto c = cfg();
  std::mt19937_64 g(7);
  for (int k = 0; k < 500; ++k) {
    const Scenario s = sample_scenario(c, g);
    ASSERT_GE(int(s.buildings.size()), c.bld_min);
    ASSERT_LE(int(s.buildings.size()), c.bld_max);
    EXPECT_LE(s.asset.head<2>().cwiseAbs().maxCoeff(), c.asset_half_range);
    for (size_t i = 0; i < s.buildings.size(); ++i) {
      const Box& b = s.buildings[i];
      const double gap = ((s.asset - b.center).head<2>().cwiseAbs() - b.half.head<2>()).cwiseMax(0.0).norm();
      EXPECT_GE(gap, c.bld_asset_clear);
      EXPECT_LE((b.center.head<2>().cwiseAbs() + b.half.head<2>()).maxCoeff(), c.half_extent - c.bld_spawn_clear);
      for (size_t j = i + 1; j < s.buildings.size(); ++j)
        EXPECT_TRUE(((b.center - s.buildings[j].center).head<2>().cwiseAbs().array() >= (b.half + s.buildings[j].half).head<2>().array()).any());
    }
    for (const Vec3& p : s.blue) {
      const double d = (p - s.asset).head<2>().norm();
      EXPECT_GE(d, c.blue_dist_min); EXPECT_LE(d, c.blue_dist_max);
    }
    for (const RedSpawn& r : s.red) {
      EXPECT_NEAR(r.pos.head<2>().cwiseAbs().maxCoeff(), c.half_extent, 1e-9);
      EXPECT_LT(r.cruise, c.limits.v_max);
    }
  }
}

TEST(Red, EvasiveJinksRespectCooldown) {
  auto c = cfg();
  Env e(c);
  Scenario s;
  s.asset = {480, 0, 0};
  s.blue = {{0, 0, 40}};
  s.red = {{Vec3(-200, 0, 40), 0.0, 8.0, Behaviour::Evasive}};
  e.reset(s, 3);
  const std::vector<Vec3> cmd{Vec3(-10, 0, 0)};
  double last = -100;
  int jinks = 0;
  for (int i = 0; i < 300 && !e.done(); ++i) e.step(cmd);
  for (const Event& ev : e.events())
    if (ev.kind == EventKind::Jink) { EXPECT_GE(ev.t - last, c.ev_duration + c.ev_cooldown - 1e-6); last = ev.t; ++jinks; }
  EXPECT_GE(jinks, 1);
}

TEST(Control, FirstOrderStepResponse) {
  auto c = cfg();
  const Vec3 a = core::velocity_control(Vec3(1, 0, 0), Vec3::Zero(), c.k_v, c.limits);
  EXPECT_NEAR(a.x(), c.k_v, 1e-12);
  const Vec3 sat = core::velocity_control(Vec3(100, 0, 100), Vec3::Zero(), c.k_v, c.limits);
  EXPECT_NEAR(sat.x(), c.limits.a_max, 1e-12);
  EXPECT_NEAR(sat.z(), c.limits.az_max, 1e-12);
}

// Independent slab test: does segment a-b pass through box b inflated by r?
static bool hits_box(const Vec3& a, const Vec3& b, const Box& box, double r) {
  double t0 = 0, t1 = 1;
  for (int k = 0; k < 3; ++k) {
    const double lo = box.center[k] - box.half[k] - r, hi = box.center[k] + box.half[k] + r, d = b[k] - a[k];
    if (std::abs(d) < 1e-12) { if (a[k] < lo || a[k] > hi) return false; continue; }
    double u = (lo - a[k]) / d, v = (hi - a[k]) / d;
    if (u > v) std::swap(u, v);
    t0 = std::max(t0, u); t1 = std::min(t1, v);
    if (t0 > t1) return false;
  }
  return true;
}

TEST(Red, WaypointSegmentsClearOfBuildings) {
  auto c = cfg();
  c.behaviour_weights = {0, 1, 0};
  Env e(c);
  int bad_exact = 0, bad_radius = 0, empty = 0, routes = 0;
  for (int k = 0; k < 500; ++k) {
    e.reset(k);
    for (int r = 0; r < e.n_red(); ++r) {
      const auto& w = e.waypoints()[r];
      ++routes;
      empty += w.empty();
      std::vector<Vec3> pts{e.scenario().red[r].pos};
      pts.insert(pts.end(), w.begin(), w.end());
      pts.push_back({e.scenario().asset.x(), e.scenario().asset.y(), pts[0].z()});
      for (size_t s = 0; s + 1 < pts.size(); ++s)
        for (const Box& b : e.scenario().buildings) {
          bad_exact += hits_box(pts[s], pts[s + 1], b, 0.0);
          bad_radius += hits_box(pts[s], pts[s + 1], b, c.drone_radius);
        }
    }
  }
  RecordProperty("routes", routes);
  std::cout << "routes=" << routes << " empty=" << empty << " exact=" << bad_exact << " inflated=" << bad_radius << "\n";
  EXPECT_EQ(bad_exact, 0);
  EXPECT_EQ(bad_radius, 0);
  EXPECT_EQ(empty, 0);
}


TEST(Red, SampledDirectApproachesClearAtEveryAltitude) {
  auto c = cfg();
  c.blue_count = 0;
  std::mt19937_64 rng(0);
  for (int seed = 0; seed < 1000; ++seed) {
    const auto s = sim::sample_scenario(c, rng);
    for (const auto& red : s.red) {
      Vec3 end = s.asset;
      end.z() = red.pos.z();
      for (auto building : s.buildings) {
        // Extend the box upward: the entire cruise/dive corridor must be clear,
        // including height errors from the velocity controller.
        building.center.z() = c.z_max / 2;
        building.half.z() = c.z_max;
        EXPECT_FALSE(hits_box(red.pos, end, building, c.wp_clearance));
      }
    }
  }
}
