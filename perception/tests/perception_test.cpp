#include <gtest/gtest.h>

#include "perception/obs_layout.hpp"
#include "perception/perception.hpp"

using namespace perception;
namespace {
core::EnvConfig cfg() { return core::load_env_config(SWARM_ROOT "/scenarios/v0/env.yaml"); }
Detection det(const Vec3& p) { return {p, Eigen::Matrix3d::Identity(), 1.0}; }
}  // namespace

TEST(Tracker, ReproducesConstantVelocity) {
  auto c = cfg();
  Tracker t(c);
  const Vec3 p0(10, -20, 30), v(7, -3, 1);
  for (int i = 0; i <= 50; ++i) {
    const std::vector<Detection> d{det(p0 + v * (i * 0.1))};
    t.update(i * 0.1, d);
  }
  ASSERT_EQ(t.tracks().size(), 1u);
  EXPECT_LT((t.tracks()[0].pos - (p0 + v * 5.0)).norm(), 1e-3);
  EXPECT_LT((t.tracks()[0].vel - v).norm(), 1e-3);
  EXPECT_NEAR(t.tracks()[0].age, 5.0, 1e-9);
}

TEST(Tracker, CoastsThenDrops) {
  auto c = cfg();
  Tracker t(c);
  const Vec3 v(10, 0, 0);
  for (int i = 0; i <= 10; ++i) {
    const std::vector<Detection> d{det(v * (i * 0.1))};
    t.update(i * 0.1, d);
  }
  const std::vector<Detection> none;
  for (int i = 1; i <= 19; ++i) t.update(1.0 + i * 0.1, none);
  ASSERT_EQ(t.tracks().size(), 1u);
  EXPECT_NEAR(t.tracks()[0].since_update, 1.9, 1e-9);
  EXPECT_NEAR(t.tracks()[0].pos.x(), 29.0, 1e-2);
  t.update(3.0 + 1e-6, none);
  t.update(3.1, none);
  EXPECT_TRUE(t.tracks().empty());
}

TEST(Tracker, KeepsCrossingTargetsSeparate) {
  auto c = cfg();
  Tracker t(c);
  for (int i = 0; i <= 20; ++i) {
    const double s = i * 0.1;
    const std::vector<Detection> d{det(Vec3(100 + 10 * s, 0, 30)), det(Vec3(-100 - 10 * s, 0, 30))};
    t.update(s, d);
  }
  EXPECT_EQ(t.tracks().size(), 2u);
}

TEST(Fusion, MergesDuplicatesOnly) {
  auto c = cfg();
  core::Track a, b, far, fast;
  a.pos = {0, 0, 10}; a.since_update = 1.0;
  b.pos = {2, 0, 10}; b.since_update = 0.0;
  far.pos = {50, 0, 10};
  fast.pos = {1, 0, 10}; fast.vel = {10, 0, 0};
  const std::vector<core::Track> l1{a, far}, l2{b, fast};
  const std::vector<const std::vector<core::Track>*> lists{&l1, &l2};
  const auto f = fuse(lists, c);
  ASSERT_EQ(f.size(), 3u);
  EXPECT_NEAR(f[0].pos.x(), 1.0, 1e-12);
  EXPECT_EQ(f[0].since_update, 0.0);
}

TEST(Graph, EgoFrameAndUncappedCounts) {
  auto c = cfg();
  namespace L = obs::local_threat_;
  core::Track t;
  t.pos = {100, 100, 40};  // Due north of a defender at (100, 0) facing north: straight ahead.
  std::vector<core::Track> many(20, t);
  const std::vector<Ego> team{{Vec3(100, 0, 40), Vec3(0, 5, 0), M_PI / 2}, {Vec3(0, 0, 40), Vec3::Zero(), 0.0}};
  ObsBatch b;
  build_agent(b, 0, 0, 0, true, many, many, Vec3(0, 0, 0), team, c.limits);
  ASSERT_EQ(b.local_owner.size(), 20u);
  ASSERT_EQ(b.global_owner.size(), 20u);
  ASSERT_EQ(b.teammate_src.size(), 1u);
  EXPECT_NEAR(b.local_threat[L::rel_pos], 0.1, 1e-6);
  EXPECT_NEAR(b.local_threat[L::rel_pos + 1], 0.0, 1e-6);
  EXPECT_NEAR(b.self[obs::self_::vel], 5 / c.limits.v_max, 1e-6);  // flying "forward" in ego frame
  EXPECT_NEAR(b.self[obs::self_::vel + 1], 0.0, 1e-6);
  EXPECT_EQ(b.self[obs::self_::g], 1.0f);
}

TEST(Graph, NoGlobalNodesWhenViewUnavailable) {
  auto c = cfg();
  core::Track t;
  const std::vector<core::Track> one{t};
  const std::vector<Ego> team{{Vec3::Zero(), Vec3::Zero(), 0.0}, {Vec3(5, 0, 0), Vec3::Zero(), 0.0}};
  ObsBatch b;
  build_agent(b, 0, 0, 0, false, one, one, Vec3(9, 9, 0), team, c.limits);
  EXPECT_EQ(b.local_owner.size(), 1u);
  EXPECT_TRUE(b.global_owner.empty());
  EXPECT_TRUE(b.teammate_src.empty());
}
