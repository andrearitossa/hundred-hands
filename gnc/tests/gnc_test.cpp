#include "gnc/gnc.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

static core::EnvConfig config() {
  return core::load_env_config(SWARM_ROOT "/scenarios/v0/env.yaml");
}
TEST(GNC, ControllerAndPrimitives) {
  auto c = config();
  gnc::Primitive p;
  for (int a = 0; a < 27; ++a) {
    gnc::Primitive q;
    q.speed = 6;
    q.apply(a, c);
    EXPECT_NEAR(q.psi, (a / 9 - 1) * c.act_turn, 1e-12);
    EXPECT_DOUBLE_EQ(q.vz, (a / 3 % 3 - 1) * c.act_climb);
    EXPECT_DOUBLE_EQ(q.speed, 6 + (a % 3 - 1) * c.act_speed);
  }
  for (int i = 0; i < 100; ++i)
    p.apply(26, c);
  EXPECT_DOUBLE_EQ(p.speed, c.limits.v_max);
  EXPECT_DOUBLE_EQ(p.vz, c.limits.vz_max);
  auto psi = p.psi;
  p.apply(gnc::kNoop, c);
  EXPECT_DOUBLE_EQ(psi, p.psi);
  EXPECT_THROW(p.apply(27, c), std::invalid_argument);
  core::Vec3 v = core::Vec3::Zero(), cmd(10, 0, 2);
  for (int i = 0; i < 1000; ++i) {
    auto a = core::velocity_control(cmd, v, c.k_v, c.limits);
    EXPECT_LE(a.head<2>().norm(), c.limits.a_max + 1e-10);
    EXPECT_LE(std::abs(a.z()), c.limits.az_max);
    v += c.control_dt() * a;
  }
  EXPECT_LT((v - cmd).norm(), 1e-6);
}
TEST(GNC, HungarianBruteForce) {
  std::mt19937 rng(4);
  for (int n = 1; n <= 5; ++n)
    for (int m = 1; m <= 5; ++m)
      for (int trial = 0; trial < 10; ++trial) {
        std::vector<std::vector<double>> cost(n, std::vector<double>(m));
        for (auto &row : cost)
          for (auto &x : row)
            x = int(rng() % 100) - 30;
        auto assignment = gnc::hungarian(cost);
        double got = 0;
        int count = 0;
        std::vector<bool> used(m);
        for (int i = 0; i < n; ++i)
          if (assignment[i] >= 0) {
            EXPECT_FALSE(used[assignment[i]]);
            used[assignment[i]] = true;
            got += cost[i][assignment[i]];
            ++count;
          }
        EXPECT_EQ(count, std::min(n, m));
        double best = INFINITY;
        std::function<void(int, int, double)> search = [&](int i, int selected,
                                                           double total) {
          if (i == n) {
            if (selected == std::min(n, m))
              best = std::min(best, total);
            return;
          }
          search(i + 1, selected, total);
          for (int j = 0; j < m; ++j)
            if (!used[j]) {
              used[j] = true;
              search(i + 1, selected + 1, total + cost[i][j]);
              used[j] = false;
            }
        };
        std::fill(used.begin(), used.end(), false);
        search(0, 0, 0);
        EXPECT_DOUBLE_EQ(got, best);
      }
  EXPECT_EQ(gnc::hungarian({{INFINITY}, {1}}), (std::vector<int>{-1, 0}));
  EXPECT_THROW(gnc::hungarian({{1}, {}}), std::invalid_argument);
}
TEST(GNC, PNIntercept) {
  const auto c = config();
  core::EntityState own;
  own.pos = {0, 0, 30};
  core::Track target;
  target.pos = {200, 40, 30};
  target.vel = {-8, 1, 0};
  core::Vec3 previous = core::Vec3::Zero();
  double closest = INFINITY;
  for (int i = 0; i < 3000; ++i) {
    if (i % 10 == 0)
      previous = gnc::guide(own, target, previous, c, gnc::Guidance::PN,
                            c.decision_dt());
    own.vel += c.control_dt() *
               core::velocity_control(previous, own.vel, c.k_v, c.limits);
    own.pos += c.control_dt() * own.vel;
    target.pos += c.control_dt() * target.vel;
    closest = std::min(closest, (own.pos - target.pos).norm());
  }
  EXPECT_LT(closest, c.capture_radius);
}
TEST(TeamEnv, RewardDeathAndReset) {
  auto c = config();
  c.max_decisions = 1;
  gnc::TeamEnv e(c);
  auto b = e.reset(7);
  EXPECT_EQ(b.n_agents(), c.blue_count);
  auto r = e.step(std::vector<int>(b.n_agents(), gnc::kNoop));
  EXPECT_TRUE(r.truncated);
  EXPECT_FALSE(r.terminated);
  EXPECT_EQ(r.reward, 0);
  EXPECT_THROW(e.step({}), std::logic_error);
  sim::Scenario s;
  s.asset = {0, 0, 0};
  s.blue = {{100, 0, 0.1}};
  s.red = {{{400, 0, 30}, 0, 8, sim::Behaviour::Straight}};
  c.max_decisions = 3000;
  gnc::TeamEnv death(c);
  death.reset(s, 1);
  auto d = death.step(std::vector<int>{13});
  EXPECT_EQ(d.observation.n_agents(), 0);
  EXPECT_NO_THROW(death.step({}));
  s.blue = {{100, 0, 30}};
  s.red = {{{5, 0, 1}, 0, 8, sim::Behaviour::Straight}};
  gnc::TeamEnv hit(c);
  hit.reset(s, 1);
  auto h = hit.step(std::vector<int>{13});
  EXPECT_TRUE(h.terminated);
  EXPECT_DOUBLE_EQ(h.reward, -1);
}
TEST(VecEnv, SchedulingAndAutoReset) {
  auto c = config();
  c.max_decisions = 2;
  gnc::VecEnv serial(c, 3, 1), parallel(c, 3, 3);
  auto a = serial.reset(9), b = parallel.reset(9);
  EXPECT_EQ(a.self, b.self);
  EXPECT_THROW(parallel.step(std::vector<int>{27}), std::invalid_argument);
  for (int t = 0; t < 8; ++t) {
    std::vector<int> actions(a.n_agents(), gnc::kNoop);
    auto x = serial.step(actions), y = parallel.step(actions);
    EXPECT_EQ(x.observation.self, y.observation.self);
    EXPECT_EQ(x.observation.teammate_src, y.observation.teammate_src);
    EXPECT_EQ(x.transitions[0].truncated, t % 2 == 1);
    a = x.observation;
    for (size_t i = 0; i < a.teammate_src.size(); ++i)
      EXPECT_EQ(a.agent_env[a.teammate_src[i]],
                a.agent_env[a.teammate_owner[i]]);
  }
}

TEST(TeamEnv, InterceptRewardAndTerminalBonus) {
  auto c = config();
  c.p_n = 1;
  sim::Scenario s;
  s.asset = {0, 0, 0};
  s.blue = {{100, 0, 30}};
  s.red = {{{101, 0, 30}, 0, 8, sim::Behaviour::Straight}};
  gnc::TeamEnv e(c);
  e.reset(s, 3);
  auto r = e.step(std::vector<int>{13});
  EXPECT_EQ(r.intercepted, 1);
  EXPECT_TRUE(r.terminated);
  EXPECT_FALSE(r.truncated);
  EXPECT_DOUBLE_EQ(r.reward, 2);
}

TEST(TeamEnv, FusedIdentityIsUniqueAcrossSensors) {
  auto c = config();
  sim::Scenario s;
  s.asset = {0, 0, 0};
  s.blue = {{-400, 0, 30}, {400, 0, 30}};
  s.red = {{{-450, 0, 30}, 0, 8, sim::Behaviour::Straight},
           {{450, 0, 30}, 0, 8, sim::Behaviour::Straight}};
  gnc::TeamEnv e(c);
  e.reset(s, 3);
  e.step(std::vector<int>{13, 13});
  ASSERT_EQ(e.fused().size(), 2);
  EXPECT_NE(e.fused()[0].id, e.fused()[1].id);
  const int first = e.fused()[0].id, second = e.fused()[1].id;
  e.step(std::vector<int>{13, 13});
  ASSERT_EQ(e.fused().size(), 2);
  EXPECT_EQ(e.fused()[0].id, first);
  EXPECT_EQ(e.fused()[1].id, second);
}

TEST(PrimitiveTuning, SweepOnNonManoeuvringTargets) {
  auto original=config();
  std::cout << "turn_deg,speed_step,climb_step,captures,mean_episode_s\n";
  int best=0;
  for(double turn:{10.0,20.0,30.0}) for(double speed:{1.5,3.0}) for(double climb:{1.5,3.0}) {
    auto c=original;c.act_turn=turn*std::acos(-1.0)/180;c.act_speed=speed;c.act_climb=climb;c.p_n=1;
    int hits=0;double total=0;
    for(double lateral:{40.0,80.0,120.0}) {
      sim::Scenario s;s.asset={200,0,0};s.blue={{-80,lateral,35}};
      s.red={{{-220,0,40},0,8,sim::Behaviour::Straight}};
      sim::Env env(c);env.reset(s,1);gnc::Primitive decoder;core::Vec3 pn_command=core::Vec3::Zero();
      while(!env.done()) {
        core::Track target;const auto red=env.state(1);target.pos=red.pos;target.vel=red.vel;
        const auto desired=gnc::guide(env.state(0),target,pn_command,c,gnc::Guidance::PN,c.decision_dt());
        pn_command=desired;
        double cost=INFINITY;int chosen=gnc::kNoop;
        for(int action=0;action<gnc::kNumActions;++action) {
          auto candidate=decoder;candidate.apply(action,c);
          const double error=(candidate.setpoint()-desired).squaredNorm();
          if(error<cost) {cost=error;chosen=action;}
        }
        decoder.apply(chosen,c);
        const auto r=env.step(std::vector<core::Vec3>{decoder.setpoint()});
        hits+=r.intercepted;
      }
      total+=env.time();
    }
    best=std::max(best,hits);
    std::cout << turn << ',' << speed << ',' << climb << ',' << hits << ',' << total/3 << '\n';
  }
  EXPECT_EQ(best,3);
}
