// Python module `swarmsim`.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/config.hpp"
#include "sim/env.hpp"
#include "gnc/gnc.hpp"
#include "runtime/runtime.hpp"
#include "perception/obs_layout.hpp"

namespace py = pybind11;
using core::Vec3;

// Arrays own the immutable graph through a capsule; subsequent steps cannot invalidate them.
static py::dict graph(perception::ObsBatch batch) {
  auto storage = std::make_unique<perception::ObsBatch>(std::move(batch));
  auto* b = storage.get();
  py::capsule owner(storage.release(), [](void* p) { delete static_cast<perception::ObsBatch*>(p); });
  py::dict d;
  auto floats = [&](const char* name, std::vector<float>& v, int dim) {
    py::array a(py::dtype::of<float>(), {py::ssize_t(v.size()/dim), py::ssize_t(dim)},
                {py::ssize_t(dim*sizeof(float)), py::ssize_t(sizeof(float))}, v.data(), owner);
    a.attr("setflags")(false); d[name]=a;
  };
  auto ints = [&](const char* name, std::vector<int64_t>& v) {
    py::array a(py::dtype::of<int64_t>(), {py::ssize_t(v.size())}, {py::ssize_t(sizeof(int64_t))}, v.data(), owner);
    a.attr("setflags")(false); d[name]=a;
  };
  floats("self",b->self,perception::obs::self_::kDim);
  floats("asset",b->asset,perception::obs::asset_::kDim);
  floats("local_threat",b->local_threat,perception::obs::local_threat_::kDim);
  floats("global_threat",b->global_threat,perception::obs::global_threat_::kDim);
  floats("teammate",b->teammate,perception::obs::teammate_::kDim);
  ints("local_owner",b->local_owner); ints("global_owner",b->global_owner);
  ints("teammate_owner",b->teammate_owner); ints("teammate_src",b->teammate_src);
  ints("agent_env",b->agent_env); ints("agent_drone",b->agent_drone); ints("alive_mask",b->alive_mask);
  return d;
}
static py::dict transition(gnc::TeamStep r) {
  return py::dict(py::arg("observation")=graph(std::move(r.observation)), py::arg("reward")=r.reward,
                  py::arg("terminated")=r.terminated, py::arg("truncated")=r.truncated,
                  py::arg("intercepted")=r.intercepted);
}

static perception::ObsBatch unpack(const py::dict& d) {
  perception::ObsBatch b;
  auto floats = [&](const char* key, std::vector<float>& out, int dim) {
    auto a=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(d[key]);
    if(!a || a.ndim()!=2 || a.shape(1)!=dim) throw py::value_error(std::string("invalid feature shape: ")+key);
    out.assign(a.data(),a.data()+a.size());
  };
  auto ints = [&](const char* key,std::vector<int64_t>& out) {
    auto a=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(d[key]);
    if(!a || a.ndim()!=1) throw py::value_error(std::string("invalid index shape: ")+key);
    out.assign(a.data(),a.data()+a.size());
  };
  floats("self",b.self,9);floats("asset",b.asset,4);floats("local_threat",b.local_threat,12);
  floats("global_threat",b.global_threat,12);floats("teammate",b.teammate,8);
  ints("local_owner",b.local_owner);ints("global_owner",b.global_owner);
  ints("teammate_owner",b.teammate_owner);ints("teammate_src",b.teammate_src);
  ints("agent_env",b.agent_env);ints("agent_drone",b.agent_drone);ints("alive_mask",b.alive_mask);
  const auto n=b.n_agents();
  if(b.self.size()!=size_t(n*9) || b.asset.size()!=size_t(n*4) || b.local_threat.size()!=b.local_owner.size()*12 ||
     b.global_threat.size()!=b.global_owner.size()*12 || b.teammate.size()!=b.teammate_owner.size()*8 ||
     b.teammate_src.size()!=b.teammate_owner.size()) throw py::value_error("inconsistent graph lengths");
  for(const auto* indices:{&b.local_owner,&b.global_owner,&b.teammate_owner,&b.teammate_src})
    for(auto i:*indices) if(i<0 || i>=n) throw py::value_error("agent index out of range");
  return b;
}
static py::dict output(runtime::Output out, const perception::ObsBatch& b) {
  auto array=[](const std::vector<float>& values,int64_t rows,int64_t cols) {
    py::array_t<float> a({rows,cols});
    std::copy(values.begin(),values.end(),a.mutable_data());return a;
  };
  return py::dict(py::arg("q")=array(out.q,b.n_agents(),27),
    py::arg("messages")=array(out.messages,b.n_agents(),out.message_dim),
    py::arg("attention_local")=array(out.local_attention,b.local_owner.size(),out.heads),
    py::arg("attention_global")=array(out.global_attention,b.global_owner.size(),out.heads),
    py::arg("attention_teammate")=array(out.teammate_attention,b.teammate_owner.size(),out.heads));
}

PYBIND11_MODULE(swarmsim, mod) {
  py::class_<core::Track>(mod,"Track")
      .def_readonly("pos",&core::Track::pos).def_readonly("vel",&core::Track::vel)
      .def_readonly("age",&core::Track::age).def_readonly("since_update",&core::Track::since_update);
  py::class_<core::EnvConfig>(mod, "EnvConfig")
      .def_static("load", &core::load_env_config)
      .def_readwrite("red_min", &core::EnvConfig::red_min)
      .def_readwrite("red_max", &core::EnvConfig::red_max)
      .def_readwrite("blue_count", &core::EnvConfig::blue_count)
      .def_readwrite("behaviour_weights", &core::EnvConfig::behaviour_weights)
      .def_readwrite("max_decisions", &core::EnvConfig::max_decisions)
      .def_readwrite("p_n", &core::EnvConfig::p_n)
      .def_readwrite("capture_radius", &core::EnvConfig::capture_radius)
      .def_readwrite("act_turn", &core::EnvConfig::act_turn)
      .def_readwrite("act_speed", &core::EnvConfig::act_speed)
      .def_readwrite("act_climb", &core::EnvConfig::act_climb)
      .def_readwrite("onboard_range", &core::EnvConfig::onboard_range)
      .def_readwrite("ground_range", &core::EnvConfig::ground_range)
      .def_readonly("half_extent", &core::EnvConfig::half_extent)
      .def_property_readonly("decision_dt", &core::EnvConfig::decision_dt);

  py::class_<sim::Env>(mod, "Env")
      .def(py::init<core::EnvConfig>())
      .def("reset", py::overload_cast<uint64_t>(&sim::Env::reset))
      .def("step", [](sim::Env& e, py::array_t<double, py::array::c_style | py::array::forcecast> v) {
        if (v.ndim() != 2 || v.shape(1) != 3 || v.shape(0) != e.n_blue()) throw py::value_error("v_cmd must be [n_blue, 3]");
        std::vector<Vec3> cmd(v.shape(0));
        for (py::ssize_t i = 0; i < v.shape(0); ++i) cmd[i] = {v.at(i, 0), v.at(i, 1), v.at(i, 2)};
        const auto r = e.step(cmd);
        return py::dict(py::arg("intercepted") = r.intercepted, py::arg("asset_hit") = r.asset_hit,
                        py::arg("done") = r.done, py::arg("truncated") = r.truncated);
      })
      .def_property_readonly("time", &sim::Env::time)
      .def_property_readonly("n_blue", &sim::Env::n_blue)
      .def_property_readonly("n_red", &sim::Env::n_red)
      .def_property_readonly("done", &sim::Env::done)
      .def("snapshot", [](const sim::Env& e) {  // [n_drones, 6] pos+vel, [n_drones] status
        const int n = e.n_drones();
        py::array_t<double> s({n, 6});
        py::array_t<int> st(n);
        for (int i = 0; i < n; ++i) {
          const auto x = e.state(i);
          for (int k = 0; k < 3; ++k) { s.mutable_at(i, k) = x.pos[k]; s.mutable_at(i, 3 + k) = x.vel[k]; }
          st.mutable_at(i) = static_cast<int>(e.status(i));
        }
        return py::make_tuple(s, st);
      })
      .def("scenario", [](const sim::Env& e) {
        const auto& s = e.scenario();
        std::vector<std::array<double, 6>> blds;
        for (const auto& b : s.buildings) blds.push_back({b.center.x(), b.center.y(), b.center.z(), b.half.x(), b.half.y(), b.half.z()});
        std::vector<std::array<double, 3>> red;
        for (const auto& r : s.red) red.push_back({r.time, r.cruise, double(static_cast<int>(r.behaviour))});
        return py::dict(py::arg("asset") = Vec3(s.asset), py::arg("buildings") = blds, py::arg("red") = red);
      })
      .def("waypoints", [](const sim::Env& e) {
        std::vector<std::vector<std::array<double, 3>>> out;
        for (const auto& route : e.waypoints()) {
          auto& o = out.emplace_back();
          for (const Vec3& p : route) o.push_back({p.x(), p.y(), p.z()});
        }
        return out;
      })
      .def("events", [](const sim::Env& e) {  // (t, kind, drone, other, cause)
        std::vector<std::tuple<double, int, int, int, int>> out;
        for (const auto& v : e.events()) out.emplace_back(v.t, int(v.kind), v.drone, v.other, int(v.cause));
        return out;
      });
  py::enum_<gnc::Guidance>(mod,"Guidance")
      .value("PP",gnc::Guidance::PurePursuit).value("PN",gnc::Guidance::PN);
  py::enum_<gnc::Assignment>(mod,"Assignment")
      .value("GREEDY",gnc::Assignment::Greedy).value("HUNGARIAN",gnc::Assignment::Hungarian);
  mod.def("hungarian", &gnc::hungarian);
  mod.def("greedy_assign", &gnc::greedy_assign);
  mod.def("time_to_intercept", &gnc::time_to_intercept);
  py::class_<gnc::TeamEnv>(mod,"TeamEnv")
      .def(py::init<core::EnvConfig>())
      .def("reset",[](gnc::TeamEnv& e,uint64_t seed){return graph(e.reset(seed));})
      .def("step",[](gnc::TeamEnv& e,const std::vector<int>& a){return transition(e.step(a));})
      .def("teacher_actions",&gnc::TeamEnv::teacher_actions)
      .def("step_baseline",[](gnc::TeamEnv& e,gnc::Guidance g,gnc::Assignment a){return transition(e.step_baseline(g,a));})
      .def_property_readonly("local_tracks",&gnc::TeamEnv::local_tracks)
      .def_property_readonly("fused_tracks",&gnc::TeamEnv::fused)
      .def_property_readonly("world", &gnc::TeamEnv::world, py::return_value_policy::reference_internal)
      .def_property_readonly("timings",[](const gnc::TeamEnv& e){auto [t,g]=e.timings();return py::dict(py::arg("tracking_ms")=t,py::arg("graph_ms")=g);})
      .def_property_readonly("headings", &gnc::TeamEnv::headings)
      .def_property_readonly("time",[](const gnc::TeamEnv& e){return e.world().time();});
  py::class_<gnc::VecEnv>(mod,"VecEnv")
      .def(py::init<core::EnvConfig,int,int>(),py::arg("config"),py::arg("count"),py::arg("threads")=0)
      .def("reset",[](gnc::VecEnv& e,uint64_t seed){
        perception::ObsBatch b; {py::gil_scoped_release release; b=e.reset(seed);} return graph(std::move(b));
      })
      .def("step",[](gnc::VecEnv& e,const std::vector<int>& a){
        gnc::VecStep r; {py::gil_scoped_release release;r=e.step(a);}
        py::list transitions;for(auto& t:r.transitions) transitions.append(transition(std::move(t)));
        return py::dict(py::arg("observation")=graph(std::move(r.observation)),py::arg("transitions")=transitions);
      });
  py::class_<runtime::Runtime>(mod,"Runtime")
      .def(py::init<const std::string&>())
      .def("forward",[](runtime::Runtime& r,const py::dict& graph,const std::vector<float>& mask){
        auto b=unpack(graph);return output(r.forward(b,mask),b);
      },py::arg("graph"),py::arg("message_mask")=std::vector<float>{});
  mod.def("evaluate_cpp",[](core::EnvConfig cfg,uint64_t seed,const std::string& policy,const std::string& path){
    runtime::EpisodeMetrics r;
    {py::gil_scoped_release release;r=runtime::evaluate(cfg,seed,policy,path);}
    py::dict d(py::arg("seed")=seed,py::arg("survival")=r.survived,py::arg("truncated")=r.truncated,
        py::arg("intercept_fraction")=r.intercept_fraction,py::arg("defender_survival")=r.defender_survival,
        py::arg("episode_s")=r.episode_s,py::arg("latency_ms")=r.latency_ms);
    d["first_intercept_s"]=r.first_intercept_s<0?py::none():py::cast(r.first_intercept_s);
    d["clear_s"]=r.clear_s<0?py::none():py::cast(r.clear_s);return d;
  },py::arg("config"),py::arg("seed"),py::arg("policy"),py::arg("model_path")="");
  mod.attr("NUM_ACTIONS")=gnc::kNumActions;
  mod.attr("NOOP")=gnc::kNoop;
  mod.attr("EVENT_KINDS") = py::make_tuple("spawn", "trial_fail", "intercept", "lost", "asset_hit", "jink", "truncated");
  mod.attr("LOSS_CAUSES") = py::make_tuple("none", "ground", "building", "out_of_area", "teammate", "intercepted");
  mod.attr("BEHAVIOURS") = py::make_tuple("straight", "waypoint", "evasive");
}
