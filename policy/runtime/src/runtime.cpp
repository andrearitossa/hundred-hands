#include "runtime/runtime.hpp"
#include "perception/obs_layout.hpp"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <chrono>
#include <random>
#include <stdexcept>

namespace runtime {
struct Runtime::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"swarm"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  Impl(const std::string& path) {
    options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
    session=std::make_unique<Ort::Session>(env,path.c_str(),options);
  }
};
Runtime::Runtime(const std::string& path):impl_(std::make_unique<Impl>(path)){}
Runtime::~Runtime()=default;
Output Runtime::forward(const perception::ObsBatch& b,std::span<const float> mask) {
  namespace o=perception::obs;
  const int64_t n=b.n_agents();
  if (!mask.empty() && mask.size()!=size_t(n)) throw std::invalid_argument("message mask must match agent count");
  auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  float dummy_float=0; int64_t dummy_int=0;
  auto floats=[&](const std::vector<float>& v,int dim){
    std::array<int64_t,2> shape{int64_t(v.size()/dim),dim};
    inputs.push_back(Ort::Value::CreateTensor<float>(memory,v.empty()?&dummy_float:const_cast<float*>(v.data()),v.size(),shape.data(),2));
  };
  auto ints=[&](const std::vector<int64_t>& v){
    std::array<int64_t,1> shape{int64_t(v.size())};
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory,v.empty()?&dummy_int:const_cast<int64_t*>(v.data()),v.size(),shape.data(),1));
  };
  floats(b.self,o::self_::kDim);floats(b.asset,o::asset_::kDim);
  floats(b.local_threat,o::local_threat_::kDim);ints(b.local_owner);
  floats(b.global_threat,o::global_threat_::kDim);ints(b.global_owner);
  floats(b.teammate,o::teammate_::kDim);ints(b.teammate_owner);ints(b.teammate_src);
  std::vector<float> ones(n,1);
  auto* data=mask.empty()?ones.data():const_cast<float*>(mask.data());
  std::array<int64_t,1> mask_shape{n};
  inputs.push_back(Ort::Value::CreateTensor<float>(memory,n?data:&dummy_float,n,mask_shape.data(),1));
  constexpr const char* names[]={"self","asset","local_threat","local_owner","global_threat","global_owner","teammate","teammate_owner","teammate_src","message_mask"};
  constexpr const char* outputs[]={"q","attention_local","attention_global","attention_teammate","messages"};
  auto values=impl_->session->Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),outputs,5);
  Output out;
  out.heads=values[1].GetTensorTypeAndShapeInfo().GetShape()[1];
  out.message_dim=values[4].GetTensorTypeAndShapeInfo().GetShape()[1];
  std::array<std::vector<float>*,5> destinations{&out.q,&out.local_attention,&out.global_attention,&out.teammate_attention,&out.messages};
  for(size_t i=0;i<values.size();++i) {
    const size_t count=values[i].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* start=values[i].GetTensorData<float>();
    if(count) destinations[i]->assign(start,start+count);
  }
  return out;
}
EpisodeMetrics evaluate(core::EnvConfig cfg,uint64_t seed,const std::string& policy,const std::string& path) {
  const bool baseline=policy=="pp_greedy"||policy=="pp_hungarian"||policy=="pn_greedy"||policy=="pn_hungarian";
  if (!baseline && policy!="idle" && policy!="random" && policy!="gqn") throw std::invalid_argument("unknown policy");
  std::unique_ptr<Runtime> runtime;
  if(policy=="gqn") runtime=std::make_unique<Runtime>(path);
  gnc::TeamEnv env(cfg);auto graph=env.reset(seed);
  std::mt19937_64 rng(seed);
  EpisodeMetrics out{};int intercepted=0;
  for(;;) {
    const auto before=std::chrono::steady_clock::now();
    gnc::TeamStep result;
    if(baseline) {
      result=env.step_baseline(policy.starts_with("pp")?gnc::Guidance::PurePursuit:gnc::Guidance::PN,
        policy.ends_with("greedy")?gnc::Assignment::Greedy:gnc::Assignment::Hungarian);
    } else {
      std::vector<int> actions(graph.n_agents(),gnc::kNoop);
      if(runtime) {
        auto output=runtime->forward(graph);
        for(size_t i=0;i<actions.size();++i)
          actions[i]=std::max_element(output.q.begin()+i*27,output.q.begin()+(i+1)*27)-(output.q.begin()+i*27);
      } else if(policy=="random") for(auto& a:actions) a=std::uniform_int_distribution<int>(0,26)(rng);
      const auto after=std::chrono::steady_clock::now();
      out.latency_ms.push_back(std::chrono::duration<double,std::milli>(after-before).count()/std::max<int64_t>(1,graph.n_agents()));
      result=env.step(actions);
    }
    if(baseline) out.latency_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count()/std::max<int64_t>(1,graph.n_agents()));
    intercepted+=result.intercepted;
    graph=std::move(result.observation);
    if(result.terminated||result.truncated) {out.truncated=result.truncated;break;}
  }
  bool asset_hit=false;double first_spawn=-1,first_intercept=-1;
  for(const auto& event:env.world().events()) {
    if(event.kind==sim::EventKind::AssetHit) asset_hit=true;
    if(event.kind==sim::EventKind::Spawn && first_spawn<0) first_spawn=event.t;
    if(event.kind==sim::EventKind::Intercept && first_intercept<0) first_intercept=event.t;
  }
  out.survived=!asset_hit&&!out.truncated;
  out.intercept_fraction=double(intercepted)/env.world().n_red();
  int alive=0;for(int i=0;i<env.world().n_blue();++i) alive+=env.world().status(i)==sim::Status::Active;
  out.defender_survival=env.world().n_blue()?double(alive)/env.world().n_blue():0;
  out.episode_s=env.world().time();
  if(first_intercept>=0) out.first_intercept_s=first_intercept-first_spawn;
  if(out.survived) out.clear_s=out.episode_s-first_spawn;
  return out;
}
} // namespace runtime
