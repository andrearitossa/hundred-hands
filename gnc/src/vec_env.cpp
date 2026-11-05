#include "gnc/gnc.hpp"
#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace gnc {
struct VecEnv::Impl {
  std::vector<std::unique_ptr<TeamEnv>> envs;
  std::vector<uint64_t> seeds;
  std::vector<std::thread> workers;
  std::mutex mutex, api_mutex;
  std::condition_variable ready, finished;
  std::function<void(size_t)> job;
  size_t next = 0, pending = 0, generation = 0;
  bool stop = false;
  std::exception_ptr error;
  Impl(core::EnvConfig c, int count, int threads) {
    if (count <= 0 || threads < 0)
      throw std::invalid_argument(
          "positive env count and nonnegative thread count required");
    for (int i = 0; i < count; ++i)
      envs.push_back(std::make_unique<TeamEnv>(c));
    seeds.resize(count);
    threads = std::min(
        count, threads
                   ? threads
                   : int(std::max(1u, std::thread::hardware_concurrency())));
    try {
      for (int i = 0; i < threads; ++i)
        workers.emplace_back([this] {
          size_t seen = 0;
          std::unique_lock lock(mutex);
          for (;;) {
            ready.wait(lock, [&] { return stop || generation != seen; });
            if (stop)
              return;
            seen = generation;
            while (next < envs.size()) {
              size_t i = next++;
              lock.unlock();
              try {
                job(i);
              } catch (...) {
                lock.lock();
                if (!error)
                  error = std::current_exception();
                lock.unlock();
              }
              lock.lock();
              if (--pending == 0)
                finished.notify_one();
            }
          }
        });
    } catch (...) {
      shutdown();
      throw;
    }
  }
  void shutdown() {
    {
      std::lock_guard lock(mutex);
      stop = true;
    }
    ready.notify_all();
    for (auto &w : workers)
      if (w.joinable())
        w.join();
  }
  ~Impl() { shutdown(); }
  void run(std::function<void(size_t)> task) {
    std::unique_lock lock(mutex);
    job = std::move(task);
    next = 0;
    pending = envs.size();
    error = nullptr;
    ++generation;
    ready.notify_all();
    finished.wait(lock, [&] { return pending == 0; });
    job = {};
    if (error)
      std::rethrow_exception(error);
  }
  perception::ObsBatch pack() const {
    perception::ObsBatch out;
    for (size_t i = 0; i < envs.size(); ++i) {
      auto b = envs[i]->observation();
      std::fill(b.agent_env.begin(), b.agent_env.end(), i);
      out.append(b);
    }
    return out;
  }
};
VecEnv::VecEnv(core::EnvConfig c, int count, int threads)
    : impl_(std::make_unique<Impl>(c, count, threads)) {}
VecEnv::~VecEnv() = default;
perception::ObsBatch VecEnv::reset(uint64_t seed) {
  std::lock_guard api_lock(impl_->api_mutex);
  impl_->run([&](size_t i) {
    impl_->seeds[i] = seed + i;
    impl_->envs[i]->reset(impl_->seeds[i]);
  });
  return impl_->pack();
}
VecStep VecEnv::step(std::span<const int> actions) {
  std::lock_guard api_lock(impl_->api_mutex);
  std::vector<size_t> offsets{0};
  for (const auto &e : impl_->envs)
    offsets.push_back(offsets.back() + e->observation().n_agents());
  if (offsets.back() != actions.size())
    throw std::invalid_argument("one action per packed agent required");
  for (int a : actions)
    if (a < 0 || a >= kNumActions)
      throw std::invalid_argument("action outside [0,27)");
  VecStep out;
  out.transitions.resize(impl_->envs.size());
  impl_->run([&](size_t i) {
    auto &r = out.transitions[i];
    r = impl_->envs[i]->step(
        actions.subspan(offsets[i], offsets[i + 1] - offsets[i]));
    std::fill(r.observation.agent_env.begin(), r.observation.agent_env.end(),
              i);
    if (r.terminated || r.truncated) {
      // Disjoint per-environment seed streams, independent of worker
      // scheduling.
      impl_->seeds[i] += impl_->envs.size();
      impl_->envs[i]->reset(impl_->seeds[i]);
    }
  });
  out.observation = impl_->pack();
  return out;
}
} // namespace gnc
