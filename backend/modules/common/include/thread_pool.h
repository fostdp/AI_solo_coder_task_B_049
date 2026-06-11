#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>

namespace tcm {
namespace common {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using ReturnType = decltype(f(args...));

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return future;
    }

    size_t size() const { return workers_.size(); }
    size_t pending_tasks() const;

    void wait_all();
    void shutdown();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};

    void worker_loop();
};

} // namespace common
} // namespace tcm
