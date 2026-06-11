#include "thread_pool.h"

namespace tcm {
namespace common {

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        active_tasks_++;
        task();
        active_tasks_--;
    }
}

size_t ThreadPool::pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}

void ThreadPool::wait_all() {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (tasks_.empty() && active_tasks_ == 0) break;
        }
        std::this_thread::yield();
    }
}

void ThreadPool::shutdown() {
    stop_ = true;
    condition_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

} // namespace common
} // namespace tcm
