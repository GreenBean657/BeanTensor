#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

#include "Helper/Hardware.h"

namespace BeanTensor::Threading {

    enum class ThreadPriority {
        Low = 1,
        Normal = 2,
        High = 3,
    };

    struct Task {
        std::function<void()> fn;
        ThreadPriority priority;
        size_t sequence;

        bool operator<(const Task& other) const {
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return sequence > other.sequence;
        }
    };

    class ThreadPool {
    public:
        explicit ThreadPool(const size_t num_threads) {
            for (size_t i = 0; i < num_threads; i++) {
                threads.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            queue_cv.wait(lock, [this] {
                                return stop.load() || !tasks.empty();
                            });
                            if (stop.load() && tasks.empty()) {
                                return;
                            }
                            task = tasks.top().fn;
                            tasks.pop();
                            ++active_tasks;
                        }
                        task();
                        --active_tasks;
                        queue_cv.notify_all();
                    }
                });
            }
        }
        // Non-copyable
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        ThreadPool(ThreadPool&&) = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

        size_t queue_size() const {
            std::unique_lock<std::mutex> lock(queue_mutex);
            return tasks.size();
        }

        [[maybe_unused]] void wait_all() {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this] { return tasks.empty() && active_tasks == 0; });        }
        template<typename F>
        std::future<void> submit(F&& f, const ThreadPriority priority = ThreadPriority::Normal) {
            auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
            std::future<void> future = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                tasks.push(Task{ [task] { (*task)(); }, priority, sequence++ });
            }
            queue_cv.notify_one();
            return future;
        }
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            queue_cv.notify_all();
            for (auto& t : threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
        }
    private:
        std::atomic<size_t> active_tasks{0};
        std::atomic<size_t> sequence{0};
        std::vector<std::thread> threads;
        std::priority_queue<Task> tasks;
        mutable std::mutex queue_mutex;
        std::condition_variable queue_cv;
        std::atomic<bool> stop{false};
    };

    inline ThreadPool& get_cpu_thread_pool() {
        static ThreadPool cpu_thread_pool(Hardware::CPU().threads);
        return cpu_thread_pool;
    }
    inline size_t get_cpu_thread_pool_queue_size() {
        return get_cpu_thread_pool().queue_size();
    }
}
