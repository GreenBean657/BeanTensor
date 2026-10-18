/**
 * @file CPUThreading.h
 * @author GreenBean
 * @brief CPU thread pool and task queue with prioritised scheduling.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>

#include "Helpers/Hardware.h"

namespace BeanTensor::Threading {
    /**
     * Task queueing priority, where higher values get claimed by the pool manager first.
     * NOTE: this is queue-pop order only — it does NOT set OS scheduling priority.
     * @brief Low, Normal, High - Higher gets claimed first.
     */
    enum class TaskPriority {
        Low = 1,
        Normal = 2,
        High = 3,
    };

    /**
     * CPU Threading Task, inserted into the thread pool for execution. Holds priority and function.
     * @brief A task to be executed by the thread pool.
     */
    struct Task {
        std::function<void()> fn;
        TaskPriority priority;
        size_t sequence;

        bool operator<(const Task& other) const {
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return sequence > other.sequence; // FIFO within a priority level
        }
    };

    /**
     * Global CPU Thread Pool, one per BeanTensor instance. Manages threads and executes in order.
     * @brief A thread pool for executing tasks on CPU threads.
     */
    class ThreadPool {
        public:
        /**
         * @brief BACKEND ONLY - DO NOT EXECUTE DIRECTLY, USE "GetCPUThreadPool()" INSTEAD.
         * Construct thread pool.
         * @param num_threads
         */
        explicit ThreadPool(const size_t num_threads) {
                for (size_t i = 0; i < num_threads; i++) {
                    this->threads.emplace_back([this] {
                        while (true) {
                            std::function<void()> task;
                            {
                                std::unique_lock<std::mutex> lock(this->queue_mutex);
                                this->queue_cv.wait(lock, [this] {
                                    return this->stop.load() || !this->tasks.empty();
                                });
                                if (this->stop.load() && this->tasks.empty()) {
                                    return;
                                }
                                task = this->tasks.top().fn;
                                this->tasks.pop();
                                ++active_tasks;
                            }
                            task();
                            // If this was the last in-flight task and the queue is
                            // drained, wake wait_all() waiters. Skip the notify_all
                            // otherwise to avoid a thundering herd per task.
                            if (--active_tasks == 0) {
                                std::unique_lock<std::mutex> lock(this->queue_mutex);
                                if (this->tasks.empty()) {
                                    this->queue_cv.notify_all();
                                }
                            }
                        }
                    });
                }
            }

        /**
         * Shut down the pool. Tasks already queued are drained to completion
         * before workers exit; submit() after this point throws.
         * @brief BACKEND ONLY - DO NOT EXECUTE DIRECTLY, USE "GetCPUThreadPool()" INSTEAD.
         */
        ~ThreadPool() {
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->stop = true;
                }
                queue_cv.notify_all();
                for (auto& t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
            }

        /**
         * Get the number of tasks in the queue.
         * @brief Get the queue size.
         * @return size_t - Number of tasks in the queue.
         */
        size_t queue_size() const {
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                return this->tasks.size();
            }

        /**
         * Pause the calling thread until all tasks are completed. This is a blocking call.
         * @brief Wait for all threads.
         */
        [[maybe_unused]] void wait_all() {
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                this->queue_cv.wait(lock, [this] {
                    return tasks.empty() && active_tasks == 0;
                });
            }

        /**
         * Submit a task to the thread pool, receiving a waitable future to tell if its complete.
         * @brief Submit a task to the thread pool for execution. Returns a future for the task.
         * @tparam Function [=] Function - A callable function or lambda to execute.
         * @param executableFunction - The function to execute in the thread pool.
         * @param priority Task queueing priority of the function being inserted into the pool.
         * @return std::future<void> - A future that can be waited on to check if the task is complete.
         * @throws std::runtime_error if the pool has been stopped.
         */
        template <typename Function>
            [[nodiscard]] std::future<void> submit(Function&& executableFunction, const TaskPriority priority = TaskPriority::Normal) {
                auto task = std::make_shared<std::packaged_task<void()>>(std::forward<Function>(executableFunction));
                std::future<void> future = task->get_future();
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    if (this->stop.load()) {
                        throw std::runtime_error("ThreadPool::submit on stopped pool");
                    }
                    this->tasks.push(Task{ [task] { (*task)(); }, priority, sequence++ });
                }
                queue_cv.notify_one();
                return future;
            }

            ThreadPool(const ThreadPool&) = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;
            ThreadPool(ThreadPool&&) = delete;
            ThreadPool& operator=(ThreadPool&&) = delete;


        private:
        std::atomic<size_t> active_tasks{0};
        std::atomic<size_t> sequence{0};
        std::vector<std::thread> threads;
        std::priority_queue<Task> tasks;
        mutable std::mutex queue_mutex;
        std::condition_variable queue_cv;
        std::atomic<bool> stop{false};

    };

    /**
     * BeanTensor global CPU Thread Pool. Function-local static so
     * IncludedCPUFeatures() is guaranteed initialized before the pool spawns
     * threads, and threads are created on first use rather than program startup.
     * @brief Access the global thread pool for CPU tasks.
     */
    inline ThreadPool& GetCPUThreadPool() {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        // HIP/CUDA complications - first path uses -fno-threadsafe-statics.
        // My IDE believes the first pass instead of watching for the second, and marks this criteria as reached.
        // This branch is dead, but present to prevent errors from a function appearing to auto-cast throw.
        // Edit the middle defined statement to modify this function.
        __builtin_unreachable();
#elif __cpp_threadsafe_static_init
        // Never compile host code with -fno-threadsafe-statics
        // NOLINTNEXTLINE(bugprone-dynamic-static-initializers)
        static ThreadPool pool(IncludedCPUFeatures().threads);
        return pool;
#else
      [[unlikely]] throw std::runtime_error("Cannot support -fno-threadsafe-statics - remove it, or you lose threading.");
#endif

    }
}