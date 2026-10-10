#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>

#include "Helpers/Hardware.h"

namespace BeanTensor::Threading {
    /**
     * CPU Threading Priority, where higher values get claimed by the pool manager first.
     * @brief Low, Normal, High - Higher gets claimed first.
     */
    enum class ThreadPriority {
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
        ThreadPriority priority;
        size_t sequence;

        bool operator<(const Task& other) const {
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return sequence < other.sequence;
        }
    };

    /**
     * Global CPU Thread Pool, one per BeanTensor instance. Manages threads and executes in order.
     * @brief A thread pool for executing tasks on CPU threads.
     */
    class ThreadPool {
        public:
        /**
         * @brief BACKEND ONLY - DO NOT EXECUTE DIRECTLY, INDEX "CPUThreadPool" INSTEAD.
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
                            --active_tasks;
                            this->queue_cv.notify_all();
                        }
                    });
                }
            }

        /**
         * Kill the current threadpool, killing threads as soon as possible.
         * @brief BACKEND ONLY - DO NOT EXECUTE DIRECTLY, INDEX "CPUThreadPool" INSTEAD.
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
         * Pause the main thread until all tasks are completed. This is a blocking call.
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
         * @param priority Thread priority of the function being inserted into the pool.
         * @return std::future<void> - A future that can be waited on to check if the task is complete.
         */
        template <typename Function>
            [[nodiscard]] std::future<void> submit(Function&& executableFunction, const ThreadPriority priority = ThreadPriority::Normal) {
                auto task = std::make_shared<std::packaged_task<void()>>(std::forward<Function>(executableFunction));
                std::future<void> future = task->get_future();
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
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

    //TODO: Try CONST
    /**
     * BeanTensor global CPU Thread Pool.
     * @brief A global thread pool for CPU tasks.
     */
    inline static auto CPUThreadPool = ThreadPool(Hardware::CurrentCPUFeatures.threads);
}
