/**
 * @file CPUThreading.h
 * @author GreenBean
 * @brief CPU work-stealing thread pool with prioritised scheduling, CCD-restricted
 * task submission, a helping wait so in-worker waits stay deadlock-free, and
 * relaxed-atomics worker telemetry with a best-effort snapshot + formatter
 * for live worker views.
 * @version 0.5
 * @date 2026-10-26
 */
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

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
     * Sentinel for "no CCD group pinned" in telemetry (WorkerSnapshot::pinned_group).
     */
    inline constexpr uint32_t no_pinned_group = std::numeric_limits<uint32_t>::max();

    /**
     * What one pool worker is doing right now, for live views.
     * @brief Live status of a single pool worker.
     */
    enum class WorkerStatus : uint32_t {
        Idle,
        RunningOwn,
        RunningStolen,
        RunningHelped,
        HelpWaiting,
    };

    /**
     * Point-in-time view of one worker. All values are best-effort relaxed
     * reads; task_sequence/task_priority/task_running_ms are valid only when
     * status != Idle (sequence 0, running 0 otherwise).
     * @brief Per-worker telemetry snapshot.
     */
    struct WorkerSnapshot {
        uint32_t worker_id;
        WorkerStatus status;
        uint32_t pinned_group;    // CCD group the current task pins to; no_pinned_group when none
        uint64_t task_sequence;   // current task's submit sequence; 0 when Idle
        TaskPriority task_priority; // valid when status != Idle
        uint64_t task_running_ms; // how long the current task has run; 0 when Idle
        uint32_t futures_awaited; // >0 only in HelpWaiting
        uint64_t tasks_completed;
        uint64_t tasks_stolen;    // stolen from another worker
        uint64_t tasks_helped;    // executed via HelpWait while blocked
        uint64_t local_queue_depth;
    };

    /**
     * Whole-pool point-in-time view, for rendering a live worker display.
     * @brief Pool-wide telemetry snapshot.
     */
    struct PoolSnapshot {
        uint64_t snapshot_ms;   // steady_clock ms timestamp
        uint32_t worker_count;
        uint64_t pending_tasks_total; // sum of local depths
        bool stopping;
        std::vector<WorkerSnapshot> workers;
    };

    /**
     * CPU Threading Task, inserted into the thread pool for execution. Holds priority and function.
     * @brief A task to be executed by the thread pool.
     */
    struct Task {
        std::function<void()> fn;
        TaskPriority priority;
        size_t sequence;
        uint32_t pinned_group = no_pinned_group; // CCD pin captured at submit_to_group time
    };

    /**
     * Multiset order: higher priority first, then lower sequence first (FIFO
     * within a priority level) — the same ordering the old priority_queue's
     * operator< produced, so pop-begin() takes the best task and stealing
     * from rbegin()-ward takes the victim's worst/newest tasks.
     * @brief Strict-weak ordering for a worker's task set.
     */
    struct TaskCompare {
        bool operator()(const Task& a, const Task& b) const {
            if (a.priority != b.priority) {
                return static_cast<uint32_t>(a.priority) > static_cast<uint32_t>(b.priority);
            }
            return a.sequence < b.sequence;
        }
    };

    /**
     * Worker-identity guard state: depth > 0 exactly while a pool worker (or a
     * helping worker) is executing a task, so nested helps and submission
     * fallbacks can be detected from any depth of nesting.
     */
    namespace detail {
        inline thread_local uint32_t worker_depth = 0;

        /**
         * The worker index of the calling thread, set at worker spawn; only
         * meaningful while worker_depth > 0. Used by HelpWait to prefer the
         * own deque when claiming work.
         */
        inline thread_local size_t worker_index = 0;

        /**
         * Current steady_clock time in milliseconds — the timebase for task
         * run durations and snapshot timestamps.
         */
        [[nodiscard]] inline uint64_t now_steady_ms() noexcept {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }

        /**
         * @brief Short fixed text for a worker status.
         */
        [[nodiscard]] inline const char* status_name(const WorkerStatus status) noexcept {
            switch (status) {
                case WorkerStatus::Idle: return "Idle";
                case WorkerStatus::RunningOwn: return "RunningOwn";
                case WorkerStatus::RunningStolen: return "RunningStolen";
                case WorkerStatus::RunningHelped: return "RunningHelped";
                case WorkerStatus::HelpWaiting: return "HelpWaiting";
            }
            return "?"; // unreachable: every enumerator is handled above
        }

        /**
         * @brief Short fixed text for a task priority.
         */
        [[nodiscard]] inline const char* priority_name(const TaskPriority priority) noexcept {
            switch (priority) {
                case TaskPriority::Low: return "Low";
                case TaskPriority::Normal: return "Normal";
                case TaskPriority::High: return "High";
            }
            return "?"; // unreachable: every enumerator is handled above
        }

        // Zero-pads a worker id to two digits ("#03").
        [[nodiscard]] inline std::string two_digit(const uint32_t v) {
            return v < 10u ? "0" + std::to_string(v) : std::to_string(v);
        }

        // Human-readable counter: 999 -> "999", 15200 -> "15.2k", 3400000 -> "3.4M".
        // Integer math (whole + one truncated decimal) — no floating point needed.
        [[nodiscard]] inline std::string human_count(const uint64_t n) {
            if (n >= 1000000u) {
                return std::to_string(n / 1000000u) + "." + std::to_string((n % 1000000u) / 100000u) + "M";
            }
            if (n >= 1000u) {
                return std::to_string(n / 1000u) + "." + std::to_string((n % 1000u) / 100u) + "k";
            }
            return std::to_string(n);
        }
    }

    /**
     * @brief True when the calling thread is a pool worker currently executing a task.
     */
    [[nodiscard]] inline bool IsInPoolWorker() noexcept {
        return detail::worker_depth > 0;
    }

    /**
     * Global CPU Thread Pool, one per BeanTensor instance. Work-stealing scheduler:
     * each worker owns an ordered task set and drains other workers' sets when idle.
     * @brief A work-stealing thread pool for executing tasks on CPU threads.
     */
    class ThreadPool {
        public:
        /**
         * @brief BACKEND ONLY - DO NOT EXECUTE DIRECTLY, USE "GetCPUThreadPool()" INSTEAD.
         * Construct thread pool.
         * @param num_threads
         */
        explicit ThreadPool(const size_t num_threads)
            : deques(num_threads), telemetry(num_threads), deque_mutexes(num_threads) {
                for (size_t i = 0; i < num_threads; i++) {
                    this->threads.emplace_back([this, i] {
                        detail::worker_index = i;
                        while (true) {
                            if (auto task = this->pop_own(i)) {
                                this->execute_task(std::move(*task), TaskProvenance::Own);
                                continue;
                            }
                            if (auto stolen = this->steal(i);
                                !stolen.empty()
                                ) {
                                for (auto& task : stolen) {
                                    this->execute_task(std::move(task), TaskProvenance::Stolen);
                                }
                                continue;
                            }
                            if (
                                const bool drained = this->stop.load() && this->queued_count.load() == 0;
                                drained
                            ) {
                                return;
                            }
                            std::unique_lock lock(this->sleep_mutex);
                            this->sleep_cv.wait(lock, [this] {
                                return this->stop.load() || this->queued_count.load() > 0;
                            });
                            if (
                                const bool drained = this->stop.load() && this->queued_count.load() == 0;
                                drained
                            ) {
                                return;
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
                    std::unique_lock lock(this->sleep_mutex);
                    this->stop = true;
                }
                sleep_cv.notify_all();
                for (auto& t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
            }

        /**
         * Get the number of tasks in the queue, summed across every worker's set.
         * @brief Get the queue size.
         * @return size_t - Number of tasks in the queue.
         */
        size_t queue_size() const {
                auto total = size_t{0};
                for (size_t i = 0; i < this->deques.size(); i++) {
                    const auto lock = std::lock_guard{this->deque_mutexes[i]};
                    total += this->deques[i].size();
                }
                return total;
            }

        /**
         * Best-effort point-in-time view of the whole pool for live worker views.
         * Never takes the sleep mutex; takes each deque mutex BRIEFLY, one at a
         * time. Safe to call from any thread, including pool workers. Telemetry
         * fields are relaxed reads and may describe instants slightly before the
         * queue depths; the caller picks their own "stuck" threshold off
         * task_running_ms.
         * @brief Collect a worker telemetry snapshot.
         * @return PoolSnapshot with per-worker status, counters and queue depths.
         */
        [[nodiscard]] PoolSnapshot snapshot() const {
                auto snap = PoolSnapshot{};
                snap.snapshot_ms = detail::now_steady_ms();
                snap.worker_count = static_cast<uint32_t>(this->deques.size());
                snap.stopping = this->stop.load(std::memory_order_relaxed);
                snap.workers.reserve(this->deques.size());
                for (size_t i = 0; i < this->deques.size(); i++) {
                    const auto lock = std::lock_guard{this->deque_mutexes[i]};
                    const auto depth = this->deques[i].size();
                    snap.pending_tasks_total += depth;
                    const auto& tele = this->telemetry[i];
                    const auto status = tele.status.load(std::memory_order_relaxed);
                    auto worker = WorkerSnapshot{};
                    worker.worker_id = static_cast<uint32_t>(i);
                    worker.status = status;
                    worker.pinned_group = tele.pinned_group.load(std::memory_order_relaxed);
                    worker.task_sequence = tele.task_sequence.load(std::memory_order_relaxed);
                    worker.task_priority = static_cast<TaskPriority>(tele.task_priority.load(std::memory_order_relaxed));
                    worker.task_running_ms = status == WorkerStatus::Idle
                        ? uint64_t{0}
                        : detail::now_steady_ms() - tele.task_start_ms.load(std::memory_order_relaxed);
                    worker.futures_awaited = tele.futures_awaited.load(std::memory_order_relaxed);
                    worker.tasks_completed = tele.tasks_completed.load(std::memory_order_relaxed);
                    worker.tasks_stolen = tele.tasks_stolen.load(std::memory_order_relaxed);
                    worker.tasks_helped = tele.tasks_helped.load(std::memory_order_relaxed);
                    worker.local_queue_depth = depth;
                    snap.workers.push_back(worker);
                }
                return snap;
            }

        /**
         * Pause the calling thread until all tasks are completed. This is a blocking call.
         * @brief Wait for all threads.
         */
        [[maybe_unused]] void wait_all() {
                std::unique_lock lock(this->sleep_mutex);
                this->sleep_cv.wait(lock, [this] {
                    return this->queued_count.load() == 0 && this->active_tasks.load() == 0;
                });
            }

        /**
         * Wait on a future from INSIDE a pool task without deadlocking: while the
         * future is not ready, run other pool tasks (own deque first, then one
         * steal round) so the work this future depends on still gets executed.
         * CALL ONLY FROM POOL TASKS. This exists to make in-worker waits
         * deadlock-free; blocking waits belong on user threads. While waiting,
         * the worker's telemetry shows HelpWaiting with futures_awaited set.
         * @brief Helping wait: wait on a future while executing pool work.
         * @param f Future to wait on.
         */
        void HelpWait(const std::shared_future<void>& f) {
                if (!IsInPoolWorker()) {
                    f.wait();
                    return;
                }
                const auto idx = detail::worker_index < this->deques.size() ? detail::worker_index : size_t{0};
                auto& tele = this->telemetry[idx];
                while (true) {
                    if (
                        const auto status = f.wait_for(std::chrono::duration<int>::zero());
                        status == std::future_status::ready
                    ) {
                        tele.futures_awaited.store(0, std::memory_order_relaxed);
                        return;
                    }
                    tele.status.store(WorkerStatus::HelpWaiting, std::memory_order_relaxed);
                    tele.futures_awaited.store(1, std::memory_order_relaxed);
                    if (auto task = this->pop_own(idx)) {
                        tele.futures_awaited.store(0, std::memory_order_relaxed);
                        this->execute_task(std::move(*task), TaskProvenance::Helped);
                        continue;
                    }
                    if (
                        auto stolen = this->steal(idx);
                        !stolen.empty()
                        ) {
                        tele.futures_awaited.store(0, std::memory_order_relaxed);
                        for (auto& task : stolen) {
                            this->execute_task(std::move(task), TaskProvenance::Helped);
                        }
                        continue;
                    }
                    static_cast<void>(f.wait_for(std::chrono::microseconds(100)));
                }
            }

        /**
         * Wait until ALL futures are ready, helping run pool tasks while waiting.
         * CALL ONLY FROM POOL TASKS — same contract as the single-future overload:
         * this exists to make in-worker waits deadlock-free; blocking waits belong
         * on user threads. While waiting, the worker's telemetry shows
         * HelpWaiting with futures_awaited set to the not-yet-ready count.
         * @brief Helping wait over a set of futures.
         * @param futures Futures to wait on.
         */
        void HelpWait(const std::vector<std::shared_future<void>>& futures) {
                if (!IsInPoolWorker()) {
                    for (const auto& f : futures) {
                        f.wait();
                    }
                    return;
                }
                if (futures.empty()) {
                    return;
                }
                const auto idx = detail::worker_index < this->deques.size() ? detail::worker_index : size_t{0};
                auto& tele = this->telemetry[idx];
                while (true) {
                    auto not_ready = uint32_t{0};
                    const std::shared_future<void>* pending = nullptr;
                    for (const auto& f : futures) {
                        if (
                            const auto status = f.wait_for(std::chrono::duration<int>::zero());
                            status != std::future_status::ready
                        ) {
                            ++not_ready;
                            if (pending == nullptr) {
                                pending = &f;
                            }
                        }
                    }
                    if (pending == nullptr) {
                        tele.futures_awaited.store(0, std::memory_order_relaxed);
                        return;
                    }
                    tele.status.store(WorkerStatus::HelpWaiting, std::memory_order_relaxed);
                    tele.futures_awaited.store(not_ready, std::memory_order_relaxed);
                    if (auto task = this->pop_own(idx)) {
                        tele.futures_awaited.store(0, std::memory_order_relaxed);
                        this->execute_task(std::move(*task), TaskProvenance::Helped);
                        continue;
                    }
                    if (
                        auto stolen = this->steal(idx); !stolen.empty()
                        ) {
                        tele.futures_awaited.store(0, std::memory_order_relaxed);
                        for (auto& task : stolen) {
                            this->execute_task(std::move(task), TaskProvenance::Helped);
                        }
                        continue;
                    }
                    static_cast<void>(pending->wait_for(std::chrono::microseconds(100)));
                }
            }

        /**
         * Submit a task to the thread pool, receiving a waitable future to tell if its complete.
         * Pushes to a worker chosen round-robin; the deque lock is released before
         * the sleep mutex is taken for the wake, so deque locks are never held
         * under the sleep mutex (lock order: sleep-mutex -> deque-mutex is forbidden).
         * @brief Submit a task to the thread pool for execution. Returns a future for the task.
         * @tparam Function [=] Function - A callable function or lambda to execute.
         * @param executableFunction - The function to execute in the thread pool.
         * @param priority Task queueing priority of the function being inserted into the pool.
         * @return std::future<void> - A future that can be waited on to check if the task is complete.
         * @throws BeanTensor::ErrorHandling::ThreadPoolStopped if the pool has been stopped.
         */
        template <typename Function>
            [[nodiscard]] std::future<void> submit(Function&& executableFunction, const TaskPriority priority = TaskPriority::Normal) {
                auto task = std::make_shared<std::packaged_task<void()>>(std::forward<Function>(executableFunction));
                std::future<void> future = task->get_future();
                {
                    const auto target = this->push_counter.fetch_add(1) % this->deques.size();
                    std::lock_guard lock(this->deque_mutexes[target]);
                    if (this->stop.load()) {
                        throw BeanTensor::ErrorHandling::ThreadPoolStopped("ThreadPool::submit on stopped pool");
                    }
                    this->deques[target].insert(Task{ [task] { (*task)(); }, priority, this->sequence++, no_pinned_group });
                    ++this->queued_count;
                }
                {
                    std::lock_guard lock(this->sleep_mutex);
                }
                sleep_cv.notify_all();
                return future;
            }

        /**
         * Submit a task restricted to the given CCD group's CPUs. The worker saves
         * its current affinity, temporarily pins itself to the group's CPU set for
         * the duration of the task, then restores the SAVED set (not the full-union
         * set), so the same workers remain available for unrestricted work and an
         * outer pin survives a nested (helped) inner pinned task.
         * The topology is captured by reference (static storage, outlives the pool).
         * On non-Linux builds, or if the group index is out of range, the restriction
         * is ignored and the task runs as a plain submit. Affinity-call failures never
         * prevent the task from running.
         * @brief Submit a task pinned to one CCD group's CPUs for its duration.
         * @tparam Function [=] Function - A callable function or lambda to execute.
         * @param executableFunction - The function to execute in the thread pool.
         * @param group CCD group index from IncludedCcdTopology().groups.
         * @param priority Task queueing priority of the function being inserted into the pool.
         * @return std::future<void> - A future that can be waited on to check if the task is complete.
         * @throws BeanTensor::ErrorHandling::ThreadPoolStopped if the pool has been stopped.
         */
        template <typename Function>
            [[nodiscard]] std::future<void> submit_to_group(Function&& executableFunction, const uint32_t group, const TaskPriority priority = TaskPriority::Normal) {
                auto task = std::make_shared<std::packaged_task<void()>>([function = std::forward<Function>(executableFunction), group]() mutable {
#if defined(__linux__)
                    const auto&[groups, cpu_to_group] = IncludedCcdTopology();
                    if (
                        const bool valid_group = group < groups.size();
                        valid_group
                    ) {
                        cpu_set_t group_set;
                        CPU_ZERO(&group_set);
                        for (const uint32_t cpu : groups[group]) {
                            // Pass the uint32_t straight through: CPU_SET's macro
                            // body converts to size_t, and casting to int first makes
                            // that an int->size_t sign change (-Werror=sign-conversion).
                            CPU_SET(cpu, &group_set);
                        }
                        cpu_set_t saved_set;
                        CPU_ZERO(&saved_set);
                        pthread_getaffinity_np(pthread_self(), sizeof(saved_set), &saved_set);
                        pthread_setaffinity_np(pthread_self(), sizeof(group_set), &group_set);
                        try {
                            function();
                        }
                        catch (...) {
                            // Restoring the saved set (not the full-union set) keeps an
                            // outer pin intact when a helping worker runs an inner pinned task.
                            pthread_setaffinity_np(pthread_self(), sizeof(saved_set), &saved_set);
                            throw;
                        }
                        pthread_setaffinity_np(pthread_self(), sizeof(saved_set), &saved_set);
                        return;
                    }
#endif
                    function();
                });
                std::future<void> future = task->get_future();
                {
                    const auto target = this->push_counter.fetch_add(1) % this->deques.size();
                    std::lock_guard lock(this->deque_mutexes[target]);
                    if (this->stop.load()) {
                        throw BeanTensor::ErrorHandling::ThreadPoolStopped("ThreadPool::submit_to_group on stopped pool");
                    }
                    this->deques[target].insert(Task{ [task] { (*task)(); }, priority, this->sequence++, group });
                    ++this->queued_count;
                }
                {
                    std::lock_guard lock(this->sleep_mutex);
                }
                sleep_cv.notify_all();
                return future;
            }

            ThreadPool(const ThreadPool&) = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;
            ThreadPool(ThreadPool&&) = delete;
            ThreadPool& operator=(ThreadPool&&) = delete;


        private:
        /**
         * Where a claimed task came from: drives the Running* status and the
         * stolen/helped counters. Own = own deque, Stolen = another worker's
         * deque, Helped = executed via HelpWait while blocked.
         */
        enum class TaskProvenance {
            Own,
            Stolen,
            Helped,
        };

        /**
         * Relaxed-atomics per-worker telemetry. Written only by the owning
         * worker (all stores relaxed); read by snapshot() from any thread.
         */
        struct WorkerTelemetry {
            std::atomic<WorkerStatus> status{WorkerStatus::Idle};
            std::atomic<uint32_t> pinned_group{no_pinned_group};
            std::atomic<uint64_t> task_sequence{0};
            std::atomic<uint32_t> task_priority{0};
            std::atomic<uint64_t> task_start_ms{0};
            std::atomic<uint32_t> futures_awaited{0};
            std::atomic<uint64_t> tasks_completed{0};
            std::atomic<uint64_t> tasks_stolen{0};
            std::atomic<uint64_t> tasks_helped{0};
        };

        /**
         * Maps a task's provenance to the status shown while it runs.
         */
        [[nodiscard]] static WorkerStatus status_for(const TaskProvenance provenance) noexcept {
            switch (provenance) {
                case TaskProvenance::Own: return WorkerStatus::RunningOwn;
                case TaskProvenance::Stolen: return WorkerStatus::RunningStolen;
                case TaskProvenance::Helped: return WorkerStatus::RunningHelped;
            }
            return WorkerStatus::Idle; // unreachable: every enumerator is handled above
        }

        /**
         * RAII worker-identity marker: increments the thread-local depth on entry,
         * decrements on exit. Nesting-safe (helped tasks re-enter).
         */
        struct WorkerDepthGuard {
            WorkerDepthGuard() noexcept {
                ++detail::worker_depth;
            }
            ~WorkerDepthGuard() {
                --detail::worker_depth;
            }
            WorkerDepthGuard(const WorkerDepthGuard&) = delete;
            WorkerDepthGuard& operator=(const WorkerDepthGuard&) = delete;
            WorkerDepthGuard(WorkerDepthGuard&&) = delete;
            WorkerDepthGuard& operator=(WorkerDepthGuard&&) = delete;
        };

        /**
         * RAII in-flight task accounting: decrements active_tasks on scope exit and,
         * when the last in-flight task finishes and the queues are empty, wakes
         * wait_all() waiters under the sleep mutex.
         */
        struct ActiveDecrementGuard {
            explicit ActiveDecrementGuard(ThreadPool& pool) : pool(pool) {
            }
            ~ActiveDecrementGuard() {
                if (this->pool.active_tasks.fetch_sub(1) == 1) {
                    std::unique_lock lock(this->pool.sleep_mutex);
                    if (this->pool.queued_count.load() == 0) {
                        this->pool.sleep_cv.notify_all();
                    }
                }
            }
            ActiveDecrementGuard(const ActiveDecrementGuard&) = delete;
            ActiveDecrementGuard& operator=(const ActiveDecrementGuard&) = delete;
            ActiveDecrementGuard(ActiveDecrementGuard&&) = delete;
            ActiveDecrementGuard& operator=(ActiveDecrementGuard&&) = delete;
            ThreadPool& pool;
        };

        /**
         * RAII task-run telemetry: on scope exit bumps tasks_completed (plus
         * tasks_stolen/tasks_helped for that provenance) and restores the
         * worker to Idle, clearing pin/sequence/priority/start/awaited. Runs
         * before the in-flight decrement so wait_all() observing completion
         * also observes the counters.
         */
        struct TelemetryRunGuard {
            TelemetryRunGuard(WorkerTelemetry& tele, const TaskProvenance provenance) : tele(tele), provenance(provenance) {
            }
            ~TelemetryRunGuard() {
                this->tele.tasks_completed.fetch_add(1, std::memory_order_relaxed);
                if (this->provenance == TaskProvenance::Stolen) {
                    this->tele.tasks_stolen.fetch_add(1, std::memory_order_relaxed);
                }
                if (this->provenance == TaskProvenance::Helped) {
                    this->tele.tasks_helped.fetch_add(1, std::memory_order_relaxed);
                }
                this->tele.futures_awaited.store(0, std::memory_order_relaxed);
                this->tele.pinned_group.store(no_pinned_group, std::memory_order_relaxed);
                this->tele.task_sequence.store(0, std::memory_order_relaxed);
                this->tele.task_priority.store(0, std::memory_order_relaxed);
                this->tele.task_start_ms.store(0, std::memory_order_relaxed);
                this->tele.status.store(WorkerStatus::Idle, std::memory_order_relaxed);
            }
            TelemetryRunGuard(const TelemetryRunGuard&) = delete;
            TelemetryRunGuard& operator=(const TelemetryRunGuard&) = delete;
            TelemetryRunGuard(TelemetryRunGuard&&) = delete;
            TelemetryRunGuard& operator=(TelemetryRunGuard&&) = delete;
            WorkerTelemetry& tele;
            TaskProvenance provenance;
        };

        /**
         * Shared execution wrapper for the main loop and HelpWait — the ONE
         * place task telemetry is written, so both paths agree: worker-identity
         * guard on, status/pin/sequence/priority/start set from the task and
         * its provenance, task runs (packaged_task captures its exceptions
         * into the future; nothing is caught here), counters bumped and the
         * worker restored to Idle on scope exit. No lock is ever held while
         * the task runs.
         * @brief Run one claimed task with guard, telemetry and accounting.
         * @param task The claimed task (priority/sequence/pin feed telemetry).
         * @param provenance Where the task was claimed from.
         */
        void execute_task(Task&& task, const TaskProvenance provenance) {
                const auto depth_guard = WorkerDepthGuard{};
                auto active_guard = ActiveDecrementGuard{*this};
                const auto idx = detail::worker_index < this->telemetry.size() ? detail::worker_index : size_t{0};
                auto& tele = this->telemetry[idx];
                tele.status.store(status_for(provenance), std::memory_order_relaxed);
                tele.pinned_group.store(task.pinned_group, std::memory_order_relaxed);
                tele.task_sequence.store(task.sequence, std::memory_order_relaxed);
                tele.task_priority.store(static_cast<uint32_t>(task.priority), std::memory_order_relaxed);
                tele.task_start_ms.store(detail::now_steady_ms(), std::memory_order_relaxed);
                auto run_guard = TelemetryRunGuard{tele, provenance};
                task.fn();
            }

        /**
         * Pop this worker's best task (multiset begin(): highest priority, then
         * lowest sequence). The queued/active accounting flips inside the deque
         * lock so wait_all() can never observe an empty queue with nothing
         * in flight while a claimed task is about to run.
         * @brief Claim the own deque's best task.
         * @param idx The worker's index.
         * @return The claimed task, or nullopt when the own deque is empty.
         */
        std::optional<Task> pop_own(const size_t idx) {
                auto& deque = this->deques[idx];
                const auto lock = std::lock_guard{this->deque_mutexes[idx]};
                if (deque.empty()) {
                    return std::nullopt;
                }
                auto it = deque.begin();
                auto task = std::optional<Task>{std::in_place, std::move(*it)};
                deque.erase(it);
                ++this->active_tasks;
                --this->queued_count;
                return task;
            }

        /**
         * Scan the other workers round-robin (offset by the thief, so thieves
         * start at different victims) and, from the first non-trivial victim,
         * steal up to half its tasks from the worst end (rbegin()-ward: lowest
         * priority, then newest sequence). The victim keeps at least half its
         * work; accounting flips inside the victim's deque lock.
         * @brief Steal up to half of one victim's worst tasks.
         * @param idx The stealing worker's index.
         * @return Stolen tasks, empty when every victim was empty/trivial.
         */
        std::vector<Task> steal(const size_t idx) {
                const auto count = this->deques.size();
                auto stolen = std::vector<Task>{};
                for (size_t step = 1; step < count; step++) {
                    const auto victim = (idx + step) % count;
                    auto& deque = this->deques[victim];
                    const auto lock = std::lock_guard{this->deque_mutexes[victim]};
                    const auto half = deque.size() / 2;
                    if (half == 0) {
                        continue;
                    }
                    auto split = deque.end();
                    for (size_t j = 0; j < half; j++) {
                        --split;
                    }
                    stolen.reserve(half);
                    for (auto it = split; it != deque.end(); ++it) {
                        stolen.emplace_back(std::move(*it));
                    }
                    deque.erase(split, deque.end());
                    this->active_tasks.fetch_add(half);
                    this->queued_count.fetch_sub(half);
                    break;
                }
                return stolen;
            }

        std::atomic<size_t> active_tasks{0};
        std::atomic<size_t> sequence{0};
        std::atomic<size_t> queued_count{0};
        std::atomic<size_t> push_counter{0};
        std::vector<std::thread> threads;
        std::vector<std::multiset<Task, TaskCompare>> deques;
        std::vector<WorkerTelemetry> telemetry;
        mutable std::vector<std::mutex> deque_mutexes;
        mutable std::mutex sleep_mutex;
        std::condition_variable sleep_cv;
        std::atomic<bool> stop{false};

    };

    /**
     * Renders a PoolSnapshot as plain text for a live worker view: one header
     * line (time, workers, pending, stopping) then one line per worker, e.g.
     *   [pool] t=61233ms workers=32 pending=3 stopping=false
     *   #03 HelpWaiting awaited=2 pin=G1 seq=1204 prio=Normal running=1842ms done=15.2k stolen=301 helped=44 queue=2
     * Counters print human-readable (k/M), status as text, pin as G<n> or -.
     * Pure function: no I/O, safe to call from any thread.
     * @brief Format a pool snapshot for display.
     * @param snap The snapshot to render.
     * @return std::string Multi-line rendering of the snapshot.
     */
    [[nodiscard]] inline std::string format_snapshot(const PoolSnapshot& snap) {
            auto out = std::string{};
            out += "[pool] t=" + std::to_string(snap.snapshot_ms) + "ms workers="
                 + std::to_string(snap.worker_count) + " pending="
                 + std::to_string(snap.pending_tasks_total) + " stopping="
                 + (snap.stopping ? "true" : "false") + "\n";
            for (const auto&[
                worker_id,
                status,
                pinned_group,
                task_sequence,
                task_priority,
                task_running_ms,
                futures_awaited,
                tasks_completed,
                tasks_stolen,
                tasks_helped,
                local_queue_depth] : snap.workers) {
                out += "#" + detail::two_digit(worker_id) + " " + detail::status_name(status);
                if (status == WorkerStatus::HelpWaiting) {
                    out += " awaited=" + std::to_string(futures_awaited);
                }
                out += pinned_group != no_pinned_group
                    ? " pin=G" + std::to_string(pinned_group)
                    : " pin=-";
                if (status != WorkerStatus::Idle) {
                    out += " seq=" + std::to_string(task_sequence);
                    out += " prio=";
                    out += detail::priority_name(task_priority);
                    out += " running=" + std::to_string(task_running_ms) + "ms";
                }
                out += " done=" + detail::human_count(tasks_completed);
                out += " stolen=" + detail::human_count(tasks_stolen);
                out += " helped=" + detail::human_count(tasks_helped);
                out += " queue=" + std::to_string(local_queue_depth);
                out += '\n';
            }
            return out;
        }

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
      [[unlikely]] throw BeanTensor::ErrorHandling::ThreadingUnsupported("Cannot support -fno-threadsafe-statics - remove it, or you lose threading.");
#endif

    }
}
