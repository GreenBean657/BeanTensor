/**
 * @file PoolMonitor.h
 * @author GreenBean
 * @brief Dependency-free ANSI terminal dashboard for the global CPU thread pool.
 * Renders PoolSnapshot frames (worker grid, statuses, counters, stuck-flagging)
 * to a stream using raw VT100/ANSI escape codes — no ncurses, no external
 * libraries, per the project's zero-dependency rule. The stuck threshold is a
 * purely presentational concern of the dashboard: scheduling never reads it.
 * @version 0.1
 * @date 2026-10-25
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Threading {

    /**
     * VT100/ANSI escape sequences used by the dashboard. Kept as named
     * constants so the rendering code reads as intent, not escape soup.
     * @brief ANSI escape fragments.
     */
    namespace ansi {
        inline constexpr const char* reset       = "\x1b[0m";
        inline constexpr const char* bold        = "\x1b[1m";
        inline constexpr const char* clear       = "\x1b[2J\x1b[H";
        inline constexpr const char* hide_cursor = "\x1b[?25l";
        inline constexpr const char* show_cursor = "\x1b[?25h";
        inline constexpr const char* alt_on      = "\x1b[?1049h";
        inline constexpr const char* alt_off     = "\x1b[?1049l";
        inline constexpr const char* home        = "\x1b[H";
    }

    /**
     * Foreground colors by worker status. Idle is dim, help-waiting is bright
     * magenta (the "someone is blocked" signal worth spotting at a glance).
     * @brief Status → color map.
     */
    [[nodiscard]] inline const char* status_color(const WorkerStatus status) noexcept {
        switch (status) {
            case WorkerStatus::Idle:          return "\x1b[90m";  // bright black (dim)
            case WorkerStatus::RunningOwn:    return "\x1b[32m";  // green
            case WorkerStatus::RunningStolen: return "\x1b[36m";  // cyan
            case WorkerStatus::RunningHelped: return "\x1b[33m";  // yellow
            case WorkerStatus::HelpWaiting:   return "\x1b[95m";  // bright magenta
        }
        return ansi::reset;
    }

    /**
     * Short status text for a worker cell.
     * @brief Two-to-five letter status tag.
     */
    [[nodiscard]] inline const char* status_tag(const WorkerStatus status) noexcept {
        switch (status) {
            case WorkerStatus::Idle:          return "IDLE";
            case WorkerStatus::RunningOwn:    return "OWN";
            case WorkerStatus::RunningStolen: return "STEAL";
            case WorkerStatus::RunningHelped: return "HELP";
            case WorkerStatus::HelpWaiting:   return "WAIT";
        }
        return "????";
    }

    /**
     * Renders ONE dashboard frame as a string (pure — no I/O, trivially testable).
     * Layout: header (time, pending, workers, stop flag), a grid of worker cells
     * (8 per row), then a per-worker detail strip with counters. A worker whose
     * current task has run longer than stuck_threshold_ms gets a red background —
     * this is a display heuristic ONLY; nothing in the pool consumes it.
     * @brief Render a single dashboard frame.
     * @param snap The pool snapshot to render.
     * @param stuck_threshold_ms Running-time beyond which a worker cell is flagged.
     * @return The full frame, ANSI escape sequences included.
     */
    [[nodiscard]] inline std::string render_dashboard(const PoolSnapshot& snap, const uint64_t stuck_threshold_ms) {
        auto out = std::ostringstream{};
        out << ansi::home << ansi::bold
            << "== BeanTensor pool  t=" << snap.snapshot_ms << "ms"
            << "  workers=" << snap.worker_count
            << "  pending=" << snap.pending_tasks_total
            << (snap.stopping ? "  STOPPING" : "")
            << "  stuck>" << stuck_threshold_ms << "ms" << ansi::reset << "\n\n";

        constexpr auto columns = size_t{8};
        for (size_t row = 0; row < snap.workers.size(); row += columns) {
            for (size_t col = row; col < row + columns && col < snap.workers.size(); ++col) {
                const auto& w = snap.workers[col];
                const auto stuck = w.status != WorkerStatus::Idle && w.task_running_ms > stuck_threshold_ms;
                out << (stuck ? "\x1b[41m\x1b[97m" : status_color(w.status));
                out << '#' << (col < 10 ? "0" : "") << col << ' ' << status_tag(w.status);
                if (w.status != WorkerStatus::Idle) {
                    out << ' ' << w.task_running_ms << "ms";
                }
                if (w.pinned_group != no_pinned_group) {
                    out << " G" << w.pinned_group;
                }
                if (w.local_queue_depth > 0) {
                    out << " q" << w.local_queue_depth;
                }
                if (w.status == WorkerStatus::HelpWaiting && w.futures_awaited > 0) {
                    out << " a" << w.futures_awaited;
                }
                out << ansi::reset << "  ";
            }
            out << "\n";
        }

        out << "\n" << ansi::bold << "id  status   running   done    stolen  helped  queue" << ansi::reset << "\n";
        for (const auto& w : snap.workers) {
            out << status_color(w.status)
                << (w.worker_id < 10 ? "0" : "") << w.worker_id << "  "
                << status_tag(w.status)
                << (w.status == WorkerStatus::HelpWaiting && w.futures_awaited > 0
                    ? " a" + std::to_string(w.futures_awaited)
                    : "")
                << "  " << w.task_running_ms << "ms"
                << "  " << w.tasks_completed
                << "  " << w.tasks_stolen
                << "  " << w.tasks_helped
                << "  " << w.local_queue_depth
                << ansi::reset << "\n";
        }
        return out.str();
    }

    namespace detail {
        struct MonitorState {
            std::atomic<bool> run{false};
            std::thread thread{};
        };

        [[nodiscard]] inline MonitorState& monitor_state() {
            static MonitorState state{};
            return state;
        }
    }

    /**
     * Starts a background thread that snapshots the global pool and renders a
     * dashboard frame to `sink` every `interval` until stop_pool_monitor() or
     * process exit. Frames go to the alternate screen buffer when the stream is
     * a TTY; plain cursor-home rewrites otherwise. No-op if a monitor is already
     * running. Safe to call from any thread; rendering never touches pool locks
     * beyond snapshot()'s brief deque probes.
     * @brief Start the live dashboard thread.
     * @param sink Stream to render to (stderr recommended — keeps stdout clean).
     * @param interval Frame interval.
     * @param stuck_threshold_ms Display-only stuck flag threshold.
     */
    inline void start_pool_monitor(std::ostream& sink,
                                   const std::chrono::milliseconds interval = std::chrono::milliseconds{150},
                                   const uint64_t stuck_threshold_ms = 10'000) {
        auto& state = detail::monitor_state();
        if (state.run.exchange(true)) {
            return;
        }
        state.thread = std::thread([&sink, &state, interval, stuck_threshold_ms] {
            sink << ansi::alt_on << ansi::hide_cursor << ansi::clear;
            while (state.run.load()) {
                sink << render_dashboard(GetCPUThreadPool().snapshot(), stuck_threshold_ms) << std::flush;
                std::this_thread::sleep_for(interval);
            }
            sink << ansi::reset << ansi::show_cursor << ansi::alt_off << std::flush;
        });
    }

    /**
     * Stops the dashboard thread and joins it. Idempotent.
     * @brief Stop the live dashboard.
     */
    inline void stop_pool_monitor() noexcept {
        auto&[run, thread] = detail::monitor_state();
        if (!run.exchange(false)) {
            return;
        }
        if (thread.joinable()) {
            thread.join();
        }
    }

    /**
     * RAII handle tying the dashboard lifetime to a scope — the early-return-
     * safe way to run it for the duration of a demo (see main.cpp).
     * @brief Scope-bound monitor session.
     */
    class MonitorSession {
        public:
        explicit MonitorSession(std::ostream& sink) {
            start_pool_monitor(sink);
        }
        MonitorSession(const MonitorSession&) = delete;
        MonitorSession& operator=(const MonitorSession&) = delete;
        ~MonitorSession() {
            stop_pool_monitor();
        }
    };
}
