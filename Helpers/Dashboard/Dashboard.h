/**
 * @file Dashboard.h
 * @author GreenBean
 * @brief TTY-gated terminal training dashboard, renderer v2: a framed,
 * fixed-geometry layout (header strip, a Training box with batch count,
 * batches/second, a compact last/mean loss readout over the bounded
 * recent-loss ring with a sparkline — non-finite cells render as '?', and an
 * epoch line; a Pool box reusing PoolMonitor::render_pool_body with the
 * OWN/STEAL/HELP/WAIT colors, counters and stuck flag; a one-line footer)
 * redrawn in place by one plain std::thread at a calm 2 Hz. No alternate
 * screen, no cursor hide: after the first frame each frame moves the cursor
 * up by the frame's line count and rewrites the whole block, and every
 * content line is padded to the frame width so the rewrite is exact. stop()
 * does NOT erase: the last complete frame persists on screen and the epoch
 * history + final summary print below it. When constructed with
 * sink_is_tty=false the dashboard is fully inert: no thread is started,
 * report_* and render_once are no-ops, stop() emits nothing, and the sink is
 * never touched — non-TTY runs are byte-identical to running without a
 * dashboard. Geometry contract: each frame re-checks the terminal size
 * (TIOCGWINSZ via the terminal_size() seam); a terminal narrower than the
 * frame or shorter than frame+2 gets ONE "terminal too small" notice line and
 * the dashboard goes render-inert for the run — the render thread exits after
 * the notice, no partial frames — while reports keep recording and stop()
 * still prints the deferred history (the demos rely on it). Mid-run resizes
 * garble at most one tick and self-correct on the next full redraw; while the
 * dashboard is active nothing else may write to the sink (the demos defer
 * their prints). stop() is idempotent, joins the render thread, and is called
 * from the destructor as a backstop (callers should still stop() explicitly).
 * The render thread is never a pool worker and never calls Tensor::sync();
 * it only reads the pool via snapshot() and this object's mutex-guarded feed
 * state. Feeding report_batch from a training loop costs whatever the caller
 * paid to obtain the scalar (typically a sync via item()) — dashboard mode is
 * a watching mode, not a benchmarking mode.
 * @version 0.4
 * @date 2026-11-19
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Tensor/DTypes.h"

namespace BeanTensor::Helpers {

    /**
     * Default render-thread frame rate. Calmer than v1's 10 Hz: the block is
     * rewritten whole each tick, so a slower cadence flickers less.
     * @brief Default dashboard refresh rate in Hz.
     */
    inline constexpr float64_t dashboard_default_refresh_hz = 2.0;
    /**
     * Fixed frame width in terminal cells. Content lines are clipped or
     * padded to fit; terminals narrower than this are too small.
     * @brief Frame width in cells.
     */
    inline constexpr uint32_t dashboard_frame_width = 76;

    /**
     * Live terminal training dashboard. Feed it from the training (user)
     * thread; rendering happens on its own thread when active.
     * @brief TTY-gated framed training dashboard with a pool worker grid.
     * @throws Nothing from report/stop paths; the constructor may throw
     * std::system_error if the render thread cannot be started (no terminal
     * state exists at that point — nothing to restore).
     */
    class Dashboard {
        public:
        /**
         * @brief Construct a dashboard; starts the render thread only when sink_is_tty. Writes nothing.
         * @throws std::system_error On thread creation failure (active mode only).
         */
        Dashboard(std::ostream&, bool, float64_t = dashboard_default_refresh_hz, uint32_t = 120);
        /**
         * @brief Backstop: calls stop() so destruction without stop() cannot leak the thread.
         */
        virtual ~Dashboard();
        Dashboard(const Dashboard&) = delete;
        Dashboard& operator=(const Dashboard&) = delete;
        Dashboard(Dashboard&&) = delete;
        Dashboard& operator=(Dashboard&&) = delete;

        /**
         * @brief Record one batch loss into the bounded recent-loss ring.
         */
        void report_batch(float64_t);
        /**
         * @brief Record one epoch's summary metrics for the frame and the stop() history.
         */
        void report_epoch(uint32_t, uint32_t, float64_t, float64_t, float64_t);
        /**
         * Test seam and snapshot hook: composes one frame and writes it to the
         * sink immediately, on the caller's thread (region-redraw bookkeeping
         * included). No-op when inert.
         * @brief Render one frame now.
         */
        void render_once();
        /**
         * Idempotent. Sets the stop flag, joins the render thread, draws one
         * final frame (unless the terminal proved too small), leaves it on
         * screen, THEN prints the epoch history and final summary below it.
         * No-op when inert or already stopped.
         * @brief Stop the dashboard; persist the final frame and print the summary.
         */
        void stop();

        /**
         * Test accessor: number of losses currently held in the ring
         * (<= loss_window; always 0 when inert).
         * @brief Current loss-ring size.
         */
        [[nodiscard]] uint64_t loss_count() const;
        /**
         * @brief Whether the dashboard is live (render thread started).
         */
        [[nodiscard]] bool active() const noexcept;
        /**
         * Test seam for sinks that are not capture-friendly (the Python
         * binding sinks to std::cerr): the exact bytes of the last frame the
         * renderer successfully WROTE — written into a member under
         * output_mutex_ at each sink write, so it races neither the render
         * thread nor render_once(). A frame that never reached the sink does
         * not count: after the too-small latch fires (one notice, then
         * render-inert for the run), no further frames are composed, so this
         * keeps holding the last complete pre-latch frame. Empty when inert
         * and before the first render.
         * @brief Last successfully written frame; empty when inert.
         */
        [[nodiscard]] std::string last_frame() const;

        protected:
        /**
         * Geometry seam for tests. The default probes TIOCGWINSZ on fd 2 (the
         * demos' sink) and returns nullopt when the size cannot be determined
         * (non-TTY) — the frame check is then skipped, which keeps the
         * stringstream tests size-agnostic. Override to inject a fixed size.
         * @brief Current terminal size as {rows, cols}, or nullopt when unknown.
         */
        [[nodiscard]] virtual std::optional<std::pair<uint32_t, uint32_t>> terminal_size() const;

        private:
        struct EpochRecord {
            uint32_t epoch;
            uint32_t epochs;
            float64_t train_mean_loss;
            float64_t test_accuracy;
            float64_t test_mean_loss;
        };

        [[nodiscard]] std::string compose_frame() const;
        void render_loop();
        void write_frame_locked();

        std::ostream& sink_;
        float64_t refresh_hz_;
        std::chrono::steady_clock::time_point started_;
        uint64_t batches_seen_ = 0;         // guarded by state_mutex_
        std::deque<float64_t> losses_;      // guarded by state_mutex_
        std::vector<EpochRecord> epochs_;   // guarded by state_mutex_
        std::string last_frame_;            // guarded by output_mutex_
        std::thread render_thread_;
        mutable std::mutex state_mutex_;
        mutable std::mutex output_mutex_;   // serializes every sink_ write and the frame bookkeeping
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> stopped_{false};
        uint32_t frame_lines_written_ = 0;  // guarded by output_mutex_
        uint32_t loss_window_;
        bool too_small_ = false;            // guarded by output_mutex_
        bool notice_written_ = false;       // guarded by output_mutex_
        bool active_;
    };
}
