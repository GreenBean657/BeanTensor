/**
 * @file Dashboard.cpp
 * @author GreenBean
 * @brief Implementation of the TTY-gated framed training dashboard (renderer
 * v2). See Dashboard.h for the inert/active contract, the geometry contract
 * (per-frame TIOCGWINSZ check; one notice then render-inert when too small),
 * and the region-redraw/persistence model. There is no terminal state to
 * restore in v2 — no alternate screen, no hidden cursor — so the constructor
 * starts the render thread plainly and a construction throw needs no undo.
 * @version 0.4
 * @date 2026-11-19
 */
#include "Helpers/Dashboard/Dashboard.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>

#include <sys/ioctl.h>
#include <unistd.h>

#include "Helpers/CPUThreading.h"
#include "Helpers/PoolMonitor.h"

namespace BeanTensor::Helpers {
namespace {

    // Interior width of a box: frame width minus the "│ " and " │" borders.
    constexpr auto kInterior = size_t{72};

    // Printable-cell width of a line, skipping CSI escape sequences. Content
    // lines are ASCII (metrics, pool grid); the multibyte box borders are
    // constructed directly and never measured here.
    [[nodiscard]] size_t visible_width(const std::string& line) {
        auto width = size_t{0};
        auto i = size_t{0};
        while (i < line.size()) {
            if (line[i] == '\x1b' && i + 1 < line.size() && line[i + 1] == '[') {
                auto j = i + 2;
                while (j < line.size() && (line[j] < '@' || line[j] > '~')) {
                    ++j;
                }
                i = j < line.size() ? j + 1 : line.size();
            } else {
                ++width;
                ++i;
            }
        }
        return width;
    }

    // Cut a line at max_visible printable cells, preserving escape sequences
    // and closing any open color so the box border is not painted.
    [[nodiscard]] std::string clip_ansi(const std::string& line, const size_t max_visible) {
        auto out = std::string{};
        auto visible = size_t{0};
        auto colored = false;
        auto i = size_t{0};
        while (i < line.size() && visible < max_visible) {
            if (line[i] == '\x1b' && i + 1 < line.size() && line[i + 1] == '[') {
                auto j = i + 2;
                while (j < line.size() && (line[j] < '@' || line[j] > '~')) {
                    ++j;
                }
                if (j >= line.size()) {
                    break;
                }
                if (line[j] == 'm') {
                    colored = true;
                }
                out.append(line, i, j - i + 1);
                i = j + 1;
            } else {
                out.push_back(line[i]);
                ++visible;
                ++i;
            }
        }
        if (colored) {
            out += Threading::ansi::reset;
        }
        return out;
    }

    [[nodiscard]] uint32_t count_lines(const std::string& text) {
        return static_cast<uint32_t>(std::count(text.begin(), text.end(), '\n'));
    }

    void append_repeated(std::string& out, const std::string& piece, const size_t count) {
        for (auto i = size_t{0}; i < count; ++i) {
            out += piece;
        }
    }

    // One box content row: clipped or space-padded to the interior width so
    // the right border stays aligned and the region redraw overwrites exactly.
    [[nodiscard]] std::string box_line(const std::string& content) {
        auto clipped = content;
        if (visible_width(clipped) > kInterior) {
            clipped = clip_ansi(clipped, kInterior);
        }
        auto out = std::string{"│ "};
        out += clipped;
        const auto pad = kInterior - visible_width(clipped);
        out.append(pad, ' ');
        out += " │";
        return out;
    }

    [[nodiscard]] std::string top_border(const std::string& title) {
        auto out = std::string{"┌─ "};
        out += title;
        out += ' ';
        const auto used = size_t{4} + title.size();
        append_repeated(out, "─", size_t{dashboard_frame_width} - used - 1);
        out += "┐";
        return out;
    }

    [[nodiscard]] std::string bottom_border() {
        auto out = std::string{"└"};
        append_repeated(out, "─", size_t{dashboard_frame_width} - 2);
        out += "┘";
        return out;
    }
}

    /**
     * @brief Construct a dashboard over a sink.
     * @param sink Stream frames and the final summary are written to (demos pass std::cerr).
     * @param sink_is_tty When false the dashboard is inert: no thread, no output, reports discarded.
     * @param refresh_hz Frame rate of the render thread (active mode only); non-positive clamps to the default.
     * @param loss_window Capacity of the recent-loss ring; 0 clamps to 1.
     */
    Dashboard::Dashboard(std::ostream& sink, const bool sink_is_tty, const float64_t refresh_hz, const uint32_t loss_window)
        : sink_(sink),
          refresh_hz_(refresh_hz > float64_t{0.0} ? refresh_hz : dashboard_default_refresh_hz),
          started_(std::chrono::steady_clock::now()),
          loss_window_(loss_window > 0 ? loss_window : 1),
          active_(sink_is_tty) {
        if (active_) {
            // v2 claims no terminal state, so a throw here leaves nothing to undo.
            render_thread_ = std::thread([this] {
                render_loop();
            });
        }
    }

    Dashboard::~Dashboard() {
        stop();
    }

    /**
     * @brief Record one batch loss; evicts the oldest entry past loss_window.
     * @param batch_loss Scalar loss for the batch (already synced by the caller).
     */
    void Dashboard::report_batch(const float64_t batch_loss) {
        if (!active_) {
            return;
        }
        std::lock_guard lock(state_mutex_);
        losses_.push_back(batch_loss);
        while (losses_.size() > loss_window_) {
            losses_.pop_front();
        }
        ++batches_seen_;
    }

    /**
     * @brief Append one epoch's metrics to the history printed by stop().
     * @param epoch 1-based epoch index.
     * @param epochs Total planned epochs.
     * @param train_mean_loss Mean training loss over the epoch.
     * @param test_accuracy Test accuracy after the epoch.
     * @param test_mean_loss Mean test loss after the epoch.
     */
    void Dashboard::report_epoch(const uint32_t epoch, const uint32_t epochs, const float64_t train_mean_loss,
                                 const float64_t test_accuracy, const float64_t test_mean_loss) {
        if (!active_) {
            return;
        }
        std::lock_guard lock(state_mutex_);
        epochs_.push_back(EpochRecord{epoch, epochs, train_mean_loss, test_accuracy, test_mean_loss});
    }

    void Dashboard::render_once() {
        if (!active_) {
            return;
        }
        std::lock_guard lock(output_mutex_);
        write_frame_locked();
    }

    void Dashboard::stop() {
        if (!active_) {
            return;
        }
        auto expected = false;
        if (!stopped_.compare_exchange_strong(expected, true)) {
            return;
        }
        stop_requested_.store(true, std::memory_order_relaxed);
        if (render_thread_.joinable()) {
            render_thread_.join();
        }
        auto epochs_copy = std::vector<EpochRecord>{};
        auto batches = uint64_t{0};
        {
            std::lock_guard lock(state_mutex_);
            epochs_copy = epochs_;
            batches = batches_seen_;
        }
        std::lock_guard lock(output_mutex_);
        if (!too_small_) {
            // One last frame so the persisting block reflects the final reports.
            write_frame_locked();
        }
        sink_ << "\n== Training summary ==\n";
        sink_ << "batches: " << batches << "\n";
        for (const auto& record : epochs_copy) {
            sink_ << "epoch " << record.epoch << "/" << record.epochs
                  << "  train_loss=" << record.train_mean_loss
                  << "  test_acc=" << record.test_accuracy
                  << "  test_loss=" << record.test_mean_loss << "\n";
        }
        sink_ << std::flush;
    }

    uint64_t Dashboard::loss_count() const {
        std::lock_guard lock(state_mutex_);
        return static_cast<uint64_t>(losses_.size());
    }

    bool Dashboard::active() const noexcept {
        return active_;
    }

    /**
     * @brief Test seam accessor for sinks that are not capture-friendly.
     * @return The exact bytes of the last frame written to the sink; empty
     *         when inert or before the first render.
     */
    std::string Dashboard::last_frame() const {
        if (!active_) {
            return {};
        }
        std::lock_guard lock(output_mutex_);
        return last_frame_;
    }

    std::optional<std::pair<uint32_t, uint32_t>> Dashboard::terminal_size() const {
        auto ws = winsize{};
        if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) != 0) {
            return std::nullopt;
        }
        return std::pair{static_cast<uint32_t>(ws.ws_row), static_cast<uint32_t>(ws.ws_col)};
    }

    /**
     * The metrics block reserves its rows unconditionally (the epoch line is
     * emitted as "epoch -/-" until the first report) and the pool worker
     * count is fixed per process, so the frame height is constant across the
     * run — the cursor-up region redraw then rewrites exactly the rows the
     * previous frame occupied. Non-finite losses render as '?' cells and are
     * excluded from the min/max/mean so no floating-to-integral conversion
     * ever sees one.
     * @brief Compose one frame: header strip, Training box, Pool box, footer.
     * @return The full frame, ANSI colors and box-drawing borders included, no cursor movement.
     */
    std::string Dashboard::compose_frame() const {
        auto losses_copy = std::deque<float64_t>{};
        auto last_epoch = EpochRecord{};
        auto have_epoch = false;
        auto batches = uint64_t{0};
        {
            std::lock_guard lock(state_mutex_);
            losses_copy = losses_;
            batches = batches_seen_;
            if (!epochs_.empty()) {
                last_epoch = epochs_.back();
                have_epoch = true;
            }
        }
        const auto elapsed = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - started_).count();
        const auto rate = elapsed > float64_t{0.0} ? static_cast<float64_t>(batches) / elapsed : float64_t{0.0};
        auto finite_sum = float64_t{0.0};
        auto finite_count = uint64_t{0};
        auto lo = std::numeric_limits<float64_t>::infinity();
        auto hi = -std::numeric_limits<float64_t>::infinity();
        for (const float64_t loss : losses_copy) {
            if (!std::isfinite(loss)) {
                continue;
            }
            finite_sum += loss;
            ++finite_count;
            lo = std::min(lo, loss);
            hi = std::max(hi, loss);
        }
        auto out = std::ostringstream{};
        out << Threading::ansi::bold << "== BeanTensor training ==" << Threading::ansi::reset
            << "  " << refresh_hz_ << " Hz\n";
        out << top_border("Training") << "\n";
        auto counts = std::ostringstream{};
        counts << "batches=" << batches
               << "  b/s=" << rate
               << "  window=" << losses_copy.size() << "/" << loss_window_;
        out << box_line(counts.str()) << "\n";
        auto readout = std::ostringstream{};
        readout << "last=";
        if (losses_copy.empty()) {
            readout << "-";
        } else {
            readout << losses_copy.back();
        }
        readout << "  mean=";
        if (finite_count > 0) {
            readout << finite_sum / static_cast<float64_t>(finite_count);
        } else {
            readout << "-";
        }
        readout << "  spark: ";
        constexpr const char* ramp = " .:-=+*#%@";
        constexpr auto ramp_last = size_t{9};
        const float64_t span = hi - lo;
        for (const float64_t loss : losses_copy) {
            if (!std::isfinite(loss)) {
                readout << '?';
                continue;
            }
            auto idx = size_t{0};
            if (span > float64_t{0.0}) {
                const float64_t t = (loss - lo) / span;
                idx = static_cast<size_t>(t * static_cast<float64_t>(ramp_last));
                if (idx > ramp_last) {
                    idx = ramp_last;
                }
            }
            readout << ramp[idx];
        }
        out << box_line(readout.str()) << "\n";
        auto epoch = std::ostringstream{};
        if (have_epoch) {
            epoch << "epoch " << last_epoch.epoch << "/" << last_epoch.epochs
                  << "  train_loss=" << last_epoch.train_mean_loss
                  << "  test_acc=" << last_epoch.test_accuracy
                  << "  test_loss=" << last_epoch.test_mean_loss;
        } else {
            epoch << "epoch -/-";
        }
        out << box_line(epoch.str()) << "\n";
        out << bottom_border() << "\n";
        out << top_border("Pool") << "\n";
        const auto body = Threading::render_pool_body(Threading::GetCPUThreadPool().snapshot(), Threading::default_stuck_threshold_ms);
        auto start = size_t{0};
        while (start < body.size()) {
            const auto nl = body.find('\n', start);
            if (nl == std::string::npos) {
                out << box_line(body.substr(start)) << "\n";
                break;
            }
            if (nl + 1 == body.size()) {
                break;
            }
            out << box_line(body.substr(start, nl - start)) << "\n";
            start = nl + 1;
        }
        out << bottom_border() << "\n";
        out << "-- dashboard live; the final frame persists after stop() --\n";
        return out.str();
    }

    /**
     * Geometry-checked frame write. With the output mutex held by the caller:
     * composes, verifies the terminal still fits the block (when the size is
     * knowable), writes the one-time notice and goes render-inert when it does
     * not, otherwise cursor-ups over the previous frame and rewrites the block.
     * @brief Compose and write one frame, honoring the geometry contract.
     */
    void Dashboard::write_frame_locked() {
        const auto frame = compose_frame();
        const auto lines = count_lines(frame);
        if (
            const auto size = terminal_size();
            size.has_value() && (size->first < lines + 2 || size->second < dashboard_frame_width)
        ) {
            too_small_ = true;
            if (!notice_written_) {
                notice_written_ = true;
                sink_ << "terminal too small for the dashboard (need "
                      << dashboard_frame_width << "x" << (lines + 2)
                      << ", have " << size->second << "x" << size->first
                      << ") - dashboard disabled\n" << std::flush;
            }
            return;
        }
        if (frame_lines_written_ > 0) {
            sink_ << "\x1b[" << frame_lines_written_ << "A";
        }
        sink_ << frame << std::flush;
        frame_lines_written_ = lines;
        last_frame_ = frame;
    }

    void Dashboard::render_loop() {
        const auto interval = std::chrono::duration<float64_t>(float64_t{1.0} / refresh_hz_);
        while (!stop_requested_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(interval);
            if (stop_requested_.load(std::memory_order_relaxed)) {
                break;
            }
            std::lock_guard lock(output_mutex_);
            write_frame_locked();
            if (too_small_) {
                // Too small is terminal for the run: the notice is out, so the
                // render thread exits instead of churning frames that never fit.
                break;
            }
        }
    }
}
