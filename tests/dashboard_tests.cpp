/**
 * @file dashboard_tests.cpp
 * @author GreenBean
 * @brief Tests for the TTY-gated training dashboard (Helpers/Dashboard),
 * renderer v2: the framed, persistent, region-redraw layout. Covers the inert
 * non-TTY path (no thread, no output, reports are no-ops), framed frame
 * composition (box-drawing borders, compact last/mean readout, batches per
 * second, sparkline, epoch line, pool worker status tags) via substring
 * checks AND a small ANSI screen model (cursor-up region redraw interpreted
 * into a line grid): the Training box must render above the Pool box with
 * both non-overlapping, the final frame must PERSIST after stop() with the
 * epoch history printed below it, the default refresh is 2 Hz, and a
 * too-small terminal yields exactly one notice line, no frames, and the
 * deferred summary at stop. Also: bounded loss ring, non-finite loss cells
 * ('?' placeholders, never a floating-to-integral conversion), stop()
 * idempotency, destruction without stop(), and a concurrent
 * feed/render/pool-load smoke. No float equality anywhere
 * (-Werror=float-equal); no alt-screen escapes may appear anywhere in v2
 * output.
 * @version 0.3
 * @date 2026-11-19
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Helpers/CPUThreading.h"
#include "Helpers/Dashboard/Dashboard.h"
#include "Helpers/PoolMonitor.h"
#include "Tensor/DTypes.h"

using BeanTensor::float64_t;
using BeanTensor::Helpers::Dashboard;
using BeanTensor::Helpers::dashboard_default_refresh_hz;
namespace bt = BeanTensor::Threading;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

static uint32_t count_occurrences(const std::string& haystack, const std::string& needle) {
    auto count = uint32_t{0};
    auto pos = size_t{0};
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

static bool contains_any_status_tag(const std::string& haystack) {
    return contains(haystack, "IDLE") || contains(haystack, "OWN") || contains(haystack, "STEAL")
        || contains(haystack, "HELP") || contains(haystack, "WAIT");
}

// Dashboard with an injectable terminal size: the geometry contract is
// exercised deterministically instead of depending on the host terminal.
class FixedSizeDashboard : public Dashboard {
    public:
    FixedSizeDashboard(std::ostream& sink, const bool tty, const uint32_t rows, const uint32_t cols,
                       const float64_t refresh_hz = dashboard_default_refresh_hz, const uint32_t loss_window = 120)
        : Dashboard(sink, tty, refresh_hz, loss_window), rows_(rows), cols_(cols) {}

    protected:
    [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>> terminal_size() const override {
        return std::pair{rows_, cols_};
    }

    private:
    uint32_t rows_;
    uint32_t cols_;
};

// Minimal ANSI screen model: a line grid interpreting cursor-home (\x1b[H),
// cursor-up (\x1b[<n>A), erase-display (\x1b[2J) and printable bytes; every
// other CSI sequence (colors, alt-screen, cursor visibility) has no visual
// effect here. Applying the dashboard's byte stream to it shows what a real
// terminal would display, so the tests assert on the final visible content.
class TerminalModel {
    public:
    void apply(const std::string& bytes) {
        auto i = size_t{0};
        while (i < bytes.size()) {
            const auto c = bytes[i];
            if (c == '\x1b' && i + 1 < bytes.size() && bytes[i + 1] == '[') {
                auto j = i + 2;
                while (j < bytes.size() && (bytes[j] < '@' || bytes[j] > '~')) {
                    ++j;
                }
                if (j >= bytes.size()) {
                    return;
                }
                handle_csi(bytes[j], bytes.substr(i + 2, j - (i + 2)));
                i = j + 1;
            } else if (c == '\n') {
                ++row_;
                col_ = 0;
                while (row_ >= lines_.size()) {
                    lines_.emplace_back();
                }
                ++i;
            } else if (c == '\r') {
                col_ = 0;
                ++i;
            } else {
                while (row_ >= lines_.size()) {
                    lines_.emplace_back();
                }
                if (col_ >= lines_[row_].size()) {
                    lines_[row_].resize(col_ + 1, ' ');
                }
                lines_[row_][col_] = c;
                ++col_;
                ++i;
            }
        }
    }

    // First row whose visible text contains the needle; -1 when absent.
    [[nodiscard]] int32_t find_row(const std::string& needle) const {
        for (auto i = size_t{0}; i < lines_.size(); ++i) {
            if (lines_[i].find(needle) != std::string::npos) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    // Last row whose visible text contains the needle; -1 when absent.
    [[nodiscard]] int32_t find_last_row(const std::string& needle) const {
        for (auto i = lines_.size(); i > 0; --i) {
            if (lines_[i - 1].find(needle) != std::string::npos) {
                return static_cast<int32_t>(i - 1);
            }
        }
        return -1;
    }

    [[nodiscard]] uint32_t count_rows_containing(const std::string& needle) const {
        auto count = uint32_t{0};
        for (const auto& line : lines_) {
            if (line.find(needle) != std::string::npos) {
                ++count;
            }
        }
        return count;
    }

    private:
    void handle_csi(const char final_ch, const std::string& params) {
        if (final_ch == 'H') {
            row_ = 0;
            col_ = 0;
        } else if (final_ch == 'A') {
            const auto n = params.empty() ? size_t{1} : static_cast<size_t>(std::stoul(params));
            row_ = row_ >= n ? row_ - n : 0;
            col_ = 0;
        } else if (final_ch == 'J' && params == "2") {
            lines_ = {""};
            row_ = 0;
            col_ = 0;
        }
    }

    std::vector<std::string> lines_{""};
    size_t row_ = 0;
    size_t col_ = 0;
};

static void test_inert_noop() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, false);
        dash.report_batch(float64_t{0.5});
        dash.report_epoch(1, 3, float64_t{0.5}, float64_t{0.9}, float64_t{0.4});
        dash.render_once();
        dash.stop();
        dash.stop();
        CHECK(dash.loss_count() == 0);
        CHECK(!dash.active());
    }
    CHECK(ss.str().empty());
    std::cout << "inert_noop ok\n";
}

static void test_frame_content() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, true);
        CHECK(dash.active());
        dash.report_batch(float64_t{0.5});
        dash.report_batch(float64_t{0.3});
        dash.report_batch(float64_t{0.1});
        CHECK(dash.loss_count() == 3);
        dash.render_once();
        const auto frame = ss.str();
        CHECK(contains(frame, "last=0.1"));
        CHECK(contains(frame, "mean=0.3"));
        CHECK(contains(frame, "spark: "));
        CHECK(contains(frame, "b/s="));
        CHECK(contains_any_status_tag(frame));
        // v2: no alt-screen enter/leave, no cursor hide/show anywhere.
        CHECK(!contains(frame, "\x1b[?1049h"));
        CHECK(!contains(frame, "\x1b[?1049l"));
        CHECK(!contains(frame, "\x1b[?25l"));
        CHECK(!contains(frame, "\x1b[?25h"));
        dash.report_epoch(1, 3, float64_t{0.3}, float64_t{0.91}, float64_t{0.25});
        dash.render_once();
        CHECK(contains(ss.str(), "1/3"));
        dash.stop();
    }
    std::cout << "frame_content ok\n";
}

static void test_ring_bound() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, true, float64_t{10.0}, 120);
        for (uint32_t i = 0; i < 300; ++i) {
            dash.report_batch(static_cast<float64_t>(i));
        }
        CHECK(dash.loss_count() == 120);
        dash.render_once();
        CHECK(contains(ss.str(), "last=299"));
        dash.stop();
    }
    std::cout << "ring_bound ok\n";
}

static void test_nonfinite_loss_cells() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, true);
        dash.report_batch(float64_t{0.5});
        dash.report_batch(std::numeric_limits<float64_t>::quiet_NaN());
        dash.report_batch(float64_t{0.3});
        dash.report_batch(std::numeric_limits<float64_t>::infinity());
        dash.report_batch(float64_t{0.1});
        dash.render_once();
        const auto frame = ss.str();
        CHECK(contains(frame, "spark: "));
        CHECK(contains(frame, "?"));
        CHECK(contains(frame, "last=0.1"));
        CHECK(contains(frame, "mean=0.3"));
        dash.stop();
    }
    std::cout << "nonfinite_loss_cells ok\n";
}

static void test_framed_layout_terminal_model() {
    auto ss = std::ostringstream{};
    {
        FixedSizeDashboard dash(ss, true, 60, 100);
        dash.report_batch(float64_t{0.5});
        dash.report_batch(float64_t{0.3});
        dash.report_epoch(1, 3, float64_t{0.3}, float64_t{0.91}, float64_t{0.25});
        dash.render_once();
        dash.report_batch(float64_t{0.2});
        dash.render_once();
        const auto frame_bytes = ss.str();
        CHECK(!contains(frame_bytes, "\x1b[?1049h"));
        auto term = TerminalModel{};
        term.apply(frame_bytes);
        const auto training_top = term.find_row("┌─ Training");
        const auto pool_top = term.find_row("┌─ Pool");
        const auto spark_row = term.find_row("spark:");
        const auto epoch_row = term.find_row("1/3");
        const auto last_row = term.find_row("last=0.2");
        const auto bottom_row = term.find_last_row("└");
        CHECK(training_top >= 0);
        CHECK(pool_top >= 0);
        CHECK(training_top < pool_top);
        CHECK(spark_row > training_top);
        CHECK(spark_row < pool_top);
        CHECK(epoch_row > training_top);
        CHECK(epoch_row < pool_top);
        // The second render_once region-redrew: the visible readout is the latest.
        CHECK(last_row > training_top);
        CHECK(last_row < pool_top);
        CHECK(bottom_row > pool_top);
        CHECK(term.find_row("IDLE") >= 0 || term.find_row("OWN") >= 0);
        dash.stop();
    }
    std::cout << "framed_layout_terminal_model ok\n";
}

static void test_persistence_after_stop() {
    auto ss = std::ostringstream{};
    {
        FixedSizeDashboard dash(ss, true, 60, 100);
        dash.report_batch(float64_t{0.5});
        dash.report_epoch(1, 2, float64_t{0.5}, float64_t{0.8}, float64_t{0.4});
        dash.report_epoch(2, 2, float64_t{0.3}, float64_t{0.9}, float64_t{0.2});
        dash.render_once();
        dash.stop();
    }
    const auto bytes = ss.str();
    CHECK(!contains(bytes, "\x1b[?1049h"));
    CHECK(!contains(bytes, "\x1b[?1049l"));
    auto term = TerminalModel{};
    term.apply(bytes);
    // The final frame persists: its borders and content stay visible...
    const auto training_top = term.find_row("┌─ Training");
    const auto bottom_row = term.find_last_row("└");
    CHECK(training_top >= 0);
    CHECK(bottom_row >= 0);
    // ...and the summary + epoch history print BELOW the frame's last row.
    const auto summary_row = term.find_row("Training summary");
    CHECK(summary_row >= 0);
    CHECK(summary_row > bottom_row);
    CHECK(term.find_last_row("epoch 2/2") > summary_row);
    CHECK(term.count_rows_containing("epoch 2/2") == 2);
    std::cout << "persistence_after_stop ok\n";
}

static void test_refresh_default_2hz() {
    CHECK(dashboard_default_refresh_hz > float64_t{1.9});
    CHECK(dashboard_default_refresh_hz < float64_t{2.1});
    std::cout << "refresh_default_2hz ok\n";
}

static void test_too_small_terminal() {
    auto ss = std::ostringstream{};
    {
        FixedSizeDashboard dash(ss, true, 10, 40);
        dash.report_batch(float64_t{0.5});
        dash.report_epoch(1, 1, float64_t{0.5}, float64_t{1.0}, float64_t{0.0});
        dash.render_once();
        CHECK(count_occurrences(ss.str(), "terminal too small") == 1);
        CHECK(!contains(ss.str(), "┌"));
        dash.render_once();
        CHECK(count_occurrences(ss.str(), "terminal too small") == 1);
        dash.stop();
        // The notice is never repeated, no frame is ever drawn, but the
        // deferred summary still prints (the demos rely on it for history).
        CHECK(count_occurrences(ss.str(), "terminal too small") == 1);
        CHECK(!contains(ss.str(), "┌"));
        CHECK(contains(ss.str(), "Training summary"));
        CHECK(contains(ss.str(), "1/1"));
    }
    std::cout << "too_small_terminal ok\n";
}

static void test_stop_idempotent_and_summary() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, true);
        dash.report_batch(float64_t{0.5});
        dash.report_epoch(1, 2, float64_t{0.5}, float64_t{0.8}, float64_t{0.4});
        dash.report_epoch(2, 2, float64_t{0.3}, float64_t{0.9}, float64_t{0.2});
        dash.stop();
        const auto size_after_first_stop = ss.str().size();
        CHECK(contains(ss.str(), "Training summary"));
        CHECK(contains(ss.str(), "1/2"));
        CHECK(contains(ss.str(), "2/2"));
        dash.stop();
        CHECK(ss.str().size() == size_after_first_stop);
    }
    std::cout << "stop_idempotent_and_summary ok\n";
}

static void test_destruction_without_stop() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, true);
        dash.report_batch(float64_t{0.5});
        dash.report_epoch(1, 1, float64_t{0.5}, float64_t{1.0}, float64_t{0.0});
    }
    // The destructor backstops stop(): no hang, the summary prints, and v2
    // has nothing to restore — no alt-screen leave escape may appear.
    CHECK(contains(ss.str(), "Training summary"));
    CHECK(contains(ss.str(), "1/1"));
    CHECK(!contains(ss.str(), "\x1b[?1049l"));
    std::cout << "destruction_without_stop ok\n";
}

static void test_frame_before_history_byte_order() {
    auto ss = std::ostringstream{};
    {
        Dashboard dash(ss, true);
        dash.report_epoch(1, 3, float64_t{0.5}, float64_t{0.9}, float64_t{0.4});
        dash.render_once();
        dash.stop();
    }
    const auto content = ss.str();
    const auto frame_pos = content.find("┌");
    const auto history_pos = content.find("Training summary");
    CHECK(frame_pos != std::string::npos);
    CHECK(history_pos != std::string::npos);
    CHECK(frame_pos < history_pos);
    CHECK(!contains(content, "\x1b[?1049h"));
    CHECK(!contains(content, "\x1b[?1049l"));
    std::cout << "frame_before_history_byte_order ok\n";
}

static void test_concurrent_feed_render_and_pool() {
    auto ss = std::ostringstream{};
    auto done = std::atomic<uint32_t>{0};
    {
        Dashboard dash(ss, true, float64_t{200.0});
        auto& pool = bt::GetCPUThreadPool();
        auto futures = std::vector<std::future<void>>{};
        futures.reserve(64);
        for (uint32_t i = 0; i < 64; ++i) {
            futures.push_back(pool.submit([&done] {
                std::this_thread::sleep_for(std::chrono::milliseconds{2});
                done.fetch_add(1, std::memory_order_relaxed);
            }));
        }
        for (uint32_t i = 0; i < 500; ++i) {
            dash.report_batch(static_cast<float64_t>(i) * float64_t{0.001});
        }
        dash.render_once();
        for (auto& f : futures) {
            f.get();
        }
        dash.stop();
    }
    CHECK(done.load(std::memory_order_relaxed) == 64);
    CHECK(contains_any_status_tag(ss.str()));
    std::cout << "concurrent_feed_render_and_pool ok\n";
}

int main() {
    test_inert_noop();
    test_frame_content();
    test_ring_bound();
    test_nonfinite_loss_cells();
    test_framed_layout_terminal_model();
    test_persistence_after_stop();
    test_refresh_default_2hz();
    test_too_small_terminal();
    test_stop_idempotent_and_summary();
    test_destruction_without_stop();
    test_frame_before_history_byte_order();
    test_concurrent_feed_render_and_pool();
    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << failures << " TEST(S) FAILED" << std::endl;
    return 1;
}
