/**
 * @file OOMPolicy.cpp
 * @author GreenBean
 * @brief Implementation of the process-wide OOM policy and append-only diagnostic log.
 * @version 0.1
 * @date 2026-11-14
 */
#include "Helpers/OOMPolicy.h"

#include <iomanip>
#include <sstream>

#include "Helpers/Errors.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Helpers {

    std::atomic<bool> OOMPolicy::compute_fallback_eligible_test_{false};

    OOMPolicy& OOMPolicy::instance() {
        static OOMPolicy policy;
        return policy;
    }

    void OOMPolicy::configure(const OOMPolicyConfig& config) {
        const auto lock = std::lock_guard{mutex_};
        if (configured_) {
            throw ErrorHandling::InvalidOOMPolicy("OOM policy has already been configured");
        }
        if (config.mode == OOMPolicyMode::CheckpointAndThrow) {
            if (!config.checkpoint_hook) {
                throw ErrorHandling::InvalidOOMPolicy("checkpoint mode requires a checkpoint_hook");
            }
            if (config.checkpoint_destination.empty()) {
                throw ErrorHandling::InvalidOOMPolicy("checkpoint mode requires a non-empty checkpoint_destination");
            }
        }
        config_ = config;
        configured_ = true;
    }

    const OOMPolicyConfig& OOMPolicy::active_config() const {
        const auto lock = std::lock_guard{mutex_};
        return config_;
    }

    bool OOMPolicy::is_configured() const {
        const auto lock = std::lock_guard{mutex_};
        return configured_;
    }

    OOMPolicyAction OOMPolicy::handle_transfer_failure(
        const Tensors::detail::TransferFailureClassification& classification,
        const std::exception& e,
        const Tensors::Tensor& tensor,
        const Tensors::Device target
    ) {
        const auto source = tensor.get_device();

        auto flags = std::ostringstream{};
        flags << "pre=" << (classification.alloc_failed_pre_change ? "1" : "0")
              << ",temp=" << (classification.copy_failed_with_temp ? "1" : "0")
              << ",h2d=" << (classification.h2d_fallback_eligible ? "1" : "0")
              << ",meaningless=" << (classification.fallback_meaningless ? "1" : "0");

        auto message = std::ostringstream{};
        message << "OOM transfer " << stringify_device_source(source)
                << " -> " << stringify_device_source(target)
                << " [" << flags.str() << "]: " << e.what();

        const auto mode = active_config().mode;

        const auto message_str = message.str();

        if (mode == OOMPolicyMode::CheckpointAndThrow) {
            if (const auto& hook = active_config().checkpoint_hook) {
                hook(e);
            }
            log(message_str);
            return OOMPolicyAction::Rethrow;
        }

        if (mode == OOMPolicyMode::H2DFallback && classification.h2d_fallback_eligible) {
            log(message_str + "; H2D fallback: leaving tensor on CPU");
            return OOMPolicyAction::Swallow;
        }

        log(message_str);
        return OOMPolicyAction::Rethrow;
    }

    OOMPolicyAction OOMPolicy::handle_compute_failure(
        const std::exception& e,
        const Tensors::Tensor& tensor
    ) {
        auto message = std::ostringstream{};
        message << "OOM compute on " << stringify_device_source(tensor.get_device())
                << ": " << e.what();
        const auto message_str = message.str();

        const auto mode = active_config().mode;

        if (mode == OOMPolicyMode::CheckpointAndThrow) {
            if (const auto& hook = active_config().checkpoint_hook) {
                hook(e);
            }
            log(message_str);
            return OOMPolicyAction::Rethrow;
        }

        if (mode == OOMPolicyMode::H2DFallback && compute_fallback_eligible_for_test()) {
            log(message_str + "; compute fallback: switching to CPU");
            return OOMPolicyAction::Swallow;
        }

        log(message_str);
        return OOMPolicyAction::Rethrow;
    }

    void OOMPolicy::log(const std::string& message) {
        const auto log_lock = std::lock_guard{log_mutex_};
        open_log_if_needed();
        if (log_stream_.is_open()) {
            write_log_line(message);
        }
    }

    void OOMPolicy::open_log_if_needed() {
        const auto& path = active_config().log_path;
        if (path.empty()) {
            return;
        }
        if (log_stream_.is_open()) {
            return;
        }
        if (log_open_failed_) {
            return;
        }
        log_stream_.open(path, std::ios::app);
        if (!log_stream_.is_open()) {
            log_open_failed_ = true;
        }
    }

    void OOMPolicy::write_log_line(const std::string& line) {
        try {
            const auto now = std::chrono::system_clock::now();
            const auto time = std::chrono::system_clock::to_time_t(now);
            // std::localtime uses a shared buffer; the log_mutex_ serializes
            // policy writers, so this is safe from races with our own logs.
            const auto* local = std::localtime(&time);
            if (local != nullptr) {
                log_stream_ << std::put_time(local, "%Y-%m-%d %H:%M:%S") << ' ' << line << '\n';
            } else {
                log_stream_ << line << '\n';
            }
            log_stream_.flush();
        } catch (...) {
            // Logging must never mask the original OOM. Swallow any formatting
            // or I/O failure silently.
        }
    }

    void OOMPolicy::reset_for_tests() {
        // Lock both mutexes in a fixed order to avoid the ABBA inversion with
        // log(), which takes log_mutex_ then mutex_ via active_config().
        const auto lock = std::scoped_lock{mutex_, log_mutex_};
        config_ = OOMPolicyConfig{};
        configured_ = false;
        if (log_stream_.is_open()) {
            log_stream_.close();
        }
        log_open_failed_ = false;
    }

    void OOMPolicy::set_compute_fallback_eligible_for_test(const bool eligible) {
        compute_fallback_eligible_test_.store(eligible);
    }

    bool OOMPolicy::compute_fallback_eligible_for_test() {
        return compute_fallback_eligible_test_.load();
    }

}
