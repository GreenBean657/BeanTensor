/**
 * @file OOMPolicy.h
 * @author GreenBean
 * @brief Process-wide out-of-memory policy: preserve/log/throw, H2D->CPU
 *        fallback, and checkpoint-and-throw via an application hook.
 * @version 0.1
 * @date 2026-11-14
 */
#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>

namespace BeanTensor::Tensors {
    class Tensor;
    enum class Device;
    namespace detail {
        struct TransferFailureClassification;
    }
}

namespace BeanTensor::Helpers {

    /**
     * Exclusive OOM handling modes. The policy is configured once at process
     * startup and is consulted by transfer and compute paths on allocation or
     * copy failure.
     * @brief Process-wide OOM policy mode.
     */
    enum class OOMPolicyMode : uint8_t {
        PreserveLogThrow,   /**< Log diagnostics and rethrow the original error (default). */
        H2DFallback,        /**< CPU->GPU failures may fall back to CPU execution. */
        CheckpointAndThrow, /**< Call the checkpoint hook, log, and rethrow. */
    };

    /**
     * Action returned to the failure site after the policy has been consulted.
     * @brief What the failure site should do with the original exception.
     */
    enum class OOMPolicyAction : uint8_t {
        Rethrow, /**< The caller must rethrow the original exception. */
        Swallow, /**< The caller may swallow the exception and continue (fallback). */
    };

    /**
     * Configuration for the process-wide OOM policy. Passed to
     * OOMPolicy::configure() once at startup; validation is performed there so
     * misconfiguration fails loud and early.
     * @brief OOM policy configuration.
     */
    struct OOMPolicyConfig {
        OOMPolicyMode mode = OOMPolicyMode::PreserveLogThrow;
        std::string log_path;
        std::function<void(const std::exception&)> checkpoint_hook;
        std::string checkpoint_destination;
    };

    /**
     * Process-wide OOM policy. A singleton consulted by the failure-atomic
     * transfer path and by compute-dispatcher boundaries. Logging is mutex-
     * serialized and append-only; a logging failure is swallowed so it can
     * never mask the original OOM.
     * @brief Process-wide OOM policy singleton.
     */
    class OOMPolicy {
    public:
        /**
         * @brief Access the process-wide OOM policy singleton.
         * @return Reference to the singleton.
         */
        static OOMPolicy& instance();

        /**
         * Configure the policy. May only be called once; repeated calls throw
         * InvalidOOMPolicy. Validates that checkpoint mode supplies both a hook
         * and a destination.
         * @brief Configure the process-wide OOM policy.
         * @param config Policy configuration.
         * @throws ErrorHandling::InvalidOOMPolicy The configuration is invalid.
         */
        void configure(const OOMPolicyConfig& config);

        /**
         * @brief Current configuration.
         * @return The active configuration.
         */
        [[nodiscard]] const OOMPolicyConfig& active_config() const;

        /**
         * @brief Whether configure() has been called.
         * @return True after a successful configure().
         */
        [[nodiscard]] bool is_configured() const;

        /**
         * Consult the policy after a transfer failure. Logs a diagnostic line
         * (unless logging is disabled or has failed). For H2DFallback mode,
         * returns Swallow when the failure was H2D-fallback-eligible. For
         * CheckpointAndThrow mode, invokes the checkpoint hook and returns
         * Rethrow. All other cases return Rethrow.
         * @brief Handle a device-transfer failure.
         * @param classification Failure classification from M1's hook.
         * @param e The exception that was thrown.
         * @param tensor The tensor being transferred.
         * @return Action the failure site should take.
         */
        [[nodiscard]] OOMPolicyAction handle_transfer_failure(
            const Tensors::detail::TransferFailureClassification& classification,
            const std::exception& e,
            const Tensors::Tensor& tensor,
            Tensors::Device target
        );

        /**
         * Consult the policy after a compute-dispatcher failure. Logs a
         * diagnostic line. For H2DFallback mode, returns Swallow when the
         * current compute boundary is marked fallback-eligible (real dispatchers
         * set this; the test seam forces it). For CheckpointAndThrow mode,
         * invokes the checkpoint hook and returns Rethrow. All other cases
         * return Rethrow.
         * @brief Handle a compute-dispatcher failure.
         * @param e The exception that was thrown.
         * @param tensor The tensor whose compute failed.
         * @return Action the failure site should take.
         */
        [[nodiscard]] OOMPolicyAction handle_compute_failure(
            const std::exception& e,
            const Tensors::Tensor& tensor
        );

        /**
         * Append a line to the diagnostic log. The call is mutex-serialized.
         * A logging failure is swallowed and never rethrown.
         * @brief Append a line to the diagnostic log.
         * @param message Message to log.
         */
        void log(const std::string& message);

        /**
         * Test-only reset: clear configuration and close the log. Allows tests
         * to reconfigure the singleton in isolation.
         * @brief Reset the policy for the next test.
         */
        void reset_for_tests();

        /**
         * Test-only seam: force the compute-dispatcher boundary to report
         * fallback eligibility. Real dispatchers compute this from op/device
         * metadata; tests use this to exercise the H2DFallback compute path.
         * @brief Set compute fallback eligibility for tests.
         * @param eligible Whether compute failures should be considered fallback-eligible.
         */
        static void set_compute_fallback_eligible_for_test(bool eligible);

        /**
         * @brief Current value of the compute fallback test seam.
         * @return True when the seam is armed.
         */
        [[nodiscard]] static bool compute_fallback_eligible_for_test();

    private:
        OOMPolicy() = default;

        void open_log_if_needed();
        void write_log_line(const std::string& line);

        mutable std::mutex mutex_;
        OOMPolicyConfig config_;
        bool configured_ = false;
        std::ofstream log_stream_;
        bool log_open_failed_ = false;
        mutable std::mutex log_mutex_;

        static std::atomic<bool> compute_fallback_eligible_test_;
    };

}
