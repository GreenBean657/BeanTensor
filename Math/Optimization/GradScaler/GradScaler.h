/**
 * @file GradScaler.h
 * @author GreenBean
 * @brief FP16 gradient scaler: training-loop arithmetic that scales the loss,
 * unscales parameter gradients in place, checks them for finite values, and
 * dynamically grows or backs off the scale factor. A skipped step never calls
 * the optimizer, so Adam's per-parameter step counter and moment buffers are
 * not touched and SGD leaves parameters unchanged.
 * @version 0.1
 * @date 2026-11-14
 */
#pragma once

#include <cstdint>

namespace BeanTensor::Modules {
    class Module;
}
#include "Tensor/Tensor.h"

namespace BeanTensor::Optimization {

    /**
     * Utility for mixed-precision training. The scaler is training-loop
     * arithmetic, not an engine precision mode: it multiplies the forward loss
     * by a scale factor, divides the resulting gradients by the same factor
     * before the optimizer sees them, and skips the optimizer step when any
     * gradient is non-finite. On a skip the scale is backed off and the
     * optimizer is not invoked, so Adam's step counter and first/second moment
     * buffers stay unchanged and SGD leaves parameters untouched.
     *
     * Typical training loop:
     * @code
     *   auto scaler = GradScaler{};
     *   for (...) {
     *       net.zero_grad();
     *       auto loss = scaler.scale(net.forward(x).softmax_ce(y));
     *       loss.backward(BeanTensor::DataTypes::DType::Float32);
     *       const auto finite = scaler.unscale_and_check(net);
     *       scaler.update(finite);
     *       if (finite) {
     *           net.adam_step(lr, beta1, beta2, eps);
     *       }
     *   }
     * @endcode
     *
     * Or use the convenience step() wrapper:
     * @code
     *   net.zero_grad();
     *   auto loss = scaler.scale(...);
     *   loss.backward(...);
     *   if (scaler.step(net, lr, beta1, beta2, eps)) { // optimizer ran }
     * @endcode
     *
     * Dynamic growth/backoff can be disabled (static mode); the scale then
     * stays at its initial value and update() does nothing.
     * @brief Loss/gradient scaler for mixed-precision training loops.
     */
    class GradScaler {
    public:
        /**
         * @brief Construct a scaler with the standard dynamic defaults.
         * @param init_scale Starting scale factor (default 2^16).
         * @param growth_factor Multiplier applied on growth_interval consecutive finite steps.
         * @param backoff_factor Multiplier applied on any non-finite step.
         * @param growth_interval Number of consecutive finite steps needed before growth.
         * @param enabled If false, scale() returns the loss unchanged and unscale_and_check() only checks finiteness.
         * @param dynamic If false, the scale never changes (static mode); update() is a no-op.
         */
        GradScaler(float64_t init_scale = float64_t{65536.0},
                   float64_t growth_factor = float64_t{2.0},
                   float64_t backoff_factor = float64_t{0.5},
                   int32_t growth_interval = 2000,
                   bool enabled = true,
                   bool dynamic = true);

        /**
         * Returns a new tensor equal to loss * scale, or the unchanged loss when
         * the scaler is disabled. The backward pass therefore produces gradients
         * that are scale times larger than the true gradients.
         * @brief Scale a scalar loss for backward.
         * @param loss A shape-{1} loss tensor.
         * @return The scaled loss tensor.
         */
        [[nodiscard]] Tensors::Tensor scale(const Tensors::Tensor& loss) const;

        /**
         * Divides every parameter gradient in the module subtree by the current
         * scale, then checks each one for finite values. Returns true only if
         * every gradient of every parameter that currently holds one is finite.
         * When the scaler is disabled no division is performed; only the finite
         * check is run.
         *
         * The finite scan is implemented with existing device ops
         * (sub/equal/min); no host fp16 arithmetic is performed. Gradients must
         * be contiguous and of a float kind.
         * @brief In-place gradient unscale + finite check.
         * @param module Module whose parameter gradients will be unscaled and checked.
         * @return True when all gradients are finite after unscaling.
         * @throws ErrorHandling::NotImplemented A gradient is non-contiguous or an integer dtype.
         */
        bool unscale_and_check(Modules::Module& module);

        /**
         * Adjusts the scale factor. When dynamic is false this is a no-op.
         * Non-finite step: scale *= backoff_factor and reset the growth tracker.
         * Finite step: increment the tracker; after growth_interval consecutive
         * finite steps, scale *= growth_factor and reset the tracker.
         * @brief Apply dynamic growth/backoff based on the last finite check.
         * @param finite Result of the preceding unscale_and_check() call.
         */
        void update(bool finite);

        /**
         * Convenience wrapper: unscale_and_check(module), update(finite), and
         * if finite call module.adam_step(lr, beta1, beta2, eps). Returns the
         * finite result so the caller knows whether the optimizer ran.
         * @brief Adam step through the scaler.
         * @param module Module to optimize.
         * @param lr Learning rate.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param eps Denominator epsilon.
         * @return True when the step was finite and Adam ran.
         */
        bool step(Modules::Module& module, float64_t lr, float64_t beta1, float64_t beta2, float64_t eps);

        /**
         * Convenience wrapper for SGD: unscale_and_check(module), update(finite),
         * and if finite call module.sgd_step(lr). Returns the finite result.
         * @brief SGD step through the scaler.
         * @param module Module to optimize.
         * @param lr Learning rate.
         * @return True when the step was finite and SGD ran.
         */
        bool step(Modules::Module& module, float64_t lr);

        /**
         * @brief Clear all parameter gradients in the module subtree.
         */
        void zero_grad(Modules::Module& module);

        /**
         * @brief Current scale factor.
         * @return The scale.
         */
        [[nodiscard]] float64_t get_scale() const;

        /**
         * @brief Current count of consecutive finite steps since the last growth or backoff.
         * @return The growth tracker.
         */
        [[nodiscard]] int32_t get_growth_tracker() const;

        /**
         * @brief Whether scaling is active.
         * @return True when scale() scales and unscale_and_check() divides.
         */
        [[nodiscard]] bool is_enabled() const;

        /**
         * @brief Whether dynamic growth/backoff is active.
         * @return True when update() can change the scale.
         */
        [[nodiscard]] bool is_dynamic() const;

        /**
         * Toggle scaling on or off. Disabling leaves the scale unchanged.
         * @brief Enable or disable scaling.
         * @param enabled New enabled state.
         */
        void set_enabled(bool enabled);

        /**
         * Toggle dynamic growth/backoff. Disabling leaves the scale unchanged.
         * @brief Enable or disable dynamic scale adjustment.
         * @param dynamic New dynamic state.
         */
        void set_dynamic(bool dynamic);

    private:
        float64_t scale_;
        float64_t growth_factor_;
        float64_t backoff_factor_;
        int32_t growth_interval_;
        int32_t growth_tracker_;
        bool enabled_;
        bool dynamic_;

        /**
         * In-place division of a single contiguous gradient tensor by the
         * current scale. Kept as high-precision compute (fp32/fp64) and narrowed
         * into fp16/bf16/packed storage, matching the elementwise division
         * contract. Float kinds only.
         * @brief Divide one gradient tensor by the scale in place.
         */
        void unscale_tensor_(Tensors::Tensor& grad);

        /**
         * True when every element of the contiguous gradient tensor is finite.
         * Uses sub/equal/min on existing device ops; no host fp16 math.
         * @brief Check one gradient tensor for finite values.
         */
        [[nodiscard]] bool check_finite_(const Tensors::Tensor& grad) const;
    };

}
