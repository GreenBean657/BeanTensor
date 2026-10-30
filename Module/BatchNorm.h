/**
 * @file BatchNorm.h
 * @author GreenBean
 * @brief Batch normalization module: per-channel normalization with gamma and
 * beta as its only registered parameters and address-stable running-statistic
 * buffers, forwarding through Tensor::batch_norm — train mode normalizes with
 * batch statistics and blends them into the buffers, eval mode normalizes
 * with the running statistics and leaves the buffers untouched.
 * @version 0.2
 * @date 2026-10-30
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>

#include "Module/Module.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Modules {

    /**
     * Batch normalization over rank-2 or rank-4 inputs, normalizing every
     * channel by its own statistics. Train forwards compute per-channel
     * statistics from the batch and blend them into the running buffers
     * (running_mean init 0, running_var init 1, biased estimator, momentum
     * blend); eval forwards normalize with the running buffers and leave
     * them untouched. Owns gamma (ones) and beta (zeros) as its only
     * registered parameters — the running statistics are buffers, NOT
     * parameters: they are excluded from named_parameters(),
     * parameter_count() and every optimizer view, and they never receive
     * gradients.
     *
     * Lifetime: the running statistics live in address-stable heap slots
     * (a deque of unique_ptr) because eval-mode graph nodes hold raw
     * Node::_saved pointers into them; module moves keep those addresses
     * valid. Like every module's parameters, the module itself — running
     * stat slots included — must outlive backward() of any forward it
     * produced; destroying the module before backward() is out of contract
     * (see the Module base class doc).
     * @brief Batch normalization module (train/eval modes, running statistics).
     */
    class BatchNorm final : public Module {
    public:
        /**
         * Validates the hyperparameters, registers gamma (ones) and beta
         * (zeros) as the module's only parameters, and allocates the two
         * running-statistic slots (mean zeros, var ones, requires_grad false)
         * on the construction device in the construction dtype.
         * @brief Construct a BatchNorm module.
         * @param num_features Channel count C (x's dim 1); must be > 0.
         * @param dtype Parameter and buffer dtype (Float32 by default).
         * @param device Device holding the parameters and buffers (CPU by default).
         * @param momentum Running-stat blend factor: new = old * (1 - m) + batch * m.
         * @param eps Epsilon inside invstd = 1/sqrt(var + eps).
         * @throws ErrorHandling::IndexOutOfRange num_features == 0.
         * @throws ErrorHandling::InvalidHyperparameter momentum not in (0, 1] or eps not > 0 — NaN fails these positive-form checks too.
         */
        BatchNorm(size_t num_features,
                  DataTypes::DType dtype = DataTypes::DType::Float32,
                  Tensors::Device device = Tensors::Device::CPU,
                  float64_t momentum = 0.1, float64_t eps = 1e-5);

        /**
         * Delegates to Tensor::batch_norm with the module's gamma/beta and
         * running-stat slots, in train or eval mode per the training flag.
         * The running-stat slots are passed by reference: eval-mode outputs
         * hold raw pointers into them (Node::_saved) that must stay valid
         * until backward() returns.
         * @brief forward: batch normalization in train or eval mode.
         * @param x Input tensor, rank 2 {N, C} or rank 4 {N, C, H, W}.
         * @return Normalized tensor, same shape/dtype/device as x.
         */
        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& x) override;

        /**
         * @brief Human-readable layer type.
         * @return "BatchNorm".
         */
        [[nodiscard]] std::string name() const override;

        /**
         * Read-only view of the running statistics (mean, var). These are
         * buffers, not parameters: they appear in no optimizer-facing view
         * and their addresses are stable for the module's lifetime (eval
         * backward nodes hold raw pointers into them).
         * @brief The running-mean / running-variance buffers.
         * @return (running_mean, running_var) as const refs.
         */
        [[nodiscard]] std::pair<const Tensors::Tensor&, const Tensors::Tensor&> running_stats() const;

    private:
        size_t num_features_;
        float64_t momentum_;
        float64_t eps_;
        // Address-stable heap slots: eval graph nodes hold raw pointers to these
        // (Node::_saved); they must survive module moves and outlive backward().
        std::deque<std::unique_ptr<Tensors::Tensor>> running_stats_;
    };
}
