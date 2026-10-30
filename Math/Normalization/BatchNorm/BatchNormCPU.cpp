/**
 * @file BatchNormCPU.cpp
 * @author GreenBean
 * @brief CPU backend for batch normalization. Hosts the Tensor::batch_norm
 * method definition (mirrors how Tensor::mean lives in MeanCPU.cpp).
 * Train-mode forward is a per-channel two-phase kernel: phase 1 accumulates
 * per-channel sum and sum-of-squares in the working precision (thread
 * partials merged in chunk order via the launch_reduce idiom), phase 2
 * normalizes into the output and applies the pinned three-step running-stat
 * blend, chained in the same task chain as the buffer writes. Eval-mode
 * forward skips the reduction phase and the buffer update: per-channel
 * mean/invstd come from the running buffers (invstd = 1/sqrt(var + eps) in
 * the working precision) and only the output is written. All arithmetic
 * follows Math/Normalization/BatchNorm/BatchNormShared.h verbatim. Train mode
 * with any requires_grad operand wires the batch_norm_g_node backward
 * trampoline (Math/Autograd/BatchNormG/) onto the output; eval mode with
 * any requires_grad operand wires batch_norm_g_eval_node, stashing raw
 * pointers to the two running buffers in the node's saved-tensor slots.
 * @version 0.6
 * @date 2026-10-30
 */
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "Math/Autograd/BatchNormG/BatchNormG.h"
#include "Math/Normalization/BatchNorm/BatchNorm.h"
#include "Math/Normalization/BatchNorm/BatchNormShared.h"
#include "Math/Reductions/ReductionsShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Tensors::detail {
    using namespace batch_norm_shared;
    using reductions_shared::drain;
    using reductions_shared::launch_reduce;
    using reductions_shared::narrow_lane;
    using reductions_shared::widen_lane;

    namespace {
        /**
         * Identity for fp32/fp64 storage, unaccel widen for fp16/bf16 lanes.
         * @brief Widen one storage element into the working precision W.
         * @tparam W Working precision (fp32 unless the storage is fp64).
         * @tparam T Storage element type.
         */
        template<typename W, typename T>
        [[nodiscard]] W to_working(const T v) {
            if constexpr (std::is_same_v<W, T>) {
                return v;
            } else {
                return widen_lane(v);
            }
        }

        /**
         * Identity for fp32/fp64 storage, RNE narrow for fp16/bf16 lanes.
         * @brief Narrow one working-precision value into the storage dtype T.
         * @tparam T Storage element type.
         * @tparam W Working precision.
         */
        template<typename T, typename W>
        [[nodiscard]] T from_working(const W v) {
            if constexpr (std::is_same_v<W, T>) {
                return v;
            } else {
                return narrow_lane<T>(v);
            }
        }

        /**
         * Phase 1 chunks per-channel sum/sum-of-squares partials in W (merged
         * on the caller in chunk order after the drain); phase 2 normalizes
         * into out and blends the running buffers in the pinned three-step
         * order, all futures registered on every tensor the tasks touch so a
         * destroy-before-sync drains them first.
         * @brief CPU train-mode batch normalization forward for one storage type.
         * @tparam T Storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         */
        template<typename T, typename W>
        void batch_norm_train_typed(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                                    const Tensor& out, Tensor& running_mean, Tensor& running_var,
                                    const T* const x_data, const T* const g_data, const T* const b_data,
                                    T* const o_data, T* const rm_data, T* const rv_data,
                                    const size_t channels, const size_t inner, const size_t count,
                                    const float64_t momentum, const float64_t eps) {
            const auto m = static_cast<W>(per_channel_count(count, channels));
            const auto eps_w = static_cast<W>(eps);
            const auto momentum_w = static_cast<W>(momentum);

            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            auto sum_partials = std::vector<W>(t_count * channels, W{0});
            auto sq_partials = std::vector<W>(t_count * channels, W{0});
            drain(launch_reduce(count, [&](const size_t slot, const size_t begin, const size_t end) {
                for (auto i = begin; i < end; ++i) {
                    const auto c = channel_of_flat(i, inner, channels);
                    const auto w = to_working<W>(x_data[i]);
                    sum_partials[slot * channels + c] += w;
                    sq_partials[slot * channels + c] += w * w;
                }
            }, x, count * sizeof(T)));

            auto means = std::make_shared<std::vector<W>>(channels);
            auto vars = std::make_shared<std::vector<W>>(channels);
            auto invstds = std::make_shared<std::vector<W>>(channels);
            for (auto c = size_t{0}; c < channels; ++c) {
                auto sum = W{0};
                auto sumsq = W{0};
                for (auto slot = size_t{0}; slot < t_count; ++slot) {
                    sum += sum_partials[slot * channels + c];
                    sumsq += sq_partials[slot * channels + c];
                }
                const auto mean = sum / m;
                const auto var = biased_variance(mean, sumsq, m);
                (*means)[c] = mean;
                (*vars)[c] = var;
                (*invstds)[c] = inv_std(var, eps_w);
            }

            const auto futures = launch_reduce(count, [=](const size_t slot, const size_t begin, const size_t end) {
                for (auto i = begin; i < end; ++i) {
                    const auto c = channel_of_flat(i, inner, channels);
                    const auto xv = to_working<W>(x_data[i]);
                    const auto y = normalize(xv, (*means)[c], (*invstds)[c],
                                             to_working<W>(g_data[c]), to_working<W>(b_data[c]));
                    o_data[i] = from_working<T>(y);
                }
                if (slot == 0) {
                    // The pinned running-stat blend rides in the first chunk's
                    // task: chained in the same task chain as the buffer
                    // writes, so any later op that drains the buffers at entry
                    // observes the update. Its storage is disjoint from the out
                    // writes above — no race with the sibling chunks.
                    for (auto c = size_t{0}; c < channels; ++c) {
                        rm_data[c] = from_working<T>(blend_running(to_working<W>(rm_data[c]), (*means)[c], momentum_w));
                        rv_data[c] = from_working<T>(blend_running(to_working<W>(rv_data[c]), (*vars)[c], momentum_w));
                    }
                }
            }, out, count * sizeof(T));
            x.give_futures(futures);
            gamma.give_futures(futures);
            beta.give_futures(futures);
            running_mean.give_futures(futures);
            running_var.give_futures(futures);
        }

        /**
         * No reduction phase and no buffer writes: per-channel mean and
         * invstd = 1/sqrt(var + eps) are precomputed from the running buffers
         * on the HOST into shared vectors the chunk workers capture (the
         * pinned inv_std), then the same chunked normalize as the train
         * phase-2 writes the output — the workers never read the buffers'
         * storage. The futures still register on every operand, buffers
         * included: registration is the destroy-before-sync ordering, so any
         * tensor this operation named drains these chunk tasks before its
         * storage can be freed (and the chain stays uniform with the train
         * path, which does mutate the buffers).
         * @brief CPU eval-mode batch normalization forward for one storage type.
         * @tparam T Storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         */
        template<typename T, typename W>
        void batch_norm_eval_typed(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                                   const Tensor& out, const Tensor& running_mean, const Tensor& running_var,
                                   const T* const x_data, const T* const g_data, const T* const b_data,
                                   T* const o_data, const T* const rm_data, const T* const rv_data,
                                   const size_t channels, const size_t inner, const size_t count,
                                   const float64_t eps) {
            const auto eps_w = static_cast<W>(eps);

            auto means = std::make_shared<std::vector<W>>(channels);
            auto invstds = std::make_shared<std::vector<W>>(channels);
            for (auto c = size_t{0}; c < channels; ++c) {
                const auto var = to_working<W>(rv_data[c]);
                (*means)[c] = to_working<W>(rm_data[c]);
                (*invstds)[c] = inv_std(var, eps_w);
            }

            const auto futures = launch_reduce(count, [=](const size_t, const size_t begin, const size_t end) {
                for (auto i = begin; i < end; ++i) {
                    const auto c = channel_of_flat(i, inner, channels);
                    const auto y = normalize(to_working<W>(x_data[i]), (*means)[c], (*invstds)[c],
                                             to_working<W>(g_data[c]), to_working<W>(b_data[c]));
                    o_data[i] = from_working<T>(y);
                }
            }, out, count * sizeof(T));
            x.give_futures(futures);
            gamma.give_futures(futures);
            beta.give_futures(futures);
            running_mean.give_futures(futures);
            running_var.give_futures(futures);
        }
    }

    /**
     * Dispatches on dtype to the typed kernel: fp64 accumulates in
     * fp64, everything else in fp32 (fp16/bf16 lanes widen/narrow through the
     * unaccel pathway, RNE on store). Train mode runs the two-phase reduction
     * + normalize and blends the running buffers; eval mode reads per-channel
     * mean/invstd from the running buffers and leaves them untouched. Private
     * member extraction happens here — the typed helpers below are not
     * friends.
     * @brief CPU batch normalization forward.
     * @param x Contiguous rank-2/4 input, float dtype, CPU device.
     * @param gamma Rank-1 per-channel scale, x's dtype.
     * @param beta Rank-1 per-channel shift, x's dtype.
     * @param out Preallocated output, x's shape/dtype/device.
     * @param running_mean Rank-1 running mean buffer; blended in train mode, read-only in eval mode.
     * @param running_var Rank-1 running variance buffer (biased); blended in train mode, read-only in eval mode.
     * @param training Train mode uses batch statistics and updates the buffers; eval mode normalizes with the running statistics.
     * @param momentum Blend factor, narrowed once into the working precision (train mode only).
     * @param eps Epsilon inside invstd = 1/sqrt(var + eps), narrowed once into the working precision.
     */
    void batch_norm_cpu(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                        const Tensor& out, Tensor& running_mean, Tensor& running_var,
                        const bool training, const float64_t momentum, const float64_t eps) {
        if (x.device != Device::CPU || gamma.device != Device::CPU || beta.device != Device::CPU ||
            out.device != Device::CPU || running_mean.device != Device::CPU || running_var.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        const auto channels = x.shape[1];
        const auto inner = x.numel / (x.shape[0] * channels);
        const auto count = x.numel;
        const auto* const x_ptr = x.data;
        const auto* const g_ptr = gamma.data;
        const auto* const b_ptr = beta.data;
        auto* const o_ptr = out.data;
        auto* const rm_ptr = running_mean.data;
        auto* const rv_ptr = running_var.data;
        const auto x_off = x.offset;
        const auto g_off = gamma.offset;
        const auto b_off = beta.offset;
        const auto o_off = out.offset;
        const auto rm_off = running_mean.offset;
        const auto rv_off = running_var.offset;
        switch (x.dtype) {
            case DataTypes::DType::Float32:
                if (training) {
                    batch_norm_train_typed<float32_t, float32_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const float32_t*>(x_ptr) + x_off, static_cast<const float32_t*>(g_ptr) + g_off,
                        static_cast<const float32_t*>(b_ptr) + b_off, static_cast<float32_t*>(o_ptr) + o_off,
                        static_cast<float32_t*>(rm_ptr) + rm_off, static_cast<float32_t*>(rv_ptr) + rv_off,
                        channels, inner, count, momentum, eps);
                } else {
                    batch_norm_eval_typed<float32_t, float32_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const float32_t*>(x_ptr) + x_off, static_cast<const float32_t*>(g_ptr) + g_off,
                        static_cast<const float32_t*>(b_ptr) + b_off, static_cast<float32_t*>(o_ptr) + o_off,
                        static_cast<const float32_t*>(rm_ptr) + rm_off, static_cast<const float32_t*>(rv_ptr) + rv_off,
                        channels, inner, count, eps);
                }
                break;
            case DataTypes::DType::Float64:
                if (training) {
                    batch_norm_train_typed<float64_t, float64_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const float64_t*>(x_ptr) + x_off, static_cast<const float64_t*>(g_ptr) + g_off,
                        static_cast<const float64_t*>(b_ptr) + b_off, static_cast<float64_t*>(o_ptr) + o_off,
                        static_cast<float64_t*>(rm_ptr) + rm_off, static_cast<float64_t*>(rv_ptr) + rv_off,
                        channels, inner, count, momentum, eps);
                } else {
                    batch_norm_eval_typed<float64_t, float64_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const float64_t*>(x_ptr) + x_off, static_cast<const float64_t*>(g_ptr) + g_off,
                        static_cast<const float64_t*>(b_ptr) + b_off, static_cast<float64_t*>(o_ptr) + o_off,
                        static_cast<const float64_t*>(rm_ptr) + rm_off, static_cast<const float64_t*>(rv_ptr) + rv_off,
                        channels, inner, count, eps);
                }
                break;
            case DataTypes::DType::Float16:
                if (training) {
                    batch_norm_train_typed<float16_t, float32_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const float16_t*>(x_ptr) + x_off, static_cast<const float16_t*>(g_ptr) + g_off,
                        static_cast<const float16_t*>(b_ptr) + b_off, static_cast<float16_t*>(o_ptr) + o_off,
                        static_cast<float16_t*>(rm_ptr) + rm_off, static_cast<float16_t*>(rv_ptr) + rv_off,
                        channels, inner, count, momentum, eps);
                } else {
                    batch_norm_eval_typed<float16_t, float32_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const float16_t*>(x_ptr) + x_off, static_cast<const float16_t*>(g_ptr) + g_off,
                        static_cast<const float16_t*>(b_ptr) + b_off, static_cast<float16_t*>(o_ptr) + o_off,
                        static_cast<const float16_t*>(rm_ptr) + rm_off, static_cast<const float16_t*>(rv_ptr) + rv_off,
                        channels, inner, count, eps);
                }
                break;
            case DataTypes::DType::BFloat16:
                if (training) {
                    batch_norm_train_typed<bfloat16_t, float32_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const bfloat16_t*>(x_ptr) + x_off, static_cast<const bfloat16_t*>(g_ptr) + g_off,
                        static_cast<const bfloat16_t*>(b_ptr) + b_off, static_cast<bfloat16_t*>(o_ptr) + o_off,
                        static_cast<bfloat16_t*>(rm_ptr) + rm_off, static_cast<bfloat16_t*>(rv_ptr) + rv_off,
                        channels, inner, count, momentum, eps);
                } else {
                    batch_norm_eval_typed<bfloat16_t, float32_t>(x, gamma, beta, out, running_mean, running_var,
                        static_cast<const bfloat16_t*>(x_ptr) + x_off, static_cast<const bfloat16_t*>(g_ptr) + g_off,
                        static_cast<const bfloat16_t*>(b_ptr) + b_off, static_cast<bfloat16_t*>(o_ptr) + o_off,
                        static_cast<const bfloat16_t*>(rm_ptr) + rm_off, static_cast<const bfloat16_t*>(rv_ptr) + rv_off,
                        channels, inner, count, eps);
                }
                break;
            [[unlikely]] default:
                throw ErrorHandling::NotImplemented();
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Syncs the input, validates rank/devices/emptiness/contiguity across all
     * operands, the channel counts, the dtypes and the eps/momentum ranges,
     * then syncs gamma/beta and the running buffers before dispatching to the
     * CPU or GPU backend. Train-mode buffer mutation chains onto the buffers'
     * own task queues, so a following op that drains them at entry — this
     * method's own entry sync included — observes the update. Train mode with
     * any requires_grad operand wires the train backward node (parents
     * x/gamma/beta, eps in Node::_scalar) onto the output; the running
     * buffers are never graph inputs of the train node: its backward
     * recomputes the channel statistics from x. Eval mode with any
     * requires_grad operand wires the eval backward node (same parents and
     * eps) and records the two running buffers in the node's saved-tensor
     * slots — implicit lifetime parents of any backward through that output,
     * which must stay alive until backward() returns, exactly like the input
     * parents (direct callers passing local running buffers must keep those
     * locals alive until backward()).
     * @brief Batch normalization forward, same shape/dtype/device as the input.
     * @param gamma Per-channel scale, rank-1 shape {C}, same dtype and device.
     * @param beta Per-channel shift, rank-1 shape {C}, same dtype and device.
     * @param running_mean Running mean buffer, rank-1 {C}; mutated in train mode, read in eval mode.
     * @param running_var Running variance buffer (BIASED estimator), rank-1 {C}; mutated in train mode, read in eval mode.
     * @param training Train mode normalizes with batch stats and updates the buffers; eval mode reads the buffers.
     * @param momentum Running-stat blend factor: new = old * (1 - m) + batch * m; must satisfy 0 < m <= 1.
     * @param eps Epsilon inside invstd = 1/sqrt(var + eps); must be > 0.
     * @return The normalized tensor.
     * @throws ErrorHandling::MismatchedShapes Rank is not 2 or 4, or a channel count disagrees.
     * @throws ErrorHandling::MismatchedDevices An operand lives on a different device than the input.
     * @throws ErrorHandling::MismatchedDTypes gamma/beta/running buffers dtype differs from the input.
     * @throws ErrorHandling::IndexOutOfRange The input is empty.
     * @throws ErrorHandling::NotImplemented Non-float dtype or non-contiguous input or operand.
     * @throws ErrorHandling::InvalidHyperparameter eps is not > 0 or momentum is not in (0, 1] — the positive-form checks reject NaN along with the out-of-range values.
     */
    Tensor Tensor::batch_norm(const Tensor& gamma, const Tensor& beta,
                              Tensor& running_mean, Tensor& running_var,
                              const bool training, const float64_t momentum, const float64_t eps) const {
        sync();
        if (this->ndim != 2 && this->ndim != 4) {
            throw ErrorHandling::MismatchedShapes(
                "batch_norm input rank must be 2 or 4, got " + std::to_string(this->ndim));
        }
        if (gamma.device != this->device || beta.device != this->device ||
            running_mean.device != this->device || running_var.device != this->device) {
            throw ErrorHandling::MismatchedDevices();
        }
        const auto channels = this->shape[1];
        if (this->numel == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (!this->is_contiguous || !gamma.is_contiguous || !beta.is_contiguous ||
            !running_mean.is_contiguous || !running_var.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        const auto check_channels = [&channels](const Tensor& t, const char* const name) {
            if (t.ndim != 1 || t.shape[0] != channels) {
                throw ErrorHandling::MismatchedShapes(
                    std::string("batch_norm ") + name + " must be rank-1 with " + std::to_string(channels) + " channels");
            }
        };
        check_channels(gamma, "gamma");
        check_channels(beta, "beta");
        check_channels(running_mean, "running_mean");
        check_channels(running_var, "running_var");
        if (!DataTypes::detail::is_float_dtype(this->dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        if (gamma.dtype != this->dtype || beta.dtype != this->dtype ||
            running_mean.dtype != this->dtype || running_var.dtype != this->dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (!(eps > float64_t{0})) {
            throw ErrorHandling::InvalidHyperparameter("batch_norm requires eps > 0");
        }
        if (!(momentum > float64_t{0} && momentum <= float64_t{1})) {
            throw ErrorHandling::InvalidHyperparameter("batch_norm requires 0 < momentum <= 1");
        }
        gamma.sync();
        beta.sync();
        running_mean.sync();
        running_var.sync();
        auto out = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::batch_norm_cpu(*this, gamma, beta, out, running_mean, running_var, training, momentum, eps);
        } else if (this->device == Device::GPU) {
            detail::batch_norm_gpu(*this, gamma, beta, out, running_mean, running_var, training, momentum, eps);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (training) {
            out.wire_node(detail::batch_norm_g_node, *this, gamma, beta);
            if (out.node != nullptr) {
                out.node->_scalar = eps;
            }
        } else {
            out.wire_node(detail::batch_norm_g_eval_node, *this, gamma, beta);
            if (out.node != nullptr) {
                out.node->_scalar = eps;
                // Running stats are graph constants of the eval backward,
                // delivered through the saved-tensor slots — implicit
                // lifetime parents that must outlive backward() (see the
                // eval node's doc comment). Set after wire_node, which
                // value-initializes the node.
                out.node->_saved[0] = &running_mean;
                out.node->_saved[1] = &running_var;
                out.node->_n_saved = 2;
            }
        }
        return out;
    }
}
