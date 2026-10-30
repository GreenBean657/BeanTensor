/**
 * @file BatchNormGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for the batch_norm backward trampolines.
 * batch_norm_g_node (train) pass 1 recomputes per-channel mean/invstd from x
 * with the forward's accumulation rules (per-slot partials merged in chunk
 * order via the launch_reduce idiom); pass 2 computes per-channel sum(dy) and
 * sum(dy * x_hat); then dgamma = sum(dy * x_hat), dbeta = sum(dy), and the
 * pinned dx element formula (BatchNormGShared.h) fill deltas allocated
 * directly in grad_dtype, so every delta leaves the node in the assigned
 * dtype by construction and needs no further boundary cast (GradDType.h).
 * batch_norm_g_eval_node (eval) treats the running statistics as graph
 * constants read live through Node::_saved (the project's first use of that
 * field): per-channel mean/invstd come from the saved buffers, one reduction
 * pass produces sum(dy) and sum(dy * x_hat) with the running-stat x_hat, and
 * the pinned eval dx element (gamma * invstd) * dy has no train correction
 * terms. Both trampolines dispatch on the input device: CPU tensors run the
 * typed kernels below, GPU tensors go to the CUDA/HIP launchers
 * (batch_norm_g_train_gpu / batch_norm_g_eval_gpu) through a POD argument
 * pack of offset-applied device pointers (BatchNormGArgs). Both trampolines
 * sync the tensors they read; every launch is drained on the caller thread
 * (a node fn runs under backward(), never inside a pool worker, so waiting
 * there is legal).
 * @version 0.3
 * @date 2026-10-30
 */
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

#include "Math/Autograd/BatchNormG/BatchNormG.h"
#include "Math/Autograd/BatchNormG/BatchNormGShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Normalization/BatchNorm/BatchNormShared.h"
#include "Math/Reductions/ReductionsShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Tensors::detail {
    using namespace batch_norm_shared;
    using batch_norm_g_shared::dx_element;
    using batch_norm_g_shared::eval_dx_element;
    using reductions_shared::drain;
    using reductions_shared::launch_reduce;
    using reductions_shared::narrow_lane;
    using reductions_shared::widen_lane;

    namespace {
        /**
         * Identity when the storage matches the working precision, unaccel
         * widen for fp16/bf16 lanes, explicit widening cast for fp32 -> fp64.
         * @brief Widen one storage element into the working precision W.
         * @tparam W Working precision (fp32 unless the input storage is fp64).
         * @tparam T Storage element type.
         */
        template<typename W, typename T>
        [[nodiscard]] W to_working(const T v) {
            if constexpr (std::is_same_v<W, T>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return static_cast<W>(widen_lane(v));
            } else {
                return static_cast<W>(v);
            }
        }

        /**
         * Identity when the working precision matches the storage, RNE narrow
         * for fp16/bf16 lanes, explicit narrowing cast for fp64 -> fp32 (RNE,
         * the same result the PERMISSIVE grad-conversion rule produces).
         * @brief Narrow one working-precision value into the storage dtype T.
         * @tparam T Storage element type.
         * @tparam W Working precision.
         */
        template<typename T, typename W>
        [[nodiscard]] T from_working(const W v) {
            if constexpr (std::is_same_v<W, T>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return narrow_lane<T>(static_cast<float32_t>(v));
            } else {
                return static_cast<T>(v);
            }
        }

        /**
         * Two reduction passes plus the dx element pass. All chunk workers
         * only touch their own partial slots / output elements and never wait
         * on futures; every launch is drained on the caller thread (this runs
         * in a node fn under backward(), never inside a pool worker, so
         * waiting there is legal). Delta buffers not requested (nullptr) are
         * skipped; dgamma/dbeta are tiny and merged on the caller thread.
         * @brief CPU train-mode batch_norm backward for one storage/grad type pair.
         * @tparam T Input storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         * @tparam G Grad storage element type (grad_dtype).
         */
        template<typename T, typename W, typename G>
        void batch_norm_g_train_typed(const Tensor& x, const Tensor& dy_t, const Tensor& gamma,
                                      Tensor* const dx_t,
                                      const T* const x_data, const G* const dy_data, const T* const g_data,
                                      G* const dx_data, G* const dg_data, G* const db_data,
                                      const size_t channels, const size_t inner, const size_t count,
                                      const float64_t eps) {
            const auto m = static_cast<W>(per_channel_count(count, channels));
            const auto eps_w = static_cast<W>(eps);
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);

            // Pass 1: recompute per-channel mean/invstd from x, same
            // accumulation rules as the forward phase 1.
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
            auto means = std::vector<W>(channels);
            auto invstds = std::vector<W>(channels);
            for (auto c = size_t{0}; c < channels; ++c) {
                auto sum = W{0};
                auto sumsq = W{0};
                for (auto slot = size_t{0}; slot < t_count; ++slot) {
                    sum += sum_partials[slot * channels + c];
                    sumsq += sq_partials[slot * channels + c];
                }
                const auto mean = sum / m;
                means[c] = mean;
                invstds[c] = inv_std(biased_variance(mean, sumsq, m), eps_w);
            }

            // Pass 2: per-channel sum(dy) and sum(dy * x_hat); dy arrives in
            // grad_dtype and widens into the working precision.
            auto dy_partials = std::vector<W>(t_count * channels, W{0});
            auto dyx_partials = std::vector<W>(t_count * channels, W{0});
            const auto sum_futures = launch_reduce(count, [&](const size_t slot, const size_t begin, const size_t end) {
                for (auto i = begin; i < end; ++i) {
                    const auto c = channel_of_flat(i, inner, channels);
                    const auto dy = to_working<W>(dy_data[i]);
                    const auto x_hat = (to_working<W>(x_data[i]) - means[c]) * invstds[c];
                    dy_partials[slot * channels + c] += dy;
                    dyx_partials[slot * channels + c] += dy * x_hat;
                }
            }, x, count * sizeof(T));
            dy_t.give_futures(sum_futures);
            drain(sum_futures);
            auto sum_dy = std::vector<W>(channels);
            auto sum_dyx = std::vector<W>(channels);
            for (auto c = size_t{0}; c < channels; ++c) {
                auto s_dy = W{0};
                auto s_dyx = W{0};
                for (auto slot = size_t{0}; slot < t_count; ++slot) {
                    s_dy += dy_partials[slot * channels + c];
                    s_dyx += dyx_partials[slot * channels + c];
                }
                sum_dy[c] = s_dy;
                sum_dyx[c] = s_dyx;
            }

            // dgamma = sum(dy * x_hat), dbeta = sum(dy): the deltas carry
            // grad_dtype by construction, narrowed from W once on store.
            if (dg_data != nullptr) {
                for (auto c = size_t{0}; c < channels; ++c) {
                    dg_data[c] = from_working<G>(sum_dyx[c]);
                }
            }
            if (db_data != nullptr) {
                for (auto c = size_t{0}; c < channels; ++c) {
                    db_data[c] = from_working<G>(sum_dy[c]);
                }
            }

            // dx = (gamma*invstd/M) * (M*dy - sum_dy - x_hat*sum_dyx), the
            // pinned element formula.
            if (dx_data != nullptr) {
                const auto dx_futures = launch_reduce(count, [&](const size_t, const size_t begin, const size_t end) {
                    for (auto i = begin; i < end; ++i) {
                        const auto c = channel_of_flat(i, inner, channels);
                        const auto dy = to_working<W>(dy_data[i]);
                        const auto x_hat = (to_working<W>(x_data[i]) - means[c]) * invstds[c];
                        dx_data[i] = from_working<G>(dx_element(to_working<W>(g_data[c]), invstds[c], m,
                                                                dy, sum_dy[c], x_hat, sum_dyx[c]));
                    }
                }, *dx_t, count * sizeof(T));
                x.give_futures(dx_futures);
                dy_t.give_futures(dx_futures);
                gamma.give_futures(dx_futures);
                drain(dx_futures);
            }
        }

        /**
         * Per-channel mean/invstd are read from the saved running buffers on
         * the caller thread (graph constants — no recompute pass); then one
         * reduction pass for per-channel sum(dy) and sum(dy * x_hat) with the
         * running-stat x_hat, and the dx element pass. All chunk workers only
         * touch their own partial slots / output elements and never wait on
         * futures; every launch is drained on the caller thread (this runs in
         * a node fn under backward(), never inside a pool worker, so waiting
         * there is legal). Delta buffers not requested (nullptr) are skipped;
         * dgamma/dbeta are tiny and merged on the caller thread.
         * @brief CPU eval-mode batch_norm backward for one storage/grad type pair.
         * @tparam T Input storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         * @tparam G Grad storage element type (grad_dtype).
         */
        template<typename T, typename W, typename G>
        void batch_norm_g_eval_typed(const Tensor& x, const Tensor& dy_t, const Tensor& gamma,
                                     Tensor* const dx_t,
                                     const T* const x_data, const G* const dy_data, const T* const g_data,
                                     const T* const rm_data, const T* const rv_data,
                                     G* const dx_data, G* const dg_data, G* const db_data,
                                     const size_t channels, const size_t inner, const size_t count,
                                     const float64_t eps) {
            const auto eps_w = static_cast<W>(eps);
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);

            // Graph constants: per-channel mean/invstd from the saved running
            // buffers, read once on the caller thread before any task launch.
            auto means = std::vector<W>(channels);
            auto invstds = std::vector<W>(channels);
            for (auto c = size_t{0}; c < channels; ++c) {
                means[c] = to_working<W>(rm_data[c]);
                invstds[c] = inv_std(to_working<W>(rv_data[c]), eps_w);
            }

            // Pass: per-channel sum(dy) and sum(dy * x_hat) with the
            // running-stat x_hat; dy arrives in grad_dtype and widens into W.
            auto dy_partials = std::vector<W>(t_count * channels, W{0});
            auto dyx_partials = std::vector<W>(t_count * channels, W{0});
            const auto sum_futures = launch_reduce(count, [&](const size_t slot, const size_t begin, const size_t end) {
                for (auto i = begin; i < end; ++i) {
                    const auto c = channel_of_flat(i, inner, channels);
                    const auto dy = to_working<W>(dy_data[i]);
                    const auto x_hat = (to_working<W>(x_data[i]) - means[c]) * invstds[c];
                    dy_partials[slot * channels + c] += dy;
                    dyx_partials[slot * channels + c] += dy * x_hat;
                }
            }, x, count * sizeof(T));
            x.give_futures(sum_futures);
            dy_t.give_futures(sum_futures);
            drain(sum_futures);
            auto sum_dy = std::vector<W>(channels);
            auto sum_dyx = std::vector<W>(channels);
            for (auto c = size_t{0}; c < channels; ++c) {
                auto s_dy = W{0};
                auto s_dyx = W{0};
                for (auto slot = size_t{0}; slot < t_count; ++slot) {
                    s_dy += dy_partials[slot * channels + c];
                    s_dyx += dyx_partials[slot * channels + c];
                }
                sum_dy[c] = s_dy;
                sum_dyx[c] = s_dyx;
            }

            // dgamma = sum(dy * x_hat), dbeta = sum(dy): the deltas carry
            // grad_dtype by construction, narrowed from W once on store.
            if (dg_data != nullptr) {
                for (auto c = size_t{0}; c < channels; ++c) {
                    dg_data[c] = from_working<G>(sum_dyx[c]);
                }
            }
            if (db_data != nullptr) {
                for (auto c = size_t{0}; c < channels; ++c) {
                    db_data[c] = from_working<G>(sum_dy[c]);
                }
            }

            // dx = (gamma * invstd) * dy, the pinned eval element formula —
            // running stats are constants, so the train correction terms
            // (sum_dy, sum_dyx) do not appear.
            if (dx_data != nullptr) {
                const auto dx_futures = launch_reduce(count, [&](const size_t, const size_t begin, const size_t end) {
                    for (auto i = begin; i < end; ++i) {
                        const auto c = channel_of_flat(i, inner, channels);
                        dx_data[i] = from_working<G>(eval_dx_element(to_working<W>(g_data[c]), invstds[c],
                                                                     to_working<W>(dy_data[i])));
                    }
                }, *dx_t, count * sizeof(T));
                dy_t.give_futures(dx_futures);
                gamma.give_futures(dx_futures);
                drain(dx_futures);
            }
        }
    }

    /**
     * inputs are (x, gamma, beta); Node::_scalar carries eps. Recomputes the
     * per-channel statistics from x and delivers dx/dgamma/dbeta for every
     * parent that requires grad. output_grad already carries grad_dtype
     * (backward()'s contract) and the delta buffers are allocated in
     * grad_dtype, so each delta leaves the node in the assigned dtype by
     * construction; accumulation goes through accumulate_grad, which syncs
     * the target first and converts into a pinned leaf buffer's dtype when a
     * later call mismatches. Syncs the tensors it reads. grad_dtype outside
     * the float set is refused: the forward only supports float inputs.
     * @brief Backward trampoline for train-mode batch_norm.
     * @param node Owning node (_scalar = eps).
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op: x, gamma, beta.
     * @param n_inputs Number of entries in inputs; must be 3.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 3.
     * @throws ErrorHandling::GPUTaskOnCPUBuild for a GPU input on a CPU-only build.
     * @throws ErrorHandling::NotImplemented for a non-float grad dtype.
     */
    void batch_norm_g_node(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 3) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        auto& gamma = *inputs[1];
        auto& beta = *inputs[2];
        if (!x.requires_grad && !gamma.requires_grad && !beta.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        gamma.sync();
        beta.sync();
        const auto channels = x.shape[1];
        const auto inner = x.numel / (x.shape[0] * channels);
        const auto count = x.numel;
        const auto eps = node->_scalar;

        // Deltas are allocated in grad_dtype only for parents that require grad.
        auto dx = std::optional<Tensor>{};
        auto dgamma = std::optional<Tensor>{};
        auto dbeta = std::optional<Tensor>{};
        if (x.requires_grad) {
            dx.emplace(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        }
        if (gamma.requires_grad) {
            dgamma.emplace(std::vector(gamma.shape, gamma.shape + gamma.ndim), grad_dtype, gamma.device, false);
        }
        if (beta.requires_grad) {
            dbeta.emplace(std::vector(beta.shape, beta.shape + beta.ndim), grad_dtype, beta.device, false);
        }

        if (x.device == Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
            // The GPU launcher takes offset-applied raw device pointers
            // (BatchNormGArgs) and stream-syncs before returning, so the
            // deltas are complete when accumulate_grad runs below.
            const auto in_eb = DataTypes::dtype_bytes(x.dtype);
            const auto g_eb = DataTypes::dtype_bytes(grad_dtype);
            const auto args = BatchNormGArgs{
                x.dtype, grad_dtype,
                static_cast<const uint8_t*>(x.data) + x.offset * in_eb,
                static_cast<const uint8_t*>(output_grad->data) + output_grad->offset * g_eb,
                static_cast<const uint8_t*>(gamma.data) + gamma.offset * in_eb,
                nullptr, nullptr,
                dx ? static_cast<uint8_t*>(dx->data) + dx->offset * g_eb : nullptr,
                dgamma ? static_cast<uint8_t*>(dgamma->data) + dgamma->offset * g_eb : nullptr,
                dbeta ? static_cast<uint8_t*>(dbeta->data) + dbeta->offset * g_eb : nullptr,
                channels, inner, count, eps};
            batch_norm_g_train_gpu(args);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else {
            auto run = [&]<typename T, typename W>() {
                const auto* const x_data = static_cast<const T*>(x.data) + x.offset;
                const auto* const g_data = static_cast<const T*>(gamma.data) + gamma.offset;
                auto run_g = [&]<typename G>() {
                    const auto* const dy_data = static_cast<const G*>(output_grad->data) + output_grad->offset;
                    auto* const dx_data = dx ? static_cast<G*>(dx->data) + dx->offset : nullptr;
                    auto* const dg_data = dgamma ? static_cast<G*>(dgamma->data) + dgamma->offset : nullptr;
                    auto* const db_data = dbeta ? static_cast<G*>(dbeta->data) + dbeta->offset : nullptr;
                    batch_norm_g_train_typed<T, W, G>(x, *output_grad, gamma, dx ? &*dx : nullptr,
                                                      x_data, dy_data, g_data, dx_data, dg_data, db_data,
                                                      channels, inner, count, eps);
                };
                switch (grad_dtype) {
                    case DataTypes::DType::Float32: run_g.template operator()<float32_t>(); break;
                    case DataTypes::DType::Float64: run_g.template operator()<float64_t>(); break;
                    case DataTypes::DType::Float16: run_g.template operator()<float16_t>(); break;
                    case DataTypes::DType::BFloat16: run_g.template operator()<bfloat16_t>(); break;
                    [[unlikely]] default: throw ErrorHandling::NotImplemented();
                }
            };
            switch (x.dtype) {
                case DataTypes::DType::Float32: run.template operator()<float32_t, float32_t>(); break;
                case DataTypes::DType::Float64: run.template operator()<float64_t, float64_t>(); break;
                case DataTypes::DType::Float16: run.template operator()<float16_t, float32_t>(); break;
                case DataTypes::DType::BFloat16: run.template operator()<bfloat16_t, float32_t>(); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }

        if (dgamma) {
            if (gamma.node != nullptr) {
                grad_probe_hit(dgamma->dtype); // test hook: an intermediate received a delta
            }
            gamma.accumulate_grad(std::move(*dgamma));
        }
        if (dbeta) {
            if (beta.node != nullptr) {
                grad_probe_hit(dbeta->dtype);
            }
            beta.accumulate_grad(std::move(*dbeta));
        }
        if (dx) {
            if (x.node != nullptr) {
                grad_probe_hit(dx->dtype);
            }
            x.accumulate_grad(std::move(*dx));
        }
    }

    /**
     * inputs are (x, gamma, beta); Node::_scalar carries eps and
     * Node::_saved[0]/_saved[1] point at the forward's running_mean and
     * running_var buffers. Running stats are graph constants AND implicit
     * parents via _saved — the tensors _saved[0]/_saved[1] point to must
     * outlive backward(). They are read live at backward() time: per-channel
     * invstd = 1/sqrt(running_var + eps) and x_hat = (x - running_mean) *
     * invstd give dgamma = sum(dy * x_hat) and dbeta = sum(dy); dx = (gamma *
     * invstd) * dy with no train correction terms, because running stats are
     * constants. Deltas are allocated in grad_dtype only for parents that
     * require grad, so each delta leaves the node in the assigned dtype by
     * construction; accumulation goes through accumulate_grad, which syncs
     * the target first and converts into a pinned leaf buffer's dtype when a
     * later call mismatches. Syncs the tensors it reads (the saved buffers
     * included). grad_dtype outside the float set is refused: the forward
     * only supports float inputs.
     * @brief Backward trampoline for eval-mode batch_norm.
     * @param node Owning node (_scalar = eps, _saved = running buffers).
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op: x, gamma, beta.
     * @param n_inputs Number of entries in inputs; must be 3.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 3 or the node carries no saved buffers.
     * @throws ErrorHandling::GPUTaskOnCPUBuild for a GPU input on a CPU-only build.
     * @throws ErrorHandling::NotImplemented for a non-float grad dtype.
     */
    void batch_norm_g_eval_node(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 3) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (node->_n_saved != uint8_t{2}) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        auto& gamma = *inputs[1];
        auto& beta = *inputs[2];
        if (!x.requires_grad && !gamma.requires_grad && !beta.requires_grad) {
            return;
        }
        auto& running_mean = *node->_saved[0];
        auto& running_var = *node->_saved[1];
        output_grad->sync();
        x.sync();
        gamma.sync();
        running_mean.sync();
        running_var.sync();
        const auto channels = x.shape[1];
        const auto inner = x.numel / (x.shape[0] * channels);
        const auto count = x.numel;
        const auto eps = node->_scalar;

        // Deltas are allocated in grad_dtype only for parents that require grad.
        auto dx = std::optional<Tensor>{};
        auto dgamma = std::optional<Tensor>{};
        auto dbeta = std::optional<Tensor>{};
        if (x.requires_grad) {
            dx.emplace(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        }
        if (gamma.requires_grad) {
            dgamma.emplace(std::vector(gamma.shape, gamma.shape + gamma.ndim), grad_dtype, gamma.device, false);
        }
        if (beta.requires_grad) {
            dbeta.emplace(std::vector(beta.shape, beta.shape + beta.ndim), grad_dtype, beta.device, false);
        }

        if (x.device == Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
            // The GPU launcher takes offset-applied raw device pointers
            // (BatchNormGArgs), the saved running buffers included, and
            // stream-syncs before returning, so the deltas are complete when
            // accumulate_grad runs below.
            const auto in_eb = DataTypes::dtype_bytes(x.dtype);
            const auto g_eb = DataTypes::dtype_bytes(grad_dtype);
            const auto args = BatchNormGArgs{
                x.dtype, grad_dtype,
                static_cast<const uint8_t*>(x.data) + x.offset * in_eb,
                static_cast<const uint8_t*>(output_grad->data) + output_grad->offset * g_eb,
                static_cast<const uint8_t*>(gamma.data) + gamma.offset * in_eb,
                static_cast<const uint8_t*>(running_mean.data) + running_mean.offset * in_eb,
                static_cast<const uint8_t*>(running_var.data) + running_var.offset * in_eb,
                dx ? static_cast<uint8_t*>(dx->data) + dx->offset * g_eb : nullptr,
                dgamma ? static_cast<uint8_t*>(dgamma->data) + dgamma->offset * g_eb : nullptr,
                dbeta ? static_cast<uint8_t*>(dbeta->data) + dbeta->offset * g_eb : nullptr,
                channels, inner, count, eps};
            batch_norm_g_eval_gpu(args);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else {
            auto run = [&]<typename T, typename W>() {
                const auto* const x_data = static_cast<const T*>(x.data) + x.offset;
                const auto* const g_data = static_cast<const T*>(gamma.data) + gamma.offset;
                const auto* const rm_data = static_cast<const T*>(running_mean.data) + running_mean.offset;
                const auto* const rv_data = static_cast<const T*>(running_var.data) + running_var.offset;
                auto run_g = [&]<typename G>() {
                    const auto* const dy_data = static_cast<const G*>(output_grad->data) + output_grad->offset;
                    auto* const dx_data = dx ? static_cast<G*>(dx->data) + dx->offset : nullptr;
                    auto* const dg_data = dgamma ? static_cast<G*>(dgamma->data) + dgamma->offset : nullptr;
                    auto* const db_data = dbeta ? static_cast<G*>(dbeta->data) + dbeta->offset : nullptr;
                    batch_norm_g_eval_typed<T, W, G>(x, *output_grad, gamma, dx ? &*dx : nullptr,
                                                     x_data, dy_data, g_data, rm_data, rv_data,
                                                     dx_data, dg_data, db_data,
                                                     channels, inner, count, eps);
                };
                switch (grad_dtype) {
                    case DataTypes::DType::Float32: run_g.template operator()<float32_t>(); break;
                    case DataTypes::DType::Float64: run_g.template operator()<float64_t>(); break;
                    case DataTypes::DType::Float16: run_g.template operator()<float16_t>(); break;
                    case DataTypes::DType::BFloat16: run_g.template operator()<bfloat16_t>(); break;
                    [[unlikely]] default: throw ErrorHandling::NotImplemented();
                }
            };
            switch (x.dtype) {
                case DataTypes::DType::Float32: run.template operator()<float32_t, float32_t>(); break;
                case DataTypes::DType::Float64: run.template operator()<float64_t, float64_t>(); break;
                case DataTypes::DType::Float16: run.template operator()<float16_t, float32_t>(); break;
                case DataTypes::DType::BFloat16: run.template operator()<bfloat16_t, float32_t>(); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }

        if (dgamma) {
            if (gamma.node != nullptr) {
                grad_probe_hit(dgamma->dtype); // test hook: an intermediate received a delta
            }
            gamma.accumulate_grad(std::move(*dgamma));
        }
        if (dbeta) {
            if (beta.node != nullptr) {
                grad_probe_hit(dbeta->dtype);
            }
            beta.accumulate_grad(std::move(*dbeta));
        }
        if (dx) {
            if (x.node != nullptr) {
                grad_probe_hit(dx->dtype);
            }
            x.accumulate_grad(std::move(*dx));
        }
    }
}
