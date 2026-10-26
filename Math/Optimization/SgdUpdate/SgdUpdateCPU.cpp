/**
 * @file SgdUpdateCPU.cpp
 * @author GreenBean
 * @brief CPU backend for sgd_update. Hosts the Tensor::sgd_update and
 * Tensor::sgd_step method definitions (mirrors how Tensor::matmul lives in
 * MatmulCPU.cpp).
 * @version 0.1.2
 * @date 2026-10-25
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Optimization/SgdUpdate/SgdUpdate.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Optimization/OptimizationShared.h"
#include "Math/Optimization/SgdUpdate/SgdUpdateKernels.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace optimization_shared;

    namespace {
        /**
         * Single rounding via fma matches the FMA-based AVX kernels bit-for-bit.
         * @note Computes w[i] = fma(neg_lr, g[i], w[i]).
         * @brief Scalar range SGD (fp32/fp64).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param neg_lr Negated learning rate.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void sgd_range_scalar(T* w, const T* g, const T neg_lr, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                if constexpr (std::is_same_v<T, float32_t>) {
                    w[i] = std::fmaf(neg_lr, g[i], w[i]);
                } else {
                    w[i] = std::fma(neg_lr, g[i], w[i]);
                }
            }
        }

        /**
         * Widen to fp32, fmaf, narrow RNE (unaccel pathway). Same conversion-pathway semantics as the GEMM half kernels.
         * @brief Scalar range SGD (fp16/bf16, fp32 compute).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param neg_lr Negated learning rate.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void sgd_range_half(H* w, const H* g, const float32_t neg_lr, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = narrow_lane<H>(std::fmaf(neg_lr, widen_lane(g[i]), widen_lane(w[i])));
            }
        }

        /**
         * Per-lane widen/fmaf/narrow, repacked on store.
         * @brief Scalar range SGD (fp16x2/bf16x2, per-lane fp32 compute).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param neg_lr Negated learning rate.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void sgd_range_packed(P* w, const P* g, const float32_t neg_lr, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto w_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(w[i]);
                const auto g_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(g[i]);
                const auto lo = narrow_lane<H>(std::fmaf(neg_lr, widen_lane(lane_lo<H>(g_bits)), widen_lane(lane_lo<H>(w_bits))));
                const auto hi = narrow_lane<H>(std::fmaf(neg_lr, widen_lane(lane_hi<H>(g_bits)), widen_lane(lane_hi<H>(w_bits))));
                w[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                     Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Computes w - lr*g in fp64 with a single fma rounding, clamps to the dtype range, narrows round-toward-zero.
         * At sub-1 magnitudes negative updates truncate up toward 0 while positive ones truncate down (up to 1 in the result).
         * A NaN update (NaN lr or grad lane) leaves the element unchanged, no integer NaN exists to store.
         * @brief Scalar range SGD (all integer dtypes, fp64 compute, clamp+truncate).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate (not negated).
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void sgd_range_int(T* w, const T* g, const float64_t lr, const size_t begin, const size_t end) {
            const auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
            const auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            for (auto i = begin; i < end; ++i) {
                const auto upd = std::fma(-lr, static_cast<float64_t>(g[i]), static_cast<float64_t>(w[i]));
                if (__builtin_isnan(upd)) {
                    continue;
                }
                if (upd <= lo) {
                    w[i] = std::numeric_limits<T>::lowest();
                } else if (upd >= hi) {
                    w[i] = std::numeric_limits<T>::max();
                } else {
                    w[i] = static_cast<T>(upd);
                }
            }
        }

        /**
         * Per-lane fp64 compute, clamp to [-8, 7], narrow round-toward-zero (same contract as sgd_range_int).
         * A NaN update leaves that lane unchanged.
         * @brief Scalar range SGD (int4x2, per-lane fp64 compute).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate (not negated).
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        void sgd_range_i4x2(int4x2_t* w, const int4x2_t* g, const float64_t lr, const size_t begin, const size_t end) {
            /**
             * Applies one clamped fp64 SGD step to a single signed int4 lane.
             * @brief Update one int4 lane.
             * @param w_lane Current weight lane, [-8, 7].
             * @param g_lane Gradient lane, [-8, 7].
             * @return Updated lane in [-8, 7]; w_lane itself when the update is NaN.
             * @note Lambda subcall. Self-execution prohibited.
             */
            const auto lane_step = [lr](const int8_t w_lane, const int8_t g_lane) {
                const auto upd = std::fma(-lr, static_cast<float64_t>(g_lane), static_cast<float64_t>(w_lane));
                if (__builtin_isnan(upd)) {
                    return w_lane;
                }
                if (upd <= -8.0) {
                    return int8_t{-8};
                }
                if (upd >= 7.0) {
                    return int8_t{7};
                }
                return static_cast<int8_t>(upd);
            };
            for (auto i = begin; i < end; ++i) {
                w[i] = i4_pack(lane_step(i4_lane_lo(w[i]), i4_lane_lo(g[i])),
                               lane_step(i4_lane_hi(w[i]), i4_lane_hi(g[i])));
            }
        }

        /**
         * Futures land on both tensors so teardown drains the workers (they hold raw buffer pointers).
         * Small inputs run inline. The worker is captured by value, the futures outlive this call.
         * CCD-aware: SGD writes the parameter buffer in place (w), so the gate is the bytes
         * written, numel * w's element size. When that fits one CCD's LLC the chunks go to
         * the calling thread's CCD group (submit_to_group) so the cache-resident parameter is
         * written locally; otherwise they go to the whole pool as before. The chunk count
         * still assumes all t_count threads — group workers simply claim multiple chunks.
         * @brief Threaded SGD launcher, chunks an elementwise two-operand range worker across the CPU thread pool.
         * @param worker Callable taking (begin, end).
         * @param numel Element count to cover.
         * @param w Weight tensor, receives the futures (its span is the write gate).
         * @param grad Gradient tensor, receives the futures.
         */
        template<typename Worker>
        void launch_sgd(Worker&& worker, const size_t numel, const Tensor& w, const Tensor& grad) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(numel * DataTypes::dtype_bytes(w.get_dtype()));
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= numel) {
                    break;
                }
                // worker is captured BY VALUE: these futures outlive this call
                // (they land on the tensors), so a reference would dangle.
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
                }
            }
            w.give_futures(futures);
            grad.give_futures(futures);
        }
    }

    /**
     * Dispatches on dtype: AVX512 > AVX2 > scalar for fp32/fp64, scalar paths for the rest.
     * Chunked across the CPU thread pool.
     * @brief CPU in-place SGD step, w -= lr * grad.
     * @param w Weights, updated in place.
     * @param grad Gradients, same shape and dtype as w.
     * @param lr Learning rate.
     */
    void sgd_update_cpu(const Tensor& w, const Tensor& grad, const float64_t lr) {
        if (w.device != Device::CPU || grad.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(w.device));
        }
        using DType = DataTypes::DType;
        const auto numel = w.numel;
        switch (w.dtype) {
            case DType::Float32: {
                auto* w_ptr = static_cast<float32_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float32_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                const SgdScalarFn<float32_t> kernel = Math::Optimization::sgd_update_avx512_available()
                    ? Math::Optimization::sgd_f32_range_avx512
                    : Math::Optimization::sgd_update_avx2_available() ? Math::Optimization::sgd_f32_range_avx2 : sgd_range_scalar<float32_t>;
                launch_sgd([=](const size_t b, const size_t e) { kernel(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Float64: {
                auto* w_ptr = static_cast<float64_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float64_t*>(grad.data) + grad.offset;
                const SgdScalarFn<float64_t> kernel = Math::Optimization::sgd_update_avx512_available()
                    ? Math::Optimization::sgd_f64_range_avx512
                    : Math::Optimization::sgd_update_avx2_available() ? Math::Optimization::sgd_f64_range_avx2 : sgd_range_scalar<float64_t>;
                launch_sgd([=](const size_t b, const size_t e) { kernel(w_ptr, g_ptr, -lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Float16: {
                auto* w_ptr = static_cast<float16_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float16_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_half<float16_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::BFloat16: {
                auto* w_ptr = static_cast<bfloat16_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const bfloat16_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_half<bfloat16_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Float16x2: {
                auto* w_ptr = static_cast<float16x2_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float16x2_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_packed<float16x2_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::BFloat16x2: {
                auto* w_ptr = static_cast<bfloat16x2_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const bfloat16x2_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_packed<bfloat16x2_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int8: {
                launch_sgd([w_ptr = static_cast<int8_t*>(w.data) + w.offset, g_ptr = static_cast<const int8_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int8_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt8: {
                launch_sgd([w_ptr = static_cast<uint8_t*>(w.data) + w.offset, g_ptr = static_cast<const uint8_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint8_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int16: {
                launch_sgd([w_ptr = static_cast<int16_t*>(w.data) + w.offset, g_ptr = static_cast<const int16_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int16_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt16: {
                launch_sgd([w_ptr = static_cast<uint16_t*>(w.data) + w.offset, g_ptr = static_cast<const uint16_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint16_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int32: {
                launch_sgd([w_ptr = static_cast<int32_t*>(w.data) + w.offset, g_ptr = static_cast<const int32_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int32_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt32: {
                launch_sgd([w_ptr = static_cast<uint32_t*>(w.data) + w.offset, g_ptr = static_cast<const uint32_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint32_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int64: {
                launch_sgd([w_ptr = static_cast<int64_t*>(w.data) + w.offset, g_ptr = static_cast<const int64_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int64_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt64: {
                launch_sgd([w_ptr = static_cast<uint64_t*>(w.data) + w.offset, g_ptr = static_cast<const uint64_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint64_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int4x2: {
                launch_sgd([w_ptr = static_cast<int4x2_t*>(w.data) + w.offset, g_ptr = static_cast<const int4x2_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_i4x2(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Both tensors are synced before dispatch; requires identical shape, device, contiguous.
     * A grad dtype differing from the weight dtype (backward(DType) makes the grad
     * dtype a per-call knob) no longer throws: a private copy of grad is converted
     * into the weight dtype via the single grad-conversion rule (GradDType.h) and
     * the update proceeds on that temp.
     * @brief In-place SGD step, this -= lr * grad.
     * @param gradient Gradient tensor.
     * @param lr Learning rate.
     */
    void Tensor::sgd_update(const Tensor& gradient, const float64_t lr) {
        if (!this->is_same_shape(gradient)) {
            throw ErrorHandling::MismatchedShapes("sgd_update requires identical shapes");
        }
        if (this->device != gradient.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !gradient.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (this->dtype != gradient.dtype) {
            const auto g = detail::grad_converted_copy(gradient, this->dtype);
            this->sgd_update(g, lr);
            return;
        }
        sync();
        gradient.sync();
        if (this->device == Device::CPU) {
            detail::sgd_update_cpu(*this, gradient, lr);
        } else if (this->device == Device::GPU) {
            detail::sgd_update_gpu(*this, gradient, lr);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * Uses the gradient attached to this tensor.
     * @brief In-place SGD step from this tensor's own gradient.
     * @param lr Learning rate.
     */
    void Tensor::sgd_step(const float64_t lr) {
        if (!this->requires_grad || this->grad == nullptr) {
            throw ErrorHandling::NoGradient();
        }
        sgd_update(*this->grad, lr);
    }
}
