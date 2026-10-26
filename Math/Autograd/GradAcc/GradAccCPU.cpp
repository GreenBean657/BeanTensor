/**
 * @file GradAccCPU.cpp
 * @author GreenBean
 * @brief CPU backend for grad accumulation (g += d).
 * @version 0.2
 * @date 2026-10-25
 */
#include <type_traits>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradAcc/GradAcc.h"
#include "Math/Autograd/GradAcc/GradAccKernels.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    namespace {
        // --------------------------- grad_acc kernels ---------------------------

        /**
         * Single-rounding add - bit-identical to the AVX kernels.
         * @brief Scalar range grad accumulation (fp32/fp64).
         * @param w Accumulated gradient buffer, updated in place (w += d).
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void grad_acc_range_scalar(T* w, const T* d, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = w[i] + d[i];
            }
        }

        /**
         * Half-dtype accumulation: widen to fp32, add, narrow RNE (unaccel
         * pathway) - same conversion-pathway semantics as the forward kernels.
         * @brief Scalar range grad accumulation (fp16/bf16, fp32 compute).
         * @param w Accumulated gradient buffer, updated in place.
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void grad_acc_range_half(H* w, const H* d, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = narrow_lane<H>(widen_lane(w[i]) + widen_lane(d[i]));
            }
        }

        /**
         * Packed-dtype accumulation: per-lane widen/add/narrow, repack on store.
         * @brief Scalar range grad accumulation (fp16x2/bf16x2, per-lane fp32 compute).
         * @param w Accumulated gradient buffer, updated in place.
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void grad_acc_range_packed(P* w, const P* d, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto w_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(w[i]);
                const auto d_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(d[i]);
                const auto lo = narrow_lane<H>(widen_lane(lane_lo<H>(w_bits)) + widen_lane(lane_lo<H>(d_bits)));
                const auto hi = narrow_lane<H>(widen_lane(lane_hi<H>(w_bits)) + widen_lane(lane_hi<H>(d_bits)));
                w[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                     Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Integer accumulation: modular (wrap) add through the unsigned twin -
         * the same wrap contract as the integer sum/bias_add kernels.
         * @brief Scalar range grad accumulation (all integer dtypes, wrap add).
         * @param w Accumulated gradient buffer, updated in place.
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void grad_acc_range_int(T* w, const T* d, const size_t begin, const size_t end) {
            using U = std::make_unsigned_t<T>;
            for (auto i = begin; i < end; ++i) {
                w[i] = static_cast<T>(static_cast<U>(static_cast<U>(w[i]) + static_cast<U>(d[i])));
            }
        }

        /**
         * Int4x2 accumulation: per-nibble wrap add, repack on store.
         * @brief Scalar range grad accumulation (int4x2, per-nibble wrap).
         * @param w Accumulated gradient buffer (one element = two nibbles), updated in place.
         * @param d Incoming gradient delta.
         * @param begin First storage element processed.
         * @param end One past the last storage element processed.
         */
        void grad_acc_range_int4x2(int4x2_t* w, const int4x2_t* d, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = pack_int4(int4_lo(w[i]) + int4_lo(d[i]),
                                 int4_hi(w[i]) + int4_hi(d[i]));
            }
        }

        /**
         * Fixed-arity function pointer type of a grad_acc range kernel (scalar or AVX).
         * @brief Range kernel signature for grad accumulation: (w, d, begin, end).
         */
        template<typename T>
        using GradAccScalarFn = void(*)(T*, const T*, size_t, size_t);

        /**
         * Local twin of autograd_shared::launch_elementwise with CCD-aware
         * scheduling: grad_acc writes the accumulated gradient buffer in place,
         * so the gate is the bytes written (numel * element size). When that
         * fits one CCD's LLC the chunks are submitted to the calling thread's
         * CCD group (submit_to_group) so the cache-resident buffer is written
         * locally; otherwise they go to the whole pool exactly as before. The
         * chunk count still assumes all t_count threads — group workers simply
         * claim multiple chunks from the queue, so restricting the submit never
         * starves the decomposition.
         * @brief Threaded grad-accumulation launcher with CCD-local routing.
         * @param worker Callable taking (begin, end); copied into each task.
         * @param numel Total element count to cover.
         * @param out_bytes Bytes of the gradient buffer actually written by the workers.
         * @param a First tensor the workers touch (receives the futures).
         * @param b Second tensor the workers touch (receives the futures).
         * @param out Output tensor (receives the futures).
         */
        template<typename Worker>
        void launch_elementwise_ccd(Worker&& worker, const size_t numel, const size_t out_bytes,
                                    const Tensor& a, const Tensor& b, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(out_bytes);
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
            a.give_futures(futures);
            b.give_futures(futures);
            out.give_futures(futures);
        }
    }

    /**
     * In-place accumulation g += d over g.numel elements, honoring both tensors' offsets.
     * d must share g's dtype and shape (not checked). Integer dtypes wrap, halfs
     * widen to fp32 and narrow RNE, fp32/fp64 add directly (AVX tiers bit-identical).
     * Work is queued on the CPU pool and returns immediately; futures land on g and d.
     * @brief CPU gradient accumulation, g += d.
     * @param g Accumulated gradient, updated in place.
     * @param d Gradient delta to add.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if either tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void grad_acc_cpu(const Tensor& g, const Tensor& d) {
        if (g.device != Device::CPU || d.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(g.device));
        }
        using DType = DataTypes::DType;
        const auto numel = g.numel;
        switch (g.dtype) {
            case DType::Float32: {
                auto* w = static_cast<float32_t*>(g.data) + g.offset;
                const auto* d_ptr = static_cast<const float32_t*>(d.data) + d.offset;
                const GradAccScalarFn<float32_t> kernel = Math::Autograd::grad_acc_avx512_available()
                    ? Math::Autograd::grad_acc_f32_range_avx512
                    : Math::Autograd::grad_acc_avx2_available() ? Math::Autograd::grad_acc_f32_range_avx2 : grad_acc_range_scalar<float32_t>;
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): w, d_ptr and the dispatched kernel.
                 * Calls kernel(w, d_ptr, b, e), where w and d_ptr already include their tensors' offsets.
                 * @brief Worker for grad_acc_cpu, float32_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([=](const size_t b, const size_t e) { kernel(w, d_ptr, b, e); }, numel, numel * sizeof(float32_t), g, d, g);
                break;
            }
            case DType::Float64: {
                auto* w = static_cast<float64_t*>(g.data) + g.offset;
                const auto* d_ptr = static_cast<const float64_t*>(d.data) + d.offset;
                const GradAccScalarFn<float64_t> kernel = Math::Autograd::grad_acc_avx512_available()
                    ? Math::Autograd::grad_acc_f64_range_avx512
                    : Math::Autograd::grad_acc_avx2_available() ? Math::Autograd::grad_acc_f64_range_avx2 : grad_acc_range_scalar<float64_t>;
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): w, d_ptr and the dispatched kernel.
                 * Calls kernel(w, d_ptr, b, e), where w and d_ptr already include their tensors' offsets.
                 * @brief Worker for grad_acc_cpu, float64_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([=](const size_t b, const size_t e) { kernel(w, d_ptr, b, e); }, numel, numel * sizeof(float64_t), g, d, g);
                break;
            }
            case DType::Float16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Lanes widen to fp32, add, and narrow RNE. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the float16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<float16_t*>(g.data) + g.offset, d_ptr = static_cast<const float16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_half<float16_t>(w, d_ptr, b, e); }, numel, numel * sizeof(float16_t), g, d, g);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Lanes widen to fp32, add, and narrow RNE. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the bfloat16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<bfloat16_t*>(g.data) + g.offset, d_ptr = static_cast<const bfloat16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_half<bfloat16_t>(w, d_ptr, b, e); }, numel, numel * sizeof(bfloat16_t), g, d, g);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Each element holds two lanes; both are processed per element. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the float16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<float16x2_t*>(g.data) + g.offset, d_ptr = static_cast<const float16x2_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_packed<float16x2_t>(w, d_ptr, b, e); }, numel, numel * sizeof(float16x2_t), g, d, g);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Each element holds two lanes; both are processed per element. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the bfloat16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<bfloat16x2_t*>(g.data) + g.offset, d_ptr = static_cast<const bfloat16x2_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_packed<bfloat16x2_t>(w, d_ptr, b, e); }, numel, numel * sizeof(bfloat16x2_t), g, d, g);
                break;
            }
            case DType::Int8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<int8_t*>(g.data) + g.offset, d_ptr = static_cast<const int8_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int8_t>(w, d_ptr, b, e); }, numel, numel * sizeof(int8_t), g, d, g);
                break;
            }
            case DType::UInt8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<uint8_t*>(g.data) + g.offset, d_ptr = static_cast<const uint8_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint8_t>(w, d_ptr, b, e); }, numel, numel * sizeof(uint8_t), g, d, g);
                break;
            }
            case DType::Int16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<int16_t*>(g.data) + g.offset, d_ptr = static_cast<const int16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int16_t>(w, d_ptr, b, e); }, numel, numel * sizeof(int16_t), g, d, g);
                break;
            }
            case DType::UInt16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<uint16_t*>(g.data) + g.offset, d_ptr = static_cast<const uint16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint16_t>(w, d_ptr, b, e); }, numel, numel * sizeof(uint16_t), g, d, g);
                break;
            }
            case DType::Int32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<int32_t*>(g.data) + g.offset, d_ptr = static_cast<const int32_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int32_t>(w, d_ptr, b, e); }, numel, numel * sizeof(int32_t), g, d, g);
                break;
            }
            case DType::UInt32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<uint32_t*>(g.data) + g.offset, d_ptr = static_cast<const uint32_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint32_t>(w, d_ptr, b, e); }, numel, numel * sizeof(uint32_t), g, d, g);
                break;
            }
            case DType::Int64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<int64_t*>(g.data) + g.offset, d_ptr = static_cast<const int64_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int64_t>(w, d_ptr, b, e); }, numel, numel * sizeof(int64_t), g, d, g);
                break;
            }
            case DType::UInt64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<uint64_t*>(g.data) + g.offset, d_ptr = static_cast<const uint64_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint64_t>(w, d_ptr, b, e); }, numel, numel * sizeof(uint64_t), g, d, g);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Each element holds two nibbles; both are processed per element. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int4x2_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise_ccd([w = static_cast<int4x2_t*>(g.data) + g.offset, d_ptr = static_cast<const int4x2_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int4x2(w, d_ptr, b, e); }, numel, numel * sizeof(int4x2_t), g, d, g);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}
