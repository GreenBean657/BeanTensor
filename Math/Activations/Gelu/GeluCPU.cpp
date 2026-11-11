/**
 * @file GeluCPU.cpp
 * @author GreenBean
 * @brief CPU backend for GELU.
 * Hosts the Tensor::gelu method definition.
 * @version 0.1
 * @date 2026-11-10
 */
#include <type_traits>
#include <vector>

#include "Math/Activations/ActivationsShared.h"
#include "Math/Activations/Gelu/Gelu.h"
#include "Math/Activations/Gelu/GeluKernels.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace activations_shared;
    using namespace BeanTensor::Math::Activations::detail;

    namespace {
        // --------------------------- gelu kernels ---------------------------

        /**
         * Computes out[i] = gelu(in[i]) using std::erf in the storage width.
         * The piecewise branches make gelu(-Inf) == 0 and gelu(+Inf) == +Inf.
         * @brief Scalar range GELU (fp32/fp64).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void gelu_range_scalar(T* out, const T* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = gelu_exact_lane(in[i]);
            }
        }

        /**
         * Widen to fp32, GELU, narrow RNE.  Non-finite and signed-zero semantics
         * come from the fp32 compute.
         * @brief Scalar range GELU (fp16/bf16, fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename H>
        void gelu_range_half(H* out, const H* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto wide = widen_lane(in[i]);
                out[i] = narrow_lane<H>(gelu_exact_lane(wide));
            }
        }

        /**
         * Per-lane widen/GELU/narrow, repack on store.
         * @brief Scalar range GELU (fp16x2/bf16x2, per-lane fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename P>
        void gelu_range_packed(P* out, const P* in, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[i]);
                const auto lo = narrow_lane<H>(gelu_exact_lane(widen_lane(lane_lo<H>(bits))));
                const auto hi = narrow_lane<H>(gelu_exact_lane(widen_lane(lane_hi<H>(bits))));
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * @brief Function pointer type shared by every GELU range kernel (scalar, AVX2, AVX512).
         */
        template<typename T>
        using GeluScalarFn = void(*)(T*, const T*, size_t, size_t);
    }

    /**
     * Float32/Float64 pick AVX512, then AVX2, then the scalar kernel.  Every other
     * supported dtype uses its scalar kernel.  Integer dtypes are rejected by
     * Tensor::gelu before reaching the backend.  Work is chunked across the CPU
     * thread pool and the futures land on both tensors.
     * @brief CPU GELU: out[i] = 0.5*in[i]*(1+erf(in[i]/sqrt(2))).
     * @param t Input tensor, CPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as t.
     */
    void gelu_cpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        switch (t.dtype) {
            case DType::Float32: {
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float32_t*>(t.data) + t.offset;
                const GeluScalarFn<float32_t> kernel = Math::Activations::gelu_avx512_available()
                    ? Math::Activations::gelu_f32_range_avx512
                    : Math::Activations::gelu_avx2_available() ? Math::Activations::gelu_f32_range_avx2 : gelu_range_scalar<float32_t>;
                /**
                 * Forwards the captured GELU pointers to the range kernel.
                 * @brief Elementwise GELU range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b, e); }, numel, numel * sizeof(float32_t), t, t, out);
                break;
            }
            case DType::Float64: {
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float64_t*>(t.data) + t.offset;
                const GeluScalarFn<float64_t> kernel = Math::Activations::gelu_avx512_available()
                    ? Math::Activations::gelu_f64_range_avx512
                    : Math::Activations::gelu_avx2_available() ? Math::Activations::gelu_f64_range_avx2 : gelu_range_scalar<float64_t>;
                /**
                 * Forwards the captured GELU pointers to the range kernel.
                 * @brief Elementwise GELU range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b, e); }, numel, numel * sizeof(float64_t), t, t, out);
                break;
            }
            case DType::Float16: {
                /**
                 * Forwards the captured GELU pointers to the range kernel.
                 * @brief Elementwise GELU range worker for float16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16_t*>(out.data) + out.offset, in_ptr = static_cast<const float16_t*>(t.data) + t.offset](const size_t b, const size_t e) { gelu_range_half<float16_t>(out_ptr, in_ptr, b, e); }, numel, numel * sizeof(float16_t), t, t, out);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Forwards the captured GELU pointers to the range kernel.
                 * @brief Elementwise GELU range worker for bfloat16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16_t*>(t.data) + t.offset](const size_t b, const size_t e) { gelu_range_half<bfloat16_t>(out_ptr, in_ptr, b, e); }, numel, numel * sizeof(bfloat16_t), t, t, out);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Forwards the captured GELU pointers to the range kernel.
                 * @brief Elementwise GELU range worker for float16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const float16x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { gelu_range_packed<float16x2_t>(out_ptr, in_ptr, b, e); }, numel, numel * sizeof(float16x2_t), t, t, out);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Forwards the captured GELU pointers to the range kernel.
                 * @brief Elementwise GELU range worker for bfloat16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { gelu_range_packed<bfloat16x2_t>(out_ptr, in_ptr, b, e); }, numel, numel * sizeof(bfloat16x2_t), t, t, out);
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
     * Validates contiguity and float-kind dtype, syncs this tensor, then routes to
     * the CPU or GPU backend and wires the autograd node.
     * @brief Elementwise Gaussian error linear unit.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::gelu() const & {
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (!detail::elementwise_shared::is_float_kind(this->dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        auto out = Tensor(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::gelu_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::gelu_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::gelu_g_node, *this);
        return out;
    }
}
