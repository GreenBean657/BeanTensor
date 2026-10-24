/**
 * @file ReluCPU.cpp
 * @author GreenBean
 * @brief CPU backend for relu.
 * Hosts the Tensor::relu method definition (mirrors how Tensor::matmul
 * lives in MatmulCPU.cpp).
 * @version 0.2
 * @date 2026-10-23
 */
#include <type_traits>
#include <vector>

#include "Math/Activations/ActivationsShared.h"
#include "Math/Activations/Relu/Relu.h"
#include "Math/Activations/Relu/ReluKernels.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace activations_shared;

    namespace {
        // --------------------------- relu kernels ---------------------------

        /**
         * Computes out[i] = in[i] < 0 ? 0 : in[i]. NaN propagates (the comparison is false for NaN) and
         * -0.0 survives. Bit-identical to the AVX max(zero, x) kernels.
         * @brief Scalar range relu (fp32/fp64).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void relu_range_scalar(T* out, const T* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = in[i] < T{0} ? T{0} : in[i];
            }
        }

        /**
         * Widen to fp32, relu, narrow RNE. NaN propagates.
         * @brief Scalar range relu (fp16/bf16, fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename H>
        void relu_range_half(H* out, const H* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto wide = widen_lane(in[i]);
                out[i] = narrow_lane<H>(wide < 0.0f ? 0.0f : wide);
            }
        }

        /**
         * Per-lane widen/relu/narrow, repack on store.
         * @brief Scalar range relu (fp16x2/bf16x2, per-lane fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename P>
        void relu_range_packed(P* out, const P* in, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[i]);
                const auto lo_wide = widen_lane(lane_lo<H>(bits));
                const auto hi_wide = widen_lane(lane_hi<H>(bits));
                const auto lo = narrow_lane<H>(lo_wide < 0.0f ? 0.0f : lo_wide);
                const auto hi = narrow_lane<H>(hi_wide < 0.0f ? 0.0f : hi_wide);
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Signed dtypes clamp to 0; unsigned dtypes copy through.
         * @brief Scalar range relu (all integer dtypes).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void relu_range_int(T* out, const T* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                if constexpr (std::is_signed_v<T>) {
                    out[i] = in[i] < T{0} ? T{0} : in[i];
                } else {
                    out[i] = in[i];
                }
            }
        }

        /**
         * Per-nibble clamp to 0, repack on store.
         * @brief Scalar range relu (int4x2, per-nibble).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        void relu_range_int4x2(int4x2_t* out, const int4x2_t* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto lo = int4_lo(in[i]);
                const auto hi = int4_hi(in[i]);
                out[i] = pack_int4(lo < 0 ? 0 : lo, hi < 0 ? 0 : hi);
            }
        }

        /**
         * @brief Function pointer type shared by every relu range kernel (scalar, AVX2, AVX512).
         */
        template<typename T>
        using ReluScalarFn = void(*)(T*, const T*, size_t, size_t);

        /**
         * wire_node's function-pointer type takes a mutable output_grad while
         * relu_g_node takes a const one (a function-pointer type mismatch the
         * Clang build accepts as an extension but GCC rejects). The trampoline
         * only reads output_grad, so forward through a matching bridge.
         * @brief Signature bridge between wire_node and relu_g_node.
         * @param node Owning node, forwarded to relu_g_node unchanged.
         * @param output_grad Gradient of the forward output, in grad_dtype.
         * @param inputs Input tensors of the forward op.
         * @param n_inputs Number of entries in inputs.
         * @param grad_dtype The calling backward()'s assigned gradient dtype.
         */
        void relu_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
            relu_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
        }
    }

    /**
     * Float32/Float64 pick AVX512, then AVX2, then the scalar kernel. Every other dtype uses its scalar kernel.
     * Work is chunked across the CPU thread pool and the futures land on both tensors.
     * @brief CPU relu: out[i] = t[i] < 0 ? 0 : t[i].
     * @param t Input tensor, CPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as t.
     */
    void relu_cpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        switch (t.dtype) {
            case DType::Float32: {
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float32_t*>(t.data) + t.offset;
                const ReluScalarFn<float32_t> kernel = Math::Activations::relu_avx512_available()
                    ? Math::Activations::relu_f32_range_avx512
                    : Math::Activations::relu_avx2_available() ? Math::Activations::relu_f32_range_avx2 : relu_range_scalar<float32_t>;
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Float64: {
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float64_t*>(t.data) + t.offset;
                const ReluScalarFn<float64_t> kernel = Math::Activations::relu_avx512_available()
                    ? Math::Activations::relu_f64_range_avx512
                    : Math::Activations::relu_avx2_available() ? Math::Activations::relu_f64_range_avx2 : relu_range_scalar<float64_t>;
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Float16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for float16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16_t*>(out.data) + out.offset, in_ptr = static_cast<const float16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_half<float16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for bfloat16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_half<bfloat16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for float16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const float16x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_packed<float16x2_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for bfloat16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_packed<bfloat16x2_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int8: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int8_t*>(out.data) + out.offset, in_ptr = static_cast<const int8_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int8_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt8: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint8_t*>(out.data) + out.offset, in_ptr = static_cast<const uint8_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint8_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int16_t*>(out.data) + out.offset, in_ptr = static_cast<const int16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint16_t*>(out.data) + out.offset, in_ptr = static_cast<const uint16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int32: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int32_t*>(out.data) + out.offset, in_ptr = static_cast<const int32_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int32_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt32: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint32_t*>(out.data) + out.offset, in_ptr = static_cast<const uint32_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint32_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int64: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int64_t*>(out.data) + out.offset, in_ptr = static_cast<const int64_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int64_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt64: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint64_t*>(out.data) + out.offset, in_ptr = static_cast<const uint64_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint64_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int4x2.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int4x2_t*>(out.data) + out.offset, in_ptr = static_cast<const int4x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int4x2(out_ptr, in_ptr, b, e); }, numel, t, t, out);
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
     * Validates contiguity, syncs this tensor, then routes to the CPU or GPU backend and wires the autograd node.
     * @brief Elementwise rectified linear unit.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::relu() const {
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        auto out = Tensor(std::vector(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::relu_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::relu_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::relu_g_bridge, *this);
        return out;
    }
}
