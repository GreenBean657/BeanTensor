/**
 * @file BiasAddCPU.cpp
 * @author GreenBean
 * @brief CPU backend for bias_add.
 * Hosts the Tensor::bias_add method definition (mirrors how Tensor::matmul
 * lives in MatmulCPU.cpp).
 * @version 0.1
 * @date 2026-10-20
 */
#include <algorithm>
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Math/LinearAlgebra/BiasAdd/BiasAdd.h"
#include "Math/LinearAlgebra/BiasAdd/BiasAddKernels.h"
#include "Math/LinearAlgebra/LinearAlgebraShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace linearalgebra_shared;

    namespace {
        // --------------------------- bias_add kernels ---------------------------

        /**
         * Single-rounding add, bit-identical to the AVX kernels.
         * @brief Scalar range bias_add (fp32/fp64).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void bias_add_range_scalar(T* out, const T* in, const T* bias, const size_t cols, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = in[i] + bias[i % cols];
            }
        }

        /**
         * Widen to fp32, add, narrow RNE (unaccel pathway).
         * Same conversion-pathway semantics as the other half kernels.
         * @brief Scalar range bias_add (fp16/bf16, fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename H>
        void bias_add_range_half(H* out, const H* in, const H* bias, const size_t cols, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = narrow_lane<H>(widen_lane(in[i]) + widen_lane(bias[i % cols]));
            }
        }

        /**
         * Per-lane widen/add/narrow, repack on store.
         * @brief Scalar range bias_add (fp16x2/bf16x2, per-lane fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename P>
        void bias_add_range_packed(P* out, const P* in, const P* bias, const size_t cols, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto in_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[i]);
                const auto b_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(bias[i % cols]);
                const auto lo = narrow_lane<H>(widen_lane(lane_lo<H>(in_bits)) + widen_lane(lane_lo<H>(b_bits)));
                const auto hi = narrow_lane<H>(widen_lane(lane_hi<H>(in_bits)) + widen_lane(lane_hi<H>(b_bits)));
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Modular (wrap) add through the unsigned twin. Same wrap contract as the integer sum kernels,
         * bit-identical on every backend and chunking.
         * @brief Scalar range bias_add (all integer dtypes, wrap add).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void bias_add_range_int(T* out, const T* in, const T* bias, const size_t cols, const size_t begin, const size_t end) {
            using U = std::make_unsigned_t<T>;
            for (auto i = begin; i < end; ++i) {
                out[i] = static_cast<T>(static_cast<U>(static_cast<U>(in[i]) + static_cast<U>(bias[i % cols])));
            }
        }

        /**
         * Per-nibble wrap add, repack on store.
         * @brief Scalar range bias_add (int4x2, per-nibble wrap).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        void bias_add_range_int4x2(int4x2_t* out, const int4x2_t* in, const int4x2_t* bias, const size_t cols, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = pack_int4(int4_lo(in[i]) + int4_lo(bias[i % cols]),
                                   int4_hi(in[i]) + int4_hi(bias[i % cols]));
            }
        }

        /**
         * @brief Function pointer type shared by every bias_add range kernel (scalar, AVX2, AVX512).
         */
        template<typename T>
        using BiasAddScalarFn = void(*)(T*, const T*, const T*, size_t, size_t, size_t);
    }

    /**
     * Float32/Float64 pick AVX512, then AVX2, then the scalar kernel. Every other dtype uses its scalar kernel.
     * Work is chunked across the CPU thread pool and the futures land on all three tensors.
     * @brief CPU bias_add: out[i] = t[i] + bias[i % cols].
     * @param t Input tensor, CPU resident, contiguous.
     * @param bias Rank-1 bias tensor matching the last dimension of t.
     * @param out Output tensor, same shape and dtype as t.
     */
    void bias_add_cpu(const Tensor& t, const Tensor& bias, Tensor& out) {
        if (t.device != Device::CPU || bias.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        const auto cols = t.shape[t.ndim - 1];
        switch (t.dtype) {
            case DType::Float32: {
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float32_t*>(t.data) + t.offset;
                const auto* b_ptr = static_cast<const float32_t*>(bias.data) + bias.offset;
                const BiasAddScalarFn<float32_t> kernel = Math::LinearAlgebra::bias_add_avx512_available()
                    ? Math::LinearAlgebra::bias_add_f32_range_avx512
                    : (Math::LinearAlgebra::bias_add_avx2_available() ? Math::LinearAlgebra::bias_add_f32_range_avx2 : bias_add_range_scalar<float32_t>);
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Float64: {
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float64_t*>(t.data) + t.offset;
                const auto* b_ptr = static_cast<const float64_t*>(bias.data) + bias.offset;
                const BiasAddScalarFn<float64_t> kernel = Math::LinearAlgebra::bias_add_avx512_available()
                    ? Math::LinearAlgebra::bias_add_f64_range_avx512
                    : (Math::LinearAlgebra::bias_add_avx2_available() ? Math::LinearAlgebra::bias_add_f64_range_avx2 : bias_add_range_scalar<float64_t>);
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Float16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for float16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16_t*>(out.data) + out.offset, in_ptr = static_cast<const float16_t*>(t.data) + t.offset, b_ptr = static_cast<const float16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_half<float16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for bfloat16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16_t*>(t.data) + t.offset, b_ptr = static_cast<const bfloat16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_half<bfloat16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for float16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const float16x2_t*>(t.data) + t.offset, b_ptr = static_cast<const float16x2_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_packed<float16x2_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for bfloat16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16x2_t*>(t.data) + t.offset, b_ptr = static_cast<const bfloat16x2_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_packed<bfloat16x2_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int8: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int8_t*>(out.data) + out.offset, in_ptr = static_cast<const int8_t*>(t.data) + t.offset, b_ptr = static_cast<const int8_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int8_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt8: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint8_t*>(out.data) + out.offset, in_ptr = static_cast<const uint8_t*>(t.data) + t.offset, b_ptr = static_cast<const uint8_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint8_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int16_t*>(out.data) + out.offset, in_ptr = static_cast<const int16_t*>(t.data) + t.offset, b_ptr = static_cast<const int16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint16_t*>(out.data) + out.offset, in_ptr = static_cast<const uint16_t*>(t.data) + t.offset, b_ptr = static_cast<const uint16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int32: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int32_t*>(out.data) + out.offset, in_ptr = static_cast<const int32_t*>(t.data) + t.offset, b_ptr = static_cast<const int32_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int32_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt32: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint32_t*>(out.data) + out.offset, in_ptr = static_cast<const uint32_t*>(t.data) + t.offset, b_ptr = static_cast<const uint32_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint32_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int64: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int64_t*>(out.data) + out.offset, in_ptr = static_cast<const int64_t*>(t.data) + t.offset, b_ptr = static_cast<const int64_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int64_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt64: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint64_t*>(out.data) + out.offset, in_ptr = static_cast<const uint64_t*>(t.data) + t.offset, b_ptr = static_cast<const uint64_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint64_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int4x2.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int4x2_t*>(out.data) + out.offset, in_ptr = static_cast<const int4x2_t*>(t.data) + t.offset, b_ptr = static_cast<const int4x2_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int4x2(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
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
     * Validates rank/dtype/device/contiguity, syncs both operands, then routes to the CPU or GPU backend
     * and wires the autograd node.
     * @brief Add a rank-1 bias along the last dimension.
     * @param bias Rank-1 tensor matching the last dimension, same dtype and device.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::bias_add(const Tensor& bias) const {
        if (bias.ndim != 1 || bias.shape[0] != this->shape[this->ndim - 1]) {
            throw ErrorHandling::MismatchedShapes("bias_add requires a rank-1 bias matching the last dimension");
        }
        if (this->dtype != bias.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != bias.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !bias.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        bias.sync();
        auto out = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::bias_add_cpu(*this, bias, out);
        } else if (this->device == Device::GPU) {
            detail::bias_add_gpu(*this, bias, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::bias_add_g_node, *this, &bias);
        return out;
    }
}
