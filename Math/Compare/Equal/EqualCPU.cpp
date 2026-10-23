/**
 * @file EqualCPU.cpp
 * @author GreenBean
 * @brief CPU backend for elementwise equal. Hosts the Tensor::equal method
 * definition (mirrors how Tensor::matmul lives in MatmulCPU.cpp).
 * The comparison result is a UInt8 (0/1) mask: DType has no Bool, and UInt8 is
 * the smallest dtype with full conversion/reduction support.
 * @version 0.1
 * @date 2026-10-22
 */
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Math/Compare/Equal/Equal.h"
#include "Math/Compare/CompareShared.h"
#include "Tensor/Tensor.h"

#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace compare_shared;

    namespace {
        // --------------------------- equal kernels ---------------------------

        /**
         * IEEE equality via float_eq: NaN never equals anything (not even NaN), -0.0 equals 0.0.
         * @brief Scalar range equal (fp32/fp64, widened to fp64 keys).
         * @param out Output UInt8 mask.
         * @param a Left input elements.
         * @param b Right input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         */
        template<typename T>
        void equal_range_scalar(uint8_t* out, const T* a, const T* b, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = float_eq(static_cast<float64_t>(a[i]), static_cast<float64_t>(b[i])) ? uint8_t{1} : uint8_t{0};
            }
        }

        /**
         * Lanes widen to fp32 (exact), then compare as fp64 keys.
         * @brief Scalar range equal (fp16/bf16).
         * @param out Output UInt8 mask.
         * @param a Left input elements.
         * @param b Right input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         */
        template<typename H>
        void equal_range_half(uint8_t* out, const H* a, const H* b, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = float_eq(static_cast<float64_t>(widen_lane(a[i])), static_cast<float64_t>(widen_lane(b[i]))) ? uint8_t{1} : uint8_t{0};
            }
        }

        /**
         * An element is equal when BOTH widened lanes are equal (NaN lane rules per lane).
         * @brief Scalar range equal (fp16x2/bf16x2, per-lane).
         * @param out Output UInt8 mask.
         * @param a Left input elements.
         * @param b Right input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         */
        template<typename P>
        void equal_range_packed(uint8_t* out, const P* a, const P* b, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto a_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a[i]);
                const auto b_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(b[i]);
                const auto lo = float_eq(static_cast<float64_t>(widen_lane(lane_lo<H>(a_bits))), static_cast<float64_t>(widen_lane(lane_lo<H>(b_bits))));
                const auto hi = float_eq(static_cast<float64_t>(widen_lane(lane_hi<H>(a_bits))), static_cast<float64_t>(widen_lane(lane_hi<H>(b_bits))));
                out[i] = lo && hi ? uint8_t{1} : uint8_t{0};
            }
        }

        /**
         * Plain integer equality.
         * @brief Scalar range equal (all integer dtypes).
         * @param out Output UInt8 mask.
         * @param a Left input elements.
         * @param b Right input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         */
        template<typename T>
        void equal_range_int(uint8_t* out, const T* a, const T* b, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = a[i] == b[i] ? uint8_t{1} : uint8_t{0};
            }
        }

        /**
         * Both nibbles equal iff the raw bytes are equal.
         * @brief Scalar range equal (int4x2, raw byte compare).
         * @param out Output UInt8 mask.
         * @param a Left input elements.
         * @param b Right input elements.
         * @param begin First element compared.
         * @param end One past the last element compared.
         */
        void equal_range_i4x2(uint8_t* out, const int4x2_t* a, const int4x2_t* b, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = Intrinsics::detail::bit_cast_memcpy<uint8_t>(a[i]) == Intrinsics::detail::bit_cast_memcpy<uint8_t>(b[i]) ? uint8_t{1} : uint8_t{0};
            }
        }
    }

    /**
     * Dispatches on dtype; both operands share shape/dtype/device (Tensor::equal checks first).
     * @brief CPU elementwise equality into a UInt8 mask.
     * @param a Left input tensor.
     * @param b Right input tensor.
     * @param out Output UInt8 tensor, same shape/device as the inputs.
     */
    void equal_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.device != Device::CPU || b.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(a.device));
        }
        using DType = DataTypes::DType;
        const auto numel = a.numel;
        auto* out_data = static_cast<uint8_t*>(out.data);
        switch (a.dtype) {
            case DType::Float32: case DType::Float64: case DType::Float16: case DType::BFloat16: {
                /**
                 * Launches the scalar/half equal range kernel by T over the whole tensor.
                 * @brief Threaded float equal for one concrete element type.
                 * @param pa Left base pointer, offset already applied.
                 * @param pb Right base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_f = [&]<typename T>(const T* pa, const T* pb) {
                    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                        launch_compare([&](const size_t s, const size_t e) { equal_range_scalar<T>(out_data, pa, pb, s, e); }, numel, a, b, out);
                    } else {
                        launch_compare([&](const size_t s, const size_t e) { equal_range_half<T>(out_data, pa, pb, s, e); }, numel, a, b, out);
                    }
                };
                switch (a.dtype) {
                    case DType::Float32: launch_f(static_cast<const float32_t*>(a.data) + a.offset, static_cast<const float32_t*>(b.data) + b.offset); break;
                    case DType::Float64: launch_f(static_cast<const float64_t*>(a.data) + a.offset, static_cast<const float64_t*>(b.data) + b.offset); break;
                    case DType::Float16: launch_f(static_cast<const float16_t*>(a.data) + a.offset, static_cast<const float16_t*>(b.data) + b.offset); break;
                    case DType::BFloat16: launch_f(static_cast<const bfloat16_t*>(a.data) + a.offset, static_cast<const bfloat16_t*>(b.data) + b.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            case DType::Float16x2: case DType::BFloat16x2: case DType::Int4x2: {
                /**
                 * Launches the packed equal range kernel by P over the whole tensor.
                 * @brief Threaded packed equal for one concrete packed type.
                 * @param pa Left base pointer, offset already applied.
                 * @param pb Right base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_p = [&]<typename P>(const P* pa, const P* pb) {
                    launch_compare([&](const size_t s, const size_t e) { equal_range_packed<P>(out_data, pa, pb, s, e); }, numel, a, b, out);
                };
                if (a.dtype == DType::Float16x2) {
                    launch_p(static_cast<const float16x2_t*>(a.data) + a.offset, static_cast<const float16x2_t*>(b.data) + b.offset);
                } else if (a.dtype == DType::BFloat16x2) {
                    launch_p(static_cast<const bfloat16x2_t*>(a.data) + a.offset, static_cast<const bfloat16x2_t*>(b.data) + b.offset);
                } else {
                    const auto* pa = static_cast<const int4x2_t*>(a.data) + a.offset;
                    const auto* pb = static_cast<const int4x2_t*>(b.data) + b.offset;
                    launch_compare([&](const size_t s, const size_t e) { equal_range_i4x2(out_data, pa, pb, s, e); }, numel, a, b, out);
                }
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                /**
                 * Launches equal_range_int<T> over the whole tensor.
                 * @brief Threaded integer equal for one concrete element type.
                 * @param pa Left base pointer, offset already applied.
                 * @param pb Right base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_i = [&]<typename T>(const T* pa, const T* pb) {
                    launch_compare([&](const size_t s, const size_t e) { equal_range_int<T>(out_data, pa, pb, s, e); }, numel, a, b, out);
                };
                switch (a.dtype) {
                    case DType::Int8: launch_i(static_cast<const int8_t*>(a.data) + a.offset, static_cast<const int8_t*>(b.data) + b.offset); break;
                    case DType::UInt8: launch_i(static_cast<const uint8_t*>(a.data) + a.offset, static_cast<const uint8_t*>(b.data) + b.offset); break;
                    case DType::Int16: launch_i(static_cast<const int16_t*>(a.data) + a.offset, static_cast<const int16_t*>(b.data) + b.offset); break;
                    case DType::UInt16: launch_i(static_cast<const uint16_t*>(a.data) + a.offset, static_cast<const uint16_t*>(b.data) + b.offset); break;
                    case DType::Int32: launch_i(static_cast<const int32_t*>(a.data) + a.offset, static_cast<const int32_t*>(b.data) + b.offset); break;
                    case DType::UInt32: launch_i(static_cast<const uint32_t*>(a.data) + a.offset, static_cast<const uint32_t*>(b.data) + b.offset); break;
                    case DType::Int64: launch_i(static_cast<const int64_t*>(a.data) + a.offset, static_cast<const int64_t*>(b.data) + b.offset); break;
                    case DType::UInt64: launch_i(static_cast<const uint64_t*>(a.data) + a.offset, static_cast<const uint64_t*>(b.data) + b.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
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
     * Both operands must share shape, dtype and device and be contiguous. The compute is queued
     * asynchronously, but the returned tensor is moved out, and the move drains pending work, so
     * equal effectively blocks before returning (same as matmul()); a drained failure is
     * re-queued on the result as a poisoned future, surfacing at its next sync().
     * @brief Elementwise equality mask (UInt8 0/1).
     * @param other Right operand.
     * @return UInt8 tensor, same shape/device as this.
     */
    Tensor Tensor::equal(const Tensor& other) const {
        sync();
        other.sync();
        if (!is_same_shape(other)) {
            throw ErrorHandling::MismatchedShapes("elementwise equal requires identical shapes");
        }
        if (this->dtype != other.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != other.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !other.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        auto out = Tensor(std::vector(this->shape, this->shape + this->ndim), DataTypes::DType::UInt8, this->device, false);
        if (this->device == Device::CPU) {
            detail::equal_cpu(*this, other, out);
        } else if (this->device == Device::GPU) {
            detail::equal_gpu(*this, other, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }
}
