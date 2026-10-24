/**
 * @file DTypes.h
 * @author GreenBean
 * @brief Fixed-width dtype aliases, packed dtypes, dtype metadata, and the
 * narrowing predicate used by gradient-dtype changes.
 * @version 0.3
 * @date 2026-10-23
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <type_traits>

#include "Helpers/Errors.h"

#if USE_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#elif USE_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#endif

namespace BeanTensor {
#if USE_HIP
    using float16_t = __half;
    using float16x2_t = __half2;
    using bfloat16_t = __hip_bfloat16;
    using bfloat16x2_t = __hip_bfloat162;
#elif USE_CUDA
    using float16_t = __half;
    using float16x2_t = __half2;
    using bfloat16_t = __nv_bfloat16;
    using bfloat16x2_t = __nv_bfloat162;
#else
    enum class float16_t : uint16_t {};
    enum class float16x2_t : uint32_t {};
    enum class bfloat16_t : uint16_t {};
    enum class bfloat16x2_t : uint32_t {};
#endif
    /**
     * Packed pair of signed 4-bit lanes in one byte.
     * CUDA/HIP do NOT provide 4-bit type for INT (only FP4).
     * @note (lo = bits[3:0], hi = bits[7:4])
     */
    enum class int4x2_t : uint8_t {};
    using float32_t = float;
    using float64_t = double;
}

namespace BeanTensor::DataTypes {

    //std::is_floating_point_v<decltype(dt)>

    /**
     * @brief [FP16x2, BF16x2]
     * Is a type that requires unpacking to be used in standard operations.
     * @tparam T Type to check
     */
    template <class T> constexpr bool requires_unpack_v = std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>;

    /**
     * @brief [FP16, BF16]
     * Is a type that can be packed into a wider type.
     * @tparam T Type to check
     */
    template <class T> constexpr bool can_be_packed_v = std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

    /**
     * @brief [[BF16, FP16, BF16x2, FP16x2]]
     * Is non-standard float kind, introduced by HIP/CUDA
     * @tparam T Type to check
     */
    template <class T> constexpr bool is_nonstandard_float_kind_v =
        std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>;

    /**
     * @brief [[BF16, FP16, BF16x2, FP16x2], [FP32, FP64]]
     * Is any float kind, both native and HIP/CUDA
     * @tparam T Type to check
     */
    template <class T> constexpr bool is_float_kind_v =
    std::is_floating_point_v<T> || is_nonstandard_float_kind_v<T>;

    enum class DType {
        BFloat16,
        BFloat16x2,
        Float16,
        Float16x2,
        Float32,
        Float64,
        Int64,
        Int32,
        Int16,
        Int8,
        UInt64,
        UInt32,
        UInt16,
        UInt8,
        Int4x2
    };

    /**
     * Policy for a gradient-dtype change that would narrow an existing gradient.
     * Reject (the default) refuses any change that can lose values or range;
     * Allow converts the gradient anyway. See is_narrowing_conversion.
     * @brief Whether a gradient-dtype change may narrow.
     */
    enum class GradNarrowing {
        Reject,
        Allow
    };

    inline size_t dtype_bytes(const DType& dt) {
        switch (dt) {
            case DType::Float16: return sizeof(float16_t);
            case DType::Float16x2: return sizeof(float16x2_t);
            case DType::BFloat16: return sizeof(bfloat16_t);
            case DType::BFloat16x2: return sizeof(bfloat16x2_t);
            case DType::Float32: return sizeof(float32_t);
            case DType::Float64: return sizeof(float64_t);
            case DType::Int64: return sizeof(int64_t);
            case DType::Int32: return sizeof(int32_t);
            case DType::Int16: return sizeof(int16_t);
            case DType::Int8: return sizeof(int8_t);
            case DType::UInt32: return sizeof(uint32_t);
            case DType::UInt64: return sizeof(uint64_t);
            case DType::UInt16: return sizeof(uint16_t);
            case DType::UInt8: return sizeof(uint8_t);
            case DType::Int4x2: return sizeof(int4x2_t);
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    inline std::pmr::string dtype_to_string(const DType& dt) {
        switch (dt) {
            case DType::Float16: return "Float16";
            case DType::Float16x2: return "Float16x2";
            case DType::BFloat16: return "BFloat16";
            case DType::BFloat16x2: return "BFloat16x2";
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            case DType::Int64: return "Int64";
            case DType::Int32: return "Int32";
            case DType::Int16: return "Int16";
            case DType::Int8: return "Int8";
            case DType::UInt32: return "UInt32";
            case DType::UInt64: return "UInt64";
            case DType::UInt16: return "UInt16";
            case DType::UInt8: return "UInt8";
            case DType::Int4x2: return "Int4x2";
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * Mapping shared by every DTypeOf specialization.
     * @tparam T C++ element type.
     * @tparam D DType that T is stored as.
     */
    template <class T, DType D> struct DTypeMapping {
        static constexpr bool mapped = true;
        static constexpr DType value = D;
    };

    /**
     * Compile-time map from a C++ element type to its DType. Each DType has
     * exactly one specialization, so the mapping is one-to-one. Any type
     * without a specialization (bool, long long, ...) keeps mapped == false
     * and is rejected at compile time by DTypeMapped.
     *
     * Packed and sub-byte types map to their own DType, not to a lane type:
     * Float16x2/BFloat16x2 are one 4-byte element holding two lanes, and
     * Int4x2 is one byte holding two signed 4-bit lanes. For each of these,
     * dtype_bytes() equals sizeof(T), so a write of T lands on the element it
     * addresses. Individual lanes of a packed element are never addressed.
     * @tparam T Candidate element type.
     */
    template <class T> struct DTypeOf {
        static constexpr bool mapped = false;
    };
    template <> struct DTypeOf<float16_t> : DTypeMapping<float16_t, DType::Float16> { };
    template <> struct DTypeOf<float16x2_t> : DTypeMapping<float16x2_t, DType::Float16x2> { };
    template <> struct DTypeOf<bfloat16_t> : DTypeMapping<bfloat16_t, DType::BFloat16> { };
    template <> struct DTypeOf<bfloat16x2_t> : DTypeMapping<bfloat16x2_t, DType::BFloat16x2> { };
    template <> struct DTypeOf<float32_t> : DTypeMapping<float32_t, DType::Float32> { };
    template <> struct DTypeOf<float64_t> : DTypeMapping<float64_t, DType::Float64> { };
    template <> struct DTypeOf<int64_t> : DTypeMapping<int64_t, DType::Int64> { };
    template <> struct DTypeOf<int32_t> : DTypeMapping<int32_t, DType::Int32> { };
    template <> struct DTypeOf<int16_t> : DTypeMapping<int16_t, DType::Int16> { };
    template <> struct DTypeOf<int8_t> : DTypeMapping<int8_t, DType::Int8> { };
    template <> struct DTypeOf<uint64_t> : DTypeMapping<uint64_t, DType::UInt64> { };
    template <> struct DTypeOf<uint32_t> : DTypeMapping<uint32_t, DType::UInt32> { };
    template <> struct DTypeOf<uint16_t> : DTypeMapping<uint16_t, DType::UInt16> { };
    template <> struct DTypeOf<uint8_t> : DTypeMapping<uint8_t, DType::UInt8> { };
    template <> struct DTypeOf<int4x2_t> : DTypeMapping<int4x2_t, DType::Int4x2> { };

    /**
     * @brief True when element type T has a DType mapping (see DTypeOf).
     * @tparam T Candidate element type.
     */
    template <class T> concept DTypeMapped = DTypeOf<T>::mapped;

    /**
     * @brief DType that element type T is stored as. Only meaningful when DTypeMapped<T>.
     * @tparam T Element type with a DTypeOf specialization.
     */
    template <class T> constexpr DType dtype_of_v = DTypeOf<T>::value;
}
namespace BeanTensor::DataTypes::detail {
    constexpr float32_t FP16_MAX      =  65504.0f;
    constexpr float32_t FP16_MIN      = -65504.0f;
    constexpr float32_t FP16_MIN_POS  =  0.000060975552f;
    constexpr float32_t FP16_EPSILON  =  0.000977f;

    constexpr float32_t BF16_MAX     =  3.38953139e+38f;
    constexpr float32_t BF16_MIN     = -3.38953139e+38f;
    constexpr float32_t BF16_MIN_POS =  1.175494351e-38f;
    constexpr float32_t BF16_EPSILON =  0.0078125f;

    // numeric_limits<float32_t>::max()/lowest(); nvcc rejects numeric_limits in device code.
    constexpr float64_t FP32_MAX_D =  3.4028234663852886e+38;
    constexpr float64_t FP32_MIN_D = -3.4028234663852886e+38;

    /**
     * Significand width (implicit bit included), largest exponent and smallest
     * subnormal exponent of a non-packed float dtype: the three numbers that
     * decide whether a float -> float conversion is exact.
     */
    struct FloatLayout {
        int32_t significand_bits;
        int32_t max_exponent;
        int32_t min_subnormal_exponent;
    };

    /**
     * Signedness and bit width of a non-packed integer dtype.
     */
    struct IntLayout {
        bool is_signed;
        int32_t bits;
    };

    /**
     * @brief Layout of a non-packed float dtype.
     * @param dt Float16, BFloat16, Float32 or Float64.
     * @return The float layout.
     * @throws ErrorHandling::NotImplemented for any other dtype.
     */
    inline FloatLayout float_layout(const DType& dt) {
        switch (dt) {
            case DType::Float16: return {11, 15, -24};
            case DType::BFloat16: return {8, 127, -133};
            case DType::Float32: return {24, 127, -149};
            case DType::Float64: return {53, 1023, -1074};
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * @brief Signedness and width of a non-packed integer dtype.
     * @param dt Int8..Int64 or UInt8..UInt64.
     * @return The integer layout.
     * @throws ErrorHandling::NotImplemented for any other dtype.
     */
    inline IntLayout int_layout(const DType& dt) {
        switch (dt) {
            case DType::Int8: return {true, 8};
            case DType::Int16: return {true, 16};
            case DType::Int32: return {true, 32};
            case DType::Int64: return {true, 64};
            case DType::UInt8: return {false, 8};
            case DType::UInt16: return {false, 16};
            case DType::UInt32: return {false, 32};
            case DType::UInt64: return {false, 64};
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * @brief Is dt one of the non-packed float dtypes (Float16, BFloat16, Float32, Float64)?
     * @param dt Dtype to test.
     * @return True for a non-packed float dtype.
     */
    inline bool is_float_dtype(const DType& dt) {
        return dt == DType::Float16 || dt == DType::BFloat16 || dt == DType::Float32 || dt == DType::Float64;
    }

    /**
     * @brief Is dt a packed dtype (Float16x2, BFloat16x2 or Int4x2)?
     * @param dt Dtype to test.
     * @return True for a packed dtype.
     */
    inline bool is_packed_dtype(const DType& dt) {
        return dt == DType::Float16x2 || dt == DType::BFloat16x2 || dt == DType::Int4x2;
    }
}

namespace BeanTensor::DataTypes {

    /**
     * True when converting a value of dtype src to dtype dst can lose value or
     * range, so the destination cannot hold every value the source can. Rules:
     * - same dtype: never narrowing;
     * - a packed source (Float16x2, BFloat16x2, Int4x2) keeps only its low lane
     *   when converted to anything else: always narrowing;
     * - an unpacked source into a packed destination duplicates the value into
     *   both lanes, so it is narrowing exactly when it is narrowing into the
     *   lane type (Int4x2 lanes are 4-bit: anything but Int4x2 is narrowing);
     * - float into integer always truncates: narrowing;
     * - float into float is exact when the source's significand, max exponent
     *   and smallest subnormal exponent all fit the destination's;
     * - integer into float is exact when the integer's bit width fits the
     *   float's significand (Int8 fits Float16/BFloat16; Int32 fits Float64 only;
     *   Int64 fits no float exactly);
     * - integer into integer is exact when the destination can hold every source
     *   value: same signedness with at least the source width, unsigned into
     *   signed with strictly more bits, signed into unsigned never.
     * @brief Does src -> dst lose values or range?
     * @param src Source dtype.
     * @param dst Destination dtype.
     * @return True when the conversion can lose data.
     * @throws ErrorHandling::NotImplemented for a dtype outside the supported set.
     */
    inline bool is_narrowing_conversion(const DType& src, const DType& dst) {
        if (src == dst) {
            return false;
        }
        if (detail::is_packed_dtype(src)) {
            return true;
        }
        if (dst == DType::Int4x2) {
            return true;
        }
        if (dst == DType::Float16x2) {
            return is_narrowing_conversion(src, DType::Float16);
        }
        if (dst == DType::BFloat16x2) {
            return is_narrowing_conversion(src, DType::BFloat16);
        }
        if (detail::is_float_dtype(dst)) {
            if (detail::is_float_dtype(src)) {
                const auto from = detail::float_layout(src);
                const auto to = detail::float_layout(dst);
                return !(
                    from.significand_bits <= to.significand_bits &&
                    from.max_exponent <= to.max_exponent &&
                    from.min_subnormal_exponent >= to.min_subnormal_exponent
                );
            }
            return detail::int_layout(src).bits > detail::float_layout(dst).significand_bits;
        }
        if (detail::is_float_dtype(src)) {
            return true;
        }
        const auto from = detail::int_layout(src);
        const auto to = detail::int_layout(dst);
        if (from.is_signed && !to.is_signed) {
            return true;
        }
        if (from.is_signed == to.is_signed) {
            return to.bits < from.bits;
        }
        return to.bits <= from.bits;
    }
}