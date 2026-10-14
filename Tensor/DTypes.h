#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <utility>

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
        UInt8
    };

    inline size_t dtype_bytes(const DType& dt) {
        switch (dt) {
            case DType::Float16: return sizeof(float16_t);
            case DType::Float16x2: return sizeof(float16x2_t);
            case DType::BFloat16: return sizeof(bfloat16_t);
            case DType::BFloat16x2: return sizeof(bfloat16x2_t);
            case DType::Float32: return sizeof(float32_t);
            case DType::Float64: return sizeof(float64_t);
            case DType::Int64:   return sizeof(int64_t);
            case DType::Int32:   return sizeof(int32_t);
            case DType::Int16:   return sizeof(int16_t);
            case DType::Int8:    return sizeof(int8_t);
            case DType::UInt32:  return sizeof(uint32_t);
            case DType::UInt64:  return sizeof(uint64_t);
            case DType::UInt16:  return sizeof(uint16_t);
            case DType::UInt8:   return sizeof(uint8_t);
        }
        assert(false && "Invalid DType"); std::unreachable();
    }

    inline std::pmr::string dtype_to_string(const DType& dt) {
        switch (dt) {
            case DType::Float16:  return "Float16";
            case DType::Float16x2: return "Float16x2";
            case DType::BFloat16:  return "BFloat16";
            case DType::BFloat16x2: return "BFloat16x2";
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            case DType::Int64:   return "Int64";
            case DType::Int32:   return "Int32";
            case DType::Int16:   return "Int16";
            case DType::Int8:    return "Int8";
            case DType::UInt32:  return "UInt32";
            case DType::UInt64:  return "UInt64";
            case DType::UInt16:  return "UInt16";
            case DType::UInt8:   return "UInt8";
        }
        assert(false && "Invalid DType"); std::unreachable();
    }
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
}