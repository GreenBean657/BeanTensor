#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

//&& defined(__clang__)
#if defined(USE_HIP) && defined(__clang__)
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#elif defined(USE_CUDA)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace BeanTensor::Casting {
#if defined(USE_HIP) && defined(__clang__)
    using float16_t  = __half;
    using bfloat16_t = __hip_bfloat16;

    inline float to_float(const bfloat16_t x) { return static_cast<float>(x); }
    inline float to_float(const float16_t x)  { return __half2float(x); }

#elif defined(USE_CUDA)
    using float16_t  = __half;
    using bfloat16_t = __nv_bfloat16;
    inline float to_float(const bfloat16_t x) { return __bfloat162float(x); }
    inline float to_float(const float16_t x)  { return __half2float(x); }
#else
    enum class float16_t  : uint16_t {};
    enum class bfloat16_t : uint16_t {};
    inline uint16_t raw(float16_t  x) { return static_cast<uint16_t>(x); }
    inline uint16_t raw(bfloat16_t x) { return static_cast<uint16_t>(x); }

    inline float16_t  as_f16 (uint16_t x) { return static_cast<float16_t> (x); }
    inline bfloat16_t as_bf16(uint16_t x) { return static_cast<bfloat16_t>(x); }
#endif
    template <class T> constexpr bool is_float_kind_v =
    std::is_floating_point_v<T>
    || std::is_same_v<T, bfloat16_t>
    || std::is_same_v<T, float16_t>;

    using float32_t = float;
    using float64_t = double;
    enum class DType {
        BFloat16,
        Float16,
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




    inline size_t dtype_size(const DType& dt) {
        switch (dt) {
            case DType::Float16: return sizeof(float16_t);
            case DType::BFloat16: return sizeof(bfloat16_t);
            case DType::Float32: return sizeof(float);
            case DType::Float64: return sizeof(double);
            case DType::Int32:   return sizeof(int32_t);
            case DType::Int64:   return sizeof(int64_t);
            case DType::Int16:   return sizeof(int16_t);
            case DType::Int8:    return sizeof(int8_t);
            case DType::UInt32:  return sizeof(uint32_t);
            case DType::UInt64:  return sizeof(uint64_t);
            case DType::UInt16:  return sizeof(uint16_t);
            case DType::UInt8:   return sizeof(uint8_t);
                
        }
        return 0;
    }

    inline std::pmr::string dtype_to_string(const DType& dt) {
        switch (dt) {
            case DType::Float16:  return "Float16";
            case DType::BFloat16:  return "BFloat16";
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            case DType::Int32:   return "Int32";
            case DType::Int64:   return "Int64";
            case DType::Int16:   return "Int16";
            case DType::Int8:    return "Int8";
            case DType::UInt32:  return "UInt32";
            case DType::UInt64:  return "UInt64";
            case DType::UInt16:  return "UInt16";
            case DType::UInt8:   return "UInt8";
        }
        return "Unknown";
    }

    inline bool dtype_is_float(const DType& dt) {
        switch (dt) {
            case DType::Float16:  return true;
            case DType::BFloat16:  return true;
            case DType::Float32:  return true;
            case DType::Float64:  return true;
            default: return false;
        }
    }
}

namespace BeanTensor::Casting::detail {

    constexpr float32_t F16_MAX      =  65504.0f;
    constexpr float32_t F16_MIN      = -65504.0f;
    constexpr float32_t F16_MIN_POS  =  0.000060975552f;
    constexpr float32_t F16_EPSILON  =  0.000977f;

    constexpr float32_t BF16_MAX     =  3.38953139e+38f;
    constexpr float32_t BF16_MIN     = -3.38953139e+38f;
    constexpr float32_t BF16_MIN_POS =  1.175494351e-38f;
    constexpr float32_t BF16_EPSILON =  0.0078125f;
}
