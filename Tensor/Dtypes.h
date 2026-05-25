#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#include "../../../../opt/rocm-7.2.0/include/hip/amd_detail/amd_hip_fp16.h"

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

    inline float to_float(const bfloat16_t x) {
        const uint32_t expanded = static_cast<uint32_t>(raw(x)) << 16;
        float result;
        std::memcpy(&result, &expanded, sizeof(result));
        return result;
    }
    inline float to_float(const float16_t x) {
        const uint16_t bits = raw(x);
        const uint32_t sign     = (bits & 0x8000u) << 16;
        const uint32_t exponent = (bits & 0x7C00u) >> 10;
        const uint32_t mantissa = (bits & 0x03FFu);
        uint32_t expanded;
        if (exponent == 31u) {
            expanded = sign | 0x7F800000u | (mantissa << 13);
        } else if (exponent == 0u) {
            expanded = sign | (mantissa << 13);
        } else {
            expanded = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &expanded, sizeof(result));
        return result;
    }
#endif
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

    inline bool dtype_is_float(const DType dt) {
        return
           (dt == DType::Float16)
        || (dt == DType::BFloat16)
        || (dt == DType::Float32)
        || (dt == DType::Float64);
    }

    inline bool dtype_is_int(const DType dt) {
        return (dt == DType::Int8)
            || (dt == DType::Int16)
            || (dt == DType::Int32)
            || (dt == DType::Int64);
    }

    inline bool dtype_is_uint(const DType dt) {
        return (dt == DType::UInt8)
        || (dt == DType::UInt16)
        || (dt == DType::UInt32)
        || (dt == DType::UInt64);
    }

}
namespace BeanTensor::Casting::detail {
    inline uint8_t dtype_rank(const DType& dt) {
        switch (dt) {
            case DType::Float16:  return 0;
            case DType::BFloat16:  return 1;
            case DType::Float32: return 2;
            case DType::Float64: return 3;

            case DType::Int8:   return 0;
            case DType::Int16:  return 1;
            case DType::Int32:  return 2;
            case DType::Int64:  return 3;

            case DType::UInt8:  return 10;
            case DType::UInt16: return 11;
            case DType::UInt32: return 12;
            case DType::UInt64: return 13;


            default: {
                __builtin_unreachable();
            }
        }
    }
    inline bool operator<(const DType& a, const DType& b) {
        if (a == b) return false;

        if (dtype_is_int(a) && dtype_is_int(b)) {
            return dtype_rank(a) < dtype_rank(b);
        }
        if (dtype_is_uint(a) && dtype_is_uint(b)) {
            return dtype_rank(a) < dtype_rank(b);
        }
        if (dtype_is_float(a) && dtype_is_float(b)) {
            return dtype_rank(a) < dtype_rank(b);
        }

        if ((dtype_is_int(a) && dtype_is_float(b)) || (dtype_is_float(a) && dtype_is_int(b))) {
            // Float always wins over int
            return dtype_is_int(a);
        }
        if ((dtype_is_uint(a) && dtype_is_float(b)) || (dtype_is_float(a) && dtype_is_uint(b))) {
            // Float always wins over uint
            return dtype_is_uint(a);
        }
        if ((dtype_is_int(a) && dtype_is_uint(b)) || (dtype_is_uint(a) && dtype_is_int(b))) {
            // Int vs uint: signed int wins only if it has strictly more bits (can represent
            // all uint values); otherwise unsigned wins (larger positive range at equal width).
            DType int_dt  = dtype_is_int(a)  ? a : b;
            DType uint_dt = dtype_is_uint(a) ? a : b;
            bool int_wins = dtype_size(int_dt) > dtype_size(uint_dt);
            return int_wins ? dtype_is_uint(a) : dtype_is_int(a);
        }

        return false;
    }
}
