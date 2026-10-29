/**
 * @file SoftmaxHipShared.h
 * @author GreenBean
 * @brief HIP/ROCm-side shared helpers for the split standalone softmax
 * translation units (forward in SoftmaxHIP.cpp, backward in SoftmaxGHIP.cpp):
 * device-side lane widening/narrowing and row reduction primitives.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#if defined(USE_HIP)
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "Math/Activations/ActivationsHipShared.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::softmax_gpu_helpers {
    using namespace activations_gpu;

    __device__ __forceinline__ float32_t acc_exp(const float32_t v) { return expf(v); }
    __device__ __forceinline__ float64_t acc_exp(const float64_t v) { return exp(v); }

    template<typename T, typename Acc>
    __device__ __forceinline__ Acc sm_widen(const T* row, const size_t j) {
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            return row[j];
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return __half2float(row[j]);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return __bfloat162float(row[j]);
        } else if constexpr (std::is_same_v<T, float16x2_t>) {
            const auto v = row[j >> 1];
            return (j & 1) != 0 ? __half2float(__high2half(v)) : __half2float(__low2half(v));
        } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
            const auto v = row[j >> 1];
            return (j & 1) != 0 ? __bfloat162float(__high2bfloat16(v)) : __bfloat162float(__low2bfloat16(v));
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            return static_cast<float64_t>((j & 1) != 0 ? dev_int4_hi(row[j >> 1]) : dev_int4_lo(row[j >> 1]));
        } else {
            return static_cast<float64_t>(row[j]);
        }
    }

    template<typename T>
    __device__ __forceinline__ float64_t int_dtype_lowest();
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<int8_t>() { return -128.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<uint8_t>() { return 0.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<int16_t>() { return -32768.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<uint16_t>() { return 0.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<int32_t>() { return -2147483648.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<uint32_t>() { return 0.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<int64_t>() { return -9223372036854775808.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_lowest<uint64_t>() { return 0.0; }

    template<typename T>
    __device__ __forceinline__ float64_t int_dtype_max();
    template<> __device__ __forceinline__ float64_t int_dtype_max<int8_t>() { return 127.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<uint8_t>() { return 255.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<int16_t>() { return 32767.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<uint16_t>() { return 65535.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<int32_t>() { return 2147483647.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<uint32_t>() { return 4294967295.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<int64_t>() { return 9223372036854775807.0; }
    template<> __device__ __forceinline__ float64_t int_dtype_max<uint64_t>() { return 18446744073709551615.0; }

    template<typename T, typename Acc>
    __device__ __forceinline__ T sm_narrow(const Acc v) {
        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
            return v;
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return __float2half_rn(static_cast<float32_t>(v));
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return dev_narrow_bf16(static_cast<float32_t>(v));
        } else if constexpr (std::is_same_v<T, float16x2_t>) {
            const auto lane = __float2half_rn(static_cast<float32_t>(v));
            return __halves2half2(lane, lane);
        } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
            const auto lane = dev_narrow_bf16(static_cast<float32_t>(v));
            return __halves2bfloat162(lane, lane);
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            auto f = static_cast<float64_t>(v);
            if (isnan(f)) {
                f = 0.0;
            }
            if (f <= -8.0) {
                f = -8.0;
            } else if (f >= 7.0) {
                f = 7.0;
            }
            return dev_pack_int4(static_cast<int32_t>(f), static_cast<int32_t>(f));
        } else {
            auto f = static_cast<float64_t>(v);
            if (isnan(f)) {
                f = 0.0;
            }
            const auto lo = int_dtype_lowest<T>();
            const auto hi = int_dtype_max<T>();
            if (f <= lo) {
                f = lo;
            } else if (f >= hi) {
                f = hi;
            }
            return static_cast<T>(f);
        }
    }

    template<typename Acc>
    __device__ __forceinline__ Acc acc_lowest() {
        if constexpr (std::is_same_v<Acc, float32_t>) {
            return -3.4028234663852886e+38f;
        } else {
            return -1.7976931348623157e+308;
        }
    }
}

#endif
