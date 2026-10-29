/**
 * @file SoftmaxCudaShared.h
 * @author GreenBean
 * @brief CUDA-side shared helpers for the split standalone softmax translation
 * units (forward in SoftmaxCUDA.cu, backward in SoftmaxGCUDA.cu): device-side
 * lane widening/narrowing and row reduction primitives.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#if defined(USE_CUDA)
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Activations/ActivationsCudaShared.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::softmax_gpu_helpers {
    using namespace activations_gpu;

    /**
     * @brief exp() in the accumulator width (expf for fp32, exp for fp64).
     */
    __device__ __forceinline__ float32_t acc_exp(const float32_t v) { return expf(v); }
    __device__ __forceinline__ float64_t acc_exp(const float64_t v) { return exp(v); }

    /**
     * fp16/bf16 and packed x2 (lanes unpack as consecutive classes) widen to fp32.
     * fp64 and the integer dtypes (int4x2 unpacks nibbles as consecutive classes) widen to fp64.
     * @brief Widen class j of a logits row to the accumulator.
     * @param row First element of the logits row.
     * @param j Effective class index within the row.
     * @return Class j widened to Acc.
     */
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

    /**
     * Device-safe integer bounds: std::numeric_limits is host-only in nvcc, so
     * the integer clamping path uses these explicit specializations.
     * @brief Lowest finite value of a plain integer dtype, as fp64.
     */
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

    /**
     * @brief Highest finite value of a plain integer dtype, as fp64.
     */
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

    /**
     * @brief Store one accumulated probability value into the output dtype.
     * Floats RNE; integers clamp/truncate and map NaN to 0.
     */
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

    /**
     * Literal constants, since nvcc rejects numeric_limits in device code.
     * @brief Lowest finite value of an accumulator type.
     */
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
