/**
 * @file DropoutCudaShared.h
 * @brief CUDA-side shared helpers for dropout mask construction: grid constants
 * and device-side threshold-scale helpers built from existing lane primitives.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#if defined(USE_CUDA)
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::dropout_gpu {

    /**
     * @brief Threads per block for every dropout mask kernel.
     */
    constexpr size_t BT_GPU_THREADS = 256;

    /**
     * @brief Upper bound on the block count of one launch.
     */
    constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

    /**
     * @brief Block count for a grid-stride kernel over numel elements.
     */
    [[nodiscard]] inline __host__ size_t grid_for(const size_t numel) {
        const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
        return blocks < 1 ? 1 : blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks;
    }

    /**
     * @brief Device FP32 -> BF16 narrowing mirroring the CPU unaccel pathway.
     */
    __device__ __forceinline__ bfloat16_t dev_narrow_bf16(const float32_t f) {
        if (
            const auto bits = __builtin_bit_cast(uint32_t, f);
            (bits & 0x7F800000u) == 0u
        ) {
            return __float2bfloat16_rn(__builtin_bit_cast(float32_t, bits & 0x80000000u));
        }
        return __float2bfloat16_rn(f);
    }

    /**
     * @brief Native float threshold-scale on device.
     */
    template<typename T>
    __device__ __forceinline__ T dev_mask_native(const T rand, const float64_t p, const float64_t scale) {
        return rand >= static_cast<T>(p) ? static_cast<T>(scale) : T{0};
    }

    /**
     * @brief Half-kind threshold-scale on device.
     */
    template<typename H>
    __device__ __forceinline__ H dev_mask_half(const H rand, const float64_t p, const float64_t scale) {
        if constexpr (std::is_same_v<H, float16_t>) {
            const auto wide = __half2float(rand);
            const auto out = wide >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
            return __float2half_rn(out);
        } else {
            const auto wide = __bfloat162float(rand);
            const auto out = wide >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
            return dev_narrow_bf16(out);
        }
    }

    /**
     * @brief Packed half-kind threshold-scale per lane on device.
     */
    template<typename P>
    __device__ __forceinline__ P dev_mask_packed(const P rand, const float64_t p, const float64_t scale) {
        if constexpr (std::is_same_v<P, float16x2_t>) {
            const auto lo = __half2float(__low2half(rand));
            const auto hi = __half2float(__high2half(rand));
            const auto lo_out = lo >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
            const auto hi_out = hi >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
            return __halves2half2(__float2half_rn(lo_out), __float2half_rn(hi_out));
        } else {
            const auto lo = __bfloat162float(__low2bfloat16(rand));
            const auto hi = __bfloat162float(__high2bfloat16(rand));
            const auto lo_out = lo >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
            const auto hi_out = hi >= static_cast<float32_t>(p) ? static_cast<float32_t>(scale) : 0.0f;
            return __halves2bfloat162(dev_narrow_bf16(lo_out), dev_narrow_bf16(hi_out));
        }
    }

}

#endif
