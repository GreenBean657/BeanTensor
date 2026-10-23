/**
 * @file ReductionsHipShared.h
 * @author GreenBean
 * @brief HIP shared helpers for the split reduction GPU backends (sum, argmax):
 * block sizing, grid-stride launcher, BF16 narrowing, int4x2 lane helpers.
 * Divergent twin of Math/Reductions/ReductionsCudaShared.h — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN-wins argmax rule, same uint64 wrap integer sums.
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#if defined(USE_HIP)
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors::detail::reductions_gpu {
    /**
     * @brief Threads per block for every kernel here (also the shared-memory tree width).
     */
    constexpr size_t BT_GPU_THREADS = 256;
    /**
     * @brief Upper bound on the grid size; grid-stride loops cover the remainder.
     */
    constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

    /**
     * Clamped to [1, BT_GPU_MAX_BLOCKS].
     * @brief Block count for a grid-stride kernel.
     * @param numel Element count to cover.
     * @return Number of blocks to launch.
     */
    [[nodiscard]] inline __host__ size_t grid_for(const size_t numel) {
        const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
        return blocks < 1 ? 1 : blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks;
    }

    /**
     * FP32-subnormal inputs flush to signed zero, everything else rounds RNE (mirrors the CPU unaccel pathway).
     * @brief FP32 -> BF16 narrowing with CPU-matching flush.
     * @param f FP32 value.
     * @return The value narrowed to bf16.
     */
    __device__ __forceinline__ bfloat16_t dev_narrow_bf16(const float32_t f) {
        if (
            const auto bits = __builtin_bit_cast(uint32_t, f);
            (bits & 0x7F800000u) == 0u
        ) {
            return __float2bfloat16(__builtin_bit_cast(float32_t, bits & 0x80000000u)); // RNE (CUDA parity: __float2bfloat16_rn)
        }
        return __float2bfloat16(f); // RNE (CUDA parity: __float2bfloat16_rn)
    }

    /**
     * @brief Signed low lane of an int4x2 byte (bits[3:0], sign-extended).
     * @param v Packed int4x2 byte.
     * @return Lane value in [-8, 7].
     */
    __device__ __forceinline__ int8_t dev_i4_lo(const int4x2_t v) {
        const auto n = static_cast<int8_t>(__builtin_bit_cast(uint8_t, v) & 0x0Fu);
        return n > 7 ? static_cast<int8_t>(n - 16) : n;
    }

    /**
     * @brief Signed high lane of an int4x2 byte (bits[7:4], sign-extended).
     * @param v Packed int4x2 byte.
     * @return Lane value in [-8, 7].
     */
    __device__ __forceinline__ int8_t dev_i4_hi(const int4x2_t v) {
        const auto n = static_cast<int8_t>(__builtin_bit_cast(uint8_t, v) >> 4);
        return n > 7 ? static_cast<int8_t>(n - 16) : n;
    }

    /**
     * Each lane keeps only its low 4 bits, so out-of-range values wrap mod 16.
     * @brief Pack two signed lanes into an int4x2 byte.
     * @param lo Low lane value.
     * @param hi High lane value.
     * @return The packed int4x2 byte.
     */
    __device__ __forceinline__ int4x2_t dev_i4_pack(const int8_t lo, const int8_t hi) {
        const auto lo_bits = static_cast<uint8_t>(lo) & 0x0Fu;
        const auto hi_bits = static_cast<uint8_t>(hi) & 0x0Fu;
        return __builtin_bit_cast(int4x2_t, static_cast<uint8_t>(lo_bits | static_cast<uint8_t>(hi_bits << 4)));
    }
}

#endif
