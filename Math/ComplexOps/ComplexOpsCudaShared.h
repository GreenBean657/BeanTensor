/**
 * @file ComplexOpsCudaShared.h
 * @author GreenBean
 * @brief CUDA-side shared helpers for the split complex-op translation units
 * (bias_add, relu, softmax_ce): grid constants, grid sizing, device FP32->BF16
 * narrowing, and int4x2 pack/unpack helpers.
 * Divergent twin of Math/ComplexOps/ComplexOpsHipShared.h — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same fp32/fp64 compute widths, same RNE narrowing,
 * same NaN propagation, same wrap integer adds.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

namespace BeanTensor::Tensors::detail::complexops_gpu {
    /**
     * Also the shared-memory reduction width of the softmax_ce kernels (power of two required).
     * @brief Threads per block for every complex-op kernel.
     */
    constexpr size_t BT_GPU_THREADS = 256;
    /**
     * Grid-stride loops cover any remainder, so large inputs never need more blocks than this.
     * @brief Upper bound on the block count of one launch.
     */
    constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

    /**
     * Clamped to [1, BT_GPU_MAX_BLOCKS]; the grid-stride loop covers any remainder.
     * @brief Block count for a grid-stride kernel over numel elements.
     * @param numel Element count the kernel walks.
     * @return Block count of BT_GPU_THREADS wide blocks.
     */
    [[nodiscard]] inline __host__ size_t grid_for(const size_t numel) {
        const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
        return blocks < 1 ? 1 : (blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks);
    }

    /**
     * Narrow FP32 to BF16 mirroring the CPU unaccel pathway: FP32-subnormal
     * inputs flush to signed zero, everything else rounds RNE.
     * @brief FP32 -> BF16 narrowing with CPU-matching flush.
     * @param f FP32 value to narrow.
     * @return BF16 value, RNE, subnormal inputs flushed to signed zero.
     */
    __device__ __forceinline__ bfloat16_t dev_narrow_bf16(const float32_t f) {
        const auto bits = __builtin_bit_cast(uint32_t, f);
        if ((bits & 0x7F800000u) == 0u) {
            return __float2bfloat16_rn(__builtin_bit_cast(float32_t, bits & 0x80000000u));
        }
        return __float2bfloat16_rn(f);
    }

    // --------------------------- int4x2 helpers ---------------------------

    /**
     * Low 4 bits, sign-extended from bit 3.
     * @brief Sign-extended low nibble of a packed int4 pair.
     * @param v Packed int4 pair.
     * @return Low nibble in [-8, 7].
     */
    __device__ __forceinline__ int32_t dev_int4_lo(const int4x2_t v) {
        const auto n = static_cast<int32_t>(static_cast<uint8_t>(v) & 0x0Fu);
        return n >= 8 ? n - 16 : n;
    }

    /**
     * High 4 bits, sign-extended from bit 3.
     * @brief Sign-extended high nibble of a packed int4 pair.
     * @param v Packed int4 pair.
     * @return High nibble in [-8, 7].
     */
    __device__ __forceinline__ int32_t dev_int4_hi(const int4x2_t v) {
        const auto n = static_cast<int32_t>(static_cast<uint8_t>(v) >> 4);
        return n >= 8 ? n - 16 : n;
    }

    /**
     * Only the low 4 bits of each input survive, so out-of-range sums wrap.
     * @brief Pack two signed values into an int4 pair (modular, low 4 bits each).
     * @param lo Value for the low nibble.
     * @param hi Value for the high nibble.
     * @return Packed int4 pair.
     */
    __device__ __forceinline__ int4x2_t dev_pack_int4(const int32_t lo, const int32_t hi) {
        return static_cast<int4x2_t>(static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
    }
}
