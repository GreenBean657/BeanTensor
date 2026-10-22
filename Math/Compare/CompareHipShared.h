/**
 * @file CompareHipShared.h
 * @author GreenBean
 * @brief HIP/ROCm-side shared helpers for the comparison translation units
 * (equal): grid constants and grid sizing.
 * Divergent twin of Math/Compare/CompareCudaShared.h — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical: same IEEE equality semantics (NaN never equal).
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace BeanTensor::Tensors::detail::compare_gpu {
    /**
     * @brief Threads per block for every comparison kernel.
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
}
