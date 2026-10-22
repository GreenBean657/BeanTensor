/**
 * @file SumKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for sum.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Reductions {
    /**
     * @brief Runtime probe: are the AVX2+FMA sum kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool sum_avx2_available();

    /**
     * @brief Runtime probe: are the AVX512F sum kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool sum_avx512_available();

    // Sum range kernels: tree reduction over [begin, end); order differs from the
    // scalar path, so float sums are not bit-identical across dispatch tiers.
    /**
     * @brief AVX2 tree-reduction sum over a range (FP32).
     * @return Sum of the elements in [begin, end).
     */
    [[nodiscard]] float32_t sum_f32_range_avx2(const float32_t*, size_t, size_t);
    /**
     * @brief AVX2 tree-reduction sum over a range (FP64).
     * @return Sum of the elements in [begin, end).
     */
    [[nodiscard]] float64_t sum_f64_range_avx2(const float64_t*, size_t, size_t);
    /**
     * @brief AVX512F tree-reduction sum over a range (FP32).
     * @return Sum of the elements in [begin, end).
     */
    [[nodiscard]] float32_t sum_f32_range_avx512(const float32_t*, size_t, size_t);
    /**
     * @brief AVX512F tree-reduction sum over a range (FP64).
     * @return Sum of the elements in [begin, end).
     */
    [[nodiscard]] float64_t sum_f64_range_avx512(const float64_t*, size_t, size_t);
}
