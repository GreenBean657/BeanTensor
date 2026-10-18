/**
 * @file BasicOpsKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for basic ops (sgd_update, sum).
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::BasicOps {
    /**
     * @brief Runtime probe: are the AVX2+FMA basic-op kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool basicops_avx2_available();

    /**
     * @brief Runtime probe: are the AVX512F basic-op kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool basicops_avx512_available();

    // In-place SGD row kernels: w[i] = fma(neg_lr, g[i], w[i]) over [begin, end).
    /**
     * @brief AVX2+FMA in-place SGD range kernel (FP32).
     */
    void sgd_f32_range_avx2(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * @brief AVX2+FMA in-place SGD range kernel (FP64).
     */
    void sgd_f64_range_avx2(float64_t*, const float64_t*, float64_t, size_t, size_t);
    /**
     * @brief AVX512F in-place SGD range kernel (FP32).
     */
    void sgd_f32_range_avx512(float32_t*, const float32_t*, float32_t, size_t, size_t);
    /**
     * @brief AVX512F in-place SGD range kernel (FP64).
     */
    void sgd_f64_range_avx512(float64_t*, const float64_t*, float64_t, size_t, size_t);

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
