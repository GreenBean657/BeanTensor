/**
 * @file SgdUpdateKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for sgd_update.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Optimization {
    /**
     * @brief Runtime probe: are the AVX2+FMA sgd_update kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool sgd_update_avx2_available();

    /**
     * @brief Runtime probe: are the AVX512F sgd_update kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool sgd_update_avx512_available();

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
}
