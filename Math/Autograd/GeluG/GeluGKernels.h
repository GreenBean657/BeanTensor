/**
 * @file GeluGKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for GELU backward:
 *   dx = dy * (0.5*(1+erf(x/sqrt(2))) + x*exp(-0.5*x^2)/sqrt(2*pi)).
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Autograd {
    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+FMA GELU backward kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool gelu_g_avx2_available();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512F GELU backward kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool gelu_g_avx512_available();

    /**
     * Computes dx[i] = dy[i] * gelu'(x[i]) over [begin, end), 8 lanes per step.
     * @brief AVX2 range GELU backward (fp32).
     */
    void gelu_g_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t);

    /**
     * Computes dx[i] = dy[i] * gelu'(x[i]) over [begin, end), 4 lanes per step.
     * @brief AVX2 range GELU backward (fp64).
     */
    void gelu_g_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t);

    /**
     * Computes dx[i] = dy[i] * gelu'(x[i]) over [begin, end), 16 lanes per step.
     * @brief AVX512F range GELU backward (fp32).
     */
    void gelu_g_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t);

    /**
     * Computes dx[i] = dy[i] * gelu'(x[i]) over [begin, end), 8 lanes per step.
     * @brief AVX512F range GELU backward (fp64).
     */
    void gelu_g_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
}
