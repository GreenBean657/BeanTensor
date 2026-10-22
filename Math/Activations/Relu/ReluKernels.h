/**
 * @file ReluKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for relu.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Activations {
    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+FMA relu kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool relu_avx2_available();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512F relu kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool relu_avx512_available();

    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i] over [begin, end). max(zero, x) keeps -0.0 and propagates NaN,
     * matching the scalar kernels.
     * @brief AVX2 range relu (fp32).
     */
    void relu_f32_range_avx2(float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i] over [begin, end). max(zero, x) keeps -0.0 and propagates NaN,
     * matching the scalar kernels.
     * @brief AVX2 range relu (fp64).
     */
    void relu_f64_range_avx2(float64_t*, const float64_t*, size_t, size_t);
    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i] over [begin, end). max(zero, x) keeps -0.0 and propagates NaN,
     * matching the scalar kernels.
     * @brief AVX512F range relu (fp32).
     */
    void relu_f32_range_avx512(float32_t*, const float32_t*, size_t, size_t);
    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i] over [begin, end). max(zero, x) keeps -0.0 and propagates NaN,
     * matching the scalar kernels.
     * @brief AVX512F range relu (fp64).
     */
    void relu_f64_range_avx512(float64_t*, const float64_t*, size_t, size_t);
}
