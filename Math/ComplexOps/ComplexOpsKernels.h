/**
 * @file ComplexOpsKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for complex ops (bias_add, relu).
 * softmax_ce has no SIMD kernels: its cost is scalar exp()/log() calls, which
 * the vector tiers cannot accelerate, so it stays on the scalar row kernels.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::ComplexOps {
    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+FMA complex-op kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool complexops_avx2_available();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512F complex-op kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool complexops_avx512_available();

    /**
     * Computes out[i] = in[i] + bias[i % cols] over [begin, end). Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range bias_add (fp32).
     */
    void bias_add_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t, size_t);
    /**
     * Computes out[i] = in[i] + bias[i % cols] over [begin, end). Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range bias_add (fp64).
     */
    void bias_add_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t, size_t);
    /**
     * Computes out[i] = in[i] + bias[i % cols] over [begin, end). Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range bias_add (fp32).
     */
    void bias_add_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t, size_t);
    /**
     * Computes out[i] = in[i] + bias[i % cols] over [begin, end). Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range bias_add (fp64).
     */
    void bias_add_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t, size_t);

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
