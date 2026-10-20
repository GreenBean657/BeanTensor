/**
 * @file BiasAddKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for bias_add.
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::ComplexOps {
    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+FMA bias_add kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool bias_add_avx2_available();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512F bias_add kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool bias_add_avx512_available();

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
}
