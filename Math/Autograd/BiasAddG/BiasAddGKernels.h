/**
 * @file BiasAddGKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for bias_add backward column sums (fp32/fp64).
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Autograd {
    [[nodiscard]] bool bias_add_g_avx2_available();

    [[nodiscard]] bool bias_add_g_avx512_available();

    void bias_add_g_f32_colsum_avx2(float32_t*, const float32_t*, size_t, size_t, size_t);
    void bias_add_g_f64_colsum_avx2(float64_t*, const float64_t*, size_t, size_t, size_t);
    void bias_add_g_f32_colsum_avx512(float32_t*, const float32_t*, size_t, size_t, size_t);
    void bias_add_g_f64_colsum_avx512(float64_t*, const float64_t*, size_t, size_t, size_t);
}
