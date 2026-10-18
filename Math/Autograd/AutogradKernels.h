/**
 * @file AutogradKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for autograd backward ops (grad
 * accumulation, relu_g, bias_add_g column sums), fp32/fp64 only. softmax_ce_g
 * has no SIMD kernels for the same reason as forward softmax_ce (scalar
 * exp()/log() dominate), and transpose is a memory-bound byte copy.
 * @version 0.1
 * @date 2026-10-18
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Autograd {
    [[nodiscard]] bool autograd_avx2_available();

    [[nodiscard]] bool autograd_avx512_available();

    void grad_acc_f32_range_avx2(float32_t*, const float32_t*, size_t, size_t);
    void grad_acc_f64_range_avx2(float64_t*, const float64_t*, size_t, size_t);
    void grad_acc_f32_range_avx512(float32_t*, const float32_t*, size_t, size_t);
    void grad_acc_f64_range_avx512(float64_t*, const float64_t*, size_t, size_t);

    void relu_g_f32_range_avx2(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    void relu_g_f64_range_avx2(float64_t*, const float64_t*, const float64_t*, size_t, size_t);
    void relu_g_f32_range_avx512(float32_t*, const float32_t*, const float32_t*, size_t, size_t);
    void relu_g_f64_range_avx512(float64_t*, const float64_t*, const float64_t*, size_t, size_t);

    void bias_add_g_f32_colsum_avx2(float32_t*, const float32_t*, size_t, size_t, size_t);
    void bias_add_g_f64_colsum_avx2(float64_t*, const float64_t*, size_t, size_t, size_t);
    void bias_add_g_f32_colsum_avx512(float32_t*, const float32_t*, size_t, size_t, size_t);
    void bias_add_g_f64_colsum_avx512(float64_t*, const float64_t*, size_t, size_t, size_t);
}
