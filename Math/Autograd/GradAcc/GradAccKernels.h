/**
 * @file GradAccKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for grad accumulation (fp32/fp64).
 * @version 0.1
 * @date 2026-10-20
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Autograd {
    [[nodiscard]] bool grad_acc_avx2_available();

    [[nodiscard]] bool grad_acc_avx512_available();

    void grad_acc_f32_range_avx2(float32_t*, const float32_t*, size_t, size_t);
    void grad_acc_f64_range_avx2(float64_t*, const float64_t*, size_t, size_t);
    void grad_acc_f32_range_avx512(float32_t*, const float32_t*, size_t, size_t);
    void grad_acc_f64_range_avx512(float64_t*, const float64_t*, size_t, size_t);
}
