/**
 * @file AvgPool2dKernels.h
 * @author GreenBean
 * @brief Kernel declarations and availability probes for AvgPool2d SIMD paths.
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Spatial {

    /**
     * @brief True when the AVX2 AvgPool2d path is compiled and the CPU supports it.
     */
    [[nodiscard]] bool avg_pool2d_avx2_available();
    /**
     * @brief True when the AVX512 AvgPool2d path is compiled and the CPU supports it.
     */
    [[nodiscard]] bool avg_pool2d_avx512_available();
}

namespace BeanTensor::Math::Spatial {
    /**
     * @brief Scalar-style fp32 window-range forward kernel compiled in the AVX2 TU.
     */
    void avg_pool2d_f32_window_avx2(const float32_t*, float32_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
    /**
     * @brief Scalar-style fp64 window-range forward kernel compiled in the AVX2 TU.
     */
    void avg_pool2d_f64_window_avx2(const float64_t*, float64_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

    /**
     * @brief Scalar-style fp32 window-range forward kernel compiled in the AVX512 TU.
     */
    void avg_pool2d_f32_window_avx512(const float32_t*, float32_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
    /**
     * @brief Scalar-style fp64 window-range forward kernel compiled in the AVX512 TU.
     */
    void avg_pool2d_f64_window_avx512(const float64_t*, float64_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
}
