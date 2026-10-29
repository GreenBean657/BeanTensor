/**
 * @file SoftmaxKernels.h
 * @author GreenBean
 * @brief SIMD kernels + runtime probes for standalone rank-2 softmax.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Math::Activations {
    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+FMA softmax kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool softmax_avx2_available();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512F softmax kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool softmax_avx512_available();

    /**
     * Numerically stable row softmax over [begin, end) rows of a rank-2 fp32 tensor
     * with cols classes per row. Rows are independent; the kernel uses scalar exp
     * to preserve the 1e-5 accuracy contract.
     * @brief AVX2 range softmax (fp32).
     */
    void softmax_f32_range_avx2(float32_t* out, const float32_t* in, size_t rows, size_t cols, size_t begin, size_t end);
    /**
     * Numerically stable row softmax over [begin, end) rows of a rank-2 fp64 tensor.
     * @brief AVX2 range softmax (fp64).
     */
    void softmax_f64_range_avx2(float64_t* out, const float64_t* in, size_t rows, size_t cols, size_t begin, size_t end);
    /**
     * Numerically stable row softmax over [begin, end) rows of a rank-2 fp32 tensor.
     * @brief AVX512 range softmax (fp32).
     */
    void softmax_f32_range_avx512(float32_t* out, const float32_t* in, size_t rows, size_t cols, size_t begin, size_t end);
    /**
     * Numerically stable row softmax over [begin, end) rows of a rank-2 fp64 tensor.
     * @brief AVX512 range softmax (fp64).
     */
    void softmax_f64_range_avx512(float64_t* out, const float64_t* in, size_t rows, size_t cols, size_t begin, size_t end);
}
