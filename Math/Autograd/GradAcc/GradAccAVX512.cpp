/**
 * @file GradAccAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for grad accumulation (fp32/fp64). Compiled with
 * -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * grad_acc_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Autograd/GradAcc/GradAccKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for autograd kernels.
     * @return True when the CPU supports AVX512F.
     */
    bool grad_acc_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * Processes 16 lanes per step over [begin, end); the remainder is handled by one masked load/store step.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range grad accumulation (fp32).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f32_range_avx512(float32_t* w, const float32_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            _mm512_storeu_ps(w + i, _mm512_add_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(d + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
            _mm512_mask_storeu_ps(w + i, mask, _mm512_add_ps(_mm512_maskz_loadu_ps(mask, w + i), _mm512_maskz_loadu_ps(mask, d + i)));
        }
    }

    /**
     * Processes 8 lanes per step over [begin, end); the remainder is handled by one masked load/store step.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range grad accumulation (fp64).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f64_range_avx512(float64_t* w, const float64_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            _mm512_storeu_pd(w + i, _mm512_add_pd(_mm512_loadu_pd(w + i), _mm512_loadu_pd(d + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
            _mm512_mask_storeu_pd(w + i, mask, _mm512_add_pd(_mm512_maskz_loadu_pd(mask, w + i), _mm512_maskz_loadu_pd(mask, d + i)));
        }
    }
}
