/**
 * @file GradAccAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for grad accumulation (fp32/fp64). Compiled with
 * -mavx2 -mfma -mf16c; execution is gated at runtime via
 * grad_acc_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Autograd/GradAcc/GradAccKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for autograd kernels.
     * @return True when the CPU supports AVX2 and FMA.
     */
    bool grad_acc_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Processes 8 lanes per step over [begin, end); the remainder runs scalar.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range grad accumulation (fp32).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f32_range_avx2(float32_t* w, const float32_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            _mm256_storeu_ps(w + i, _mm256_add_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(d + i)));
        }
        for (; i < end; ++i) {
            w[i] = w[i] + d[i];
        }
    }

    /**
     * Processes 4 lanes per step over [begin, end); the remainder runs scalar.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range grad accumulation (fp64).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f64_range_avx2(float64_t* w, const float64_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            _mm256_storeu_pd(w + i, _mm256_add_pd(_mm256_loadu_pd(w + i), _mm256_loadu_pd(d + i)));
        }
        for (; i < end; ++i) {
            w[i] = w[i] + d[i];
        }
    }
}
