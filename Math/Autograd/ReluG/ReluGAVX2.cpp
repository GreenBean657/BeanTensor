/**
 * @file ReluGAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for relu backward (fp32/fp64). Compiled with
 * -mavx2 -mfma -mf16c; execution is gated at runtime via
 * relu_g_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Autograd/ReluG/ReluGKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for autograd kernels.
     * @return True when the CPU supports AVX2 and FMA.
     */
    bool relu_g_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Processes 8 lanes per step over [begin, end); the remainder runs scalar.
     * Uses an ordered, quiet x > 0 compare and a select (not a multiply): NaN in x gives 0, NaN in a selected dy lane propagates, bit-identical to the scalar kernels.
     * @brief AVX2 range relu backward (fp32).
     * @param dx Output gradient buffer.
     * @param x Forward input buffer.
     * @param dy Upstream gradient buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_g_f32_range_avx2(float32_t* dx, const float32_t* x, const float32_t* dy, const size_t begin, const size_t end) {
        const auto zero = _mm256_setzero_ps();
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            // Ordered, quiet x > 0: NaN lanes compare false and are zeroed.
            const auto mask = _mm256_cmp_ps(_mm256_loadu_ps(x + i), zero, _CMP_GT_OQ);
            _mm256_storeu_ps(dx + i, _mm256_and_ps(mask, _mm256_loadu_ps(dy + i)));
        }
        for (; i < end; ++i) {
            dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
        }
    }

    /**
     * Processes 4 lanes per step over [begin, end); the remainder runs scalar.
     * Uses an ordered, quiet x > 0 compare and a select (not a multiply): NaN in x gives 0, NaN in a selected dy lane propagates, bit-identical to the scalar kernels.
     * @brief AVX2 range relu backward (fp64).
     * @param dx Output gradient buffer.
     * @param x Forward input buffer.
     * @param dy Upstream gradient buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_g_f64_range_avx2(float64_t* dx, const float64_t* x, const float64_t* dy, const size_t begin, const size_t end) {
        const auto zero = _mm256_setzero_pd();
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            const auto mask = _mm256_cmp_pd(_mm256_loadu_pd(x + i), zero, _CMP_GT_OQ);
            _mm256_storeu_pd(dx + i, _mm256_and_pd(mask, _mm256_loadu_pd(dy + i)));
        }
        for (; i < end; ++i) {
            dx[i] = x[i] > 0.0 ? dy[i] : 0.0;
        }
    }
}
