/**
 * @file ReluAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for relu (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * relu_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Activations/Relu/ReluKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Activations {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for relu kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool relu_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i], 8 lanes per step. The tail runs scalar. NaN propagates and -0.0 survives, matching the scalar kernel.
     * @brief AVX2 range relu (fp32).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_f32_range_avx2(float32_t* out, const float32_t* in, const size_t begin, const size_t end) {
        const auto zero = _mm256_setzero_ps();
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            // max(zero, x): unordered (NaN) or equal-signed zero keeps SRC2 = x,
            // so NaN propagates and -0.0 survives, matching the scalar kernel.
            _mm256_storeu_ps(out + i, _mm256_max_ps(zero, _mm256_loadu_ps(in + i)));
        }
        for (; i < end; ++i) {
            out[i] = in[i] < 0.0f ? 0.0f : in[i];
        }
    }

    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i], 4 lanes per step. The tail runs scalar. NaN propagates and -0.0 survives, matching the scalar kernel.
     * @brief AVX2 range relu (fp64).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_f64_range_avx2(float64_t* out, const float64_t* in, const size_t begin, const size_t end) {
        const auto zero = _mm256_setzero_pd();
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            _mm256_storeu_pd(out + i, _mm256_max_pd(zero, _mm256_loadu_pd(in + i)));
        }
        for (; i < end; ++i) {
            out[i] = in[i] < 0.0 ? 0.0 : in[i];
        }
    }
}
