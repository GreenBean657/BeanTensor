/**
 * @file ReluAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for relu (fp32/fp64).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * relu_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Activations/Relu/ReluKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Activations {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for relu kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    bool relu_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i], 16 lanes per step. The tail runs under a lane mask. NaN propagates and -0.0 survives, matching the scalar kernel.
     * @brief AVX512F range relu (fp32).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_f32_range_avx512(float32_t* out, const float32_t* in, const size_t begin, const size_t end) {
        const auto zero = _mm512_setzero_ps();
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            // max(zero, x): unordered (NaN) or equal-signed zero keeps SRC2 = x,
            // so NaN propagates and -0.0 survives, matching the scalar kernel.
            _mm512_storeu_ps(out + i, _mm512_max_ps(zero, _mm512_loadu_ps(in + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
            _mm512_mask_storeu_ps(out + i, mask, _mm512_max_ps(zero, _mm512_maskz_loadu_ps(mask, in + i)));
        }
    }

    /**
     * Computes out[i] = in[i] < 0 ? 0 : in[i], 8 lanes per step. The tail runs under a lane mask. NaN propagates and -0.0 survives, matching the scalar kernel.
     * @brief AVX512F range relu (fp64).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_f64_range_avx512(float64_t* out, const float64_t* in, const size_t begin, const size_t end) {
        const auto zero = _mm512_setzero_pd();
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            _mm512_storeu_pd(out + i, _mm512_max_pd(zero, _mm512_loadu_pd(in + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
            _mm512_mask_storeu_pd(out + i, mask, _mm512_max_pd(zero, _mm512_maskz_loadu_pd(mask, in + i)));
        }
    }
}
