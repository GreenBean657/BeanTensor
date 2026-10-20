/**
 * @file ReluGAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for relu backward (fp32/fp64). Compiled with
 * -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * relu_g_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Autograd/ReluG/ReluGKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for autograd kernels.
     * @return True when the CPU supports AVX512F.
     */
    bool relu_g_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * Processes 16 lanes per step over [begin, end); the remainder is handled by one masked load/store step.
     * Uses an ordered, quiet x > 0 compare and a select (not a multiply): NaN in x gives 0, NaN in a selected dy lane propagates, bit-identical to the scalar kernels.
     * @brief AVX512F range relu backward (fp32).
     * @param dx Output gradient buffer.
     * @param x Forward input buffer.
     * @param dy Upstream gradient buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_g_f32_range_avx512(float32_t* dx, const float32_t* x, const float32_t* dy, const size_t begin, const size_t end) {
        const auto zero = _mm512_setzero_ps();
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            // Ordered, quiet x > 0: NaN lanes compare false and are zeroed.
            const auto gt = _mm512_cmp_ps_mask(_mm512_loadu_ps(x + i), zero, _CMP_GT_OQ);
            _mm512_storeu_ps(dx + i, _mm512_maskz_mov_ps(gt, _mm512_loadu_ps(dy + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
            const auto gt = _mm512_mask_cmp_ps_mask(mask, _mm512_maskz_loadu_ps(mask, x + i), zero, _CMP_GT_OQ);
            _mm512_mask_storeu_ps(dx + i, mask, _mm512_maskz_mov_ps(gt, _mm512_maskz_loadu_ps(mask, dy + i)));
        }
    }

    /**
     * Processes 8 lanes per step over [begin, end); the remainder is handled by one masked load/store step.
     * Uses an ordered, quiet x > 0 compare and a select (not a multiply): NaN in x gives 0, NaN in a selected dy lane propagates, bit-identical to the scalar kernels.
     * @brief AVX512F range relu backward (fp64).
     * @param dx Output gradient buffer.
     * @param x Forward input buffer.
     * @param dy Upstream gradient buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void relu_g_f64_range_avx512(float64_t* dx, const float64_t* x, const float64_t* dy, const size_t begin, const size_t end) {
        const auto zero = _mm512_setzero_pd();
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            const auto gt = _mm512_cmp_pd_mask(_mm512_loadu_pd(x + i), zero, _CMP_GT_OQ);
            _mm512_storeu_pd(dx + i, _mm512_maskz_mov_pd(gt, _mm512_loadu_pd(dy + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
            const auto gt = _mm512_mask_cmp_pd_mask(mask, _mm512_maskz_loadu_pd(mask, x + i), zero, _CMP_GT_OQ);
            _mm512_mask_storeu_pd(dx + i, mask, _mm512_maskz_mov_pd(gt, _mm512_maskz_loadu_pd(mask, dy + i)));
        }
    }
}
