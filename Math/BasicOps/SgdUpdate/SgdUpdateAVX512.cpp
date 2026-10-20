/**
 * @file SgdUpdateAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for sgd_update (fp32/fp64).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * sgd_update_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <cmath>
#include <immintrin.h>

#include "Math/BasicOps/SgdUpdate/SgdUpdateKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::BasicOps {

    /**
     * Cached runtime variable - Gates AVX512F sgd_update kernels on CPU support.
     * @brief AVX512F verification, for sgd_update kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    bool sgd_update_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * In-place SGD step over [begin, end): w[i] = fma(neg_lr, g[i], w[i]), 16 lanes per iteration.
     * Remainder lanes run once under a mask; masked-off lanes are never touched.
     * @note Computes w[i] += neg_lr * g[i]. Fused multiply-add, single rounding.
     * @brief AVX512F in-place SGD range kernel (FP32).
     * @param w Weights, updated in place.
     * @param g Gradients, read only.
     * @param neg_lr Negated learning rate.
     * @param begin First element updated.
     * @param end One past the last element updated.
     */
    void sgd_f32_range_avx512(float32_t* w, const float32_t* g, const float32_t neg_lr, const size_t begin, const size_t end) {
        const auto lr_vec = _mm512_set1_ps(neg_lr);
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            const auto g_vec = _mm512_loadu_ps(g + i);
            const auto w_vec = _mm512_loadu_ps(w + i);
            _mm512_storeu_ps(w + i, _mm512_fmadd_ps(lr_vec, g_vec, w_vec));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
            const auto g_vec = _mm512_maskz_loadu_ps(mask, g + i);
            const auto w_vec = _mm512_maskz_loadu_ps(mask, w + i);
            _mm512_mask_storeu_ps(w + i, mask, _mm512_fmadd_ps(lr_vec, g_vec, w_vec));
        }
    }

    /**
     * In-place SGD step over [begin, end): w[i] = fma(neg_lr, g[i], w[i]), 8 lanes per iteration.
     * Remainder lanes run once under a mask; masked-off lanes are never touched.
     * @note Computes w[i] += neg_lr * g[i]. Fused multiply-add, single rounding.
     * @brief AVX512F in-place SGD range kernel (FP64).
     * @param w Weights, updated in place.
     * @param g Gradients, read only.
     * @param neg_lr Negated learning rate.
     * @param begin First element updated.
     * @param end One past the last element updated.
     */
    void sgd_f64_range_avx512(float64_t* w, const float64_t* g, const float64_t neg_lr, const size_t begin, const size_t end) {
        const auto lr_vec = _mm512_set1_pd(neg_lr);
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            const auto g_vec = _mm512_loadu_pd(g + i);
            const auto w_vec = _mm512_loadu_pd(w + i);
            _mm512_storeu_pd(w + i, _mm512_fmadd_pd(lr_vec, g_vec, w_vec));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
            const auto g_vec = _mm512_maskz_loadu_pd(mask, g + i);
            const auto w_vec = _mm512_maskz_loadu_pd(mask, w + i);
            _mm512_mask_storeu_pd(w + i, mask, _mm512_fmadd_pd(lr_vec, g_vec, w_vec));
        }
    }
}
