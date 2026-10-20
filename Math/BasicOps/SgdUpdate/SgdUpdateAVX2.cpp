/**
 * @file SgdUpdateAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for sgd_update (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * sgd_update_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <cmath>
#include <immintrin.h>

#include "Math/BasicOps/SgdUpdate/SgdUpdateKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::BasicOps {

    /**
     * Cached runtime variable - Gates AVX2+FMA sgd_update kernels on CPU support.
     * @brief AVX2 && FMA verification, for sgd_update kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool sgd_update_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * In-place SGD step over [begin, end): w[i] = fma(neg_lr, g[i], w[i]), 8 lanes per iteration.
     * Remainder lanes fall back to scalar fmaf.
     * @note Computes w[i] += neg_lr * g[i]. Fused multiply-add, single rounding.
     * @brief AVX2+FMA in-place SGD range kernel (FP32).
     * @param w Weights, updated in place.
     * @param g Gradients, read only.
     * @param neg_lr Negated learning rate.
     * @param begin First element updated.
     * @param end One past the last element updated.
     */
    void sgd_f32_range_avx2(float32_t* w, const float32_t* g, const float32_t neg_lr, const size_t begin, const size_t end) {
        const auto lr_vec = _mm256_set1_ps(neg_lr);
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            const auto g_vec = _mm256_loadu_ps(g + i);
            const auto w_vec = _mm256_loadu_ps(w + i);
            _mm256_storeu_ps(w + i, _mm256_fmadd_ps(lr_vec, g_vec, w_vec));
        }
        for (; i < end; ++i) {
            w[i] = std::fmaf(neg_lr, g[i], w[i]);
        }
    }

    /**
     * In-place SGD step over [begin, end): w[i] = fma(neg_lr, g[i], w[i]), 4 lanes per iteration.
     * Remainder lanes fall back to scalar fma.
     * @note Computes w[i] += neg_lr * g[i]. Fused multiply-add, single rounding.
     * @brief AVX2+FMA in-place SGD range kernel (FP64).
     * @param w Weights, updated in place.
     * @param g Gradients, read only.
     * @param neg_lr Negated learning rate.
     * @param begin First element updated.
     * @param end One past the last element updated.
     */
    void sgd_f64_range_avx2(float64_t* w, const float64_t* g, const float64_t neg_lr, const size_t begin, const size_t end) {
        const auto lr_vec = _mm256_set1_pd(neg_lr);
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            const auto g_vec = _mm256_loadu_pd(g + i);
            const auto w_vec = _mm256_loadu_pd(w + i);
            _mm256_storeu_pd(w + i, _mm256_fmadd_pd(lr_vec, g_vec, w_vec));
        }
        for (; i < end; ++i) {
            w[i] = std::fma(neg_lr, g[i], w[i]);
        }
    }
}
