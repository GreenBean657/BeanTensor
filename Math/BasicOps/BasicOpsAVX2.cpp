/**
 * @file BasicOpsAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for basic ops (fp32/fp64 sgd_update and sum).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * basicops_avx2_available().
 * @version 0.1
 * @date 2026-10-18
 */
#include <cmath>
#include <immintrin.h>

#include "Math/BasicOps/BasicOpsKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::BasicOps {

    /**
     * Cached runtime variable - Gates AVX2+FMA basic-op kernels on CPU support.
     * @brief AVX2 && FMA verification, for basic-op kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool basicops_avx2_available() {
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

    /**
     * Four independent 8-lane accumulators hide add latency, then collapse to one vector.
     * Horizontal reduce via 128-bit halves; remainder lanes are added scalar.
     * @note Order differs from the scalar path, so the result is not bit-identical.
     * @brief AVX2 tree-reduction sum over a range (FP32).
     * @param data Input elements.
     * @param begin First element summed.
     * @param end One past the last element summed.
     * @return Sum of the elements in [begin, end).
     */
    float32_t sum_f32_range_avx2(const float32_t* data, const size_t begin, const size_t end) {
        auto acc0 = _mm256_setzero_ps();
        auto acc1 = _mm256_setzero_ps();
        auto acc2 = _mm256_setzero_ps();
        auto acc3 = _mm256_setzero_ps();
        auto i = begin;
        for (; i + 32 <= end; i += 32) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data + i + 16));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data + i + 24));
        }
        for (; i + 8 <= end; i += 8) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        }
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        const auto lo = _mm256_castps256_ps128(acc0);
        const auto hi = _mm256_extractf128_ps(acc0, 1);
        const auto sum4 = _mm_add_ps(lo, hi);
        const auto shuf = _mm_movehl_ps(sum4, sum4);
        const auto sum2 = _mm_add_ps(sum4, shuf);
        const auto lane = _mm_shuffle_ps(sum2, sum2, 1);
        auto total = _mm_cvtss_f32(_mm_add_ss(sum2, lane));
        for (; i < end; ++i) {
            total += data[i];
        }
        return total;
    }

    /**
     * Four independent 4-lane accumulators hide add latency, then collapse to one vector.
     * Horizontal reduce via 128-bit halves; remainder lanes are added scalar.
     * @note Order differs from the scalar path, so the result is not bit-identical.
     * @brief AVX2 tree-reduction sum over a range (FP64).
     * @param data Input elements.
     * @param begin First element summed.
     * @param end One past the last element summed.
     * @return Sum of the elements in [begin, end).
     */
    float64_t sum_f64_range_avx2(const float64_t* data, const size_t begin, const size_t end) {
        auto acc0 = _mm256_setzero_pd();
        auto acc1 = _mm256_setzero_pd();
        auto acc2 = _mm256_setzero_pd();
        auto acc3 = _mm256_setzero_pd();
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(data + i));
            acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(data + i + 4));
            acc2 = _mm256_add_pd(acc2, _mm256_loadu_pd(data + i + 8));
            acc3 = _mm256_add_pd(acc3, _mm256_loadu_pd(data + i + 12));
        }
        for (; i + 4 <= end; i += 4) {
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(data + i));
        }
        acc0 = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        const auto lo = _mm256_castpd256_pd128(acc0);
        const auto hi = _mm256_extractf128_pd(acc0, 1);
        const auto sum2 = _mm_add_pd(lo, hi);
        auto total = _mm_cvtsd_f64(_mm_add_sd(sum2, _mm_unpackhi_pd(sum2, sum2)));
        for (; i < end; ++i) {
            total += data[i];
        }
        return total;
    }
}
