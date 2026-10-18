/**
 * @file BasicOpsAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for basic ops (fp32/fp64 sgd_update and sum).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * basicops_avx512_available().
 * @version 0.1
 * @date 2026-10-18
 */
#include <cmath>
#include <immintrin.h>

#include "Math/BasicOps/BasicOpsKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::BasicOps {

    /**
     * Cached runtime variable - Gates AVX512F basic-op kernels on CPU support.
     * @brief AVX512F verification, for basic-op kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    bool basicops_avx512_available() {
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

    /**
     * Four independent 16-lane accumulators hide add latency, then collapse to one vector.
     * Horizontal reduce via _mm512_reduce_add_ps; remainder lanes are added scalar.
     * @note Order differs from the scalar path, so the result is not bit-identical.
     * @brief AVX512F tree-reduction sum over a range (FP32).
     * @param data Input elements.
     * @param begin First element summed.
     * @param end One past the last element summed.
     * @return Sum of the elements in [begin, end).
     */
    float32_t sum_f32_range_avx512(const float32_t* data, const size_t begin, const size_t end) {
        auto acc0 = _mm512_setzero_ps();
        auto acc1 = _mm512_setzero_ps();
        auto acc2 = _mm512_setzero_ps();
        auto acc3 = _mm512_setzero_ps();
        auto i = begin;
        for (; i + 64 <= end; i += 64) {
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(data + i));
            acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(data + i + 16));
            acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(data + i + 32));
            acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(data + i + 48));
        }
        for (; i + 16 <= end; i += 16) {
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(data + i));
        }
        acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        auto total = _mm512_reduce_add_ps(acc0);
        for (; i < end; ++i) {
            total += data[i];
        }
        return total;
    }

    /**
     * Four independent 8-lane accumulators hide add latency, then collapse to one vector.
     * Horizontal reduce via _mm512_reduce_add_pd; remainder lanes are added scalar.
     * @note Order differs from the scalar path, so the result is not bit-identical.
     * @brief AVX512F tree-reduction sum over a range (FP64).
     * @param data Input elements.
     * @param begin First element summed.
     * @param end One past the last element summed.
     * @return Sum of the elements in [begin, end).
     */
    float64_t sum_f64_range_avx512(const float64_t* data, const size_t begin, const size_t end) {
        auto acc0 = _mm512_setzero_pd();
        auto acc1 = _mm512_setzero_pd();
        auto acc2 = _mm512_setzero_pd();
        auto acc3 = _mm512_setzero_pd();
        auto i = begin;
        for (; i + 32 <= end; i += 32) {
            acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(data + i));
            acc1 = _mm512_add_pd(acc1, _mm512_loadu_pd(data + i + 8));
            acc2 = _mm512_add_pd(acc2, _mm512_loadu_pd(data + i + 16));
            acc3 = _mm512_add_pd(acc3, _mm512_loadu_pd(data + i + 24));
        }
        for (; i + 8 <= end; i += 8) {
            acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(data + i));
        }
        acc0 = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));
        auto total = _mm512_reduce_add_pd(acc0);
        for (; i < end; ++i) {
            total += data[i];
        }
        return total;
    }
}
