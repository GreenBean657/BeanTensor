/**
 * @file SumAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for sum (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * sum_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <cmath>
#include <immintrin.h>

#include "Math/BasicOps/Sum/SumKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::BasicOps {

    /**
     * Cached runtime variable - Gates AVX2+FMA sum kernels on CPU support.
     * @brief AVX2 && FMA verification, for sum kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool sum_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
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
