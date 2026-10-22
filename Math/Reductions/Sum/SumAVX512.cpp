/**
 * @file SumAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for sum (fp32/fp64).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * sum_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <cmath>
#include <immintrin.h>

#include "Math/Reductions/Sum/SumKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Reductions {

    /**
     * Cached runtime variable - Gates AVX512F sum kernels on CPU support.
     * @brief AVX512F verification, for sum kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    bool sum_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
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
