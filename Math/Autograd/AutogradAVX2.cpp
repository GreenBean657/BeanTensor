/**
 * @file AutogradAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for autograd backward ops (fp32/fp64 grad
 * accumulation, relu_g, bias_add_g column sums). Compiled with
 * -mavx2 -mfma -mf16c; execution is gated at runtime via
 * autograd_avx2_available().
 * @version 0.1
 * @date 2026-10-18
 */
#include <immintrin.h>

#include "Math/Autograd/AutogradKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for autograd kernels.
     * @return True when the CPU supports AVX2 and FMA.
     */
    bool autograd_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Processes 8 lanes per step over [begin, end); the remainder runs scalar.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range grad accumulation (fp32).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f32_range_avx2(float32_t* w, const float32_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            _mm256_storeu_ps(w + i, _mm256_add_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(d + i)));
        }
        for (; i < end; ++i) {
            w[i] = w[i] + d[i];
        }
    }

    /**
     * Processes 4 lanes per step over [begin, end); the remainder runs scalar.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range grad accumulation (fp64).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f64_range_avx2(float64_t* w, const float64_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            _mm256_storeu_pd(w + i, _mm256_add_pd(_mm256_loadu_pd(w + i), _mm256_loadu_pd(d + i)));
        }
        for (; i < end; ++i) {
            w[i] = w[i] + d[i];
        }
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

    /**
     * For each row r in [rbegin, rend), adds that row of dy into partial, 8 columns per step; the column tail runs scalar.
     * Vectorized across columns only, so each column's accumulation order (row by row) matches the scalar kernels exactly, bit-identical.
     * @brief AVX2 bias_add backward column sum (fp32).
     * @param partial Per-column accumulator, cols elements, accumulated in place.
     * @param dy Upstream gradient, row-major with cols columns.
     * @param cols Number of columns (last-dimension size).
     * @param rbegin First row accumulated.
     * @param rend One past the last row accumulated.
     */
    void bias_add_g_f32_colsum_avx2(float32_t* partial, const float32_t* dy, const size_t cols, const size_t rbegin, const size_t rend) {
        for (auto r = rbegin; r < rend; ++r) {
            const auto* row = dy + r * cols;
            auto c = size_t{0};
            for (; c + 8 <= cols; c += 8) {
                _mm256_storeu_ps(partial + c, _mm256_add_ps(_mm256_loadu_ps(partial + c), _mm256_loadu_ps(row + c)));
            }
            for (; c < cols; ++c) {
                partial[c] = partial[c] + row[c];
            }
        }
    }

    /**
     * For each row r in [rbegin, rend), adds that row of dy into partial, 4 columns per step; the column tail runs scalar.
     * Vectorized across columns only, so each column's accumulation order (row by row) matches the scalar kernels exactly, bit-identical.
     * @brief AVX2 bias_add backward column sum (fp64).
     * @param partial Per-column accumulator, cols elements, accumulated in place.
     * @param dy Upstream gradient, row-major with cols columns.
     * @param cols Number of columns (last-dimension size).
     * @param rbegin First row accumulated.
     * @param rend One past the last row accumulated.
     */
    void bias_add_g_f64_colsum_avx2(float64_t* partial, const float64_t* dy, const size_t cols, const size_t rbegin, const size_t rend) {
        for (auto r = rbegin; r < rend; ++r) {
            const auto* row = dy + r * cols;
            auto c = size_t{0};
            for (; c + 4 <= cols; c += 4) {
                _mm256_storeu_pd(partial + c, _mm256_add_pd(_mm256_loadu_pd(partial + c), _mm256_loadu_pd(row + c)));
            }
            for (; c < cols; ++c) {
                partial[c] = partial[c] + row[c];
            }
        }
    }
}
