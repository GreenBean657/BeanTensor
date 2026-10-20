/**
 * @file BiasAddGAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for bias_add backward column sums (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * bias_add_g_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Autograd/BiasAddG/BiasAddGKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for autograd kernels.
     * @return True when the CPU supports AVX2 and FMA.
     */
    bool bias_add_g_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
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
