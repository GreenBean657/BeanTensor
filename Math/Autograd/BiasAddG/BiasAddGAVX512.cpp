/**
 * @file BiasAddGAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for bias_add backward column sums (fp32/fp64).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime
 * via bias_add_g_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <immintrin.h>

#include "Math/Autograd/BiasAddG/BiasAddGKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for autograd kernels.
     * @return True when the CPU supports AVX512F.
     */
    bool bias_add_g_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * For each row r in [rbegin, rend), adds that row of dy into partial, 16 columns per step; the column tail uses a masked load/store.
     * Vectorized across columns only, so each column's accumulation order (row by row) matches the scalar kernels exactly, bit-identical.
     * @brief AVX512F bias_add backward column sum (fp32).
     * @param partial Per-column accumulator, cols elements, accumulated in place.
     * @param dy Upstream gradient, row-major with cols columns.
     * @param cols Number of columns (last-dimension size).
     * @param rbegin First row accumulated.
     * @param rend One past the last row accumulated.
     */
    void bias_add_g_f32_colsum_avx512(float32_t* partial, const float32_t* dy, const size_t cols, const size_t rbegin, const size_t rend) {
        for (auto r = rbegin; r < rend; ++r) {
            const auto* row = dy + r * cols;
            auto c = size_t{0};
            for (; c + 16 <= cols; c += 16) {
                _mm512_storeu_ps(partial + c, _mm512_add_ps(_mm512_loadu_ps(partial + c), _mm512_loadu_ps(row + c)));
            }
            if (c < cols) {
                const auto mask = static_cast<__mmask16>((1u << (cols - c)) - 1u);
                _mm512_mask_storeu_ps(partial + c, mask, _mm512_add_ps(_mm512_maskz_loadu_ps(mask, partial + c), _mm512_maskz_loadu_ps(mask, row + c)));
            }
        }
    }

    /**
     * For each row r in [rbegin, rend), adds that row of dy into partial, 8 columns per step; the column tail uses a masked load/store.
     * Vectorized across columns only, so each column's accumulation order (row by row) matches the scalar kernels exactly, bit-identical.
     * @brief AVX512F bias_add backward column sum (fp64).
     * @param partial Per-column accumulator, cols elements, accumulated in place.
     * @param dy Upstream gradient, row-major with cols columns.
     * @param cols Number of columns (last-dimension size).
     * @param rbegin First row accumulated.
     * @param rend One past the last row accumulated.
     */
    void bias_add_g_f64_colsum_avx512(float64_t* partial, const float64_t* dy, const size_t cols, const size_t rbegin, const size_t rend) {
        for (auto r = rbegin; r < rend; ++r) {
            const auto* row = dy + r * cols;
            auto c = size_t{0};
            for (; c + 8 <= cols; c += 8) {
                _mm512_storeu_pd(partial + c, _mm512_add_pd(_mm512_loadu_pd(partial + c), _mm512_loadu_pd(row + c)));
            }
            if (c < cols) {
                const auto mask = static_cast<__mmask8>((1u << (cols - c)) - 1u);
                _mm512_mask_storeu_pd(partial + c, mask, _mm512_add_pd(_mm512_maskz_loadu_pd(mask, partial + c), _mm512_maskz_loadu_pd(mask, row + c)));
            }
        }
    }
}
