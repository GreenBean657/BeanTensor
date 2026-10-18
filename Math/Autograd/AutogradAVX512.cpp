/**
 * @file AutogradAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for autograd backward ops (fp32/fp64 grad
 * accumulation, relu_g, bias_add_g column sums). Compiled with
 * -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * autograd_avx512_available().
 * @version 0.1
 * @date 2026-10-18
 */
#include <immintrin.h>

#include "Math/Autograd/AutogradKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for autograd kernels.
     * @return True when the CPU supports AVX512F.
     */
    bool autograd_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * Processes 16 lanes per step over [begin, end); the remainder is handled by one masked load/store step.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range grad accumulation (fp32).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f32_range_avx512(float32_t* w, const float32_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            _mm512_storeu_ps(w + i, _mm512_add_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(d + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
            _mm512_mask_storeu_ps(w + i, mask, _mm512_add_ps(_mm512_maskz_loadu_ps(mask, w + i), _mm512_maskz_loadu_ps(mask, d + i)));
        }
    }

    /**
     * Processes 8 lanes per step over [begin, end); the remainder is handled by one masked load/store step.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range grad accumulation (fp64).
     * @param w Gradient buffer, accumulated in place.
     * @param d Delta buffer added into w.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void grad_acc_f64_range_avx512(float64_t* w, const float64_t* d, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            _mm512_storeu_pd(w + i, _mm512_add_pd(_mm512_loadu_pd(w + i), _mm512_loadu_pd(d + i)));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
            _mm512_mask_storeu_pd(w + i, mask, _mm512_add_pd(_mm512_maskz_loadu_pd(mask, w + i), _mm512_maskz_loadu_pd(mask, d + i)));
        }
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
