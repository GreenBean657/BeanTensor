/**
 * @file BiasAddAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for bias_add (fp32/fp64).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * bias_add_avx512_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <algorithm>
#include <immintrin.h>

#include "Math/LinearAlgebra/BiasAdd/BiasAddKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::LinearAlgebra {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for bias_add kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    bool bias_add_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * Walks [begin, end) one bias row segment at a time, 16 lanes per step.
     * The column tail runs under a lane mask; masked-off lanes are never touched.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range bias_add (fp32).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param bias Bias vector, cols elements.
     * @param cols Last-dimension size; bias index is i % cols.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void bias_add_f32_range_avx512(float32_t* out, const float32_t* in, const float32_t* bias, const size_t cols, const size_t begin, const size_t end) {
        auto i = begin;
        while (i < end) {
            const auto col0 = i % cols;
            const auto seg_end = std::min(end, i + (cols - col0));
            auto c = col0;
            for (; c + 16 <= cols && i + 16 <= seg_end; c += 16, i += 16) {
                _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(in + i), _mm512_loadu_ps(bias + c)));
            }
            if (i < seg_end) {
                const auto mask = static_cast<__mmask16>((1u << (seg_end - i)) - 1u);
                const auto in_vec = _mm512_maskz_loadu_ps(mask, in + i);
                const auto bias_vec = _mm512_maskz_loadu_ps(mask, bias + c);
                _mm512_mask_storeu_ps(out + i, mask, _mm512_add_ps(in_vec, bias_vec));
                i = seg_end;
            }
        }
    }

    /**
     * Walks [begin, end) one bias row segment at a time, 8 lanes per step.
     * The column tail runs under a lane mask; masked-off lanes are never touched.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX512F range bias_add (fp64).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param bias Bias vector, cols elements.
     * @param cols Last-dimension size; bias index is i % cols.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void bias_add_f64_range_avx512(float64_t* out, const float64_t* in, const float64_t* bias, const size_t cols, const size_t begin, const size_t end) {
        auto i = begin;
        while (i < end) {
            const auto col0 = i % cols;
            const auto seg_end = std::min(end, i + (cols - col0));
            auto c = col0;
            for (; c + 8 <= cols && i + 8 <= seg_end; c += 8, i += 8) {
                _mm512_storeu_pd(out + i, _mm512_add_pd(_mm512_loadu_pd(in + i), _mm512_loadu_pd(bias + c)));
            }
            if (i < seg_end) {
                const auto mask = static_cast<__mmask8>((1u << (seg_end - i)) - 1u);
                const auto in_vec = _mm512_maskz_loadu_pd(mask, in + i);
                const auto bias_vec = _mm512_maskz_loadu_pd(mask, bias + c);
                _mm512_mask_storeu_pd(out + i, mask, _mm512_add_pd(in_vec, bias_vec));
                i = seg_end;
            }
        }
    }
}
