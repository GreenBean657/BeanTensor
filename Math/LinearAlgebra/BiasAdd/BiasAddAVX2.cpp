/**
 * @file BiasAddAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for bias_add (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * bias_add_avx2_available().
 * @version 0.1
 * @date 2026-10-20
 */
#include <algorithm>
#include <immintrin.h>

#include "Math/LinearAlgebra/BiasAdd/BiasAddKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::LinearAlgebra {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for bias_add kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool bias_add_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Walks [begin, end) one bias row segment at a time, 8 lanes per step.
     * The column tail runs scalar.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range bias_add (fp32).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param bias Bias vector, cols elements.
     * @param cols Last-dimension size; bias index is i % cols.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void bias_add_f32_range_avx2(float32_t* out, const float32_t* in, const float32_t* bias, const size_t cols, const size_t begin, const size_t end) {
        auto i = begin;
        while (i < end) {
            const auto col0 = i % cols;
            const auto seg_end = std::min(end, i + (cols - col0));
            auto c = col0;
            for (; c + 8 <= cols && i + 8 <= seg_end; c += 8, i += 8) {
                _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(in + i), _mm256_loadu_ps(bias + c)));
            }
            for (; i < seg_end; ++i, ++c) {
                out[i] = in[i] + bias[c];
            }
        }
    }

    /**
     * Walks [begin, end) one bias row segment at a time, 4 lanes per step.
     * The column tail runs scalar.
     * Single-rounding add, bit-identical to the scalar kernels.
     * @brief AVX2 range bias_add (fp64).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param bias Bias vector, cols elements.
     * @param cols Last-dimension size; bias index is i % cols.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void bias_add_f64_range_avx2(float64_t* out, const float64_t* in, const float64_t* bias, const size_t cols, const size_t begin, const size_t end) {
        auto i = begin;
        while (i < end) {
            const auto col0 = i % cols;
            const auto seg_end = std::min(end, i + (cols - col0));
            auto c = col0;
            for (; c + 4 <= cols && i + 4 <= seg_end; c += 4, i += 4) {
                _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(in + i), _mm256_loadu_pd(bias + c)));
            }
            for (; i < seg_end; ++i, ++c) {
                out[i] = in[i] + bias[c];
            }
        }
    }
}
