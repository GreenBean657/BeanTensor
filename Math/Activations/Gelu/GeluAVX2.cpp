/**
 * @file GeluAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for GELU (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * gelu_avx2_available().
 *
 * The SIMD path uses the A&S 7.1.26 polynomial erf evaluated lane-wise with
 * std::exp.  Vector loads/stores amortize memory traffic; the arithmetic is
 * lane-wise scalar so the result stays within the polynomial's documented
 * absolute error bound.
 * @version 0.1
 * @date 2026-11-10
 */
#include <immintrin.h>

#include "Math/Activations/Gelu/GeluKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Activations {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for GELU kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool gelu_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end), 8 lanes per step.  The
     * tail runs scalar.  Uses the polynomial-erf backend (A&S 7.1.26) with the
     * same saturation branches as the scalar kernel.
     * @brief AVX2 range GELU (fp32).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void gelu_f32_range_avx2(float32_t* out, const float32_t* in, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            alignas(32) float32_t buf_out[8];
            for (auto j = size_t{0}; j < 8; ++j) {
                buf_out[j] = detail::gelu_poly_lane(in[i + j]);
            }
            _mm256_storeu_ps(out + i, _mm256_load_ps(buf_out));
        }
        for (; i < end; ++i) {
            out[i] = detail::gelu_poly_lane(in[i]);
        }
    }

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end), 4 lanes per step.  The
     * tail runs scalar.  Uses the polynomial-erf backend (A&S 7.1.26) with the
     * same saturation branches as the scalar kernel.
     * @brief AVX2 range GELU (fp64).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void gelu_f64_range_avx2(float64_t* out, const float64_t* in, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            alignas(32) float64_t buf_out[4];
            for (auto j = size_t{0}; j < 4; ++j) {
                buf_out[j] = detail::gelu_exact_lane(in[i + j]);
            }
            _mm256_storeu_pd(out + i, _mm256_load_pd(buf_out));
        }
        for (; i < end; ++i) {
            out[i] = detail::gelu_exact_lane(in[i]);
        }
    }
}
