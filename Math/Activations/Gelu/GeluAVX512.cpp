/**
 * @file GeluAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for GELU (fp32/fp64).
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * gelu_avx512_available().
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
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief AVX512F verification, for GELU kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    bool gelu_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end), 16 lanes per step.  The
     * tail runs under a lane mask.  Uses the polynomial-erf backend (A&S 7.1.26)
     * with the same saturation branches as the scalar kernel.
     * @brief AVX512F range GELU (fp32).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void gelu_f32_range_avx512(float32_t* out, const float32_t* in, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 16 <= end; i += 16) {
            alignas(64) float32_t buf_out[16];
            for (auto j = size_t{0}; j < 16; ++j) {
                buf_out[j] = detail::gelu_poly_lane(in[i + j]);
            }
            _mm512_storeu_ps(out + i, _mm512_load_ps(buf_out));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask16>((1u << (end - i)) - 1u);
            alignas(64) float32_t buf_out[16] = {};
            for (auto j = size_t{0}; j < end - i; ++j) {
                buf_out[j] = detail::gelu_poly_lane(in[i + j]);
            }
            _mm512_mask_storeu_ps(out + i, mask, _mm512_load_ps(buf_out));
        }
    }

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end), 8 lanes per step.  The
     * tail runs under a lane mask.  Uses the polynomial-erf backend (A&S 7.1.26)
     * with the same saturation branches as the scalar kernel.
     * @brief AVX512F range GELU (fp64).
     * @param out Output buffer.
     * @param in Input buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void gelu_f64_range_avx512(float64_t* out, const float64_t* in, const size_t begin, const size_t end) {
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            alignas(64) float64_t buf_out[8];
            for (auto j = size_t{0}; j < 8; ++j) {
                buf_out[j] = detail::gelu_exact_lane(in[i + j]);
            }
            _mm512_storeu_pd(out + i, _mm512_load_pd(buf_out));
        }
        if (i < end) {
            const auto mask = static_cast<__mmask8>((1u << (end - i)) - 1u);
            alignas(64) float64_t buf_out[8] = {};
            for (auto j = size_t{0}; j < end - i; ++j) {
                buf_out[j] = detail::gelu_exact_lane(in[i + j]);
            }
            _mm512_mask_storeu_pd(out + i, mask, _mm512_load_pd(buf_out));
        }
    }
}
