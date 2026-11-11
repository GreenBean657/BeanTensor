/**
 * @file GeluGAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for GELU backward (fp32/fp64).
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * gelu_g_avx2_available().
 *
 * The SIMD path uses the A&S 7.1.26 polynomial-erf derivative evaluated lane-wise
 * with std::exp.  Vector loads/stores amortize memory traffic; the arithmetic is
 * lane-wise scalar so the result stays within the polynomial's documented
 * absolute error bound.
 * @version 0.1
 * @date 2026-11-10
 */
#include <immintrin.h>

#include "Math/Autograd/GeluG/GeluGKernels.h"
#include "Math/Activations/Gelu/GeluKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Autograd {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief AVX2 && FMA verification, for GELU backward kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool gelu_g_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
    }

    /**
     * Computes dx[i] = dy[i] * gelu'(x[i]) over [begin, end), 8 lanes per step.
     * The tail runs scalar.  Uses the polynomial-erf derivative backend.
     * @brief AVX2 range GELU backward (fp32).
     * @param dx Output gradient buffer.
     * @param x Forward input buffer.
     * @param dy Upstream gradient buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void gelu_g_f32_range_avx2(float32_t* dx, const float32_t* x, const float32_t* dy, const size_t begin, const size_t end) {
        using BeanTensor::Math::Activations::detail::gelu_deriv_poly_lane;
        auto i = begin;
        for (; i + 8 <= end; i += 8) {
            alignas(32) float32_t buf_dx[8];
            for (auto j = size_t{0}; j < 8; ++j) {
                buf_dx[j] = dy[i + j] * gelu_deriv_poly_lane(x[i + j]);
            }
            _mm256_storeu_ps(dx + i, _mm256_load_ps(buf_dx));
        }
        for (; i < end; ++i) {
            dx[i] = dy[i] * gelu_deriv_poly_lane(x[i]);
        }
    }

    /**
     * Computes dx[i] = dy[i] * gelu'(x[i]) over [begin, end), 4 lanes per step.
     * The tail runs scalar.  Uses the polynomial-erf derivative backend.
     * @brief AVX2 range GELU backward (fp64).
     * @param dx Output gradient buffer.
     * @param x Forward input buffer.
     * @param dy Upstream gradient buffer.
     * @param begin First element computed.
     * @param end One past the last element computed.
     */
    void gelu_g_f64_range_avx2(float64_t* dx, const float64_t* x, const float64_t* dy, const size_t begin, const size_t end) {
        using BeanTensor::Math::Activations::detail::gelu_deriv_exact_lane;
        auto i = begin;
        for (; i + 4 <= end; i += 4) {
            alignas(32) float64_t buf_dx[4];
            for (auto j = size_t{0}; j < 4; ++j) {
                buf_dx[j] = dy[i + j] * gelu_deriv_exact_lane(x[i + j]);
            }
            _mm256_storeu_pd(dx + i, _mm256_load_pd(buf_dx));
        }
        for (; i < end; ++i) {
            dx[i] = dy[i] * gelu_deriv_exact_lane(x[i]);
        }
    }
}
