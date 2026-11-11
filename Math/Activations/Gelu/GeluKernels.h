/**
 * @file GeluKernels.h
 * @author GreenBean
 * @brief SIMD kernel declarations and shared scalar GELU helpers for the
 * polynomial-erf SIMD tier and the exact std::erf scalar tier.
 *
 * The polynomial follows Abramowitz & Stegun 7.1.26:
 *   erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)*exp(-x^2),
 *   t = 1 / (1 + p*|x|), p = 0.3275911,
 *   a1..a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429.
 * Documented absolute error bound: |err| <= 1.5e-7.  GELU uses erf(x/sqrt(2)).
 *
 * The piecewise non-finite contract (tested in tests/gelu_tests.cpp):
 *   NaN -> NaN, +Inf -> +Inf, -Inf -> 0,
 *   x >= +c -> x, x <= -c -> 0,
 *   with c chosen where erf(c/sqrt(2)) rounds to +1 in the compute width
 *   (c = 5.7 for fp32, c = 9.0 for fp64).
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>

#include "Tensor/DTypes.h"

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define BT_GELU_HD __host__ __device__
#else
#define BT_GELU_HD
#endif

namespace BeanTensor::Math::Activations {

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX2 kernel on this.
     * @brief Runtime probe: are the AVX2+FMA GELU kernels safe to execute?
     * @return True when the CPU supports AVX2 and FMA.
     */
    [[nodiscard]] bool gelu_avx2_available();

    /**
     * Cached through IncludedCPUFeatures(); callers gate every AVX512 kernel on this.
     * @brief Runtime probe: are the AVX512F GELU kernels safe to execute?
     * @return True when the CPU supports AVX512F.
     */
    [[nodiscard]] bool gelu_avx512_available();

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end).  The fp32/fp64 SIMD
     * variants use a lane-wise polynomial erf (A&S 7.1.26) with the same
     * saturation branches as the scalar kernel, so the tiers stay within the
     * activation family's documented tolerance.
     * @brief AVX2 range GELU (fp32).
     */
    void gelu_f32_range_avx2(float32_t*, const float32_t*, size_t, size_t);

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end).  The fp32/fp64 SIMD
     * variants use a lane-wise polynomial erf (A&S 7.1.26) with the same
     * saturation branches as the scalar kernel, so the tiers stay within the
     * activation family's documented tolerance.
     * @brief AVX2 range GELU (fp64).
     */
    void gelu_f64_range_avx2(float64_t*, const float64_t*, size_t, size_t);

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end).  The fp32/fp64 SIMD
     * variants use a lane-wise polynomial erf (A&S 7.1.26) with the same
     * saturation branches as the scalar kernel, so the tiers stay within the
     * activation family's documented tolerance.
     * @brief AVX512F range GELU (fp32).
     */
    void gelu_f32_range_avx512(float32_t*, const float32_t*, size_t, size_t);

    /**
     * Computes out[i] = gelu(in[i]) over [begin, end).  The fp32/fp64 SIMD
     * variants use a lane-wise polynomial erf (A&S 7.1.26) with the same
     * saturation branches as the scalar kernel, so the tiers stay within the
     * activation family's documented tolerance.
     * @brief AVX512F range GELU (fp64).
     */
    void gelu_f64_range_avx512(float64_t*, const float64_t*, size_t, size_t);

    namespace detail {

        /**
         * Cutoff where erf(c/sqrt(2)) rounds to +1 in the compute width.
         * @brief Saturation cutoff for GELU piecewise branches.
         */
        template<typename T>
        [[nodiscard]] BT_GELU_HD inline T gelu_cutoff() {
            if constexpr (std::is_same_v<T, float32_t>) {
                return T{5.7f};
            } else {
                return T{9.0f};
            }
        }

        /**
         * Polynomial error-function approximation (A&S 7.1.26) for a single lane.
         * Odd function: poly_erf(-x) = -poly_erf(x).  Returns +0 for x = +/-0.
         * @brief Lane-wise polynomial erf used by the SIMD GELU tiers.
         * @param x Input value.
         * @return Approximate erf(x).
         */
        template<typename T>
        [[nodiscard]] BT_GELU_HD inline T poly_erf(const T x) {
            using std::abs;
            using std::exp;
            const auto ax = abs(x);
            const auto p = static_cast<T>(0.3275911);
            const auto t = T{1} / (T{1} + p * ax);
            const auto a1 = static_cast<T>(0.254829592);
            const auto a2 = static_cast<T>(-0.284496736);
            const auto a3 = static_cast<T>(1.421413741);
            const auto a4 = static_cast<T>(-1.453152027);
            const auto a5 = static_cast<T>(1.061405429);
            const auto poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
            const auto erf_abs = T{1} - poly * exp(-ax * ax);
            if (x < T{0}) {
                return -erf_abs;
            }
            return erf_abs;
        }

        /**
         * Exact std::erf-based GELU for a single lane, including the saturation
         * branches that make gelu(-Inf) == 0 and gelu(+Inf) == +Inf.
         * @brief Lane-wise exact GELU used by the scalar CPU tier.
         * @param x Input value.
         * @return GELU(x).
         */
        template<typename T>
        [[nodiscard]] BT_GELU_HD inline T gelu_exact_lane(const T x) {
            using std::erf;
            using std::isinf;
            using std::isnan;
            if (isnan(x)) {
                return x;
            }
            if (isinf(x)) {
                if (x > T{0}) {
                    return x;
                }
                return T{0};
            }
            const auto cutoff = gelu_cutoff<T>();
            if (x >= cutoff) {
                return x;
            }
            if (x <= -cutoff) {
                return T{0};
            }
            const auto sqrt2 = std::numbers::sqrt2_v<T>;
            return T{0.5} * x * (T{1} + erf(x / sqrt2));
        }

        /**
         * Polynomial-erf GELU for a single lane, with the same piecewise
         * saturation branches as gelu_exact_lane.  Used by the SIMD tiers so
         * the scalar/SIMD difference is bounded by the polynomial's documented
         * absolute error.
         * @brief Lane-wise polynomial GELU used by the SIMD tiers.
         * @param x Input value.
         * @return Approximate GELU(x).
         */
        template<typename T>
        [[nodiscard]] BT_GELU_HD inline T gelu_poly_lane(const T x) {
            using std::isinf;
            using std::isnan;
            if (isnan(x)) {
                return x;
            }
            if (isinf(x)) {
                if (x > T{0}) {
                    return x;
                }
                return T{0};
            }
            const auto cutoff = gelu_cutoff<T>();
            if (x >= cutoff) {
                return x;
            }
            if (x <= -cutoff) {
                return T{0};
            }
            const auto sqrt2 = std::numbers::sqrt2_v<T>;
            return T{0.5} * x * (T{1} + poly_erf(x / sqrt2));
        }

        /**
         * Exact std::erf-based GELU derivative for a single lane.  The branches
         * match gelu_exact_lane so forward and backward agree at the limits.
         * @brief Lane-wise exact GELU derivative.
         * @param x Input value.
         * @return d/dx GELU(x).
         */
        template<typename T>
        [[nodiscard]] BT_GELU_HD inline T gelu_deriv_exact_lane(const T x) {
            using std::erf;
            using std::exp;
            using std::isinf;
            using std::isnan;
            if (isnan(x)) {
                return x;
            }
            if (isinf(x)) {
                if (x > T{0}) {
                    return T{1};
                }
                return T{0};
            }
            const auto cutoff = gelu_cutoff<T>();
            if (x >= cutoff) {
                return T{1};
            }
            if (x <= -cutoff) {
                return T{0};
            }
            const auto sqrt2 = std::numbers::sqrt2_v<T>;
            const auto sqrt2pi = static_cast<T>(2.5066282746310002);
            const auto phi = exp(T{-0.5} * x * x) / sqrt2pi;
            return T{0.5} * (T{1} + erf(x / sqrt2)) + x * phi;
        }

        /**
         * Polynomial-erf GELU derivative for a single lane.  Uses poly_erf in
         * the erf term so it matches the forward SIMD tier's backend.
         * @brief Lane-wise polynomial GELU derivative.
         * @param x Input value.
         * @return Approximate d/dx GELU(x).
         */
        template<typename T>
        [[nodiscard]] BT_GELU_HD inline T gelu_deriv_poly_lane(const T x) {
            using std::exp;
            using std::isinf;
            using std::isnan;
            if (isnan(x)) {
                return x;
            }
            if (isinf(x)) {
                if (x > T{0}) {
                    return T{1};
                }
                return T{0};
            }
            const auto cutoff = gelu_cutoff<T>();
            if (x >= cutoff) {
                return T{1};
            }
            if (x <= -cutoff) {
                return T{0};
            }
            const auto sqrt2 = std::numbers::sqrt2_v<T>;
            const auto sqrt2pi = static_cast<T>(2.5066282746310002);
            const auto phi = exp(T{-0.5} * x * x) / sqrt2pi;
            return T{0.5} * (T{1} + poly_erf(x / sqrt2)) + x * phi;
        }

    }
}

#undef BT_GELU_HD
