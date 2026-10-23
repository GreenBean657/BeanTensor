/**
 * @file GemmAVX512.cpp
 * @author GreenBean
 * @brief AVX512F GEMM row-range kernels.
 * @version 0.1.1
 * @date 2026-10-18
 */
#include "Math/LinearAlgebra/MatMul/GemmKernels.h"

#include <immintrin.h>

#include <cassert>
#include <cstring>
#include <vector>

#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Math::LinearAlgebra {
    /**
     * Cached runtime variable - Gates AVX512F GEMM kernels on CPU support.
     * @brief AVX512F verification, for FP32/FP64/32-bit GEMM kernels.
     * @return T/F - AVX512F is supported by the CPU.
     */
    [[nodiscard]] bool gemm_avx512_available() noexcept {
        static const auto supported =
            IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        return supported;
    }

    /**
     * Cached runtime variable - Gates AVX512BF16 GEMM kernels on CPU support.
     * @brief AVX512BF16 && AVX512BW verification, for BF16/BF16x2 kernels.
     * @return T/F - AVX512BF16 && AVX512BW  is supported by the CPU.
     */
    [[nodiscard]] bool gemm_avx512_bf16_available() noexcept {
        static const auto supported =
            IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BF16) &&
                IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BW);
        return supported;
    }

    /**
     * Cached runtime variable - Gates AVX512F16C GEMM kernels on CPU support.
     * @brief AVX512F16C && AVX512BW && AVX512F verification, for F16C/F16Cx2 kernels.
     * @return T/F - AVX512F16C && AVX512BW && AVX512F is supported by the CPU.
     */
    [[nodiscard]] bool gemm_avx512_f16c_available() noexcept {
        static const auto supported =
            IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F) &&
                IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BW) &&
                    IncludedCPUFeatures().has_feature(CPUFeatureSetType_F16C);
        return supported;
    }

    /**
     * Cached runtime variable - Gates AVX512VNNI GEMM kernels on CPU support.
     * @brief AVX512VNNI && AVX512F && AVX512BW verification, for 16 bit masked kernels.
     * @return T/F - AVX512VNNI && AVX512F && AVX512BW is supported by the CPU.
     */
    [[nodiscard]] bool gemm_avx512_vnni_available() noexcept {
        static const auto supported =
            IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512VNNI) &&
            IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F) &&
                IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BW);
        return supported;
    }

    /**
     * Cached runtime variable - Gates AVX512DQ GEMM kernels on CPU support.
     * @brief AVX512F && AVX512DQ verification, for 16 bit masked kernels.
     * @return T/F - AVX512F && AVX512DQ is supported by the CPU.
     */
    [[nodiscard]] bool gemm_avx512_dq_available() noexcept {
        static const auto supported =
            IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F) &&
                IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512DQ);
        return supported;
    }

    namespace {
        /**
         * Indices 0..31 select from the first source, 32..63 select from the second source.
         * Consumed by the BF16 kernel and every VNNI kernel.
         * @note Lane equation "out[2l] = a16[l], out[2l+1] = b16[l]"
         * @brief Interleave two sources low 16 lanes into dot-product k-pairs.
         */
        alignas(64) constexpr uint16_t ZIP2[32] = {
            0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
            8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47
        };
        // PAIR_LO/PAIR_HI: packed dtypes hold (lo, hi) lanes in one 32-bit
        // unit; extract one lane from each of two k-rows into dpbf16 pairs:
        // PAIR_LO out[2l] = a16[2l] (lo lane of a's packed element l).

        /**
         * Index vectors for the packed bf16x2 layout, where each 32-bit unit holds (LO, HI) lanes.
         * Dot instruction consumes one lane per k-slice. PAIR_LO picks the lo half of every packed unit.
         * @note Lane equation "out[2l] = a16[2l], out[2l+1] = b16[2l]"
         * @brief Extract the lo lane of each packed unit into dpbf16 pairs.
         */
        alignas(64) constexpr uint16_t PAIR_LO[32] = {
            0, 32, 2, 34, 4, 36, 6, 38, 8, 40, 10, 42, 12, 44, 14, 46,
            16, 48, 18, 50, 20, 52, 22, 54, 24, 56, 26, 58, 28, 60, 30, 62
        };
        /**
         * Index vectors for the packed bf16x2 layout, where each 32-bit unit holds (LO, HI) lanes.
         * Dot instruction consumes one lane per k-slice. PAIR_HI picks the hi half of every packed unit.
         * @note Lane equation "out[2l] = a16[2l+1], out[2l+1] = b16[2l+1]"
         * @brief Extract the hi lane of each packed unit into dpbf16 pairs.
         */
        alignas(64) constexpr uint16_t PAIR_HI[32] = {
            1, 33, 3, 35, 5, 37, 7, 39, 9, 41, 11, 43, 13, 45, 15, 47,
            17, 49, 19, 51, 21, 53, 23, 55, 25, 57, 27, 59, 29, 61, 31, 63
        };

        /**
         * Index vectors for the packed bf16x2 B rows. One lane of every packed unit contiguously into the low 256 bits.
         * A single CVTPH_PS will widen it.
         * @note Lane equation "out[l] = in[2l]"
         * @brief Collapse the lo (resp. hi) FP16 lane of 16 packed units.
         */
        alignas(64) constexpr uint16_t COLLAPSE_LO[32] = {
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        /**
         * Index vectors for the packed bf16x2 B rows. One lane of every packed unit contiguously into the hi 256 bits.
         * A single CVTPH_PS will widen it.
         * @note Lane equation "out[l] = in[2l+1]"
         * @brief Collapse the hi FP16 lane of 16 packed units.
         */
        alignas(64) constexpr uint16_t COLLAPSE_HI[32] = {
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        /**
         * Index vector for the pakced stores. Input is [lo 0..15 | hi 0..15] across the zmm's two 256-bit halves.
         * Output restores packed (lo, hi) unit order.
         * @note Lane Equation "out[2l] = in[l], out[2l+1] = in[16+l]"
         * @brief ZIP lo/hi accumulator planes for packed unit order.
         */
        alignas(64) constexpr uint16_t ZIP_STORE[32] = {
            0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
        };

        /**
         * Low bits set over 16x 16-bit lanes; cols == 16 yields all-ones without invoking undefined behavior.
         * (Shift stays under 32)
         * @brief Column mask over 16 16-bit elements.
         * @param cols Valid column count of [0, 16]
         * @return Mask with the low col bits set.
         */
        [[nodiscard]] __mmask32 col_mask16(const size_t cols) noexcept {
            assert(cols <= 16 && "col_mask16: cols must be in [0, 16]"); //Private namespace - compile out is OK.
            return (uint32_t{1} << cols) - 1u;
        }

        /**
         * Packed dtypes hold two 16-bit lanes per column. Cols set 2*cols bits, and cols == 16 saturates to all-ones.
         * @brief Column mask over 32 16-bit elements.
         * @param cols Valid packed column count [0, 16]
         * @return Mask with the low col bits set.
         */
        [[nodiscard]] __mmask32 col_mask32(const size_t cols) noexcept {
            assert(cols <= 16 && "col_mask32: cols must be in [0, 16]"); //Private namespace - compile out is OK.
            return cols >= 16 ? 0xFFFFFFFFu : (uint32_t{1} << (2 * cols)) - 1u;
        }

        /**
         * bit based memcpy handles single objects, while here we need two adjacent elements.
         * Function keeps unique memcpy with a static_assert for elemnt size instead of a length-blind void* load.
         * @brief Load the 32-bit pair formed by two adjacent bf16 elements.
         * @return uint32_t with p[0] in the low half, p[1] in the high half.
         */
        [[nodiscard]] uint32_t load_bf16_pair(const bfloat16_t* p) {
            static_assert(sizeof(bfloat16_t) == sizeof(uint16_t));
            auto v = uint32_t{0};
            memcpy(&v, p, sizeof v);
            return v;
        }
    }

    /**
     * Four A rows share each loaded "B" vector (4x16 registers, one accumulator per row-block)
     * FP32 accumulate via FMA in order. Column tails run m asked, row tails single-row. B loads are unaligned.
     * Masked-off lanes are zero and contibute nothing (present but "skipped").
     * @brief AVX512F row-range GEMM (FP32 FMA accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_f32_rows_avx512(const float32_t* a, const float32_t* b, float32_t* c,
                              const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            for (auto j = size_t{0}; j < n; j += 16) {
                const auto remaining = n - j;
                const auto mask = static_cast<__mmask16>(remaining >= 16 ? 0xFFFFu : (uint32_t{1} << remaining) - 1u);
                auto acc0 = _mm512_setzero_ps();
                auto acc1 = _mm512_setzero_ps();
                auto acc2 = _mm512_setzero_ps();
                auto acc3 = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bv = _mm512_maskz_loadu_ps(mask, b + kk * n + j);
                    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[kk]), bv, acc0);
                    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[kk]), bv, acc1);
                    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[kk]), bv, acc2);
                    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[kk]), bv, acc3);
                }
                _mm512_mask_storeu_ps(c + i * n + j, mask, acc0);
                _mm512_mask_storeu_ps(c + (i + 1) * n + j, mask, acc1);
                _mm512_mask_storeu_ps(c + (i + 2) * n + j, mask, acc2);
                _mm512_mask_storeu_ps(c + (i + 3) * n + j, mask, acc3);
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            for (auto j = size_t{0}; j < n; j += 16) {
                const auto remaining = n - j;
                const auto mask = static_cast<__mmask16>(remaining >= 16 ? 0xFFFFu : (uint32_t{1} << remaining) - 1u);
                auto acc = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    acc = _mm512_fmadd_ps(_mm512_set1_ps(a0[kk]), _mm512_maskz_loadu_ps(mask, b + kk * n + j), acc);
                }
                _mm512_mask_storeu_ps(c + i * n + j, mask, acc);
            }
        }
    }
    /**
     * Four A rows share each loaded "B" vector (4x8 registers, one accumulator per row-block)
     * FP64 accumulate via FMA in order. Column tails run m asked, row tails single-row. B loads are unaligned.
     * Masked-off lanes are zero and contibute nothing (present but "skipped").
     * @brief AVX512F row-range GEMM (FP64 FMA accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_f64_rows_avx512(const float64_t* a, const float64_t* b, float64_t* c,
                              const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            for (auto j = size_t{0}; j < n; j += 8) {
                const auto remaining = n - j;
                const auto mask = static_cast<__mmask8>(remaining >= 8 ? 0xFFu : (uint32_t{1} << remaining) - 1u);
                auto acc0 = _mm512_setzero_pd();
                auto acc1 = _mm512_setzero_pd();
                auto acc2 = _mm512_setzero_pd();
                auto acc3 = _mm512_setzero_pd();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bv = _mm512_maskz_loadu_pd(mask, b + kk * n + j);
                    acc0 = _mm512_fmadd_pd(_mm512_set1_pd(a0[kk]), bv, acc0);
                    acc1 = _mm512_fmadd_pd(_mm512_set1_pd(a1[kk]), bv, acc1);
                    acc2 = _mm512_fmadd_pd(_mm512_set1_pd(a2[kk]), bv, acc2);
                    acc3 = _mm512_fmadd_pd(_mm512_set1_pd(a3[kk]), bv, acc3);
                }
                _mm512_mask_storeu_pd(c + i * n + j, mask, acc0);
                _mm512_mask_storeu_pd(c + (i + 1) * n + j, mask, acc1);
                _mm512_mask_storeu_pd(c + (i + 2) * n + j, mask, acc2);
                _mm512_mask_storeu_pd(c + (i + 3) * n + j, mask, acc3);
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            for (auto j = size_t{0}; j < n; j += 8) {
                const auto remaining = n - j;
                const auto mask = static_cast<__mmask8>(remaining >= 8 ? 0xFFu : (uint32_t{1} << remaining) - 1u);
                auto acc = _mm512_setzero_pd();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    acc = _mm512_fmadd_pd(_mm512_set1_pd(a0[kk]), _mm512_maskz_loadu_pd(mask, b + kk * n + j), acc);
                }
                _mm512_mask_storeu_pd(c + i * n + j, mask, acc);
            }
        }
    }

    /**
     * bf16 GEMM via VDPBF16PS: native bf16 multiply with FP32 accumulate.
     * Probably works.
     * @brief AVX512BF16 row-range GEMM (bf16, FP32 accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_bf16_rows_avx512bf16(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* c,
                                   const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        const auto zip2 = _mm512_loadu_si512(ZIP2);
        const auto k2 = k & ~size_t{1};
        /**
         * Narrow 16 FP32 lanes to bf16. Full 32-byte store when the block is
         * complete; the tail round-trips through a stack buffer so columns
         * past n are never touched.
         * @brief Store one 16-column fp32 accumulator block as bf16.
         * @param dst First output element of the C row block.
         * @param acc FP32 accumulators, 16 lanes.
         * @param cols Valid column count, [0, 16]; less than 16 stores via a bounded memcpy.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto store_block = [](bfloat16_t* dst, const __m512 acc, const size_t cols) {
            const auto narrowed = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(acc));
            if (cols >= 16) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), narrowed);
            } else {
                alignas(32) uint16_t tmp[16];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), narrowed);
                memcpy(reinterpret_cast<uint16_t*>(dst), tmp, cols * sizeof(uint16_t));
            }
        };
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            for (auto j = size_t{0}; j < n; j += 32) {
                const auto rem = n - j;
                const auto cols1 = rem >= 16 ? size_t{16} : rem;
                const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                const auto mask1 = col_mask16(cols1);
                const auto mask2 = col_mask16(cols2);
                auto acc0a = _mm512_setzero_ps();
                auto acc1a = _mm512_setzero_ps();
                auto acc2a = _mm512_setzero_ps();
                auto acc3a = _mm512_setzero_ps();
                auto acc0b = _mm512_setzero_ps();
                auto acc1b = _mm512_setzero_ps();
                auto acc2b = _mm512_setzero_ps();
                auto acc3b = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k2; kk += 2) {
                    const auto p0 = load_bf16_pair(a0 + kk);
                    const auto p1 = load_bf16_pair(a1 + kk);
                    const auto p2 = load_bf16_pair(a2 + kk);
                    const auto p3 = load_bf16_pair(a3 + kk);
                    const auto a0v = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p0)));
                    const auto a1v = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p1)));
                    const auto a2v = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p2)));
                    const auto a3v = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p3)));
                    const auto bk = _mm512_maskz_loadu_epi16(mask1, b + kk * n + j);
                    const auto bk1 = _mm512_maskz_loadu_epi16(mask1, b + (kk + 1) * n + j);
                    const auto bpa = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(bk, zip2, bk1));
                    acc0a = _mm512_dpbf16_ps(acc0a, a0v, bpa);
                    acc1a = _mm512_dpbf16_ps(acc1a, a1v, bpa);
                    acc2a = _mm512_dpbf16_ps(acc2a, a2v, bpa);
                    acc3a = _mm512_dpbf16_ps(acc3a, a3v, bpa);
                    if (cols2 != 0) {
                        const auto ck = _mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16);
                        const auto ck1 = _mm512_maskz_loadu_epi16(mask2, b + (kk + 1) * n + j + 16);
                        const auto bpb = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(ck, zip2, ck1));
                        acc0b = _mm512_dpbf16_ps(acc0b, a0v, bpb);
                        acc1b = _mm512_dpbf16_ps(acc1b, a1v, bpb);
                        acc2b = _mm512_dpbf16_ps(acc2b, a2v, bpb);
                        acc3b = _mm512_dpbf16_ps(acc3b, a3v, bpb);
                    }
                }
                if (k2 != k) {
                    const auto zero = _mm512_setzero_si512();
                    const auto p0 = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(a0[k2]));
                    const auto p1 = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(a1[k2]));
                    const auto p2 = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(a2[k2]));
                    const auto p3 = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(a3[k2]));
                    const auto bk = _mm512_maskz_loadu_epi16(mask1, b + k2 * n + j);
                    const auto bpa = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(bk, zip2, zero));
                    acc0a = _mm512_dpbf16_ps(acc0a, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p0))), bpa);
                    acc1a = _mm512_dpbf16_ps(acc1a, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p1))), bpa);
                    acc2a = _mm512_dpbf16_ps(acc2a, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p2))), bpa);
                    acc3a = _mm512_dpbf16_ps(acc3a, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p3))), bpa);
                    if (cols2 != 0) {
                        const auto ck = _mm512_maskz_loadu_epi16(mask2, b + k2 * n + j + 16);
                        const auto bpb = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(ck, zip2, zero));
                        acc0b = _mm512_dpbf16_ps(acc0b, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p0))), bpb);
                        acc1b = _mm512_dpbf16_ps(acc1b, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p1))), bpb);
                        acc2b = _mm512_dpbf16_ps(acc2b, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p2))), bpb);
                        acc3b = _mm512_dpbf16_ps(acc3b, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p3))), bpb);
                    }
                }
                store_block(c + i * n + j, acc0a, cols1);
                store_block(c + (i + 1) * n + j, acc1a, cols1);
                store_block(c + (i + 2) * n + j, acc2a, cols1);
                store_block(c + (i + 3) * n + j, acc3a, cols1);
                if (cols2 != 0) {
                    store_block(c + i * n + j + 16, acc0b, cols2);
                    store_block(c + (i + 1) * n + j + 16, acc1b, cols2);
                    store_block(c + (i + 2) * n + j + 16, acc2b, cols2);
                    store_block(c + (i + 3) * n + j + 16, acc3b, cols2);
                }
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            for (auto j = size_t{0}; j < n; j += 32) {
                const auto rem = n - j;
                const auto cols1 = rem >= 16 ? size_t{16} : rem;
                const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                const auto mask1 = col_mask16(cols1);
                const auto mask2 = col_mask16(cols2);
                auto acc0a = _mm512_setzero_ps();
                auto acc0b = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k2; kk += 2) {
                    const auto p0 = load_bf16_pair(a0 + kk);
                    const auto a0v = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p0)));
                    const auto bk = _mm512_maskz_loadu_epi16(mask1, b + kk * n + j);
                    const auto bk1 = _mm512_maskz_loadu_epi16(mask1, b + (kk + 1) * n + j);
                    acc0a = _mm512_dpbf16_ps(acc0a, a0v, reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(bk, zip2, bk1)));
                    if (cols2 != 0) {
                        const auto ck = _mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16);
                        const auto ck1 = _mm512_maskz_loadu_epi16(mask2, b + (kk + 1) * n + j + 16);
                        acc0b = _mm512_dpbf16_ps(acc0b, a0v, reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(ck, zip2, ck1)));
                    }
                }
                if (k2 != k) {
                    const auto p0 = static_cast<uint32_t>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(a0[k2]));
                    const auto a0v = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(p0)));
                    const auto zero = _mm512_setzero_si512();
                    const auto bk = _mm512_maskz_loadu_epi16(mask1, b + k2 * n + j);
                    acc0a = _mm512_dpbf16_ps(acc0a, a0v, reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(bk, zip2, zero)));
                    if (cols2 != 0) {
                        const auto ck = _mm512_maskz_loadu_epi16(mask2, b + k2 * n + j + 16);
                        acc0b = _mm512_dpbf16_ps(acc0b, a0v, reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(ck, zip2, zero)));
                    }
                }
                store_block(c + i * n + j, acc0a, cols1);
                if (cols2 != 0) {
                    store_block(c + i * n + j + 16, acc0b, cols2);
                }
            }
        }
    }

    /**
     * Packed bf16x2 GEMM. the packed 32-bit unit's two lanes are
     * independent outputs, so B's lanes are extracted per k-row pair (PAIR_LO/PAIR_HI)
     * and A's lane pairs are rebuilt by scalar bit ops into broadcast pairs.
     * Dual FP32 accumulators per column lane, zipped back to packed order on store.
     * @brief AVX512BF16 row-range GEMM (bf16x2, per-lane FP32 accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_bf16x2_rows_avx512bf16(const bfloat16x2_t* a, const bfloat16x2_t* b, bfloat16x2_t* c,
                                     const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        const auto pair_lo = _mm512_loadu_si512(PAIR_LO);
        const auto pair_hi = _mm512_loadu_si512(PAIR_HI);
        const auto zip_store = _mm512_loadu_si512(ZIP_STORE);
        const auto k2 = k & ~size_t{1};
        /**
         * Narrow both lane accumulators to bf16, then zip the lo/hi
         * planes back to packed unit order. Full 64-byte store when the block is complete.
         * @brief Store one 16-column packed bf16x2 accumulator block.
         * @param dst First packed output element of the C row block.
         * @param acc_lo FP32 accumulators for the lo bf16 lane, 16 columns.
         * @param acc_hi FP32 accumulators for the hi bf16 lane, 16 columns.
         * @param cols Valid packed-column count, [0, 16]; less than 16 stores via a bounded memcpy.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto store_block = [zip_store](bfloat16x2_t* dst, const __m512 acc_lo, const __m512 acc_hi, const size_t cols) {
            const auto lo = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(acc_lo));
            const auto hi = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(acc_hi));
            const auto zipped = _mm512_permutexvar_epi16(zip_store, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
            if (cols >= 16) {
                _mm512_storeu_si512(dst, zipped);
            } else {
                alignas(64) uint16_t tmp[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp), zipped);
                memcpy(reinterpret_cast<uint32_t*>(dst), tmp, cols * sizeof(uint32_t));
            }
        };
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            for (auto j = size_t{0}; j < n; j += 16) {
                const auto cols = n - j >= 16 ? size_t{16} : n - j;
                const auto mask = col_mask32(cols);
                auto acc_lo0 = _mm512_setzero_ps();
                auto acc_lo1 = _mm512_setzero_ps();
                auto acc_lo2 = _mm512_setzero_ps();
                auto acc_lo3 = _mm512_setzero_ps();
                auto acc_hi0 = _mm512_setzero_ps();
                auto acc_hi1 = _mm512_setzero_ps();
                auto acc_hi2 = _mm512_setzero_ps();
                auto acc_hi3 = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k2; kk += 2) {
                    const auto v0 = _mm512_maskz_loadu_epi16(mask, b + kk * n + j);
                    const auto v1 = _mm512_maskz_loadu_epi16(mask, b + (kk + 1) * n + j);
                    const auto bp_lo = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_lo, v1));
                    const auto bp_hi = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_hi, v1));
                    const auto e0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[kk]);
                    const auto f0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[kk + 1]);
                    const auto e1 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a1[kk]);
                    const auto f1 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a1[kk + 1]);
                    const auto e2 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a2[kk]);
                    const auto f2 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a2[kk + 1]);
                    const auto e3 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a3[kk]);
                    const auto f3 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a3[kk + 1]);
                    const auto a0_lo = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e0 & 0xFFFFu) | (f0 << 16))));
                    const auto a0_hi = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e0 >> 16) | (f0 & 0xFFFF0000u))));
                    const auto a1_lo = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e1 & 0xFFFFu) | (f1 << 16))));
                    const auto a1_hi = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e1 >> 16) | (f1 & 0xFFFF0000u))));
                    const auto a2_lo = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e2 & 0xFFFFu) | (f2 << 16))));
                    const auto a2_hi = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e2 >> 16) | (f2 & 0xFFFF0000u))));
                    const auto a3_lo = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e3 & 0xFFFFu) | (f3 << 16))));
                    const auto a3_hi = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e3 >> 16) | (f3 & 0xFFFF0000u))));
                    acc_lo0 = _mm512_dpbf16_ps(acc_lo0, a0_lo, bp_lo);
                    acc_hi0 = _mm512_dpbf16_ps(acc_hi0, a0_hi, bp_hi);
                    acc_lo1 = _mm512_dpbf16_ps(acc_lo1, a1_lo, bp_lo);
                    acc_hi1 = _mm512_dpbf16_ps(acc_hi1, a1_hi, bp_hi);
                    acc_lo2 = _mm512_dpbf16_ps(acc_lo2, a2_lo, bp_lo);
                    acc_hi2 = _mm512_dpbf16_ps(acc_hi2, a2_hi, bp_hi);
                    acc_lo3 = _mm512_dpbf16_ps(acc_lo3, a3_lo, bp_lo);
                    acc_hi3 = _mm512_dpbf16_ps(acc_hi3, a3_hi, bp_hi);
                }
                if (k2 != k) {
                    const auto zero = _mm512_setzero_si512();
                    const auto v0 = _mm512_maskz_loadu_epi16(mask, b + k2 * n + j);
                    const auto bp_lo = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_lo, zero));
                    const auto bp_hi = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_hi, zero));
                    const auto e0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[k2]);
                    const auto e1 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a1[k2]);
                    const auto e2 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a2[k2]);
                    const auto e3 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a3[k2]);
                    acc_lo0 = _mm512_dpbf16_ps(acc_lo0, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e0 & 0xFFFFu))), bp_lo);
                    acc_hi0 = _mm512_dpbf16_ps(acc_hi0, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e0 >> 16))), bp_hi);
                    acc_lo1 = _mm512_dpbf16_ps(acc_lo1, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e1 & 0xFFFFu))), bp_lo);
                    acc_hi1 = _mm512_dpbf16_ps(acc_hi1, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e1 >> 16))), bp_hi);
                    acc_lo2 = _mm512_dpbf16_ps(acc_lo2, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e2 & 0xFFFFu))), bp_lo);
                    acc_hi2 = _mm512_dpbf16_ps(acc_hi2, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e2 >> 16))), bp_hi);
                    acc_lo3 = _mm512_dpbf16_ps(acc_lo3, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e3 & 0xFFFFu))), bp_lo);
                    acc_hi3 = _mm512_dpbf16_ps(acc_hi3, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e3 >> 16))), bp_hi);
                }
                store_block(c + i * n + j, acc_lo0, acc_hi0, cols);
                store_block(c + (i + 1) * n + j, acc_lo1, acc_hi1, cols);
                store_block(c + (i + 2) * n + j, acc_lo2, acc_hi2, cols);
                store_block(c + (i + 3) * n + j, acc_lo3, acc_hi3, cols);
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            for (auto j = size_t{0}; j < n; j += 16) {
                const auto cols = n - j >= 16 ? size_t{16} : n - j;
                const auto mask = col_mask32(cols);
                auto acc_lo0 = _mm512_setzero_ps();
                auto acc_hi0 = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k2; kk += 2) {
                    const auto v0 = _mm512_maskz_loadu_epi16(mask, b + kk * n + j);
                    const auto v1 = _mm512_maskz_loadu_epi16(mask, b + (kk + 1) * n + j);
                    const auto bp_lo = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_lo, v1));
                    const auto bp_hi = reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_hi, v1));
                    const auto e0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[kk]);
                    const auto f0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[kk + 1]);
                    acc_lo0 = _mm512_dpbf16_ps(acc_lo0, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e0 & 0xFFFFu) | (f0 << 16)))), bp_lo);
                    acc_hi0 = _mm512_dpbf16_ps(acc_hi0, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>((e0 >> 16) | (f0 & 0xFFFF0000u)))), bp_hi);
                }
                if (k2 != k) {
                    const auto zero = _mm512_setzero_si512();
                    const auto v0 = _mm512_maskz_loadu_epi16(mask, b + k2 * n + j);
                    const auto e0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[k2]);
                    acc_lo0 = _mm512_dpbf16_ps(acc_lo0, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e0 & 0xFFFFu))),
                                               reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_lo, zero)));
                    acc_hi0 = _mm512_dpbf16_ps(acc_hi0, reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int32_t>(e0 >> 16))),
                                               reinterpret_cast<__m512bh>(_mm512_permutex2var_epi16(v0, pair_hi, zero)));
                }
                store_block(c + i * n + j, acc_lo0, acc_hi0, cols);
            }
        }
    }

    //TODO: AVX512-FP16 workaround (not supported on testing CPU.)
    /**
     * FP16 GEMM via AVX512F+F16C (avoiding 512-FP16): native fp16 multiply with FP32 accumulate.
     * Halves widened with F16C and multiplied in FP32 with FMA.
     * @brief AVX512F+F16C row-range GEMM (fp16, FP32 accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_f16_rows_avx512f16c(const float16_t* a, const float16_t* b, float16_t* c,
                                  const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        const auto k16 = k + 15 & ~size_t{15};
        std::vector<float32_t> panel_storage(4 * k16);
        /**
         * Widens one A row of fp16 into FP32 in 16-lane steps, so the inner loop broadcasts plain floats.
         * Tail lanes past kk_max load as zero through the mask, and the panel is padded to a multiple of 16.
         * @note Lambda subcall. Self-execution prohibited.
         * @brief Widen one fp16 A row into an FP32 panel.
         * @param src First element of the fp16 A row.
         * @param dst FP32 panel destination, at least kk_max rounded up to 16 elements.
         * @param kk_max Valid element count of the row (the inner dimension k).
         */
        const auto load_panel = [](const float16_t* src, float32_t* dst, const size_t kk_max) {
            for (auto kk = size_t{0}; kk < kk_max; kk += 16) {
                const auto rem = kk_max - kk;
                const auto mask = rem >= 16 ? 0xFFFFu : (uint32_t{1} << rem) - 1u;
                const auto h = _mm512_castsi512_si256(_mm512_maskz_loadu_epi16(mask, src + kk));
                _mm512_storeu_ps(dst + kk, _mm512_cvtph_ps(h));
            }
        };
        /**
         * Narrow 16 FP32 lanes to fp16 (RNE) as two 256-bit halves.
         * _mm512_cvtps_ph is coded HORRIBLY due to developers who don't know how to write simple safe conversions.
         * Split matches full behavior without unsafe sign-conversion.
         * @brief Store one 16-column fp32 accumulator block as fp16.
         * @param dst First output element of the C row block.
         * @param acc FP32 accumulators, 16 lanes.
         * @param cols Valid column count, [0, 16]; less than 16 stores via a bounded memcpy.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto store_block = [](float16_t* dst, const __m512 acc, const size_t cols) {
            // Unsafe macro -  _mm512_cvtps_ph uses __mask16-1, which trips sign-conversion. Narrow safely via the two 256-bit halves.
            const auto lo128 = _mm256_cvtps_ph(_mm512_castps512_ps256(acc), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const auto hi128 = _mm256_cvtps_ph(_mm512_extractf32x8_ps(acc, 1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const auto narrowed = _mm256_inserti128_si256(_mm256_castsi128_si256(lo128), hi128, 1);
            if (cols >= 16) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), narrowed);
            } else {
                alignas(32) uint16_t tmp[16];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), narrowed);
                memcpy(dst, tmp, cols * sizeof(uint16_t));
            }
        };
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            load_panel(a + i * k, panel_storage.data(), k);
            load_panel(a + (i + 1) * k, panel_storage.data() + k16, k);
            load_panel(a + (i + 2) * k, panel_storage.data() + 2 * k16, k);
            load_panel(a + (i + 3) * k, panel_storage.data() + 3 * k16, k);
            const auto* p0 = panel_storage.data();
            const auto* p1 = p0 + k16;
            const auto* p2 = p1 + k16;
            const auto* p3 = p2 + k16;
            for (auto j = size_t{0}; j < n; j += 32) {
                const auto rem = n - j;
                const auto cols1 = rem >= 16 ? size_t{16} : rem;
                const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                const auto mask1 = col_mask16(cols1);
                const auto mask2 = col_mask16(cols2);
                auto acc0a = _mm512_setzero_ps();
                auto acc1a = _mm512_setzero_ps();
                auto acc2a = _mm512_setzero_ps();
                auto acc3a = _mm512_setzero_ps();
                auto acc0b = _mm512_setzero_ps();
                auto acc1b = _mm512_setzero_ps();
                auto acc2b = _mm512_setzero_ps();
                auto acc3b = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bva = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_maskz_loadu_epi16(mask1, b + kk * n + j)));
                    acc0a = _mm512_fmadd_ps(_mm512_set1_ps(p0[kk]), bva, acc0a);
                    acc1a = _mm512_fmadd_ps(_mm512_set1_ps(p1[kk]), bva, acc1a);
                    acc2a = _mm512_fmadd_ps(_mm512_set1_ps(p2[kk]), bva, acc2a);
                    acc3a = _mm512_fmadd_ps(_mm512_set1_ps(p3[kk]), bva, acc3a);
                    if (cols2 != 0) {
                        const auto bvb = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16)));
                        acc0b = _mm512_fmadd_ps(_mm512_set1_ps(p0[kk]), bvb, acc0b);
                        acc1b = _mm512_fmadd_ps(_mm512_set1_ps(p1[kk]), bvb, acc1b);
                        acc2b = _mm512_fmadd_ps(_mm512_set1_ps(p2[kk]), bvb, acc2b);
                        acc3b = _mm512_fmadd_ps(_mm512_set1_ps(p3[kk]), bvb, acc3b);
                    }
                }
                store_block(c + i * n + j, acc0a, cols1);
                store_block(c + (i + 1) * n + j, acc1a, cols1);
                store_block(c + (i + 2) * n + j, acc2a, cols1);
                store_block(c + (i + 3) * n + j, acc3a, cols1);
                if (cols2 != 0) {
                    store_block(c + i * n + j + 16, acc0b, cols2);
                    store_block(c + (i + 1) * n + j + 16, acc1b, cols2);
                    store_block(c + (i + 2) * n + j + 16, acc2b, cols2);
                    store_block(c + (i + 3) * n + j + 16, acc3b, cols2);
                }
            }
        }
        for (; i < row_end; ++i) {
            load_panel(a + i * k, panel_storage.data(), k);
            const auto* p0 = panel_storage.data();
            for (auto j = size_t{0}; j < n; j += 32) {
                const auto rem = n - j;
                const auto cols1 = rem >= 16 ? size_t{16} : rem;
                const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                const auto mask1 = col_mask16(cols1);
                const auto mask2 = col_mask16(cols2);
                auto acc0a = _mm512_setzero_ps();
                auto acc0b = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bva = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_maskz_loadu_epi16(mask1, b + kk * n + j)));
                    acc0a = _mm512_fmadd_ps(_mm512_set1_ps(p0[kk]), bva, acc0a);
                    if (cols2 != 0) {
                        const auto bvb = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16)));
                        acc0b = _mm512_fmadd_ps(_mm512_set1_ps(p0[kk]), bvb, acc0b);
                    }
                }
                store_block(c + i * n + j, acc0a, cols1);
                if (cols2 != 0) {
                    store_block(c + i * n + j + 16, acc0b, cols2);
                }
            }
        }
    }

    /*
     * Packed fp16x2 GEMM: the packed unit's two lanes are collapsed out of each
     * B vector with permutexvar (COLLAPSE_LO/HI), widened with F16C, and FMA'd
     * into dual FP32 accumulators; A lanes are scalar-extracted broadcasts.
     * Zipped back to packed order on store.
     * @brief AVX512F+F16C row-range GEMM (fp16x2, per-lane FP32 accumulate).
     */

    /**
     * Packed FP16x2 GEMM: the packed unit's two lanes are collapsed out of each B vector, widened,
     * and FMA'ed into 2x FP32 accum. Zipped back into packed order during store.
     * @brief AVX512F+F16C row-range GEMM (fp16x2, per-lane FP32 accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_f16x2_rows_avx512f16c(const float16x2_t* a, const float16x2_t* b, float16x2_t* c,
                                    const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        const auto collapse_lo = _mm512_loadu_si512(COLLAPSE_LO);
        const auto collapse_hi = _mm512_loadu_si512(COLLAPSE_HI);
        const auto zip_store = _mm512_loadu_si512(ZIP_STORE);
        /**
         * Narrow both lane accumulators to fp16 as two 256-bit halves
         * then zip the lo/hi planes back to packed unit order
         * Full 64-byte store when complete!
         * Correct function call for this is again coded by idiots. sign-conversion trips.
         * @brief Store one 16-column packed fp16x2 accumulator block.
         * @param dst First packed output element of the C row block.
         * @param acc_lo FP32 accumulators for the lo fp16 lane, 16 columns.
         * @param acc_hi FP32 accumulators for the hi fp16 lane, 16 columns.
         * @param cols Valid packed-column count, [0, 16]; less than 16 stores via a bounded memcpy.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto store_block = [zip_store](float16x2_t* dst, const __m512 acc_lo, const __m512 acc_hi, const size_t cols) {
            const auto lo = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_cvtps_ph(_mm512_castps512_ps256(acc_lo), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
                                                    _mm256_cvtps_ph(_mm512_extractf32x8_ps(acc_lo, 1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 1);
            const auto hi = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_cvtps_ph(_mm512_castps512_ps256(acc_hi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
                                                    _mm256_cvtps_ph(_mm512_extractf32x8_ps(acc_hi, 1), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 1);
            const auto zipped = _mm512_permutexvar_epi16(zip_store, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
            if (cols >= 16) {
                _mm512_storeu_si512(dst, zipped);
            } else {
                alignas(64) uint16_t tmp[32];
                _mm512_storeu_si512(tmp, zipped);
                memcpy(dst, tmp, cols * sizeof(uint32_t));
            }
        };
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            for (auto j = size_t{0}; j < n; j += 16) {
                const auto cols = n - j >= 16 ? size_t{16} : n - j;
                const auto mask = col_mask32(cols);
                auto acc_lo0 = _mm512_setzero_ps();
                auto acc_lo1 = _mm512_setzero_ps();
                auto acc_lo2 = _mm512_setzero_ps();
                auto acc_lo3 = _mm512_setzero_ps();
                auto acc_hi0 = _mm512_setzero_ps();
                auto acc_hi1 = _mm512_setzero_ps();
                auto acc_hi2 = _mm512_setzero_ps();
                auto acc_hi3 = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto v = _mm512_maskz_loadu_epi16(mask, b + kk * n + j);
                    const auto b_lo = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_permutexvar_epi16(collapse_lo, v)));
                    const auto b_hi = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_permutexvar_epi16(collapse_hi, v)));
                    const auto p0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[kk]);
                    const auto p1 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a1[kk]);
                    const auto p2 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a2[kk]);
                    const auto p3 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a3[kk]);
                    acc_lo0 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p0 & 0xFFFFu))), b_lo, acc_lo0);
                    acc_hi0 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p0 >> 16))), b_hi, acc_hi0);
                    acc_lo1 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p1 & 0xFFFFu))), b_lo, acc_lo1);
                    acc_hi1 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p1 >> 16))), b_hi, acc_hi1);
                    acc_lo2 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p2 & 0xFFFFu))), b_lo, acc_lo2);
                    acc_hi2 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p2 >> 16))), b_hi, acc_hi2);
                    acc_lo3 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p3 & 0xFFFFu))), b_lo, acc_lo3);
                    acc_hi3 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p3 >> 16))), b_hi, acc_hi3);
                }
                store_block(c + i * n + j, acc_lo0, acc_hi0, cols);
                store_block(c + (i + 1) * n + j, acc_lo1, acc_hi1, cols);
                store_block(c + (i + 2) * n + j, acc_lo2, acc_hi2, cols);
                store_block(c + (i + 3) * n + j, acc_lo3, acc_hi3, cols);
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            for (auto j = size_t{0}; j < n; j += 16) {
                const auto cols = n - j >= 16 ? size_t{16} : n - j;
                const auto mask = col_mask32(cols);
                auto acc_lo0 = _mm512_setzero_ps();
                auto acc_hi0 = _mm512_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto v = _mm512_maskz_loadu_epi16(mask, b + kk * n + j);
                    const auto b_lo = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_permutexvar_epi16(collapse_lo, v)));
                    const auto b_hi = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_permutexvar_epi16(collapse_hi, v)));
                    const auto p0 = Intrinsics::detail::bit_cast_memcpy<uint32_t>(a0[kk]);
                    acc_lo0 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p0 & 0xFFFFu))), b_lo, acc_lo0);
                    acc_hi0 = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(static_cast<uint16_t>(p0 >> 16))), b_hi, acc_hi0);
                }
                store_block(c + i * n + j, acc_lo0, acc_hi0, cols);
            }
        }
    }

    namespace {
        /**
         * Low-cols bits set over 64 byte lanes, where cols <= 16 per half-block.
         * Shift stays under 64.
         * @brief Mask of the first cols bytes of a zmm (8-bit B loads).
         * @param cols Valid byte count, [0, 16].
         * @return Mask with the low cols bits set.
         */
        [[nodiscard]] __mmask64 col_mask_bytes(const size_t cols) noexcept {
            assert(cols <= 16 && "col_mask_bytes only supports 0..16");
            return (uint64_t{1} << cols) - uint64_t{1};
        }

        /**
         * Replicate an (lo, hi) s16 pair across all 32-bit lanes.
         * @brief Broadcast two s16 k-slices of one A row to all lanes.
         * @param lo Low s16 of each pair (k-slice kk).
         * @param hi High s16 of each pair (k-slice kk+1).
         * @return zmm with the packed (lo, hi) pair in every 32-bit lane.
         */
        [[nodiscard]] __m512i broadcast_s16_pair(const int16_t lo, const int16_t hi) {
            const auto packed =
                static_cast<uint32_t>(static_cast<uint16_t>(lo)) |
                    (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16);
            return _mm512_set1_epi32(static_cast<int32_t>(packed));
        }

        /**
         * Zero-extends: [0,255] fits s16 exactly, so we can compute
         * the true product against the sign-extended int8 side. Masked-off lanes
         * are zero (never -1 as with sign extension) and dead to the zip.
         * @brief Masked load + widen of 16 B elements to s16, zero-extended.
         * @param p Pointer to the first of up to 16 B-row bytes.
         * @param mask Byte-lane mask over the 64-byte load.
         * @return zmm of 32 s16 lanes; masked-off lanes are zero.
         * @note Input int8_t.
         */
        [[nodiscard]] __m512i widen8_to_s16(const int8_t* p, const __mmask64 mask) {
            const auto raw = _mm512_maskz_loadu_epi8(mask, p);
            return _mm512_cvtepi8_epi16(_mm512_castsi512_si256(raw));
        }
        /**
         * Zero-extends: [0,255] fits s16 exactly, so we can compute
         * the true product against the sign-extended int8 side. Masked-off lanes
         * are zero (never -1 as with sign extension) and dead to the zip.
         * @brief Masked load + widen of 16 B elements to s16, zero-extended.
         * @param p Pointer to the first of up to 16 B-row bytes.
         * @param mask Byte-lane mask over the 64-byte load.
         * @return zmm of 32 s16 lanes; masked-off lanes are zero.
         * @note Input uint8_t.
         */
        [[nodiscard]] __m512i widen8_to_s16(const uint8_t* p, const __mmask64 mask) {
            const auto raw = _mm512_maskz_loadu_epi8(mask, p);
            return _mm512_cvtepu8_epi16(_mm512_castsi512_si256(raw));
        }

        /**
         * Store path for the integer kernels: the accumulator lanes are
         * 32-bit (wrapping), the output dtype T is 8 or 16 bits, and masked
         * stores cannot narrow. Dump to stack array and then take the firsts cols lanes,
         * storing individually with static_cast<T>(lane).
         * @brief Store the low cols lanes of a wrapping 32-bit accumulator, truncated to T.
         * @param dst Output position; receives exactly cols elements.
         * @param acc 32-bit wrapping accumulators; lane jj becomes dst[jj].
         * @param cols Elements to store, [0, 16].
         */
        template<typename T>
        void store_truncated(T* dst, const __m512i acc, const size_t cols) {
            assert(cols <= 16 && "store_truncated only supports 0..16");
            alignas(64) int32_t tmp[16];
            _mm512_store_si512(tmp, acc);
            for (auto jj = size_t{0}; jj < cols; ++jj) {
                dst[jj] = static_cast<T>(tmp[jj]);
            }
        }

        /**
         * Multiplies the A rows [row_begin, row_end) by B, accumulating in
         * wrapping 32-bit and truncating each result to T on store. Exact
         * for both signed and unsigned inputs, as each operand widens to 16
         * bits losslessly before the product, so every result is
         * bit-identical.
         * @brief Row-range GEMM (8-bit A and B, wrapping 32-bit accumulate).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         * @note Computes C[i][j] = sum_k A[i][k]*B[k][j].
         */
        template<typename T>
        void gemm_rows_vnni_w8(const T* a, const T* b, T* c, const size_t k, const size_t n,
                               const size_t row_begin, const size_t row_end) {
            const auto zip2 = _mm512_loadu_si512(ZIP2);
            const auto k2 = k & ~size_t{1};
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                const auto* a0 = a + i * k;
                const auto* a1 = a0 + k;
                const auto* a2 = a1 + k;
                const auto* a3 = a2 + k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask_bytes(cols1);
                    const auto mask2 = col_mask_bytes(cols2);
                    auto acc0a = _mm512_setzero_si512();
                    auto acc1a = _mm512_setzero_si512();
                    auto acc2a = _mm512_setzero_si512();
                    auto acc3a = _mm512_setzero_si512();
                    auto acc0b = _mm512_setzero_si512();
                    auto acc1b = _mm512_setzero_si512();
                    auto acc2b = _mm512_setzero_si512();
                    auto acc3b = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0v = broadcast_s16_pair(static_cast<int16_t>(a0[kk]), static_cast<int16_t>(a0[kk + 1]));
                        const auto a1v = broadcast_s16_pair(static_cast<int16_t>(a1[kk]), static_cast<int16_t>(a1[kk + 1]));
                        const auto a2v = broadcast_s16_pair(static_cast<int16_t>(a2[kk]), static_cast<int16_t>(a2[kk + 1]));
                        const auto a3v = broadcast_s16_pair(static_cast<int16_t>(a3[kk]), static_cast<int16_t>(a3[kk + 1]));
                        const auto bk = widen8_to_s16(b + kk * n + j, mask1);
                        const auto bk1 = widen8_to_s16(b + (kk + 1) * n + j, mask1);
                        const auto bpa = _mm512_permutex2var_epi16(bk, zip2, bk1);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        acc1a = _mm512_dpwssd_epi32(acc1a, a1v, bpa);
                        acc2a = _mm512_dpwssd_epi32(acc2a, a2v, bpa);
                        acc3a = _mm512_dpwssd_epi32(acc3a, a3v, bpa);
                        if (cols2 != 0) {
                            const auto ck = widen8_to_s16(b + kk * n + j + 16, mask2);
                            const auto ck1 = widen8_to_s16(b + (kk + 1) * n + j + 16, mask2);
                            const auto bpb = _mm512_permutex2var_epi16(ck, zip2, ck1);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                            acc1b = _mm512_dpwssd_epi32(acc1b, a1v, bpb);
                            acc2b = _mm512_dpwssd_epi32(acc2b, a2v, bpb);
                            acc3b = _mm512_dpwssd_epi32(acc3b, a3v, bpb);
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0v = broadcast_s16_pair(static_cast<int16_t>(a0[k2]), int16_t{0});
                        const auto a1v = broadcast_s16_pair(static_cast<int16_t>(a1[k2]), int16_t{0});
                        const auto a2v = broadcast_s16_pair(static_cast<int16_t>(a2[k2]), int16_t{0});
                        const auto a3v = broadcast_s16_pair(static_cast<int16_t>(a3[k2]), int16_t{0});
                        const auto bpa = _mm512_permutex2var_epi16(widen8_to_s16(b + k2 * n + j, mask1), zip2, zero);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        acc1a = _mm512_dpwssd_epi32(acc1a, a1v, bpa);
                        acc2a = _mm512_dpwssd_epi32(acc2a, a2v, bpa);
                        acc3a = _mm512_dpwssd_epi32(acc3a, a3v, bpa);
                        if (cols2 != 0) {
                            const auto bpb = _mm512_permutex2var_epi16(widen8_to_s16(b + k2 * n + j + 16, mask2), zip2, zero);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                            acc1b = _mm512_dpwssd_epi32(acc1b, a1v, bpb);
                            acc2b = _mm512_dpwssd_epi32(acc2b, a2v, bpb);
                            acc3b = _mm512_dpwssd_epi32(acc3b, a3v, bpb);
                        }
                    }
                    store_truncated(c + i * n + j, acc0a, cols1);
                    store_truncated(c + (i + 1) * n + j, acc1a, cols1);
                    store_truncated(c + (i + 2) * n + j, acc2a, cols1);
                    store_truncated(c + (i + 3) * n + j, acc3a, cols1);
                    if (cols2 != 0) {
                        store_truncated(c + i * n + j + 16, acc0b, cols2);
                        store_truncated(c + (i + 1) * n + j + 16, acc1b, cols2);
                        store_truncated(c + (i + 2) * n + j + 16, acc2b, cols2);
                        store_truncated(c + (i + 3) * n + j + 16, acc3b, cols2);
                    }
                }
            }
            for (; i < row_end; ++i) {
                const auto* a0 = a + i * k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask_bytes(cols1);
                    const auto mask2 = col_mask_bytes(cols2);
                    auto acc0a = _mm512_setzero_si512();
                    auto acc0b = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0v = broadcast_s16_pair(static_cast<int16_t>(a0[kk]), static_cast<int16_t>(a0[kk + 1]));
                        const auto bk = widen8_to_s16(b + kk * n + j, mask1);
                        const auto bk1 = widen8_to_s16(b + (kk + 1) * n + j, mask1);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, _mm512_permutex2var_epi16(bk, zip2, bk1));
                        if (cols2 != 0) {
                            const auto ck = widen8_to_s16(b + kk * n + j + 16, mask2);
                            const auto ck1 = widen8_to_s16(b + (kk + 1) * n + j + 16, mask2);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, _mm512_permutex2var_epi16(ck, zip2, ck1));
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0v = broadcast_s16_pair(static_cast<int16_t>(a0[k2]), int16_t{0});
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v,
                                                    _mm512_permutex2var_epi16(widen8_to_s16(b + k2 * n + j, mask1), zip2, zero));
                        if (cols2 != 0) {
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v,
                                                        _mm512_permutex2var_epi16(widen8_to_s16(b + k2 * n + j + 16, mask2), zip2, zero));
                        }
                    }
                    store_truncated(c + i * n + j, acc0a, cols1);
                    if (cols2 != 0) {
                        store_truncated(c + i * n + j + 16, acc0b, cols2);
                    }
                }
            }
        }

        /**
         * Multiplies the A rows [row_begin, row_end) by B, accumulating in
         * wrapping 32-bit and truncating each result to int16_t on store.
         * Every product is exact in 32 bits, so every result is
         * bit-identical.
         * @brief Row-range GEMM (int16, wrapping 32-bit accumulate).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         * @note Computes C[i][j] = sum_k A[i][k]*B[k][j].
         */
        void gemm_rows_vnni_i16(const int16_t* a, const int16_t* b, int16_t* c, const size_t k, const size_t n,
                                const size_t row_begin, const size_t row_end) {
            const auto zip2 = _mm512_loadu_si512(ZIP2);
            const auto k2 = k & ~size_t{1};
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                const auto* a0 = a + i * k;
                const auto* a1 = a0 + k;
                const auto* a2 = a1 + k;
                const auto* a3 = a2 + k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask16(cols1);
                    const auto mask2 = col_mask16(cols2);
                    auto acc0a = _mm512_setzero_si512();
                    auto acc1a = _mm512_setzero_si512();
                    auto acc2a = _mm512_setzero_si512();
                    auto acc3a = _mm512_setzero_si512();
                    auto acc0b = _mm512_setzero_si512();
                    auto acc1b = _mm512_setzero_si512();
                    auto acc2b = _mm512_setzero_si512();
                    auto acc3b = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0v = broadcast_s16_pair(a0[kk], a0[kk + 1]);
                        const auto a1v = broadcast_s16_pair(a1[kk], a1[kk + 1]);
                        const auto a2v = broadcast_s16_pair(a2[kk], a2[kk + 1]);
                        const auto a3v = broadcast_s16_pair(a3[kk], a3[kk + 1]);
                        const auto bk = _mm512_maskz_loadu_epi16(mask1, b + kk * n + j);
                        const auto bk1 = _mm512_maskz_loadu_epi16(mask1, b + (kk + 1) * n + j);
                        const auto bpa = _mm512_permutex2var_epi16(bk, zip2, bk1);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        acc1a = _mm512_dpwssd_epi32(acc1a, a1v, bpa);
                        acc2a = _mm512_dpwssd_epi32(acc2a, a2v, bpa);
                        acc3a = _mm512_dpwssd_epi32(acc3a, a3v, bpa);
                        if (cols2 != 0) {
                            const auto ck = _mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16);
                            const auto ck1 = _mm512_maskz_loadu_epi16(mask2, b + (kk + 1) * n + j + 16);
                            const auto bpb = _mm512_permutex2var_epi16(ck, zip2, ck1);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                            acc1b = _mm512_dpwssd_epi32(acc1b, a1v, bpb);
                            acc2b = _mm512_dpwssd_epi32(acc2b, a2v, bpb);
                            acc3b = _mm512_dpwssd_epi32(acc3b, a3v, bpb);
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0v = broadcast_s16_pair(a0[k2], int16_t{0});
                        const auto a1v = broadcast_s16_pair(a1[k2], int16_t{0});
                        const auto a2v = broadcast_s16_pair(a2[k2], int16_t{0});
                        const auto a3v = broadcast_s16_pair(a3[k2], int16_t{0});
                        const auto bpa = _mm512_permutex2var_epi16(_mm512_maskz_loadu_epi16(mask1, b + k2 * n + j), zip2, zero);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        acc1a = _mm512_dpwssd_epi32(acc1a, a1v, bpa);
                        acc2a = _mm512_dpwssd_epi32(acc2a, a2v, bpa);
                        acc3a = _mm512_dpwssd_epi32(acc3a, a3v, bpa);
                        if (cols2 != 0) {
                            const auto bpb = _mm512_permutex2var_epi16(_mm512_maskz_loadu_epi16(mask2, b + k2 * n + j + 16), zip2, zero);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                            acc1b = _mm512_dpwssd_epi32(acc1b, a1v, bpb);
                            acc2b = _mm512_dpwssd_epi32(acc2b, a2v, bpb);
                            acc3b = _mm512_dpwssd_epi32(acc3b, a3v, bpb);
                        }
                    }
                    store_truncated(c + i * n + j, acc0a, cols1);
                    store_truncated(c + (i + 1) * n + j, acc1a, cols1);
                    store_truncated(c + (i + 2) * n + j, acc2a, cols1);
                    store_truncated(c + (i + 3) * n + j, acc3a, cols1);
                    if (cols2 != 0) {
                        store_truncated(c + i * n + j + 16, acc0b, cols2);
                        store_truncated(c + (i + 1) * n + j + 16, acc1b, cols2);
                        store_truncated(c + (i + 2) * n + j + 16, acc2b, cols2);
                        store_truncated(c + (i + 3) * n + j + 16, acc3b, cols2);
                    }
                }
            }
            for (; i < row_end; ++i) {
                const auto* a0 = a + i * k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask16(cols1);
                    const auto mask2 = col_mask16(cols2);
                    auto acc0a = _mm512_setzero_si512();
                    auto acc0b = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0v = broadcast_s16_pair(a0[kk], a0[kk + 1]);
                        const auto bk = _mm512_maskz_loadu_epi16(mask1, b + kk * n + j);
                        const auto bk1 = _mm512_maskz_loadu_epi16(mask1, b + (kk + 1) * n + j);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, _mm512_permutex2var_epi16(bk, zip2, bk1));
                        if (cols2 != 0) {
                            const auto ck = _mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16);
                            const auto ck1 = _mm512_maskz_loadu_epi16(mask2, b + (kk + 1) * n + j + 16);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, _mm512_permutex2var_epi16(ck, zip2, ck1));
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0v = broadcast_s16_pair(a0[k2], int16_t{0});
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v,
                                                    _mm512_permutex2var_epi16(_mm512_maskz_loadu_epi16(mask1, b + k2 * n + j), zip2, zero));
                        if (cols2 != 0) {
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v,
                                                        _mm512_permutex2var_epi16(_mm512_maskz_loadu_epi16(mask2, b + k2 * n + j + 16), zip2, zero));
                        }
                    }
                    store_truncated(c + i * n + j, acc0a, cols1);
                    if (cols2 != 0) {
                        store_truncated(c + i * n + j + 16, acc0b, cols2);
                    }
                }
            }
        }

        /**
         * Multiplies the A rows [row_begin, row_end) by B, accumulating in
         * wrapping 32-bit and truncating each result to uint16_t on store.
         * Exact for unsigned inputs: both operands are offset into signed
         * range before the product, and the offset correction is applied
         * mod 2^32, so every result is bit-identical.
         * @brief Row-range GEMM (uint16, wrapping 32-bit accumulate).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. With a' = a - 32768
         * and b' = b - 32768: sum(a*b) = sum(a'*b') + 32768*(sum(a') + sum(b'))
         * + k*2^30 (mod 2^32).
         */
        void gemm_rows_vnni_u16(const uint16_t* a, const uint16_t* b, uint16_t* c, const size_t k, const size_t n,
                                const size_t row_begin, const size_t row_end) {
            const auto zip2 = _mm512_loadu_si512(ZIP2);
            const auto flip = _mm512_set1_epi16(static_cast<int16_t>(uint16_t{0x8000}));
            const auto ones = _mm512_set1_epi32(0x00010001);
            const auto kterm = _mm512_set1_epi32(static_cast<int32_t>(static_cast<uint32_t>(k % size_t{4}) << 30));
            const auto k2 = k & ~size_t{1};
            /**
             * Flips the sign bit of both u16 k-slices (x ^ 0x8000), which biases them into the s16 domain.
             * The bias is undone by the correction terms after the k loop.
             * @note Lambda subcall. Self-execution prohibited.
             * @brief Bias two u16 k-slices to s16 and broadcast them to all lanes.
             * @param lo u16 A element at k.
             * @param hi u16 A element at k + 1.
             * @return __m512i holding the biased (lo, hi) pair in every 32-bit lane.
             */
            const auto flip_pair = [](const uint16_t lo, const uint16_t hi) {
                return broadcast_s16_pair(static_cast<int16_t>(static_cast<uint16_t>(lo ^ uint16_t{0x8000})),
                                          static_cast<int16_t>(static_cast<uint16_t>(hi ^ uint16_t{0x8000})));
            };
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                const auto* a0 = a + i * k;
                const auto* a1 = a0 + k;
                const auto* a2 = a1 + k;
                const auto* a3 = a2 + k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask16(cols1);
                    const auto mask2 = col_mask16(cols2);
                    auto acc0a = _mm512_setzero_si512();
                    auto acc1a = _mm512_setzero_si512();
                    auto acc2a = _mm512_setzero_si512();
                    auto acc3a = _mm512_setzero_si512();
                    auto acc0b = _mm512_setzero_si512();
                    auto acc1b = _mm512_setzero_si512();
                    auto acc2b = _mm512_setzero_si512();
                    auto acc3b = _mm512_setzero_si512();
                    auto asum0 = _mm512_setzero_si512();
                    auto asum1 = _mm512_setzero_si512();
                    auto asum2 = _mm512_setzero_si512();
                    auto asum3 = _mm512_setzero_si512();
                    auto bsum_a = _mm512_setzero_si512();
                    auto bsum_b = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0v = flip_pair(a0[kk], a0[kk + 1]);
                        const auto a1v = flip_pair(a1[kk], a1[kk + 1]);
                        const auto a2v = flip_pair(a2[kk], a2[kk + 1]);
                        const auto a3v = flip_pair(a3[kk], a3[kk + 1]);
                        const auto bk = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask1, b + kk * n + j), flip);
                        const auto bk1 = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask1, b + (kk + 1) * n + j), flip);
                        const auto bpa = _mm512_permutex2var_epi16(bk, zip2, bk1);
                        bsum_a = _mm512_dpwssd_epi32(bsum_a, ones, bpa);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        acc1a = _mm512_dpwssd_epi32(acc1a, a1v, bpa);
                        acc2a = _mm512_dpwssd_epi32(acc2a, a2v, bpa);
                        acc3a = _mm512_dpwssd_epi32(acc3a, a3v, bpa);
                        asum0 = _mm512_dpwssd_epi32(asum0, a0v, ones);
                        asum1 = _mm512_dpwssd_epi32(asum1, a1v, ones);
                        asum2 = _mm512_dpwssd_epi32(asum2, a2v, ones);
                        asum3 = _mm512_dpwssd_epi32(asum3, a3v, ones);
                        if (cols2 != 0) {
                            const auto ck = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16), flip);
                            const auto ck1 = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask2, b + (kk + 1) * n + j + 16), flip);
                            const auto bpb = _mm512_permutex2var_epi16(ck, zip2, ck1);
                            bsum_b = _mm512_dpwssd_epi32(bsum_b, ones, bpb);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                            acc1b = _mm512_dpwssd_epi32(acc1b, a1v, bpb);
                            acc2b = _mm512_dpwssd_epi32(acc2b, a2v, bpb);
                            acc3b = _mm512_dpwssd_epi32(acc3b, a3v, bpb);
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0v = flip_pair(a0[k2], uint16_t{0x8000});
                        const auto a1v = flip_pair(a1[k2], uint16_t{0x8000});
                        const auto a2v = flip_pair(a2[k2], uint16_t{0x8000});
                        const auto a3v = flip_pair(a3[k2], uint16_t{0x8000});
                        const auto bk = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask1, b + k2 * n + j), flip);
                        const auto bpa = _mm512_permutex2var_epi16(bk, zip2, zero);
                        bsum_a = _mm512_dpwssd_epi32(bsum_a, ones, bpa);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        acc1a = _mm512_dpwssd_epi32(acc1a, a1v, bpa);
                        acc2a = _mm512_dpwssd_epi32(acc2a, a2v, bpa);
                        acc3a = _mm512_dpwssd_epi32(acc3a, a3v, bpa);
                        asum0 = _mm512_dpwssd_epi32(asum0, a0v, ones);
                        asum1 = _mm512_dpwssd_epi32(asum1, a1v, ones);
                        asum2 = _mm512_dpwssd_epi32(asum2, a2v, ones);
                        asum3 = _mm512_dpwssd_epi32(asum3, a3v, ones);
                        if (cols2 != 0) {
                            const auto ck = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask2, b + k2 * n + j + 16), flip);
                            const auto bpb = _mm512_permutex2var_epi16(ck, zip2, zero);
                            bsum_b = _mm512_dpwssd_epi32(bsum_b, ones, bpb);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                            acc1b = _mm512_dpwssd_epi32(acc1b, a1v, bpb);
                            acc2b = _mm512_dpwssd_epi32(acc2b, a2v, bpb);
                            acc3b = _mm512_dpwssd_epi32(acc3b, a3v, bpb);
                        }
                    }
                    const auto res0a = _mm512_add_epi32(_mm512_add_epi32(acc0a, _mm512_slli_epi32(_mm512_add_epi32(asum0, bsum_a), 15)), kterm);
                    const auto res1a = _mm512_add_epi32(_mm512_add_epi32(acc1a, _mm512_slli_epi32(_mm512_add_epi32(asum1, bsum_a), 15)), kterm);
                    const auto res2a = _mm512_add_epi32(_mm512_add_epi32(acc2a, _mm512_slli_epi32(_mm512_add_epi32(asum2, bsum_a), 15)), kterm);
                    const auto res3a = _mm512_add_epi32(_mm512_add_epi32(acc3a, _mm512_slli_epi32(_mm512_add_epi32(asum3, bsum_a), 15)), kterm);
                    store_truncated(c + i * n + j, res0a, cols1);
                    store_truncated(c + (i + 1) * n + j, res1a, cols1);
                    store_truncated(c + (i + 2) * n + j, res2a, cols1);
                    store_truncated(c + (i + 3) * n + j, res3a, cols1);
                    if (cols2 != 0) {
                        const auto res0b = _mm512_add_epi32(_mm512_add_epi32(acc0b, _mm512_slli_epi32(_mm512_add_epi32(asum0, bsum_b), 15)), kterm);
                        const auto res1b = _mm512_add_epi32(_mm512_add_epi32(acc1b, _mm512_slli_epi32(_mm512_add_epi32(asum1, bsum_b), 15)), kterm);
                        const auto res2b = _mm512_add_epi32(_mm512_add_epi32(acc2b, _mm512_slli_epi32(_mm512_add_epi32(asum2, bsum_b), 15)), kterm);
                        const auto res3b = _mm512_add_epi32(_mm512_add_epi32(acc3b, _mm512_slli_epi32(_mm512_add_epi32(asum3, bsum_b), 15)), kterm);
                        store_truncated(c + i * n + j + 16, res0b, cols2);
                        store_truncated(c + (i + 1) * n + j + 16, res1b, cols2);
                        store_truncated(c + (i + 2) * n + j + 16, res2b, cols2);
                        store_truncated(c + (i + 3) * n + j + 16, res3b, cols2);
                    }
                }
            }
            for (; i < row_end; ++i) {
                const auto* a0 = a + i * k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask16(cols1);
                    const auto mask2 = col_mask16(cols2);
                    auto acc0a = _mm512_setzero_si512();
                    auto acc0b = _mm512_setzero_si512();
                    auto asum0 = _mm512_setzero_si512();
                    auto bsum_a = _mm512_setzero_si512();
                    auto bsum_b = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0v = flip_pair(a0[kk], a0[kk + 1]);
                        const auto bk = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask1, b + kk * n + j), flip);
                        const auto bk1 = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask1, b + (kk + 1) * n + j), flip);
                        const auto bpa = _mm512_permutex2var_epi16(bk, zip2, bk1);
                        bsum_a = _mm512_dpwssd_epi32(bsum_a, ones, bpa);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        asum0 = _mm512_dpwssd_epi32(asum0, a0v, ones);
                        if (cols2 != 0) {
                            const auto ck = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask2, b + kk * n + j + 16), flip);
                            const auto ck1 = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask2, b + (kk + 1) * n + j + 16), flip);
                            const auto bpb = _mm512_permutex2var_epi16(ck, zip2, ck1);
                            bsum_b = _mm512_dpwssd_epi32(bsum_b, ones, bpb);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0v = flip_pair(a0[k2], uint16_t{0x8000});
                        const auto bk = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask1, b + k2 * n + j), flip);
                        const auto bpa = _mm512_permutex2var_epi16(bk, zip2, zero);
                        bsum_a = _mm512_dpwssd_epi32(bsum_a, ones, bpa);
                        acc0a = _mm512_dpwssd_epi32(acc0a, a0v, bpa);
                        asum0 = _mm512_dpwssd_epi32(asum0, a0v, ones);
                        if (cols2 != 0) {
                            const auto ck = _mm512_xor_si512(_mm512_maskz_loadu_epi16(mask2, b + k2 * n + j + 16), flip);
                            const auto bpb = _mm512_permutex2var_epi16(ck, zip2, zero);
                            bsum_b = _mm512_dpwssd_epi32(bsum_b, ones, bpb);
                            acc0b = _mm512_dpwssd_epi32(acc0b, a0v, bpb);
                        }
                    }
                    const auto res0a = _mm512_add_epi32(_mm512_add_epi32(acc0a, _mm512_slli_epi32(_mm512_add_epi32(asum0, bsum_a), 15)), kterm);
                    store_truncated(c + i * n + j, res0a, cols1);
                    if (cols2 != 0) {
                        const auto res0b = _mm512_add_epi32(_mm512_add_epi32(acc0b, _mm512_slli_epi32(_mm512_add_epi32(asum0, bsum_b), 15)), kterm);
                        store_truncated(c + i * n + j + 16, res0b, cols2);
                    }
                }
            }
        }

        /**
         * Multiplies the A rows [row_begin, row_end) by B, keeping the low
         * 32 bits of each product and accumulating mod 2^32. The low bits
         * are sign-agnostic, so one implementation serves int32 and uint32
         * and every result is bit-identical to a scalar wrapping reference.
         * @brief Row-range GEMM (32-bit elements, mod-2^32 arithmetic).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^32.
         */
        template<typename T>
        void gemm_rows_w32_avx512(const T* a, const T* b, T* c, const size_t k, const size_t n,
                                  const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                const auto* a0 = a + i * k;
                const auto* a1 = a0 + k;
                const auto* a2 = a1 + k;
                const auto* a3 = a2 + k;
                for (auto j = size_t{0}; j < n; j += 16) {
                    const auto remaining = n - j;
                    const auto mask = static_cast<__mmask16>(remaining >= 16 ? 0xFFFFu : (uint32_t{1} << remaining) - 1u);
                    auto acc0 = _mm512_setzero_si512();
                    auto acc1 = _mm512_setzero_si512();
                    auto acc2 = _mm512_setzero_si512();
                    auto acc3 = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        const auto bv = _mm512_maskz_loadu_epi32(mask, static_cast<const void*>(b + kk * n + j));
                        acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(a0[kk])), bv));
                        acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(a1[kk])), bv));
                        acc2 = _mm512_add_epi32(acc2, _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(a2[kk])), bv));
                        acc3 = _mm512_add_epi32(acc3, _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(a3[kk])), bv));
                    }
                    _mm512_mask_storeu_epi32(static_cast<void*>(c + i * n + j), mask, acc0);
                    _mm512_mask_storeu_epi32(static_cast<void*>(c + (i + 1) * n + j), mask, acc1);
                    _mm512_mask_storeu_epi32(static_cast<void*>(c + (i + 2) * n + j), mask, acc2);
                    _mm512_mask_storeu_epi32(static_cast<void*>(c + (i + 3) * n + j), mask, acc3);
                }
            }
            for (; i < row_end; ++i) {
                const auto* a0 = a + i * k;
                for (auto j = size_t{0}; j < n; j += 16) {
                    const auto remaining = n - j;
                    const auto mask = static_cast<__mmask16>(remaining >= 16 ? 0xFFFFu : (uint32_t{1} << remaining) - 1u);
                    auto acc = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(_mm512_set1_epi32(static_cast<int32_t>(a0[kk])),
                                                                         _mm512_maskz_loadu_epi32(mask, static_cast<const void*>(b + kk * n + j))));
                    }
                    _mm512_mask_storeu_epi32(static_cast<void*>(c + i * n + j), mask, acc);
                }
            }
        }


        /**
         * Multiplies the A rows [row_begin, row_end) by B, keeping the low
         * 64 bits of each product and accumulating mod 2^64. The low bits
         * are sign-agnostic, so one implementation serves int64 and uint64
         * and every result is bit-identical to a scalar wrapping reference.
         * @brief Row-range GEMM (64-bit elements, mod-2^64 arithmetic).
         * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n.
         * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^64.
         */
        template<typename T>
        void gemm_rows_w64_avx512dq(const T* a, const T* b, T* c, const size_t k, const size_t n,
                                    const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                const auto* a0 = a + i * k;
                const auto* a1 = a0 + k;
                const auto* a2 = a1 + k;
                const auto* a3 = a2 + k;
                for (auto j = size_t{0}; j < n; j += 8) {
                    const auto remaining = n - j;
                    const auto mask = static_cast<__mmask8>(remaining >= 8 ? 0xFFu : (uint32_t{1} << remaining) - 1u);
                    auto acc0 = _mm512_setzero_si512();
                    auto acc1 = _mm512_setzero_si512();
                    auto acc2 = _mm512_setzero_si512();
                    auto acc3 = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        const auto bv = _mm512_maskz_loadu_epi64(mask, static_cast<const void*>(b + kk * n + j));
                        acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi64(_mm512_set1_epi64(static_cast<int64_t>(a0[kk])), bv));
                        acc1 = _mm512_add_epi64(acc1, _mm512_mullo_epi64(_mm512_set1_epi64(static_cast<int64_t>(a1[kk])), bv));
                        acc2 = _mm512_add_epi64(acc2, _mm512_mullo_epi64(_mm512_set1_epi64(static_cast<int64_t>(a2[kk])), bv));
                        acc3 = _mm512_add_epi64(acc3, _mm512_mullo_epi64(_mm512_set1_epi64(static_cast<int64_t>(a3[kk])), bv));
                    }
                    _mm512_mask_storeu_epi64(static_cast<void*>(c + i * n + j), mask, acc0);
                    _mm512_mask_storeu_epi64(static_cast<void*>(c + (i + 1) * n + j), mask, acc1);
                    _mm512_mask_storeu_epi64(static_cast<void*>(c + (i + 2) * n + j), mask, acc2);
                    _mm512_mask_storeu_epi64(static_cast<void*>(c + (i + 3) * n + j), mask, acc3);
                }
            }
            for (; i < row_end; ++i) {
                const auto* a0 = a + i * k;
                for (auto j = size_t{0}; j < n; j += 8) {
                    const auto remaining = n - j;
                    const auto mask = static_cast<__mmask8>(remaining >= 8 ? 0xFFu : (uint32_t{1} << remaining) - 1u);
                    auto acc = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k; ++kk) {
                        acc = _mm512_add_epi64(acc, _mm512_mullo_epi64(_mm512_set1_epi64(static_cast<int64_t>(a0[kk])),
                                                                         _mm512_maskz_loadu_epi64(mask, static_cast<const void*>(b + kk * n + j))));
                    }
                    _mm512_mask_storeu_epi64(static_cast<void*>(c + i * n + j), mask, acc);
                }
            }
        }
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to int8_t on store.
     * @brief AVX512VNNI row-range GEMM (int8_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal template to provide a concrete function pointer.
     */
    void gemm_i8_rows_avx512vnni(const int8_t* a, const int8_t* b, int8_t* c,
                                 const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_vnni_w8(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to uint8_t on store.
     * @brief AVX512VNNI row-range GEMM (uint8_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal template to provide a concrete function pointer.
     */
    void gemm_u8_rows_avx512vnni(const uint8_t* a, const uint8_t* b, uint8_t* c,
                                 const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_vnni_w8(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to int16_t on store.
     * @brief AVX512VNNI row-range GEMM (int16_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i16_rows_avx512vnni(const int16_t* a, const int16_t* b, int16_t* c,
                                  const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_vnni_i16(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to uint16_t on store.
     * @brief AVX512VNNI row-range GEMM (uint16_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_u16_rows_avx512vnni(const uint16_t* a, const uint16_t* b, uint16_t* c,
                                  const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_vnni_u16(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 32 bits of each product and accumulating mod 2^32.
     * @brief AVX512F row-range GEMM (int32_t elements, mod-2^32 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^32. Public wrapper around the internal template to provide a concrete function pointer.
     */
    void gemm_i32_rows_avx512(const int32_t* a, const int32_t* b, int32_t* c,
                              const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w32_avx512(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 32 bits of each product and accumulating mod 2^32.
     * @brief AVX512F row-range GEMM (uint32_t elements, mod-2^32 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^32. Public wrapper around the internal template to provide a concrete function pointer.
     */
    void gemm_u32_rows_avx512(const uint32_t* a, const uint32_t* b, uint32_t* c,
                              const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w32_avx512(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 64 bits of each product and accumulating mod 2^64.
     * @brief AVX512DQ row-range GEMM (int64_t elements, mod-2^64 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^64. Public wrapper around the internal template to provide a concrete function pointer.
     */
    void gemm_i64_rows_avx512dq(const int64_t* a, const int64_t* b, int64_t* c,
                                const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w64_avx512dq(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 64 bits of each product and accumulating mod 2^64.
     * @brief AVX512DQ row-range GEMM (uint64_t elements, mod-2^64 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^64. Public wrapper around the internal template to provide a concrete function pointer.
     */
    void gemm_u64_rows_avx512dq(const uint64_t* a, const uint64_t* b, uint64_t* c,
                                const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w64_avx512dq(a, b, c, k, n, row_begin, row_end);
    }

    namespace {
        /**
         * @brief Sign-extend the lo nibble (bits [3:0]) of a packed int4x2 byte to int16_t.
         * @param v The packed byte.
         * @return The sign-extended lo nibble in [-8, 7].
         */
        [[nodiscard]] int16_t nib_lo_s16(const uint8_t v) noexcept {
            const auto nib = static_cast<int32_t>(v & 0x0Fu);
            return static_cast<int16_t>(nib >= 8 ? nib - 16 : nib);
        }
        /**
         * @brief Sign-extend the hi nibble (bits [7:4]) of a packed int4x2 byte to int16_t.
         * @param v The packed byte.
         * @return The sign-extended hi nibble in [-8, 7].
         */
        [[nodiscard]] int16_t nib_hi_s16(const uint8_t v) noexcept {
            const auto nib = v >> 4;
            return static_cast<int16_t>(nib >= 8 ? nib - 16 : nib);
        }

        /**
         * Widened form of 16 packed int4x2 bytes: one 512-bit plane per
         * nibble lane, 32 lanes each. Produced by widen_nibbles_to_s16 and
         * consumed only by the int4x2 GEMM inner loop.
         * @brief Lo and hi nibble planes of 16 packed bytes, as s16 lanes.
         */
        struct alignas(64) S16NibblePair {
            __m512i lo;
            __m512i hi;
        };
        static_assert(alignof(S16NibblePair) == 64, "S16NibblePair pair must be 64-byte aligned");
        static_assert(sizeof(S16NibblePair) == 128, "two __m512i, no padding");

        /**
         * Reads up to 16 packed bytes (one per column) and widens each into
         * two s16 lanes: the lo nibble in the lo plane, the hi nibble in
         * the hi plane. Each byte widens to 16 bits before the hi nibble is
         * shifted down.
         * @brief Masked load of 16 packed bytes, widened lo/hi lanes to s16.
         * @param p Pointer to the first of 16 packed int4x2 bytes.
         * @param mask Byte-lane mask over the 64-byte load.
         * @return The lo/hi s16 planes, one byte's nibbles per lane pair.
         */
        [[nodiscard]] S16NibblePair widen_nibbles_to_s16(const int4x2_t* p, const __mmask64 mask) {
            const auto raw = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_maskz_loadu_epi8(mask, p)));
            const auto nmask = _mm512_set1_epi16(int16_t{0x0F});
            const auto sbit = _mm512_set1_epi16(int16_t{0x08});
            return S16NibblePair{
                .lo = _mm512_sub_epi16(_mm512_xor_si512(_mm512_and_si512(raw, nmask), sbit), sbit),
                .hi = _mm512_sub_epi16(_mm512_xor_si512(_mm512_srli_epi16(raw, 4), sbit), sbit),
            };
        }

        /**
         * Writes dst[0..cols) as packed bytes: lane jj of acc_lo supplies
         * the lo nibble of byte jj and lane jj of acc_hi the hi nibble —
         * bit-identical to truncating each wrapping accumulator to its
         * nibble and packing, as the scalar reference does.
         *
         * Masked stores can neither narrow to 8 bits nor interleave
         * two sources, so both accumulators dump to stack arrays and
         * are packed lane by lane.
         * @brief Store cols packed bytes from the lo/hi accumulator lanes.
         * @param dst Output position; receives exactly cols packed bytes.
         * @param acc_lo 32-bit wrapping accumulators for the lo nibble lane.
         * @param acc_hi 32-bit wrapping accumulators for the hi nibble lane.
         * @param cols Bytes to store, [0, 16].
         */
        void store_packed_nibbles(int4x2_t* dst, const __m512i acc_lo, const __m512i acc_hi, const size_t cols) {
            assert(cols <= 16 && "store_packed_nibbles: cols must be <= 16");
            alignas(64) int32_t tmp_lo[16];
            alignas(64) int32_t tmp_hi[16];
            _mm512_store_si512(tmp_lo, acc_lo);
            _mm512_store_si512(tmp_hi, acc_hi);
            for (auto jj = size_t{0}; jj < cols; ++jj) {
                const auto packed = static_cast<uint8_t>(
                    (static_cast<uint32_t>(tmp_lo[jj]) & 0x0Fu) |
                    ((static_cast<uint32_t>(tmp_hi[jj]) & 0x0Fu) << 4));
                dst[jj] = static_cast<int4x2_t>(packed);
            }
        }

        /**
         * Runs two independent signed-4-bit GEMMs in one pass! The lo
         * nibble lane of each packed input byte forms one output matrix,
         * the hi nibble lane the other. Accumulates in wrapping 32-bit per
         * lane and truncates each result to its nibble on store so the output is
         * bit-identical to the scalar wrapping reference.
         * @brief Row-range GEMM (packed int4x2, per-lane wrapping accumulate).
         * @param a Row-major A, m x k packed bytes; only rows [row_begin, row_end) are read.
         * @param b Row-major B, k x n packed bytes.
         * @param c Row-major C, m x n packed bytes; only rows [row_begin, row_end) are written.
         * @param k Inner dimension (A's column count, B's row count).
         * @param n Column count of B and C.
         * @param row_begin First row computed.
         * @param row_end One past the last row computed.
         * @note Computes C_lo[i][j] = sum_k lo(A[i][k])*lo(B[k][j]).
         */
        void gemm_rows_vnni_i4x2(const int4x2_t* a, const int4x2_t* b, int4x2_t* c, const size_t k, const size_t n,
                                 const size_t row_begin, const size_t row_end) {
            const auto zip2 = _mm512_loadu_si512(ZIP2);
            const auto k2 = k & ~size_t{1};
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                const auto* a0 = a + i * k;
                const auto* a1 = a0 + k;
                const auto* a2 = a1 + k;
                const auto* a3 = a2 + k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask_bytes(cols1);
                    const auto mask2 = col_mask_bytes(cols2);
                    auto acc0a_lo = _mm512_setzero_si512();
                    auto acc1a_lo = _mm512_setzero_si512();
                    auto acc2a_lo = _mm512_setzero_si512();
                    auto acc3a_lo = _mm512_setzero_si512();
                    auto acc0a_hi = _mm512_setzero_si512();
                    auto acc1a_hi = _mm512_setzero_si512();
                    auto acc2a_hi = _mm512_setzero_si512();
                    auto acc3a_hi = _mm512_setzero_si512();
                    auto acc0b_lo = _mm512_setzero_si512();
                    auto acc1b_lo = _mm512_setzero_si512();
                    auto acc2b_lo = _mm512_setzero_si512();
                    auto acc3b_lo = _mm512_setzero_si512();
                    auto acc0b_hi = _mm512_setzero_si512();
                    auto acc1b_hi = _mm512_setzero_si512();
                    auto acc2b_hi = _mm512_setzero_si512();
                    auto acc3b_hi = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto a0_lo = nib_lo_s16(static_cast<uint8_t>(a0[kk]));
                        const auto a0_hi = nib_hi_s16(static_cast<uint8_t>(a0[kk]));
                        const auto a1_lo = nib_lo_s16(static_cast<uint8_t>(a1[kk]));
                        const auto a1_hi = nib_hi_s16(static_cast<uint8_t>(a1[kk]));
                        const auto a2_lo = nib_lo_s16(static_cast<uint8_t>(a2[kk]));
                        const auto a2_hi = nib_hi_s16(static_cast<uint8_t>(a2[kk]));
                        const auto a3_lo = nib_lo_s16(static_cast<uint8_t>(a3[kk]));
                        const auto a3_hi = nib_hi_s16(static_cast<uint8_t>(a3[kk]));
                        const auto a0_k1 = static_cast<uint8_t>(a0[kk + 1]);
                        const auto a1_k1 = static_cast<uint8_t>(a1[kk + 1]);
                        const auto a2_k1 = static_cast<uint8_t>(a2[kk + 1]);
                        const auto a3_k1 = static_cast<uint8_t>(a3[kk + 1]);
                        const auto a0v_lo = broadcast_s16_pair(a0_lo, nib_lo_s16(a0_k1));
                        const auto a0v_hi = broadcast_s16_pair(a0_hi, nib_hi_s16(a0_k1));
                        const auto a1v_lo = broadcast_s16_pair(a1_lo, nib_lo_s16(a1_k1));
                        const auto a1v_hi = broadcast_s16_pair(a1_hi, nib_hi_s16(a1_k1));
                        const auto a2v_lo = broadcast_s16_pair(a2_lo, nib_lo_s16(a2_k1));
                        const auto a2v_hi = broadcast_s16_pair(a2_hi, nib_hi_s16(a2_k1));
                        const auto a3v_lo = broadcast_s16_pair(a3_lo, nib_lo_s16(a3_k1));
                        const auto a3v_hi = broadcast_s16_pair(a3_hi, nib_hi_s16(a3_k1));
                        const auto [bk_0_lo, bk_0_hi] = widen_nibbles_to_s16(b + kk * n + j, mask1);
                        const auto [bk_1_lo, bk_1_hi] = widen_nibbles_to_s16(b + (kk + 1) * n + j, mask1);
                        const auto bpa_lo = _mm512_permutex2var_epi16(bk_0_lo, zip2, bk_1_lo);
                        const auto bpa_hi = _mm512_permutex2var_epi16(bk_0_hi, zip2, bk_1_hi);
                        acc0a_lo = _mm512_dpwssd_epi32(acc0a_lo, a0v_lo, bpa_lo);
                        acc1a_lo = _mm512_dpwssd_epi32(acc1a_lo, a1v_lo, bpa_lo);
                        acc2a_lo = _mm512_dpwssd_epi32(acc2a_lo, a2v_lo, bpa_lo);
                        acc3a_lo = _mm512_dpwssd_epi32(acc3a_lo, a3v_lo, bpa_lo);
                        acc0a_hi = _mm512_dpwssd_epi32(acc0a_hi, a0v_hi, bpa_hi);
                        acc1a_hi = _mm512_dpwssd_epi32(acc1a_hi, a1v_hi, bpa_hi);
                        acc2a_hi = _mm512_dpwssd_epi32(acc2a_hi, a2v_hi, bpa_hi);
                        acc3a_hi = _mm512_dpwssd_epi32(acc3a_hi, a3v_hi, bpa_hi);
                        if (cols2 != 0) {
                            const auto [ck_0_lo, ck_0_hi] = widen_nibbles_to_s16(b + kk * n + j + 16, mask2);
                            const auto [ck_1_lo, ck_1_hi] = widen_nibbles_to_s16(b + (kk + 1) * n + j + 16, mask2);
                            const auto bpb_lo = _mm512_permutex2var_epi16(ck_0_lo, zip2, ck_1_lo);
                            const auto bpb_hi = _mm512_permutex2var_epi16(ck_0_hi, zip2, ck_1_hi);
                            acc0b_lo = _mm512_dpwssd_epi32(acc0b_lo, a0v_lo, bpb_lo);
                            acc1b_lo = _mm512_dpwssd_epi32(acc1b_lo, a1v_lo, bpb_lo);
                            acc2b_lo = _mm512_dpwssd_epi32(acc2b_lo, a2v_lo, bpb_lo);
                            acc3b_lo = _mm512_dpwssd_epi32(acc3b_lo, a3v_lo, bpb_lo);
                            acc0b_hi = _mm512_dpwssd_epi32(acc0b_hi, a0v_hi, bpb_hi);
                            acc1b_hi = _mm512_dpwssd_epi32(acc1b_hi, a1v_hi, bpb_hi);
                            acc2b_hi = _mm512_dpwssd_epi32(acc2b_hi, a2v_hi, bpb_hi);
                            acc3b_hi = _mm512_dpwssd_epi32(acc3b_hi, a3v_hi, bpb_hi);
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto a0_b = static_cast<uint8_t>(a0[k2]);
                        const auto a1_b = static_cast<uint8_t>(a1[k2]);
                        const auto a2_b = static_cast<uint8_t>(a2[k2]);
                        const auto a3_b = static_cast<uint8_t>(a3[k2]);
                        const auto a0v_lo = broadcast_s16_pair(nib_lo_s16(a0_b), int16_t{0});
                        const auto a0v_hi = broadcast_s16_pair(nib_hi_s16(a0_b), int16_t{0});
                        const auto a1v_lo = broadcast_s16_pair(nib_lo_s16(a1_b), int16_t{0});
                        const auto a1v_hi = broadcast_s16_pair(nib_hi_s16(a1_b), int16_t{0});
                        const auto a2v_lo = broadcast_s16_pair(nib_lo_s16(a2_b), int16_t{0});
                        const auto a2v_hi = broadcast_s16_pair(nib_hi_s16(a2_b), int16_t{0});
                        const auto a3v_lo = broadcast_s16_pair(nib_lo_s16(a3_b), int16_t{0});
                        const auto a3v_hi = broadcast_s16_pair(nib_hi_s16(a3_b), int16_t{0});
                        const auto [bk_lo, bk_hi] = widen_nibbles_to_s16(b + k2 * n + j, mask1);
                        const auto bpa_lo = _mm512_permutex2var_epi16(bk_lo, zip2, zero);
                        const auto bpa_hi = _mm512_permutex2var_epi16(bk_hi, zip2, zero);
                        acc0a_lo = _mm512_dpwssd_epi32(acc0a_lo, a0v_lo, bpa_lo);
                        acc1a_lo = _mm512_dpwssd_epi32(acc1a_lo, a1v_lo, bpa_lo);
                        acc2a_lo = _mm512_dpwssd_epi32(acc2a_lo, a2v_lo, bpa_lo);
                        acc3a_lo = _mm512_dpwssd_epi32(acc3a_lo, a3v_lo, bpa_lo);
                        acc0a_hi = _mm512_dpwssd_epi32(acc0a_hi, a0v_hi, bpa_hi);
                        acc1a_hi = _mm512_dpwssd_epi32(acc1a_hi, a1v_hi, bpa_hi);
                        acc2a_hi = _mm512_dpwssd_epi32(acc2a_hi, a2v_hi, bpa_hi);
                        acc3a_hi = _mm512_dpwssd_epi32(acc3a_hi, a3v_hi, bpa_hi);
                        if (cols2 != 0) {
                            const auto [ck_lo, ck_hi] = widen_nibbles_to_s16(b + k2 * n + j + 16, mask2);
                            const auto bpb_lo = _mm512_permutex2var_epi16(ck_lo, zip2, zero);
                            const auto bpb_hi = _mm512_permutex2var_epi16(ck_hi, zip2, zero);
                            acc0b_lo = _mm512_dpwssd_epi32(acc0b_lo, a0v_lo, bpb_lo);
                            acc1b_lo = _mm512_dpwssd_epi32(acc1b_lo, a1v_lo, bpb_lo);
                            acc2b_lo = _mm512_dpwssd_epi32(acc2b_lo, a2v_lo, bpb_lo);
                            acc3b_lo = _mm512_dpwssd_epi32(acc3b_lo, a3v_lo, bpb_lo);
                            acc0b_hi = _mm512_dpwssd_epi32(acc0b_hi, a0v_hi, bpb_hi);
                            acc1b_hi = _mm512_dpwssd_epi32(acc1b_hi, a1v_hi, bpb_hi);
                            acc2b_hi = _mm512_dpwssd_epi32(acc2b_hi, a2v_hi, bpb_hi);
                            acc3b_hi = _mm512_dpwssd_epi32(acc3b_hi, a3v_hi, bpb_hi);
                        }
                    }
                    store_packed_nibbles(c + i * n + j, acc0a_lo, acc0a_hi, cols1);
                    store_packed_nibbles(c + (i + 1) * n + j, acc1a_lo, acc1a_hi, cols1);
                    store_packed_nibbles(c + (i + 2) * n + j, acc2a_lo, acc2a_hi, cols1);
                    store_packed_nibbles(c + (i + 3) * n + j, acc3a_lo, acc3a_hi, cols1);
                    if (cols2 != 0) {
                        store_packed_nibbles(c + i * n + j + 16, acc0b_lo, acc0b_hi, cols2);
                        store_packed_nibbles(c + (i + 1) * n + j + 16, acc1b_lo, acc1b_hi, cols2);
                        store_packed_nibbles(c + (i + 2) * n + j + 16, acc2b_lo, acc2b_hi, cols2);
                        store_packed_nibbles(c + (i + 3) * n + j + 16, acc3b_lo, acc3b_hi, cols2);
                    }
                }
            }
            for (; i < row_end; ++i) {
                const auto* a0 = a + i * k;
                for (auto j = size_t{0}; j < n; j += 32) {
                    const auto rem = n - j;
                    const auto cols1 = rem >= 16 ? size_t{16} : rem;
                    const auto cols2 = rem > 16 ? (rem - 16 >= 16 ? size_t{16} : rem - 16) : size_t{0};
                    const auto mask1 = col_mask_bytes(cols1);
                    const auto mask2 = col_mask_bytes(cols2);
                    auto acc0a_lo = _mm512_setzero_si512();
                    auto acc0a_hi = _mm512_setzero_si512();
                    auto acc0b_lo = _mm512_setzero_si512();
                    auto acc0b_hi = _mm512_setzero_si512();
                    for (auto kk = size_t{0}; kk < k2; kk += 2) {
                        const auto ab = static_cast<uint8_t>(a0[kk]);
                        const auto ab1 = static_cast<uint8_t>(a0[kk + 1]);
                        const auto a0v_lo = broadcast_s16_pair(nib_lo_s16(ab), nib_lo_s16(ab1));
                        const auto a0v_hi = broadcast_s16_pair(nib_hi_s16(ab), nib_hi_s16(ab1));
                        const auto [bk_0_lo, bk_0_hi] = widen_nibbles_to_s16(b + kk * n + j, mask1);
                        const auto [bk_1_lo, bk_1_hi] = widen_nibbles_to_s16(b + (kk + 1) * n + j, mask1);
                        acc0a_lo = _mm512_dpwssd_epi32(acc0a_lo, a0v_lo, _mm512_permutex2var_epi16(bk_0_lo, zip2, bk_1_lo));
                        acc0a_hi = _mm512_dpwssd_epi32(acc0a_hi, a0v_hi, _mm512_permutex2var_epi16(bk_0_hi, zip2, bk_1_hi));

                        if (cols2 != 0) {
                            const auto [ck_0_lo, ck_0_hi] = widen_nibbles_to_s16(b + kk * n + j + 16, mask2);
                            const auto [ck_1_lo, ck_1_hi] = widen_nibbles_to_s16(b + (kk + 1) * n + j + 16, mask2);
                            acc0b_lo = _mm512_dpwssd_epi32(acc0b_lo, a0v_lo, _mm512_permutex2var_epi16(ck_0_lo, zip2, ck_1_lo));
                            acc0b_hi = _mm512_dpwssd_epi32(acc0b_hi, a0v_hi, _mm512_permutex2var_epi16(ck_0_hi, zip2, ck_1_hi));
                        }
                    }
                    if (k2 != k) {
                        const auto zero = _mm512_setzero_si512();
                        const auto ab = static_cast<uint8_t>(a0[k2]);
                        const auto a0v_lo = broadcast_s16_pair(nib_lo_s16(ab), int16_t{0});
                        const auto a0v_hi = broadcast_s16_pair(nib_hi_s16(ab), int16_t{0});
                        const auto [bk_0_lo, bk_0_hi] = widen_nibbles_to_s16(b + k2 * n + j, mask1);
                        acc0a_lo = _mm512_dpwssd_epi32(acc0a_lo, a0v_lo, _mm512_permutex2var_epi16(bk_0_lo, zip2, zero));
                        acc0a_hi = _mm512_dpwssd_epi32(acc0a_hi, a0v_hi, _mm512_permutex2var_epi16(bk_0_hi, zip2, zero));
                        if (cols2 != 0) {
                            const auto [ck_0_lo, ck_0_hi] = widen_nibbles_to_s16(b + k2 * n + j + 16, mask2);
                            acc0b_lo = _mm512_dpwssd_epi32(acc0b_lo, a0v_lo, _mm512_permutex2var_epi16(ck_0_lo, zip2, zero));
                            acc0b_hi = _mm512_dpwssd_epi32(acc0b_hi, a0v_hi, _mm512_permutex2var_epi16(ck_0_hi, zip2, zero));
                        }
                    }
                    store_packed_nibbles(c + i * n + j, acc0a_lo, acc0a_hi, cols1);
                    if (cols2 != 0) {
                        store_packed_nibbles(c + i * n + j + 16, acc0b_lo, acc0b_hi, cols2);
                    }
                }
            }
        }
    }

    /**
     * Runs two independent signed-4-bit GEMMs in one pass: the lo nibble lane
     * of each packed input byte forms one output matrix, the hi nibble lane
     * the other. Accumulates in wrapping 32-bit per lane and truncates each
     * result to its nibble on store.
     * @brief AVX512VNNI row-range GEMM (packed int4x2, per-lane wrapping accumulate).
     * @param a Row-major A, m x k packed bytes; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n packed bytes.
     * @param c Row-major C, m x n packed bytes; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C_lo[i][j] = sum_k lo(A[i][k])*lo(B[k][j]). Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i4x2_rows_avx512vnni(const int4x2_t* a, const int4x2_t* b, int4x2_t* c,
                                   const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_vnni_i4x2(a, b, c, k, n, row_begin, row_end);
    }
}
