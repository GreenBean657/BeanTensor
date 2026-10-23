/**
 * @file GemmAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA GEMM row-range kernels.
 * @version 0.1
 * @date 2026-10-18
 */
#include "Math/LinearAlgebra/MatMul/GemmKernels.h"

#include <immintrin.h>

#include <type_traits>

#include "Helpers/Hardware.h"

namespace BeanTensor::Math::LinearAlgebra {

    /**
     * Cached runtime variable - Gates AVX2+FMA GEMM kernels on CPU support.
     * @brief AVX2 && FMA verification, for GEMM kernels.
     * @return T/F - AVX2 && FMA is supported by the CPU.
     */
    bool gemm_avx2_available() noexcept {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2) &&
                                      IncludedCPUFeatures().has_feature(CPUFeatureSetType_FMA);
        return supported;
    }

    /**
     * Four A rows share each loaded "B" vector (4x8 registers, one accumulator per row-block)
     * FP32 accumulate via FMA in order. Column tails fall back to scalar, row tails single-row.
     * @brief AVX2+FMA row-range GEMM (FP32 FMA accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_f32_rows_avx2(const float32_t* a, const float32_t* b, float32_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            auto j = size_t{0};
            for (; j + 8 <= n; j += 8) {
                auto acc0 = _mm256_setzero_ps();
                auto acc1 = _mm256_setzero_ps();
                auto acc2 = _mm256_setzero_ps();
                auto acc3 = _mm256_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bv = _mm256_loadu_ps(b + kk * n + j);
                    acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[kk]), bv, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[kk]), bv, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[kk]), bv, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[kk]), bv, acc3);
                }
                _mm256_storeu_ps(c + i * n + j, acc0);
                _mm256_storeu_ps(c + (i + 1) * n + j, acc1);
                _mm256_storeu_ps(c + (i + 2) * n + j, acc2);
                _mm256_storeu_ps(c + (i + 3) * n + j, acc3);
            }
            for (; j < n; ++j) {
                auto s0 = 0.0f;
                auto s1 = 0.0f;
                auto s2 = 0.0f;
                auto s3 = 0.0f;
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bv = b[kk * n + j];
                    s0 += a0[kk] * bv;
                    s1 += a1[kk] * bv;
                    s2 += a2[kk] * bv;
                    s3 += a3[kk] * bv;
                }
                c[i * n + j] = s0;
                c[(i + 1) * n + j] = s1;
                c[(i + 2) * n + j] = s2;
                c[(i + 3) * n + j] = s3;
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            auto j = size_t{0};
            for (; j + 8 <= n; j += 8) {
                auto acc = _mm256_setzero_ps();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(a0[kk]), _mm256_loadu_ps(b + kk * n + j), acc);
                }
                _mm256_storeu_ps(c + i * n + j, acc);
            }
            for (; j < n; ++j) {
                auto s = 0.0f;
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    s += a0[kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    /**
     * Four A rows share each loaded "B" vector (4x4 registers, one accumulator per row-block)
     * FP64 accumulate via FMA in order. Column tails fall back to scalar, row tails single-row.
     * @brief AVX2+FMA row-range GEMM (FP64 FMA accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     */
    void gemm_f64_rows_avx2(const float64_t* a, const float64_t* b, float64_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        auto i = row_begin;
        for (; i + 4 <= row_end; i += 4) {
            const auto* a0 = a + i * k;
            const auto* a1 = a0 + k;
            const auto* a2 = a1 + k;
            const auto* a3 = a2 + k;
            auto j = size_t{0};
            for (; j + 4 <= n; j += 4) {
                auto acc0 = _mm256_setzero_pd();
                auto acc1 = _mm256_setzero_pd();
                auto acc2 = _mm256_setzero_pd();
                auto acc3 = _mm256_setzero_pd();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bv = _mm256_loadu_pd(b + kk * n + j);
                    acc0 = _mm256_fmadd_pd(_mm256_set1_pd(a0[kk]), bv, acc0);
                    acc1 = _mm256_fmadd_pd(_mm256_set1_pd(a1[kk]), bv, acc1);
                    acc2 = _mm256_fmadd_pd(_mm256_set1_pd(a2[kk]), bv, acc2);
                    acc3 = _mm256_fmadd_pd(_mm256_set1_pd(a3[kk]), bv, acc3);
                }
                _mm256_storeu_pd(c + i * n + j, acc0);
                _mm256_storeu_pd(c + (i + 1) * n + j, acc1);
                _mm256_storeu_pd(c + (i + 2) * n + j, acc2);
                _mm256_storeu_pd(c + (i + 3) * n + j, acc3);
            }
            for (; j < n; ++j) {
                auto s0 = 0.0;
                auto s1 = 0.0;
                auto s2 = 0.0;
                auto s3 = 0.0;
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto bv = b[kk * n + j];
                    s0 += a0[kk] * bv;
                    s1 += a1[kk] * bv;
                    s2 += a2[kk] * bv;
                    s3 += a3[kk] * bv;
                }
                c[i * n + j] = s0;
                c[(i + 1) * n + j] = s1;
                c[(i + 2) * n + j] = s2;
                c[(i + 3) * n + j] = s3;
            }
        }
        for (; i < row_end; ++i) {
            const auto* a0 = a + i * k;
            auto j = size_t{0};
            for (; j + 4 <= n; j += 4) {
                auto acc = _mm256_setzero_pd();
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(a0[kk]), _mm256_loadu_pd(b + kk * n + j), acc);
                }
                _mm256_storeu_pd(c + i * n + j, acc);
            }
            for (; j < n; ++j) {
                auto s = 0.0;
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    s += a0[kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    namespace {
        /**
         * Replicate an (lo, hi) s16 pair across all 32-bit lanes, matching the
         * VPMADDWD k-pair layout produced by the unpacklo/hi interleave below.
         * @brief Broadcast two s16 k-slices of one A row to all lanes.
         */
        __m256i broadcast_s16_pair(const int16_t lo, const int16_t hi) {
            const auto packed = static_cast<uint32_t>(static_cast<uint16_t>(lo)) |
                                (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16);
            return _mm256_set1_epi32(static_cast<int32_t>(packed));
        }

        /**
         * Widen 16 bytes of one B row to s16 lanes. Signed bytes sign-extend,
         * unsigned bytes zero-extend; 0..255 fits s16, so both are exact.
         * @brief Load + widen 16 B elements to s16.
         */
        template<typename T>
        __m256i widen8_to_s16(const T* p) {
            const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            if constexpr (std::is_signed_v<T>) {
                return _mm256_cvtepi8_epi16(raw);
            } else {
                return _mm256_cvtepu8_epi16(raw);
            }
        }

        /**
         * Sign-extend a packed int4x2 byte's lo nibble (bits [3:0]) or hi
         * nibble (bits [7:4]) to s16. Exact for 4-bit two's complement.
         * @brief Scalar nibble extract + sign-extend to int16_t.
         */
        [[nodiscard]] int16_t nib_lo_s16(const uint8_t v) {
            const auto nib = static_cast<int32_t>(v & 0x0Fu);
            return static_cast<int16_t>(nib >= 8 ? nib - 16 : nib);
        }

        /**
         * Sign-extend a packed int4x2 byte's hi nibble (bits [7:4]) to s16.
         * Exact for 4-bit two's complement.
         * @brief Scalar hi-nibble extract + sign-extend to int16_t.
         * @param v The packed byte.
         * @return The sign-extended hi nibble in [-8, 7].
         */
        [[nodiscard]] int16_t nib_hi_s16(const uint8_t v) {
            const auto nib = v >> 4;
            return static_cast<int16_t>(nib >= 8 ? nib - 16 : nib);
        }

        /**
         * Both nibble lanes of 16 packed int4x2 bytes (16 C columns), widened
         * to s16 for VPMADDWD. Values in [-8,7] are exact in s16.
         * @brief Lo and hi nibble planes of 16 packed bytes, as s16 lanes.
         */
        struct S16NibblePair {
            __m256i lo;
            __m256i hi;
        };

        /**
         * The hi nibble rides out of _mm_srli_epi16(v, 4) at each byte's low
         * half (the stray cross-byte bits are masked off); both lanes are then
         * sign-extended in 8-bit with (x ^ 0x08) - 0x08, exact for a masked
         * 4-bit two's-complement nibble.
         * @brief Load 16 packed bytes, widen lo/hi nibble lanes to s16.
         */
        [[nodiscard]] S16NibblePair widen_nibbles_to_s16(const int4x2_t* p) {
            const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const auto nmask = _mm_set1_epi32(0x0F0F0F0F);
            const auto sbit = _mm_set1_epi32(0x08080808);
            const auto lo8 = _mm_and_si128(raw, nmask);
            const auto hi8 = _mm_and_si128(_mm_srli_epi16(raw, 4), nmask);
            return S16NibblePair{
                .lo = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_xor_si128(lo8, sbit), sbit)),
                .hi = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_xor_si128(hi8, sbit), sbit)),
            };
        }

        /**
         * Scalar C row i, columns [j_begin, n), per-lane wrap semantics
         * (uint32_t accumulate per lane, nibble truncation on store), matching
         * the vector accumulators bit-exactly.
         * @brief Scalar int4x2 column tail for one row.
         */
        void gemm_row_tail_i4x2(const int4x2_t* a, const int4x2_t* b, int4x2_t* c, const size_t k, const size_t n,
                                const size_t i, const size_t j_begin) {
            for (auto j = j_begin; j < n; ++j) {
                auto acc_lo = uint32_t{0};
                auto acc_hi = uint32_t{0};
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    const auto av = static_cast<uint8_t>(a[i * k + kk]);
                    const auto bv = static_cast<uint8_t>(b[kk * n + j]);
                    acc_lo += static_cast<uint32_t>(static_cast<int32_t>(nib_lo_s16(av)) *
                                                    static_cast<int32_t>(nib_lo_s16(bv)));
                    acc_hi += static_cast<uint32_t>(static_cast<int32_t>(nib_hi_s16(av)) *
                                                    static_cast<int32_t>(nib_hi_s16(bv)));
                }
                const auto packed = static_cast<uint8_t>((acc_lo & 0x0Fu) | ((acc_hi & 0x0Fu) << 4));
                c[i * n + j] = static_cast<int4x2_t>(packed);
            }
        }

        /**
         * Modular scalar tail: wraps every add in the unsigned twin widened to
         * >=32 bits, bit-identical to the vector accumulators truncated on store.
         * @brief Scalar C row i, columns [j_begin, n), wrap semantics.
         */
        template<typename T>
        void gemm_row_tail_scalar(const T* a, const T* b, T* c, const size_t k, const size_t n,
                                  const size_t i, const size_t j_begin) {
            using AT = std::conditional_t<(sizeof(T) > 4uz), uint64_t, uint32_t>;
            for (auto j = j_begin; j < n; ++j) {
                auto acc = AT{0};
                for (auto kk = size_t{0}; kk < k; ++kk) {
                    acc += static_cast<AT>(a[i * k + kk]) * static_cast<AT>(b[kk * n + j]);
                }
                c[i * n + j] = static_cast<T>(acc);
            }
        }

        /**
         * 16 columns x ROWS rows for 8-bit inputs. Two B k-rows are widened to
         * s16 and interleaved with unpacklo/hi, so one VPMADDWD against the
         * broadcast (a[kk], a[kk+1]) pair covers two k-slices. acc_lo holds
         * columns 0-3|8-11, acc_hi 4-7|12-15 (per 128-bit lane); the lane swap
         * is undone at store time. Accumulation wraps mod 2^32, and truncation
         * to 8 bits on store makes the result bit-exact.
         * @brief 16xROWS block kernel, 8-bit A and B, wrapping 32-bit accumulate.
         */
        template<typename T, size_t ROWS>
        void gemm_block_w8(const T* a, const T* b, T* c, const size_t k, const size_t n,
                           const size_t i, const size_t j) {
            __m256i acc_lo[ROWS];
            __m256i acc_hi[ROWS];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                acc_lo[r] = _mm256_setzero_si256();
                acc_hi[r] = _mm256_setzero_si256();
            }
            for (auto kk = size_t{0}; kk < k; kk += 2) {
                const auto bw0 = widen8_to_s16(b + kk * n + j);
                const auto bw1 = kk + 1 < k ? widen8_to_s16(b + (kk + 1) * n + j) : _mm256_setzero_si256();
                const auto bp_lo = _mm256_unpacklo_epi16(bw0, bw1);
                const auto bp_hi = _mm256_unpackhi_epi16(bw0, bw1);
                for (auto r = size_t{0}; r < ROWS; ++r) {
                    const auto* ar = a + (i + r) * k;
                    const auto a_hi = kk + 1 < k ? static_cast<int16_t>(ar[kk + 1]) : int16_t{0};
                    const auto av = broadcast_s16_pair(static_cast<int16_t>(ar[kk]), a_hi);
                    acc_lo[r] = _mm256_add_epi32(acc_lo[r], _mm256_madd_epi16(av, bp_lo));
                    acc_hi[r] = _mm256_add_epi32(acc_hi[r], _mm256_madd_epi16(av, bp_hi));
                }
            }
            alignas(32) int32_t tmp[16];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp),
                                   _mm256_permute2x128_si256(acc_lo[r], acc_hi[r], 0x20));
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 8),
                                   _mm256_permute2x128_si256(acc_lo[r], acc_hi[r], 0x31));
                auto* cr = c + (i + r) * n + j;
                for (auto jj = size_t{0}; jj < 16; ++jj) {
                    cr[jj] = static_cast<T>(tmp[jj]);
                }
            }
        }

        /**
         * 16 columns x ROWS rows for 16-bit inputs, direct VPMADDWD on s16
         * lanes. The (0x8000, 0x8000) pair wraps to 0x80000000 inside VPMADDWD,
         * which equals the exact product mod 2^32, so wrapping accumulation
         * stays bit-exact. Lane interleave identical to gemm_block_w8.
         * @brief 16xROWS block kernel, signed 16-bit A and B.
         */
        template<size_t ROWS>
        void gemm_block_i16(const int16_t* a, const int16_t* b, int16_t* c, const size_t k, const size_t n,
                            const size_t i, const size_t j) {
            __m256i acc_lo[ROWS];
            __m256i acc_hi[ROWS];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                acc_lo[r] = _mm256_setzero_si256();
                acc_hi[r] = _mm256_setzero_si256();
            }
            for (auto kk = size_t{0}; kk < k; kk += 2) {
                const auto bw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + kk * n + j));
                const auto bw1 = kk + 1 < k
                                     ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + (kk + 1) * n + j))
                                     : _mm256_setzero_si256();
                const auto bp_lo = _mm256_unpacklo_epi16(bw0, bw1);
                const auto bp_hi = _mm256_unpackhi_epi16(bw0, bw1);
                for (auto r = size_t{0}; r < ROWS; ++r) {
                    const auto* ar = a + (i + r) * k;
                    const auto a_hi = kk + 1 < k ? ar[kk + 1] : int16_t{0};
                    const auto av = broadcast_s16_pair(ar[kk], a_hi);
                    acc_lo[r] = _mm256_add_epi32(acc_lo[r], _mm256_madd_epi16(av, bp_lo));
                    acc_hi[r] = _mm256_add_epi32(acc_hi[r], _mm256_madd_epi16(av, bp_hi));
                }
            }
            alignas(32) int32_t tmp[16];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp),
                                   _mm256_permute2x128_si256(acc_lo[r], acc_hi[r], 0x20));
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 8),
                                   _mm256_permute2x128_si256(acc_lo[r], acc_hi[r], 0x31));
                auto* cr = c + (i + r) * n + j;
                for (auto jj = size_t{0}; jj < 16; ++jj) {
                    cr[jj] = static_cast<int16_t>(tmp[jj]);
                }
            }
        }

        /**
         * 16 columns x ROWS rows for uint16 via the sign-flip offset trick:
         * with a' = a - 32768 (a XOR 0x8000 as s16) and likewise b',
         *   sum(a*b) = sum(a'*b') + 32768*(sum(a') + sum(b')) + k*2^30 (mod 2^32).
         * acc accumulates sum(a'*b') via VPMADDWD; bsum accumulates sum(b')
         * per column (madd against ones); asum accumulates sum(a') per row as
         * a wrapping scalar. 32768*x mod 2^32 == (uint32)x << 15 and
         * k*2^30 mod 2^32 == (k mod 4) << 30. All corrections are exact mod 2^32.
         * @brief 16xROWS block kernel, unsigned 16-bit A and B.
         */
        template<size_t ROWS>
        void gemm_block_u16(const uint16_t* a, const uint16_t* b, uint16_t* c, const size_t k, const size_t n,
                            const size_t i, const size_t j) {
            const auto flip = _mm256_set1_epi16(static_cast<int16_t>(uint16_t{0x8000}));
            const auto ones = _mm256_set1_epi16(int16_t{1});
            __m256i acc_lo[ROWS];
            __m256i acc_hi[ROWS];
            uint32_t asum[ROWS];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                acc_lo[r] = _mm256_setzero_si256();
                acc_hi[r] = _mm256_setzero_si256();
                asum[r] = 0;
            }
            auto bsum_lo = _mm256_setzero_si256();
            auto bsum_hi = _mm256_setzero_si256();
            for (auto kk = size_t{0}; kk < k; kk += 2) {
                const auto bw0 = _mm256_xor_si256(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + kk * n + j)), flip);
                const auto bw1 = kk + 1 < k
                                     ? _mm256_xor_si256(
                                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + (kk + 1) * n + j)), flip)
                                     : _mm256_setzero_si256();
                const auto bp_lo = _mm256_unpacklo_epi16(bw0, bw1);
                const auto bp_hi = _mm256_unpackhi_epi16(bw0, bw1);
                bsum_lo = _mm256_add_epi32(bsum_lo, _mm256_madd_epi16(ones, bp_lo));
                bsum_hi = _mm256_add_epi32(bsum_hi, _mm256_madd_epi16(ones, bp_hi));
                for (auto r = size_t{0}; r < ROWS; ++r) {
                    const auto* ar = a + (i + r) * k;
                    const auto a_lo = static_cast<int16_t>(static_cast<uint16_t>(ar[kk] ^ uint16_t{0x8000}));
                    const auto a_hi = kk + 1 < k
                                          ? static_cast<int16_t>(static_cast<uint16_t>(ar[kk + 1] ^ uint16_t{0x8000}))
                                          : int16_t{0};
                    asum[r] += static_cast<uint32_t>(static_cast<int32_t>(a_lo)) +
                               static_cast<uint32_t>(static_cast<int32_t>(a_hi));
                    const auto av = broadcast_s16_pair(a_lo, a_hi);
                    acc_lo[r] = _mm256_add_epi32(acc_lo[r], _mm256_madd_epi16(av, bp_lo));
                    acc_hi[r] = _mm256_add_epi32(acc_hi[r], _mm256_madd_epi16(av, bp_hi));
                }
            }
            const auto kterm = _mm256_set1_epi32(
                static_cast<int32_t>(static_cast<uint32_t>(k % size_t{4}) << 30));
            alignas(32) int32_t tmp[16];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                const auto asumv = _mm256_set1_epi32(static_cast<int32_t>(asum[r]));
                const auto res_lo = _mm256_add_epi32(
                    _mm256_add_epi32(acc_lo[r], _mm256_slli_epi32(_mm256_add_epi32(asumv, bsum_lo), 15)), kterm);
                const auto res_hi = _mm256_add_epi32(
                    _mm256_add_epi32(acc_hi[r], _mm256_slli_epi32(_mm256_add_epi32(asumv, bsum_hi), 15)), kterm);
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp),
                                   _mm256_permute2x128_si256(res_lo, res_hi, 0x20));
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp + 8),
                                   _mm256_permute2x128_si256(res_lo, res_hi, 0x31));
                auto* cr = c + (i + r) * n + j;
                for (auto jj = size_t{0}; jj < 16; ++jj) {
                    cr[jj] = static_cast<uint16_t>(tmp[jj]);
                }
            }
        }

        /**
         * 8 columns x ROWS rows for 32-bit inputs. VPMULLD keeps the low 32
         * bits, which are sign-agnostic, so one implementation serves both
         * int32 and uint32 with wrapping VPPADDD accumulation.
         * @brief 8xROWS block kernel, 32-bit A and B, mod-2^32 arithmetic.
         */
        template<typename T, size_t ROWS>
        void gemm_block_w32(const T* a, const T* b, T* c, const size_t k, const size_t n,
                            const size_t i, const size_t j) {
            __m256i acc[ROWS];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                acc[r] = _mm256_setzero_si256();
            }
            for (auto kk = size_t{0}; kk < k; ++kk) {
                const auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + kk * n + j));
                for (auto r = size_t{0}; r < ROWS; ++r) {
                    const auto av = _mm256_set1_epi32(static_cast<int32_t>(a[(i + r) * k + kk]));
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_mullo_epi32(av, bv));
                }
            }
            for (auto r = size_t{0}; r < ROWS; ++r) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + (i + r) * n + j), acc[r]);
            }
        }

        /**
         * 4 columns x ROWS rows for 64-bit inputs. VPMULUDQ decomposition of
         * the mod-2^64 product: a*b == a_lo*b_lo + ((a_lo*b_hi + a_hi*b_lo) << 32),
         * every add wraps mod 2^64, so signed and unsigned share one path.
         * @brief 4xROWS block kernel, 64-bit A and B, mod-2^64 arithmetic.
         */
        template<typename T, size_t ROWS>
        void gemm_block_w64(const T* a, const T* b, T* c, const size_t k, const size_t n,
                            const size_t i, const size_t j) {
            const auto mask_lo = _mm256_set1_epi64x(uint64_t{0xFFFFFFFF});
            __m256i acc[ROWS];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                acc[r] = _mm256_setzero_si256();
            }
            for (auto kk = size_t{0}; kk < k; ++kk) {
                const auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + kk * n + j));
                const auto b_lo = _mm256_and_si256(bv, mask_lo);
                const auto b_hi = _mm256_srli_epi64(bv, 32);
                for (auto r = size_t{0}; r < ROWS; ++r) {
                    const auto av = _mm256_set1_epi64x(static_cast<int64_t>(a[(i + r) * k + kk]));
                    const auto a_lo = _mm256_and_si256(av, mask_lo);
                    const auto a_hi = _mm256_srli_epi64(av, 32);
                    const auto p_ll = _mm256_mul_epu32(a_lo, b_lo);
                    const auto cross = _mm256_slli_epi64(
                        _mm256_add_epi64(_mm256_mul_epu32(a_lo, b_hi), _mm256_mul_epu32(a_hi, b_lo)), 32);
                    acc[r] = _mm256_add_epi64(acc[r], _mm256_add_epi64(p_ll, cross));
                }
            }
            for (auto r = size_t{0}; r < ROWS; ++r) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + (i + r) * n + j), acc[r]);
            }
        }

        /**
         * 16 columns x ROWS rows for packed int4x2: two independent i8-like
         * GEMMs (lo-nibble lane and hi-nibble lane) sharing the row loads.
         * Both B k-rows of a k-pair are nibble-split and widened to s16, then
         * interleaved with unpacklo/hi so one VPMADDWD against the broadcast
         * (a_lane[kk], a_lane[kk+1]) pair covers two k-slices, exactly like
         * gemm_block_w8. acc[lane][part] holds 32-bit wrapping sums; store
         * truncates each lane to its nibble, never a saturating pack.
         * @brief 16xROWS block kernel, packed int4x2 A and B, per-lane wrap.
         */
        template<size_t ROWS>
        void gemm_block_i4x2(const int4x2_t* a, const int4x2_t* b, int4x2_t* c, const size_t k, const size_t n,
                             const size_t i, const size_t j) {
            __m256i acc[2][2][ROWS]; // [nibble lane: 0=lo, 1=hi][unpacklo/hi][row]
            for (auto lane = size_t{0}; lane < 2; ++lane) {
                for (auto part = size_t{0}; part < 2; ++part) {
                    for (auto r = size_t{0}; r < ROWS; ++r) {
                        acc[lane][part][r] = _mm256_setzero_si256();
                    }
                }
            }
            for (auto kk = size_t{0}; kk < k; kk += 2) {
                const auto [bw0_lo, bw0_hi] = widen_nibbles_to_s16(b + kk * n + j);
                const auto [bw1_lo, bw1_hi] = kk + 1 < k ? widen_nibbles_to_s16(b + (kk + 1) * n + j) : S16NibblePair{};
                const __m256i bp[2][2] = {
                    {_mm256_unpacklo_epi16(bw0_lo, bw1_lo), _mm256_unpackhi_epi16(bw0_lo, bw1_lo)},
                    {_mm256_unpacklo_epi16(bw0_hi, bw1_hi), _mm256_unpackhi_epi16(bw0_hi, bw1_hi)},
                };
                for (auto r = size_t{0}; r < ROWS; ++r) {
                    const auto* ar = a + (i + r) * k;
                    const auto ab0 = static_cast<uint8_t>(ar[kk]);
                    const auto ab1 = kk + 1 < k ? static_cast<uint8_t>(ar[kk + 1]) : uint8_t{0};
                    const __m256i av[2] = {
                        broadcast_s16_pair(nib_lo_s16(ab0), nib_lo_s16(ab1)),
                        broadcast_s16_pair(nib_hi_s16(ab0), nib_hi_s16(ab1)),
                    };
                    for (auto lane = size_t{0}; lane < 2; ++lane) {
                        for (auto part = size_t{0}; part < 2; ++part) {
                            acc[lane][part][r] = _mm256_add_epi32(
                                acc[lane][part][r], _mm256_madd_epi16(av[lane], bp[lane][part]));
                        }
                    }
                }
            }
            alignas(32) int32_t tmp_lo[16];
            alignas(32) int32_t tmp_hi[16];
            for (auto r = size_t{0}; r < ROWS; ++r) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp_lo),
                                   _mm256_permute2x128_si256(acc[0][0][r], acc[0][1][r], 0x20));
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp_lo + 8),
                                   _mm256_permute2x128_si256(acc[0][0][r], acc[0][1][r], 0x31));
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp_hi),
                                   _mm256_permute2x128_si256(acc[1][0][r], acc[1][1][r], 0x20));
                _mm256_store_si256(reinterpret_cast<__m256i*>(tmp_hi + 8),
                                   _mm256_permute2x128_si256(acc[1][0][r], acc[1][1][r], 0x31));
                auto* cr = c + (i + r) * n + j;
                for (auto jj = size_t{0}; jj < 16; ++jj) {
                    const auto packed = static_cast<uint8_t>(
                        (static_cast<uint32_t>(tmp_lo[jj]) & 0x0Fu) |
                        ((static_cast<uint32_t>(tmp_hi[jj]) & 0x0Fu) << 4));
                    cr[jj] = static_cast<int4x2_t>(packed);
                }
            }
        }

        /**
         * Row-range driver for 8-bit A and B. Processes 4 rows at a time with
         * gemm_block_w8<4>, then single-row blocks, then scalar column tails.
         * Accumulates in wrapping 32-bit and truncates to 8 bits on store.
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
        void gemm_rows_w8(const T* a, const T* b, T* c, const size_t k, const size_t n,
                          const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                auto j = size_t{0};
                for (; j + 16 <= n; j += 16) {
                    gemm_block_w8<T, 4>(a, b, c, k, n, i, j);
                }
                for (auto r = size_t{0}; r < 4; ++r) {
                    gemm_row_tail_scalar(a, b, c, k, n, i + r, j);
                }
            }
            for (; i < row_end; ++i) {
                auto j = size_t{0};
                for (; j + 16 <= n; j += 16) {
                    gemm_block_w8<T, 1>(a, b, c, k, n, i, j);
                }
                gemm_row_tail_scalar(a, b, c, k, n, i, j);
            }
        }

        /**
         * Row-range driver for 16-bit A and B. Processes 4 rows at a time with
         * gemm_block_i16<4> or gemm_block_u16<4>, then single-row blocks, then
         * scalar column tails. Accumulates in wrapping 32-bit and truncates to
         * 16 bits on store.
         * @brief Row-range GEMM (16-bit A and B, wrapping 32-bit accumulate).
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
        void gemm_rows_w16(const T* a, const T* b, T* c, const size_t k, const size_t n,
                           const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                auto j = size_t{0};
                for (; j + 16 <= n; j += 16) {
                    if constexpr (std::is_signed_v<T>) {
                        gemm_block_i16<4>(a, b, c, k, n, i, j);
                    } else {
                        gemm_block_u16<4>(a, b, c, k, n, i, j);
                    }
                }
                for (auto r = size_t{0}; r < 4; ++r) {
                    gemm_row_tail_scalar(a, b, c, k, n, i + r, j);
                }
            }
            for (; i < row_end; ++i) {
                auto j = size_t{0};
                for (; j + 16 <= n; j += 16) {
                    if constexpr (std::is_signed_v<T>) {
                        gemm_block_i16<1>(a, b, c, k, n, i, j);
                    } else {
                        gemm_block_u16<1>(a, b, c, k, n, i, j);
                    }
                }
                gemm_row_tail_scalar(a, b, c, k, n, i, j);
            }
        }

        /**
         * Row-range driver for 32-bit A and B. Processes 4 rows at a time with
         * gemm_block_w32<4>, then single-row blocks, then scalar column tails.
         * Keeps the low 32 bits of each product, accumulating mod 2^32.
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
        void gemm_rows_w32(const T* a, const T* b, T* c, const size_t k, const size_t n,
                           const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                auto j = size_t{0};
                for (; j + 8 <= n; j += 8) {
                    gemm_block_w32<T, 4>(a, b, c, k, n, i, j);
                }
                for (auto r = size_t{0}; r < 4; ++r) {
                    gemm_row_tail_scalar(a, b, c, k, n, i + r, j);
                }
            }
            for (; i < row_end; ++i) {
                auto j = size_t{0};
                for (; j + 8 <= n; j += 8) {
                    gemm_block_w32<T, 1>(a, b, c, k, n, i, j);
                }
                gemm_row_tail_scalar(a, b, c, k, n, i, j);
            }
        }

        /**
         * Row-range driver for 64-bit A and B. Processes 4 rows at a time with
         * gemm_block_w64<4>, then single-row blocks, then scalar column tails.
         * Keeps the low 64 bits of each product, accumulating mod 2^64.
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
        void gemm_rows_w64(const T* a, const T* b, T* c, const size_t k, const size_t n,
                           const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                auto j = size_t{0};
                for (; j + 4 <= n; j += 4) {
                    gemm_block_w64<T, 4>(a, b, c, k, n, i, j);
                }
                for (auto r = size_t{0}; r < 4; ++r) {
                    gemm_row_tail_scalar(a, b, c, k, n, i + r, j);
                }
            }
            for (; i < row_end; ++i) {
                auto j = size_t{0};
                for (; j + 4 <= n; j += 4) {
                    gemm_block_w64<T, 1>(a, b, c, k, n, i, j);
                }
                gemm_row_tail_scalar(a, b, c, k, n, i, j);
            }
        }

        /**
         * Row-range driver for packed int4x2 A and B. Processes 4 rows at a time
         * with gemm_block_i4x2<4>, then single-row blocks, then scalar column
         * tails. Runs two independent signed-4-bit GEMMs in one pass; the lo
         * nibble lane of each packed input byte forms one output matrix, the hi
         * nibble lane the other.
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
        void gemm_rows_i4x2(const int4x2_t* a, const int4x2_t* b, int4x2_t* c, const size_t k, const size_t n,
                            const size_t row_begin, const size_t row_end) {
            auto i = row_begin;
            for (; i + 4 <= row_end; i += 4) {
                auto j = size_t{0};
                for (; j + 16 <= n; j += 16) {
                    gemm_block_i4x2<4>(a, b, c, k, n, i, j);
                }
                for (auto r = size_t{0}; r < 4; ++r) {
                    gemm_row_tail_i4x2(a, b, c, k, n, i + r, j);
                }
            }
            for (; i < row_end; ++i) {
                auto j = size_t{0};
                for (; j + 16 <= n; j += 16) {
                    gemm_block_i4x2<1>(a, b, c, k, n, i, j);
                }
                gemm_row_tail_i4x2(a, b, c, k, n, i, j);
            }
        }
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to int8_t on store.
     * @brief AVX2 row-range GEMM (int8_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i8_rows_avx2(const int8_t* a, const int8_t* b, int8_t* c,
                           const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w8(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to uint8_t on store.
     * @brief AVX2 row-range GEMM (uint8_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_u8_rows_avx2(const uint8_t* a, const uint8_t* b, uint8_t* c,
                           const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w8(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to int16_t on store.
     * @brief AVX2 row-range GEMM (int16_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i16_rows_avx2(const int16_t* a, const int16_t* b, int16_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w16(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, accumulating in
     * wrapping 32-bit and truncating each result to uint16_t on store.
     * @brief AVX2 row-range GEMM (uint16_t A and B, wrapping 32-bit accumulate).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j]. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_u16_rows_avx2(const uint16_t* a, const uint16_t* b, uint16_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w16(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 32 bits of each product and accumulating mod 2^32.
     * @brief AVX2 row-range GEMM (int32_t elements, mod-2^32 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^32. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i32_rows_avx2(const int32_t* a, const int32_t* b, int32_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w32(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 32 bits of each product and accumulating mod 2^32.
     * @brief AVX2 row-range GEMM (uint32_t elements, mod-2^32 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^32. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_u32_rows_avx2(const uint32_t* a, const uint32_t* b, uint32_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w32(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 64 bits of each product and accumulating mod 2^64.
     * @brief AVX2 row-range GEMM (int64_t elements, mod-2^64 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^64. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i64_rows_avx2(const int64_t* a, const int64_t* b, int64_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w64(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Multiplies the A rows [row_begin, row_end) by B, keeping the low
     * 64 bits of each product and accumulating mod 2^64.
     * @brief AVX2 row-range GEMM (uint64_t elements, mod-2^64 arithmetic).
     * @param a Row-major A, m x k; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n.
     * @param c Row-major C, m x n; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C[i][j] = sum_k A[i][k]*B[k][j] mod 2^64. Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_u64_rows_avx2(const uint64_t* a, const uint64_t* b, uint64_t* c,
                            const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_w64(a, b, c, k, n, row_begin, row_end);
    }

    /**
     * Runs two independent signed-4-bit GEMMs in one pass: the lo nibble lane
     * of each packed input byte forms one output matrix, the hi nibble lane
     * the other. Accumulates in wrapping 32-bit per lane and truncates each
     * result to its nibble on store.
     * @brief AVX2 row-range GEMM (packed int4x2, per-lane wrapping accumulate).
     * @param a Row-major A, m x k packed bytes; only rows [row_begin, row_end) are read.
     * @param b Row-major B, k x n packed bytes.
     * @param c Row-major C, m x n packed bytes; only rows [row_begin, row_end) are written.
     * @param k Inner dimension (A's column count, B's row count).
     * @param n Column count of B and C.
     * @param row_begin First row computed.
     * @param row_end One past the last row computed.
     * @note Computes C_lo[i][j] = sum_k lo(A[i][k])*lo(B[k][j]). Public wrapper around the internal kernel to provide a concrete function pointer.
     */
    void gemm_i4x2_rows_avx2(const int4x2_t* a, const int4x2_t* b, int4x2_t* c,
                             const size_t k, const size_t n, const size_t row_begin, const size_t row_end) {
        gemm_rows_i4x2(a, b, c, k, n, row_begin, row_end);
    }
}
