/**
 * @file DTypeConversions.cpp
 * @author GreenBean
 * @brief AVX2 dtype conversion kernels (fp32/fp16/bf16 and integer narrowing/widening).
 * Compiled with per-file AVX2 flags; execution is gated at runtime via IncludedCPUFeatures().
 * @version 0.1
 * @date 2026-10-14
 */
#include "DTypeConversions.h"

#include <bit>
#include <cstring>
#include <immintrin.h>
#include <limits>

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Dispatcher.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Intrinsics::detail {
#ifdef __AVX2__

    /**
     * Use AVX2 to accelerate the conversation between a FP32 block of memory to a BF16 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated FP32 > BF16. Requires AVX2.
     * @param src Source FP32 block of memory.
     * @param dst Target BF16 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Set when a conversion event the active method treats as an error occurs.
     * @param method Active ConversionClampMethod (NaN flags under any non-PERMISSIVE method, Inf under STRICT).
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_fp32_to_bf16(const float32_t* src, bfloat16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2");
        }

        const auto t = IncludedCPUFeatures().threads;
        const auto* src_f = src;

        // Runtime 32-byte alignment check — enables NT stores which bypass cache
        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src_f, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            auto convert8 = [&](const size_t at) -> __m256i {
                // Integer emulation of the scalar kernel — NaN quieting, subnormal
                // flush-to-zero and RNE are bit-exact with unaccel_fp32_to_bf16_tracked.
                const auto bits = _mm256_castps_si256(_mm256_loadu_ps(src_f + at));
                const auto exp  = _mm256_and_si256(bits, _mm256_set1_epi32(0x7F800000));
                const auto mant = _mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF));

                const auto is_special = _mm256_cmpeq_epi32(exp, _mm256_set1_epi32(0x7F800000)); // NaN or Inf
                const auto is_nan = _mm256_and_si256(is_special, _mm256_cmpgt_epi32(mant, _mm256_setzero_si256()));
                const auto is_sub = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256()); // zero or FP32 subnormal

                if (method == ConversionClampMethod::STRICT) {
                    // NaN/Inf inputs flag under STRICT.
                    if (_mm256_movemask_epi8(is_special) != 0) {
                        local_flag = true;
                    }
                } else if (method != ConversionClampMethod::PERMISSIVE) {
                    // ALLOW_INF: NaN flags, Inf does not — matches the scalar path.
                    if (_mm256_movemask_epi8(is_nan) != 0) {
                        local_flag = true;
                    }
                }

                // NaN/Inf: truncate, NaN gets the quiet bit so the payload never zeroes to Inf.
                const auto special = _mm256_or_si256(_mm256_srli_epi32(bits, 16),
                                                     _mm256_and_si256(is_nan, _mm256_set1_epi32(0x0040)));
                // Subnormal/zero: flush to zero, preserving only the sign.
                const auto sub = _mm256_srli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(static_cast<int>(0x80000000u))), 16);
                // Normal: round to nearest even — add half + the result LSB, then truncate.
                const auto lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
                const auto rounded = _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF)), lsb), 16);

                return _mm256_blendv_epi8(_mm256_blendv_epi8(rounded, sub, is_sub), special, is_special);
            };

            // Pack 8 u32 lane results (low halves) into 8 u16, in order.
            auto pack8 = [](const __m256i r) -> __m128i {
                // Results are <= 0xFFFF, so the signed saturation of packus never triggers.
                return _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
            };

            if (aligned) {
                // NT store path — bypasses cache
                for (; i + 16 <= end_l; i += 16) {
                    // packus is lane-local; permute restores element order across the 128-bit halves.
                    const auto packed = _mm256_permute4x64_epi64(
                        _mm256_packus_epi32(convert8(i), convert8(i + 8)), 0xD8);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), packed);
                }
                for (; i + 8 <= end_l; i += 8) {
                    _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), pack8(convert8(i)));
                }
            } else {
                // Unaligned fallback — no NT stores
                for (; i + 16 <= end_l; i += 16) {
                    const auto packed = _mm256_permute4x64_epi64(
                        _mm256_packus_epi32(convert8(i), convert8(i + 8)), 0xD8);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), packed);
                }
                for (; i + 8 <= end_l; i += 8) {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), pack8(convert8(i)));
                }
            }

            for (; i < end_l; ++i) {
                // Scalar tail — identical special-case handling (NaN quieting,
                // subnormal flush, RNE) to the vector loop by construction.
                dst[i] = unaccel_fp32_to_bf16_tracked(src_f[i], local_flag, method);
            }

            // sfence required after NT stores to ensure visibility to other threads/cores
            if (aligned) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 15) / 16 * 16;

        auto worker = [compute](const size_t start, const size_t w_end) {
            compute(start, w_end);
        };
        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX2 to accelerate the conversation between a BF16 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated BF16 > FP32. Requires AVX2.
     * @param src Source BF16 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the BF16 target.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_bf16_to_fp32(const bfloat16_t* src, float32_t* dst, const size_t& end) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2");
        }

        const auto t = IncludedCPUFeatures().threads;

        // Runtime 32-byte alignment check for NT stores
        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src, dst, aligned](const size_t start, const size_t end_l) {
            auto i = start;

            auto convert8 = [&](const size_t at) -> __m256 {
                const auto bf16_ints = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + at));
                const auto extended  = _mm256_cvtepu16_epi32(bf16_ints);
                return _mm256_castsi256_ps(_mm256_slli_epi32(extended, 16));
            };

            if (aligned) {
                for (; i + 16 <= end_l; i += 16) {
                    _mm256_stream_ps(dst + i,     convert8(i));
                    _mm256_stream_ps(dst + i + 8, convert8(i + 8));
                }
                for (; i + 8 <= end_l; i += 8) {
                    _mm256_stream_ps(dst + i, convert8(i));
                }
            } else {
                for (; i + 16 <= end_l; i += 16) {
                    _mm256_storeu_ps(dst + i,     convert8(i));
                    _mm256_storeu_ps(dst + i + 8, convert8(i + 8));
                }
                for (; i + 8 <= end_l; i += 8) {
                    _mm256_storeu_ps(dst + i, convert8(i));
                }
            }

            for (; i < end_l; ++i) {
                uint16_t raw;
                std::memcpy(&raw, &src[i], sizeof(raw));
                const auto bits = static_cast<uint32_t>(raw) << 16;
                dst[i] = std::bit_cast<float32_t>(bits);
            }

            // sfence after NT stores
            if (aligned) {
                _mm_sfence();
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 15) / 16 * 16;

        auto worker = [compute](const size_t start, const size_t end_l) {
            compute(start, end_l);
        };
        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_v = std::min(start + chunk_size, end);
            if (start >= end_v) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_v); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX2 to accelerate the conversation between a FP32 block of memory to a FP64 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated FP32 > FP64. Requires AVX2.
     * @note Widening is exact — vcvtps2pd is bit-identical to the scalar static_cast. NaN and Inf follow
     * standard2standard float->float semantics: NaN stays NaN, while Inf CLAMPS to +/-DBL_MAX (the scalar
     * overflow branch catches Inf before the isinf check) — both flag under any non-PERMISSIVE method.
     * @param src Source FP32 block of memory.
     * @param dst Target FP64 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_fp32_to_fp64(const float32_t* src, float64_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2");
        }

        const auto t = IncludedCPUFeatures().threads;
        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            for (; i + 4 <= end_l; i += 4) {
                const auto v = _mm_loadu_ps(src + i);
                const auto av = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(v), _mm_set1_epi32(0x7FFFFFFF)));
                const auto inf = _mm_cmp_ps(av, _mm_set1_ps(__builtin_inff()), _CMP_EQ_OQ);
                auto d = _mm256_cvtps_pd(v);
                // Inf CLAMPS to +/-DBL_MAX and flags under any non-PERMISSIVE method —
                // standard2standard's overflow branch catches Inf before the isinf check.
                if (
                    const auto inf_mask = _mm_movemask_ps(inf);
                    inf_mask != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    const auto sign = _mm_cmp_ps(v, _mm_setzero_ps(), _CMP_LT_OQ);
                    const auto clamped = _mm256_blendv_pd(
                        _mm256_set1_pd(std::numeric_limits<float64_t>::max()),
                        _mm256_set1_pd(std::numeric_limits<float64_t>::lowest()),
                        _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(sign))));
                    d = _mm256_blendv_pd(d, clamped,
                        _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(inf))));
                }
                if (method != ConversionClampMethod::PERMISSIVE) {
                    // NaN flags under any non-PERMISSIVE method.
                    if (_mm_movemask_ps(_mm_cmp_ps(v, v, _CMP_UNORD_Q)) != 0) {
                        local_flag = true;
                    }
                }
                if (aligned) {
                    _mm256_stream_pd(dst + i, d);
                } else {
                    _mm256_storeu_pd(dst + i, d);
                }
            }

            for (; i < end_l; ++i) {
                dst[i] = standard2standard<float32_t, float64_t>(src[i], local_flag, method);
            }

            if (aligned) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 3) / 4 * 4;

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX2 to accelerate the conversation between a FP64 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated FP64 > FP32. Requires AVX2.
     * @note standard2standard clamps ANY magnitude past FP32_MAX (including Inf) to +/-FP32_MAX and flags
     * under any non-PERMISSIVE method — unlike the FP16 kernels, Inf does NOT pass through. Hardware RNE
     * would produce Inf past the overflow boundary, so those lanes are blended back to the clamp.
     * @param src Source FP64 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the FP64 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_fp64_to_fp32(const float64_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2");
        }

        const auto t = IncludedCPUFeatures().threads;
        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 16 == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            for (; i + 4 <= end_l; i += 4) {
                const auto v = _mm256_loadu_pd(src + i);
                auto r = _mm256_cvtpd_ps(v); // __m128 — 4 fp32 results

                // Overflow (including Inf): scalar clamps to +/-FP32_MAX and flags under any
                // non-PERMISSIVE method — the overflow branch runs before the isinf check.
                const auto gt = _mm256_cmp_pd(v, _mm256_set1_pd(DataTypes::detail::FP32_MAX_D), _CMP_GT_OQ);
                const auto lt = _mm256_cmp_pd(v, _mm256_set1_pd(DataTypes::detail::FP32_MIN_D), _CMP_LT_OQ);
                if (
                    const auto ovf = _mm256_or_pd(gt, lt);
                    _mm256_movemask_pd(ovf) != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    alignas(16) float32_t lanes[4];
                    _mm_store_ps(lanes, r);
                    auto mask = static_cast<uint32_t>(_mm256_movemask_pd(ovf));
                    while (mask != 0) {
                        const auto lane = static_cast<uint32_t>(std::countr_zero(mask));
                        mask &= mask - 1;
                        auto ignored = false;
                        lanes[lane] = standard2standard<float64_t, float32_t>(src[i + lane], ignored, ConversionClampMethod::PERMISSIVE);
                    }
                    r = _mm_load_ps(lanes);
                }

                // NaN flags under any non-PERMISSIVE method; the hardware NaN output matches
                // the scalar static_cast (both quiet and truncate the payload identically).
                if (method != ConversionClampMethod::PERMISSIVE) {
                    if (_mm256_movemask_pd(_mm256_cmp_pd(v, v, _CMP_UNORD_Q)) != 0) {
                        local_flag = true;
                    }
                }

                if (aligned) {
                    _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), _mm_castps_si128(r));
                } else {
                    _mm_storeu_ps(dst + i, r);
                }
            }

            for (; i < end_l; ++i) {
                dst[i] = standard2standard<float64_t, float32_t>(src[i], local_flag, method);
            }

            if (aligned) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 3) / 4 * 4;

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX2 to accelerate the conversation between an INT32 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated INT32 > FP32. Requires AVX2.
     * @note vcvtdq2ps is RNE under the default MXCSR — bit-identical to the scalar static_cast. No flag semantics exist for int->float.
     * @param src Source INT32 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the INT32 target.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_int32_to_fp32(const int32_t* src, float32_t* dst, const size_t& end) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2");
        }

        const auto t = IncludedCPUFeatures().threads;
        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src, dst, aligned](const size_t start, const size_t end_l) {
            auto i = start;

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                const auto f = _mm256_cvtepi32_ps(v);
                if (aligned) {
                    _mm256_stream_ps(dst + i, f);
                } else {
                    _mm256_storeu_ps(dst + i, f);
                }
            }

            for (; i < end_l; ++i) {
                // Scalar int->float: always representable in range, plain static_cast.
                dst[i] = static_cast<float32_t>(src[i]);
            }

            if (aligned) {
                _mm_sfence();
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 7) / 8 * 8;

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX2 to accelerate the conversation between a FP32 block of memory to an INT32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated FP32 > INT32. Requires AVX2.
     * @note vcvttps2dq truncates like the scalar static_cast in range, but yields the integer indefinite
     * (0x80000000) for NaN/out-of-range — those lanes are blended to the scalar clamp results
     * (NaN -> 0, >= 2^31 -> INT32_MAX, < -2^31 -> INT32_MIN) and flag under any non-PERMISSIVE method.
     * @param src Source FP32 block of memory.
     * @param dst Target INT32 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_fp32_to_int32(const float32_t* src, int32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2");
        }

        const auto t = IncludedCPUFeatures().threads;
        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm256_loadu_ps(src + i);
                auto r = _mm256_cvttps_epi32(v);

                // Bounds mirror scalar_convert float->int: (float)INT32_MAX rounds up to 2^31, and
                // since fp32 cannot represent INT32_MAX exactly, equality with 2^31 is also overflow.
                // -2^31 is exactly representable, so only strictly-less underflows.
                const auto nan = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                const auto ge = _mm256_cmp_ps(v, _mm256_set1_ps(2147483648.0f), _CMP_GE_OQ);
                const auto lt = _mm256_cmp_ps(v, _mm256_set1_ps(-2147483648.0f), _CMP_LT_OQ);
                if (
                    const auto bad = _mm256_or_ps(_mm256_or_ps(nan, ge), lt);
                    _mm256_movemask_ps(bad) != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    // NaN/overflow/underflow masks are mutually exclusive (ordered compares reject NaN).
                    r = _mm256_blendv_epi8(r, _mm256_set1_epi32(std::numeric_limits<int32_t>::max()), _mm256_castps_si256(ge));
                    r = _mm256_blendv_epi8(r, _mm256_set1_epi32(std::numeric_limits<int32_t>::min()), _mm256_castps_si256(lt));
                    r = _mm256_blendv_epi8(r, _mm256_setzero_si256(), _mm256_castps_si256(nan));
                }

                if (aligned) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), r);
                } else {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), r);
                }
            }

            for (; i < end_l; ++i) {
                dst[i] = scalar_convert<float32_t, int32_t>(src[i], local_flag, method);
            }

            if (aligned) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 7) / 8 * 8;

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }
#else
    /**
     * @brief IllegalInstructionPresent - AVX2 support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_fp32_to_bf16(const float32_t* src, bfloat16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2");
    }
    /**
     * @brief IllegalInstructionPresent - AVX2 support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_bf16_to_fp32(const bfloat16_t* src, float32_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2");
    }
    /**
     * @brief IllegalInstructionPresent - AVX2 support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_fp32_to_fp64(const float32_t* src, float64_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2");
    }
    /**
     * @brief IllegalInstructionPresent - AVX2 support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_fp64_to_fp32(const float64_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2");
    }
    /**
     * @brief IllegalInstructionPresent - AVX2 support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_int32_to_fp32(const int32_t* src, float32_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2");
    }
    /**
     * @brief IllegalInstructionPresent - AVX2 support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_fp32_to_int32(const float32_t* src, int32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2");
    }
#endif

#if defined(__AVX2__) && defined(__F16C__)
    /**
     * Use AVX2/F16C to accelerate the conversation between a FP32 block of memory to a FP16 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated FP32 > FP16. Requires AVX2 + F16C.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @param src Source FP32 block of memory.
     * @param dst Source FP16 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Atomic flag for when conversion events occur that the active method treats as an error. Feeding a flag for discarding means you don't care about divergence.
     * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_fp32_to_fp16(const float32_t* src, float16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2) &&
                                      IncludedCPUFeatures().has_feature(CPUFeatureSetType_F16C);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2+F16C");
        }

        const auto t = IncludedCPUFeatures().threads;

        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm256_loadu_ps(src + i);
                const auto av = _mm256_castsi256_ps(_mm256_and_si256(
                    _mm256_castps_si256(v), _mm256_set1_epi32(0x7FFFFFFF)));

                auto h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Overflow: the scalar path clamps finite values to +/-FP16_MAX while hardware RNE would
                // produce Inf past 65520 — fix those up scalar-wise. Inf lanes need no value fixup (the
                // scalar kernel reproduces Inf) and flag under STRICT only; finite overflow flags under
                // any non-PERMISSIVE method.
                if (
                    const auto ovf = _mm256_cmp_ps(av, _mm256_set1_ps(DataTypes::detail::FP16_MAX), _CMP_GT_OQ);
                    _mm256_movemask_ps(ovf) != 0
                ) {
                    const auto inf_lanes = _mm256_cmp_ps(av, _mm256_set1_ps(__builtin_inff()), _CMP_EQ_OQ);
                    if (
                        const auto finite_ovf = _mm256_andnot_si256(_mm256_castps_si256(inf_lanes), _mm256_castps_si256(ovf));
                        _mm256_movemask_epi8(finite_ovf) != 0 && method != ConversionClampMethod::PERMISSIVE
                    ) {
                        local_flag = true;
                    }
                    if (_mm256_movemask_ps(inf_lanes) != 0 && method == ConversionClampMethod::STRICT) {
                        local_flag = true;
                    }
                    alignas(16) uint16_t lanes[8];
                    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), h);
                    auto mask = static_cast<uint32_t>(_mm256_movemask_ps(ovf));
                    while (mask != 0) {
                        const auto lane = static_cast<uint32_t>(std::countr_zero(mask));
                        mask &= mask - 1;
                        auto ignored = false;
                        lanes[lane] = bit_cast_memcpy<uint16_t>(
                            unaccel_fp32_to_fp16_tracked(src[i + lane], ignored, ConversionClampMethod::PERMISSIVE));
                    }
                    h = _mm_load_si128(reinterpret_cast<const __m128i*>(lanes));
                }

                // NaN flags under any non-PERMISSIVE method, matching the scalar path.
                if (method != ConversionClampMethod::PERMISSIVE) {
                    if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q)) != 0) {
                        local_flag = true;
                    }
                }

                if (aligned) {
                    _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), h);
                } else {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
                }
            }

            for (; i < end_l; ++i) {
                dst[i] = unaccel_fp32_to_fp16_tracked(src[i], local_flag, method);
            }

            if (aligned) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 7) / 8 * 8;

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX2/F16C to accelerate the conversation between a FP16 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX2 accelerated FP16 > FP32. Requires AVX2 + F16C.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @param src Source FP16 block of memory.
     * @param dst Source FP32 block of memory.
     * @param end Length of the FP16 target.
     * @param flag Atomic flag for when conversion events occur that the active method treats as an error. Feeding a flag for discarding means you don't care about divergence.
     * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx2_fp16_to_fp32(const float16_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2) &&
                                      IncludedCPUFeatures().has_feature(CPUFeatureSetType_F16C);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX2+F16C");
        }

        const auto t = IncludedCPUFeatures().threads;

        const auto aligned = reinterpret_cast<uintptr_t>(dst) % 32 == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            for (; i + 8 <= end_l; i += 8) {
                const auto h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
                if (method == ConversionClampMethod::STRICT) {
                    // NaN/Inf inputs flag under STRICT.
                    const auto exp = _mm_and_si128(h, _mm_set1_epi16(0x7C00));
                    if (
                        const auto special = _mm_cmpeq_epi16(exp, _mm_set1_epi16(0x7C00));
                        _mm_movemask_epi8(special) != 0
                    ) {
                        local_flag = true;
                    }
                } else if (method != ConversionClampMethod::PERMISSIVE) {
                    // ALLOW_INF: NaN flags, Inf does not — matches the scalar path.
                    const auto exp = _mm_and_si128(h, _mm_set1_epi16(0x7C00));
                    const auto mant = _mm_and_si128(h, _mm_set1_epi16(0x03FF));
                    const auto nan_lanes = _mm_and_si128(
                        _mm_cmpeq_epi16(exp, _mm_set1_epi16(0x7C00)),
                        _mm_cmpgt_epi16(mant, _mm_setzero_si128()));
                    if (_mm_movemask_epi8(nan_lanes) != 0) {
                        local_flag = true;
                    }
                }
                const auto f = _mm256_cvtph_ps(h);
                if (aligned) {
                    _mm256_stream_ps(dst + i, f);
                } else {
                    _mm256_storeu_ps(dst + i, f);
                }
            }

            for (; i < end_l; ++i) {
                dst[i] = unaccel_fp16_to_fp32_tracked(src[i], local_flag, method);
            }

            if (aligned) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        auto futures = std::vector<std::shared_future<void>>{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const auto chunk_size = ((end + t - 1) / t + 7) / 8 * 8;

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }
#else
    /**
     * @brief IllegalInstructionPresent - AVX2/F16C support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_fp32_to_fp16(const float32_t* src, float16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2+F16C");
    }
    /**
     * @brief IllegalInstructionPresent - AVX2/F16C support not compiled.
     */
    std::vector<std::shared_future<void>> avx2_fp16_to_fp32(const float16_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX2+F16C");
    }
#endif
}
