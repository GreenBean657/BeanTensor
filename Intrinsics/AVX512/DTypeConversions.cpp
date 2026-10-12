
#include "DTypeConversions.h"

#include <bit>
#include <cstring>
#include <immintrin.h>

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Intrinsics::detail {
#ifdef __AVX512BF16__

    /**
     *  Use AVX512BF16 to accelerate the conversation between a FP32 block of memory to a BF16 block of memory, based on the "end" length.
     *  Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP32 > BF16. Requires AVX512BF16.
     * @param src Source FP32 block of memory.
     * @param dst Target BF16 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Set when a conversion event the active method treats as an error occurs.
     * @param method Active ConversionClampMethod (NaN flags under any non-PERMISSIVE method, Inf under STRICT).
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const float32_t* src, bfloat16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const bool supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BF16);
        if (!supported) throw ErrorHandling::CPUIllegalInstruction("AVX512BF16");

        const size_t t = IncludedCPUFeatures().threads;
        const auto* src_f = static_cast<const float32_t*>(src);

        // Runtime 64-byte alignment check — enables NT stores which bypass cache
        const bool aligned = (reinterpret_cast<uintptr_t>(dst) % 64) == 0;

        auto compute = [src_f, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            bool local_flag = false;
            std::size_t i = start;

            auto convert16 = [&](const size_t at) -> __m256i {
                const __m512 v = _mm512_loadu_ps(src_f + at);
                if (method == ConversionClampMethod::STRICT) {
                    // NaN/Inf inputs flag under STRICT.
                    const __m512 av = _mm512_castsi512_ps(_mm512_and_si512(
                        _mm512_castps_si512(v), _mm512_set1_epi32(0x7FFFFFFF)));
                    if (_mm512_cmp_ps_mask(av, _mm512_set1_ps(__builtin_inff()), _CMP_NLT_UQ) != 0)
                        local_flag = true;
                } else if (method != ConversionClampMethod::PERMISSIVE) {
                    // ALLOW_INF: NaN flags, Inf does not — matches the scalar path.
                    if (_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) != 0)
                        local_flag = true;
                }
                return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v));
            };

            if (aligned) {
                // NT store path — bypasses cache
                for (; i + 32 <= end_l; i += 32) {
                    const __m256i lo = convert16(i);
                    const __m256i hi = convert16(i + 16);
                    const auto packed = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), packed);
                }
                for (; i + 16 <= end_l; i += 16) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), convert16(i));
                }
            } else {
                // Unaligned fallback — no NT stores
                for (; i + 32 <= end_l; i += 32) {
                    const __m256i lo = convert16(i);
                    const __m256i hi = convert16(i + 16);
                    const auto packed = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                    _mm512_storeu_si512(dst + i, packed);
                }
                for (; i + 16 <= end_l; i += 16) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), convert16(i));
                }
            }

            for (; i < end_l; ++i) {
                // Scalar tail — identical special-case handling (NaN quieting,
                // subnormal flush, RNE) to the vector loop by construction.
                dst[i] = unaccel_fp32_to_bf16_tracked(src_f[i], local_flag, method);
            }

            // sfence required after NT stores to ensure visibility to other threads/cores
            if (aligned) _mm_sfence();
            if (local_flag) flag.store(true, std::memory_order_relaxed);
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 31) / 32 * 32;

        auto worker = [compute](const size_t start, const size_t w_end) {
            compute(start, w_end);
        };
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX512BF16 to accelerate the conversation between a BF16 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated BF16 > FP32. Requires AVX512BF16.
     * @param src Source BF16 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the BF16 target.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_bf16_to_fp32(const bfloat16_t* src, float32_t* dst, const size_t& end) {
        static const bool supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BF16);
        if (!supported) throw ErrorHandling::CPUIllegalInstruction("AVX512BF16");

        const size_t t = IncludedCPUFeatures().threads;

        // Runtime 64-byte alignment check for NT stores
        const bool aligned = (reinterpret_cast<uintptr_t>(dst) % 64) == 0;

        auto compute = [src, dst, aligned](const size_t start, const size_t end_l) {
            std::size_t i = start;

            if (aligned) {
                for (; i + 32 <= end_l; i += 32) {
                    const __m256i lo_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                    const __m256i hi_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 16));
                    const __m512i lo_ext  = _mm512_cvtepu16_epi32(lo_bf16);
                    const __m512i hi_ext  = _mm512_cvtepu16_epi32(hi_bf16);
                    _mm512_stream_ps(dst + i,      _mm512_castsi512_ps(_mm512_slli_epi32(lo_ext, 16)));
                    _mm512_stream_ps(dst + i + 16, _mm512_castsi512_ps(_mm512_slli_epi32(hi_ext, 16)));
                }
                for (; i + 16 <= end_l; i += 16) {
                    const __m256i bf16_ints = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                    const __m512i extended  = _mm512_cvtepu16_epi32(bf16_ints);
                    _mm512_stream_ps(dst + i, _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16)));
                }
            } else {
                for (; i + 32 <= end_l; i += 32) {
                    const __m256i lo_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                    const __m256i hi_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 16));
                    const __m512i lo_ext  = _mm512_cvtepu16_epi32(lo_bf16);
                    const __m512i hi_ext  = _mm512_cvtepu16_epi32(hi_bf16);
                    _mm512_storeu_ps(dst + i,      _mm512_castsi512_ps(_mm512_slli_epi32(lo_ext, 16)));
                    _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(_mm512_slli_epi32(hi_ext, 16)));
                }
                for (; i + 16 <= end_l; i += 16) {
                    const __m256i bf16_ints = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                    const __m512i extended  = _mm512_cvtepu16_epi32(bf16_ints);
                    _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16)));
                }
            }

            for (; i < end_l; ++i) {
                uint16_t raw;
                std::memcpy(&raw, &src[i], sizeof(raw));
                uint32_t bits = static_cast<uint32_t>(raw) << 16;
                dst[i] = std::bit_cast<float32_t>(bits);
            }

            // sfence after NT stores
            if (aligned) _mm_sfence();
        };

        std::vector<std::shared_future<void>> futures;
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 31) / 32 * 32;

        auto worker = [compute](const size_t start, const size_t end_l) {
            compute(start, end_l);
        };
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_v = std::min(start + chunk_size, end);
            if (start >= end_v) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_v); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }
#else
    /**
     * @brief IllegalInstructionPresent - AVX512BF support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const float32_t* src, bfloat16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512BF16");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512BF support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_bf16_to_fp32(const bfloat16_t* src, float32_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512BF16");
    }
#endif

#ifdef __AVX512F__
    /**
     * Use AVX512F to accelerate the conversation between a FP32 block of memory to a FP16 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP32 > FP16. Requires AVX512F.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @param src Source FP32 block of memory.
     * @param dst Source FP16 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Atomic flag for when conversion events occur that the active method treats as an error. Feeding a flag for discarding means you don't care about divergence.
     * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_fp16(const float32_t* src, float16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const bool supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) throw ErrorHandling::CPUIllegalInstruction("AVX512F");

        const size_t t = IncludedCPUFeatures().threads;

        const bool aligned = (reinterpret_cast<uintptr_t>(dst) % 64) == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            bool local_flag = false;
            std::size_t i = start;

            for (; i + 16 <= end_l; i += 16) {
                const __m512 v = _mm512_loadu_ps(src + i);
                const __m512 av = _mm512_castsi512_ps(_mm512_and_si512(
                    _mm512_castps_si512(v), _mm512_set1_epi32(0x7FFFFFFF)));

                auto h = _mm512_cvtps_ph(
                    v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Overflow: the scalar path clamps finite values to +/-FP16_MAX while hardware RNE would
                // produce Inf past 65520 — fix those up scalar-wise. Inf lanes need no value fixup (the
                // scalar kernel reproduces Inf) and flag under STRICT only; finite overflow flags under
                // any non-PERMISSIVE method.
                if (
                    const __mmask16 ovf = _mm512_cmp_ps_mask(av, _mm512_set1_ps(DataTypes::detail::FP16_MAX), _CMP_GT_OQ); // If-init memory optimization for localized scope.
                    ovf != 0
                    ) {
                    const __mmask16 inf_lanes = _mm512_cmp_ps_mask(av, _mm512_set1_ps(__builtin_inff()), _CMP_EQ_OQ);
                    if ((ovf & ~inf_lanes) != 0 && method != ConversionClampMethod::PERMISSIVE) local_flag = true;
                    if (inf_lanes != 0 && method == ConversionClampMethod::STRICT) local_flag = true;
                    alignas(32) uint16_t lanes[16];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), h);
                    unsigned mask = ovf;
                    while (mask != 0) {
                        const unsigned lane = std::countr_zero(mask);
                        mask &= mask - 1;
                        bool ignored = false;
                        lanes[lane] = bit_cast_memcpy<uint16_t>(
                            unaccel_fp32_to_fp16_tracked(src[i + lane], ignored, ConversionClampMethod::PERMISSIVE));
                    }
                    h = _mm256_load_si256(reinterpret_cast<const __m256i*>(lanes));
                }

                // NaN flags under any non-PERMISSIVE method, matching the scalar path.
                if (method != ConversionClampMethod::PERMISSIVE) {
                    if (_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) != 0)
                        local_flag = true;
                }

                if (aligned) _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), h);
                else         _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), h);
            }

            for (; i < end_l; ++i) {
                dst[i] = unaccel_fp32_to_fp16_tracked(src[i], local_flag, method);
            }

            if (aligned) _mm_sfence();
            if (local_flag) flag.store(true, std::memory_order_relaxed);
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 15) / 16 * 16;

        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) break;
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }

    /**
     * Use AVX512F to accelerate the conversation between a FP16 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP16 > FP32. Requires AVX512F.
     * @note Does not throw on ConversionClampMethod due to hyperthreading divergence support. If you need to throw on a conversion event, TRACK THE FLAG ARGUMENT.
     * @param src Source FP16 block of memory.
     * @param dst Source FP32 block of memory.
     * @param end Length of the FP16 target.
     * @param flag Atomic flag for when conversion events occur that the active method treats as an error. Feeding a flag for discarding means you don't care about divergence.
     * @param method Active ConversionClampMethod - How should conversion tolerance be handled?
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp16_to_fp32(const float16_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const bool supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) throw ErrorHandling::CPUIllegalInstruction("AVX512F");

        const size_t t = IncludedCPUFeatures().threads;

        const bool aligned = (reinterpret_cast<uintptr_t>(dst) % 64) == 0;

        auto compute = [src, dst, aligned, &flag, method](const size_t start, const size_t end_l) {
            bool local_flag = false;
            std::size_t i = start;

            for (; i + 16 <= end_l; i += 16) {
                const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                if (method == ConversionClampMethod::STRICT) {
                    // NaN/Inf inputs flag under STRICT.
                    const __m256i exp = _mm256_and_si256(h, _mm256_set1_epi16(0x7C00));
                    const __m256i special = _mm256_cmpeq_epi16(exp, _mm256_set1_epi16(0x7C00));
                    if (_mm256_movemask_epi8(special) != 0) local_flag = true;
                } else if (method != ConversionClampMethod::PERMISSIVE) {
                    // ALLOW_INF: NaN flags, Inf does not — matches the scalar path.
                    const __m256i exp = _mm256_and_si256(h, _mm256_set1_epi16(0x7C00));
                    const __m256i mant = _mm256_and_si256(h, _mm256_set1_epi16(0x03FF));
                    const __m256i nan_lanes = _mm256_and_si256(
                        _mm256_cmpeq_epi16(exp, _mm256_set1_epi16(0x7C00)),
                        _mm256_cmpgt_epi16(mant, _mm256_setzero_si256()));
                    if (_mm256_movemask_epi8(nan_lanes) != 0) local_flag = true;
                }
                const __m512 f = _mm512_cvtph_ps(h);
                if (aligned) _mm512_stream_ps(dst + i, f);
                else         _mm512_storeu_ps(dst + i, f);
            }

            for (; i < end_l; ++i) {
                dst[i] = unaccel_fp16_to_fp32_tracked(src[i], local_flag, method);
            }

            if (aligned) _mm_sfence();
            if (local_flag) flag.store(true, std::memory_order_relaxed);
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 15) / 16 * 16;

        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) break;
            futures.push_back(threadMgr.submit(
                [=] { compute(start, end_worker); },
                Threading::TaskPriority::Normal
            ).share());
        }
        return futures;
    }
#else
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp32_to_fp16(const float32_t* src, float16_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp16_to_fp32(const float16_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
#endif
}
