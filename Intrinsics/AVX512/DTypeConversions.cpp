/**
 * @file DTypeConversions.cpp
 * @author GreenBean
 * @brief AVX512 dtype conversion kernels (fp32/fp16/bf16 and integer narrowing/widening).
 * NT stores are gated on pointer alignment AND NtStoreMinBytes() — measured 1.8x loss
 * below the LLC, 25-31% win past it. Chunk submission is pinned to the calling thread's
 * CCD group when the destination fits one CCD's LLC (KeepWorkLocal).
 * Compiled with per-file AVX512 flags; execution is gated at runtime via IncludedCPUFeatures().
 * Scalar prologue/tail lanes route through detail::scalar_tail so their semantics live
 * in one place beside scalar_convert (bit-exact-parity reference).
 * @version 0.3
 * @date 2026-10-25
 */
#include "DTypeConversions.h"

#include <bit>
#include <immintrin.h>
#include <limits>

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Dispatcher.h"
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
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512BF16");
        }

        const size_t t = IncludedCPUFeatures().threads;
        const auto* src_f = src;

        // NT stores are a measured 1.8x loss below the LLC and a 25-31% win past
        // it, so they are gated on BOTH the runtime 64-byte alignment check and
        // the destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(bfloat16_t) > NtStoreMinBytes();

        auto compute = [src_f, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            bool local_flag = false;
            std::size_t i = start;

            auto convert16 = [&](const size_t at) -> __m256i {
                const __m512 v = _mm512_loadu_ps(src_f + at);
                if (method == ConversionClampMethod::STRICT) {
                    // NaN/Inf inputs flag under STRICT.
                    const __m512 av = _mm512_castsi512_ps(_mm512_and_si512(
                        _mm512_castps_si512(v), _mm512_set1_epi32(0x7FFFFFFF)));
                    if (_mm512_cmp_ps_mask(av, _mm512_set1_ps(__builtin_inff()), _CMP_NLT_UQ) != 0) {
                        local_flag = true;
                    }
                } else if (method != ConversionClampMethod::PERMISSIVE) {
                    // ALLOW_INF: NaN flags, Inf does not — matches the scalar path.
                    if (_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) != 0) {
                        local_flag = true;
                    }
                }
                return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v));
            };

            // Scalar prologue: align dst to the 64-byte NT store boundary so the
            // vector loop below can stream. Vacuous today (use_nt requires a
            // 64-byte-aligned dst and chunks start vector-aligned), but keeps the
            // single loop correct if the gate ever relaxes. Same scalar converter
            // and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(bfloat16_t));
                    scalar_tail<float32_t, bfloat16_t>(src_f, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 32 <= end_l; i += 32) {
                const __m256i lo = convert16(i);
                const __m256i hi = convert16(i + 16);
                const auto packed = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
                if (use_nt) {
                    // NT store path — bypasses cache
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), packed);
                } else {
                    // Unaligned store — legal at any address
                    _mm512_storeu_si512(dst + i, packed);
                }
            }
            for (; i + 16 <= end_l; i += 16) {
                const __m256i h = convert16(i);
                if (use_nt) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), h);
                } else {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), h);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float32_t, bfloat16_t>(src_f, dst, i, end_l, local_flag, method);

            // sfence required after NT stores to ensure visibility to other threads/cores
            if (use_nt) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 31) / 32 * 32;
        const auto keep_local = KeepWorkLocal(end * sizeof(bfloat16_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        auto worker = [compute](const size_t start, const size_t w_end) {
            compute(start, w_end);
        };
        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                // Working set fits one CCD's LLC — pin chunks to this CCD's
                // group so remote-CCD writes (~8x cost) are avoided.
                futures.push_back(threadMgr.submit_to_group(
                    [=] { worker(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { worker(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
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
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512BF16");
        }

        const size_t t = IncludedCPUFeatures().threads;

        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt](const size_t start, const size_t end_l) {
            auto tail_flag = false;
            std::size_t i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (16
            // elements of 4 bytes). Same converter as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(float32_t));
                    // Widening ignores flag/method — scalar_convert's bfloat16 branch is the
                    // same shift+bit-cast the inlined loop used.
                    scalar_tail<bfloat16_t, float32_t>(src, dst, i, prologue_end, tail_flag, ConversionClampMethod::PERMISSIVE);
                    i = prologue_end;
                }
            }

            for (; i + 32 <= end_l; i += 32) {
                const __m256i lo_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                const __m256i hi_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 16));
                const __m512i lo_ext  = _mm512_cvtepu16_epi32(lo_bf16);
                const __m512i hi_ext  = _mm512_cvtepu16_epi32(hi_bf16);
                if (use_nt) {
                    _mm512_stream_ps(dst + i,      _mm512_castsi512_ps(_mm512_slli_epi32(lo_ext, 16)));
                    _mm512_stream_ps(dst + i + 16, _mm512_castsi512_ps(_mm512_slli_epi32(hi_ext, 16)));
                } else {
                    _mm512_storeu_ps(dst + i,      _mm512_castsi512_ps(_mm512_slli_epi32(lo_ext, 16)));
                    _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(_mm512_slli_epi32(hi_ext, 16)));
                }
            }
            for (; i + 16 <= end_l; i += 16) {
                const __m256i bf16_ints = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                const __m512i extended  = _mm512_cvtepu16_epi32(bf16_ints);
                if (use_nt) {
                    _mm512_stream_ps(dst + i, _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16)));
                } else {
                    _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16)));
                }
            }

            scalar_tail<bfloat16_t, float32_t>(src, dst, i, end_l, tail_flag, ConversionClampMethod::PERMISSIVE);

            // sfence after NT stores
            if (use_nt) {
                _mm_sfence();
            }
        };

        std::vector<std::shared_future<void>> futures;
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 31) / 32 * 32;
        const auto keep_local = KeepWorkLocal(end * sizeof(float32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        auto worker = [compute](const size_t start, const size_t end_l) {
            compute(start, end_l);
        };
        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_v = std::min(start + chunk_size, end);
            if (start >= end_v) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { worker(start, end_v); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { worker(start, end_v); },
                    Threading::TaskPriority::Normal
                ).share());
            }
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
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const size_t t = IncludedCPUFeatures().threads;

        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float16_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            bool local_flag = false;
            std::size_t i = start;

            // Scalar prologue: align dst to the 32-byte NT store boundary (16
            // elements of 2 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 32;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (32 - misalign) / sizeof(float16_t));
                    scalar_tail<float32_t, float16_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 16 <= end_l; i += 16) {
                const __m512 v = _mm512_loadu_ps(src + i);
                const __m512 av = _mm512_castsi512_ps(_mm512_and_si512(
                    _mm512_castps_si512(v), _mm512_set1_epi32(0x7FFFFFFF)));
#pragma GCC diagnostic push // Snapshot GCC properties.
#pragma GCC diagnostic ignored "-Wsign-conversion" //Modify flag, remove sign-conversion warning for all lines beyond the push.

                /*
                 * A brief story in basic competence. You would think system headers would be coded in a way to not include errors, and usually they are.
                 * In this example, we are shown what happens when a low level developer stops caring about quality of code, and just wants to ship whatever.
                 * _mm512_cvtps_ph is a magical function for FP32 > FP16. The amazing AVX developers decided to do this (the internal call ran by this define):
                 * __builtin_ia32_vcvtps2ph512_mask((v), (imm), (__v16hi)undefined, -1)
                 * The last argument is a mask, which should be __mmask16 = uint16_t with a different name. The header, in it's amazing and unflawed coding manner,
                 * decides to pass a bare int (int32_t) -1, so every call site must perform implicit signed->unsigned conversion.
                 * -1 becomes 655365 (correct mask), but this was NOT the right way to set it - the header should have used the correct type. The AVX developer sacrificed C++ standard compliance
                 * for SLIGHTLY better readability.
                 *
                 * I'm not going to call a __builtin_ function directly with the correct mask then watch this script explode if AVX instructions update.
                 * Instead, to ensure compliance with C++ standard, I must disable the conversion error for this line, so the rest of the program can still properly error on violations.
                 * Crazy how I have to do this for a system header, but I guess the AVX developers didn't care about C++ standard compliance!
                 *
                 * (Yes, I'm angry - but only because I spent a good hour figuring out why this function explodes with the flag. I hate following the #define chain to the source.)
                 */
                auto h = _mm512_cvtps_ph(
                    v,
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
                    );
#pragma GCC diagnostic pop //Revert to pre-recorded GCC state (via push).

                // Overflow: the scalar path clamps finite values to +/-FP16_MAX while hardware RNE would
                // produce Inf past 65520 — fix those up with masked constant blends. Inf lanes need no
                // value fixup (the scalar kernel reproduces Inf) and flag under STRICT only; finite
                // overflow flags under any non-PERMISSIVE method. The blend runs in the dword domain
                // (one fp16 lane per dword) because this TU is compiled without AVX512BW/VL, so the
                // word-granularity _mm256_mask_mov_epi16 is unavailable — cvtepu16_epi32/mask_mov_epi32/
                // cvtepi32_epi16 are all AVX512F. Exhaustively verified bit-exact vs the scalar fixup
                // over all 2^32 patterns (scratch/fp16_blend_verify/test_fp16_blend_512v2.cpp).
                if (
                    const __mmask16 ovf = _mm512_cmp_ps_mask(av, _mm512_set1_ps(DataTypes::detail::FP16_MAX), _CMP_GT_OQ);
                    ovf != 0
                    ) {
                    const __mmask16 inf_lanes = _mm512_cmp_ps_mask(av, _mm512_set1_ps(__builtin_inff()), _CMP_EQ_OQ);
                    if ((ovf & ~inf_lanes) != 0 && method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    if (inf_lanes != 0 && method == ConversionClampMethod::STRICT) {
                        local_flag = true;
                    }
                    const __mmask16 fix = static_cast<__mmask16>(ovf & static_cast<__mmask16>(~inf_lanes));
                    const __mmask16 lt = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_LT_OQ);
                    const __mmask16 neg_mask = static_cast<__mmask16>(fix & lt);
                    const __mmask16 pos_mask = static_cast<__mmask16>(fix & static_cast<__mmask16>(~lt));
                    __m512i h32 = _mm512_cvtepu16_epi32(h);
                    h32 = _mm512_mask_mov_epi32(h32, pos_mask, _mm512_set1_epi32(0x7BFF));
                    h32 = _mm512_mask_mov_epi32(h32, neg_mask, _mm512_set1_epi32(0xFBFF));
                    h = _mm512_cvtepi32_epi16(h32);
                }

                // NaN flags under any non-PERMISSIVE method, matching the scalar path.
                if (method != ConversionClampMethod::PERMISSIVE) {
                    if (_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q) != 0) {
                        local_flag = true;
                    }
                }

                if (use_nt) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), h);
                } else {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), h);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float32_t, float16_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 15) / 16 * 16;
        const auto keep_local = KeepWorkLocal(end * sizeof(float16_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
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
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const size_t t = IncludedCPUFeatures().threads;

        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            bool local_flag = false;
            std::size_t i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (16
            // elements of 4 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(float32_t));
                    scalar_tail<float16_t, float32_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 16 <= end_l; i += 16) {
                const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                if (method == ConversionClampMethod::STRICT) {
                    // NaN/Inf inputs flag under STRICT.
                    const __m256i exp = _mm256_and_si256(h, _mm256_set1_epi16(0x7C00));
                    if (
                        const __m256i special = _mm256_cmpeq_epi16(exp, _mm256_set1_epi16(0x7C00));
                        _mm256_movemask_epi8(special) != 0
                    ) {
                        local_flag = true;
                    }
                } else if (method != ConversionClampMethod::PERMISSIVE) {
                    // ALLOW_INF: NaN flags, Inf does not — matches the scalar path.
                    const __m256i exp = _mm256_and_si256(h, _mm256_set1_epi16(0x7C00));
                    const __m256i mant = _mm256_and_si256(h, _mm256_set1_epi16(0x03FF));
                    const __m256i nan_lanes = _mm256_and_si256(
                        _mm256_cmpeq_epi16(exp, _mm256_set1_epi16(0x7C00)),
                        _mm256_cmpgt_epi16(mant, _mm256_setzero_si256()));
                    if (_mm256_movemask_epi8(nan_lanes) != 0) {
                        local_flag = true;
                    }
                }
                const __m512 f = _mm512_cvtph_ps(h);
                if (use_nt) {
                    _mm512_stream_ps(dst + i, f);
                } else {
                    _mm512_storeu_ps(dst + i, f);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float16_t, float32_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
                _mm_sfence();
            }
            if (local_flag) {
                flag.store(true, std::memory_order_relaxed);
            }
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t chunk_size = ((end + t - 1) / t + 15) / 16 * 16;
        const auto keep_local = KeepWorkLocal(end * sizeof(float32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
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

#ifdef __AVX512F__
    /**
     * Use AVX512F to accelerate the conversation between a FP32 block of memory to a FP64 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP32 > FP64. Requires AVX512F.
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
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_fp64(const float32_t* src, float64_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float64_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (8
            // elements of 8 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(float64_t));
                    scalar_tail<float32_t, float64_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm256_loadu_ps(src + i);
                const auto av = _mm256_castsi256_ps(_mm256_and_si256(
                    _mm256_castps_si256(v), _mm256_set1_epi32(0x7FFFFFFF)));
                const auto inf = _mm256_cmp_ps(av, _mm256_set1_ps(__builtin_inff()), _CMP_EQ_OQ);
                auto d = _mm512_cvtps_pd(v);
                // Inf CLAMPS to +/-DBL_MAX and flags under any non-PERMISSIVE method —
                // standard2standard's overflow branch catches Inf before the isinf check.
                if (
                    const auto inf_mask = _mm256_movemask_ps(inf);
                    inf_mask != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    const auto sign = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ);
                    const auto kinf = static_cast<__mmask8>(inf_mask);
                    const auto ksign = static_cast<__mmask8>(_mm256_movemask_ps(sign));
                    const auto clamped = _mm512_mask_blend_pd(ksign,
                        _mm512_set1_pd(std::numeric_limits<float64_t>::max()),
                        _mm512_set1_pd(std::numeric_limits<float64_t>::lowest()));
                    d = _mm512_mask_blend_pd(kinf, d, clamped);
                }
                if (method != ConversionClampMethod::PERMISSIVE) {
                    // NaN flags under any non-PERMISSIVE method.
                    if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q)) != 0) {
                        local_flag = true;
                    }
                }
                if (use_nt) {
                    _mm512_stream_pd(dst + i, d);
                } else {
                    _mm512_storeu_pd(dst + i, d);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float32_t, float64_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(float64_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }

    /**
     * Use AVX512F to accelerate the conversation between a FP64 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP64 > FP32. Requires AVX512F.
     * @note standard2standard clamps ANY magnitude past FP32_MAX (including Inf) to +/-FP32_MAX and flags
     * under any non-PERMISSIVE method — unlike the FP16 kernels, Inf does NOT pass through. Hardware RNE
     * would produce Inf past the overflow boundary, so those lanes are fixed up scalar-wise.
     * @param src Source FP64 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the FP64 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp64_to_fp32(const float64_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 32-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 32 == 0
            && end * sizeof(float32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 32-byte NT store boundary (8
            // elements of 4 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 32;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (32 - misalign) / sizeof(float32_t));
                    scalar_tail<float64_t, float32_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm512_loadu_pd(src + i);
                auto r = _mm512_cvtpd_ps(v);

                // Overflow (including Inf): scalar clamps to +/-FP32_MAX and flags under any
                // non-PERMISSIVE method — the overflow branch runs before the isinf check.
                const auto gt = _mm512_cmp_pd_mask(v, _mm512_set1_pd(DataTypes::detail::FP32_MAX_D), _CMP_GT_OQ);
                const auto lt = _mm512_cmp_pd_mask(v, _mm512_set1_pd(DataTypes::detail::FP32_MIN_D), _CMP_LT_OQ);
                if (
                    const auto ovf = static_cast<uint32_t>(gt | lt);
                    ovf != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    alignas(32) float32_t lanes[8];
                    _mm256_store_ps(lanes, r);
                    auto mask = ovf;
                    while (mask != 0) {
                        const auto lane = static_cast<uint32_t>(std::countr_zero(mask));
                        mask &= mask - 1;
                        auto ignored = false;
                        lanes[lane] = standard2standard<float64_t, float32_t>(src[i + lane], ignored, ConversionClampMethod::PERMISSIVE);
                    }
                    r = _mm256_load_ps(lanes);
                }

                // NaN flags under any non-PERMISSIVE method; the hardware NaN output matches
                // the scalar static_cast (both quiet and truncate the payload identically).
                if (method != ConversionClampMethod::PERMISSIVE) {
                    if (_mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q) != 0) {
                        local_flag = true;
                    }
                }

                if (use_nt) {
                    _mm256_stream_ps(dst + i, r);
                } else {
                    _mm256_storeu_ps(dst + i, r);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float64_t, float32_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(float32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }

    /**
     * Use AVX512F to accelerate the conversation between an INT32 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated INT32 > FP32. Requires AVX512F.
     * @note vcvtdq2ps is RNE under the default MXCSR — bit-identical to the scalar static_cast. No flag semantics exist for int->float.
     * @param src Source INT32 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the INT32 target.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_int32_to_fp32(const int32_t* src, float32_t* dst, const size_t& end) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt](const size_t start, const size_t end_l) {
            auto tail_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (16
            // elements of 4 bytes). Same converter as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(float32_t));
                    // int -> float ignores flag/method in scalar_convert (plain static_cast).
                    scalar_tail<int32_t, float32_t>(src, dst, i, prologue_end, tail_flag, ConversionClampMethod::PERMISSIVE);
                    i = prologue_end;
                }
            }

            for (; i + 16 <= end_l; i += 16) {
                const auto v = _mm512_loadu_si512(src + i);
                const auto f = _mm512_cvtepi32_ps(v);
                if (use_nt) {
                    _mm512_stream_ps(dst + i, f);
                } else {
                    _mm512_storeu_ps(dst + i, f);
                }
            }

            // Scalar tail — int->float ignores flag/method in scalar_convert (plain static_cast).
            scalar_tail<int32_t, float32_t>(src, dst, i, end_l, tail_flag, ConversionClampMethod::PERMISSIVE);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(float32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }

    /**
     * Use AVX512F to accelerate the conversation between a FP32 block of memory to an INT32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP32 > INT32. Requires AVX512F.
     * @note vcvttps2dq truncates like the scalar static_cast in range, but yields the integer indefinite
     * (0x80000000) for NaN/out-of-range — those lanes are masked to the scalar clamp results
     * (NaN -> 0, >= 2^31 -> INT32_MAX, < -2^31 -> INT32_MIN) and flag under any non-PERMISSIVE method.
     * @param src Source FP32 block of memory.
     * @param dst Target INT32 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_int32(const float32_t* src, int32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(int32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (16
            // elements of 4 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(int32_t));
                    scalar_tail<float32_t, int32_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 16 <= end_l; i += 16) {
                const auto v = _mm512_loadu_ps(src + i);
                auto r = _mm512_cvttps_epi32(v);

                // Bounds mirror scalar_convert float->int: (float)INT32_MAX rounds up to 2^31, and
                // since fp32 cannot represent INT32_MAX exactly, equality with 2^31 is also overflow.
                // -2^31 is exactly representable, so only strictly-less underflows.
                const auto nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
                const auto ge = _mm512_cmp_ps_mask(v, _mm512_set1_ps(2147483648.0f), _CMP_GE_OQ);
                const auto lt = _mm512_cmp_ps_mask(v, _mm512_set1_ps(-2147483648.0f), _CMP_LT_OQ);
                if (
                    const auto bad = static_cast<uint32_t>(nan | ge | lt);
                    bad != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    // NaN/overflow/underflow masks are mutually exclusive (ordered compares reject NaN).
                    r = _mm512_mask_mov_epi32(r, ge, _mm512_set1_epi32(std::numeric_limits<int32_t>::max()));
                    r = _mm512_mask_mov_epi32(r, lt, _mm512_set1_epi32(std::numeric_limits<int32_t>::min()));
                    r = _mm512_mask_mov_epi32(r, nan, _mm512_setzero_si512());
                }

                if (use_nt) {
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), r);
                } else {
                    _mm512_storeu_si512(dst + i, r);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float32_t, int32_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(int32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }

    /**
     * Use AVX512F to accelerate the conversation between a UINT32 block of memory to a FP32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated UINT32 > FP32. Requires AVX512F.
     * @note vcvtdq2ps (unsigned) is RNE under the default MXCSR — bit-identical to the scalar static_cast. No flag semantics exist for int->float.
     * @param src Source UINT32 block of memory.
     * @param dst Target FP32 block of memory.
     * @param end Length of the UINT32 target.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_uint32_to_fp32(const uint32_t* src, float32_t* dst, const size_t& end) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt](const size_t start, const size_t end_l) {
            auto tail_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (16
            // elements of 4 bytes). Same converter as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(float32_t));
                    // int -> float ignores flag/method in scalar_convert (plain static_cast).
                    scalar_tail<uint32_t, float32_t>(src, dst, i, prologue_end, tail_flag, ConversionClampMethod::PERMISSIVE);
                    i = prologue_end;
                }
            }

            for (; i + 16 <= end_l; i += 16) {
                const auto v = _mm512_loadu_si512(src + i);
                const auto f = _mm512_cvtepu32_ps(v);
                if (use_nt) {
                    _mm512_stream_ps(dst + i, f);
                } else {
                    _mm512_storeu_ps(dst + i, f);
                }
            }

            // Scalar tail — int->float ignores flag/method in scalar_convert (plain static_cast).
            scalar_tail<uint32_t, float32_t>(src, dst, i, end_l, tail_flag, ConversionClampMethod::PERMISSIVE);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(float32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }

    /**
     * Use AVX512F to accelerate the conversation between a FP32 block of memory to a UINT32 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP32 > UINT32. Requires AVX512F.
     * @note vcvttps2udq truncates like the scalar static_cast in range; NaN/out-of-range lanes are masked
     * to the scalar clamp results (NaN -> 0, >= 2^32 -> UINT32_MAX, < 0 -> 0) and flag under any
     * non-PERMISSIVE method. -0.0 is in range (scalar lowest() is 0 and -0.0 < 0 is false).
     * @param src Source FP32 block of memory.
     * @param dst Target UINT32 block of memory.
     * @param end Length of the FP32 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_uint32(const float32_t* src, uint32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512F");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(uint32_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (16
            // elements of 4 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(uint32_t));
                    scalar_tail<float32_t, uint32_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 16 <= end_l; i += 16) {
                const auto v = _mm512_loadu_ps(src + i);
                auto r = _mm512_cvttps_epu32(v);

                // Bounds mirror scalar_convert float->int: (float)UINT32_MAX rounds up to 2^32, and
                // since fp32 cannot represent UINT32_MAX exactly, equality with 2^32 is also overflow.
                const auto nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
                const auto ge = _mm512_cmp_ps_mask(v, _mm512_set1_ps(4294967296.0f), _CMP_GE_OQ);
                const auto lt = _mm512_cmp_ps_mask(v, _mm512_set1_ps(0.0f), _CMP_LT_OQ);
                if (
                    const auto bad = static_cast<uint32_t>(nan | ge | lt);
                    bad != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    r = _mm512_mask_mov_epi32(r, ge, _mm512_set1_epi32(-1)); // UINT32_MAX
                    r = _mm512_mask_mov_epi32(r, lt, _mm512_setzero_si512());
                    r = _mm512_mask_mov_epi32(r, nan, _mm512_setzero_si512());
                }

                if (use_nt) {
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), r);
                } else {
                    _mm512_storeu_si512(dst + i, r);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float32_t, uint32_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(uint32_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }
#else
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp32_to_fp64(const float32_t* src, float64_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp64_to_fp32(const float64_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_int32_to_fp32(const int32_t* src, float32_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp32_to_int32(const float32_t* src, int32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_uint32_to_fp32(const uint32_t* src, float32_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512F support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp32_to_uint32(const float32_t* src, uint32_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512F");
    }
#endif

#ifdef __AVX512DQ__
    /**
     * Use AVX512DQ to accelerate the conversation between an INT64 block of memory to a FP64 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated INT64 > FP64. Requires AVX512DQ.
     * @note vcvtqq2pd is RNE under the default MXCSR — bit-identical to the scalar static_cast. No flag semantics exist for int->float.
     * @param src Source INT64 block of memory.
     * @param dst Target FP64 block of memory.
     * @param end Length of the INT64 target.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_int64_to_fp64(const int64_t* src, float64_t* dst, const size_t& end) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512DQ);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512DQ");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(float64_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt](const size_t start, const size_t end_l) {
            auto tail_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (8
            // elements of 8 bytes). Same converter as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(float64_t));
                    // int -> float ignores flag/method in scalar_convert (plain static_cast).
                    scalar_tail<int64_t, float64_t>(src, dst, i, prologue_end, tail_flag, ConversionClampMethod::PERMISSIVE);
                    i = prologue_end;
                }
            }

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm512_loadu_si512(src + i);
                const auto d = _mm512_cvtepi64_pd(v);
                if (use_nt) {
                    _mm512_stream_pd(dst + i, d);
                } else {
                    _mm512_storeu_pd(dst + i, d);
                }
            }

            // Scalar tail — int->float ignores flag/method in scalar_convert (plain static_cast).
            scalar_tail<int64_t, float64_t>(src, dst, i, end_l, tail_flag, ConversionClampMethod::PERMISSIVE);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(float64_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }

    /**
     * Use AVX512DQ to accelerate the conversation between a FP64 block of memory to an INT64 block of memory, based on the "end" length.
     * Returns a vector of the shared futures containing threads running conversion. Vector is empty if the block was small enough for a single pin, else the futures are given.
     * @brief AVX512 accelerated FP64 > INT64. Requires AVX512DQ.
     * @note vcvttpd2qq truncates like the scalar static_cast in range; NaN/out-of-range lanes are masked
     * to the scalar clamp results (NaN -> 0, >= 2^63 -> INT64_MAX, < -2^63 -> INT64_MIN) and flag under
     * any non-PERMISSIVE method. -2^63 is exactly representable, so only strictly-less underflows.
     * @param src Source FP64 block of memory.
     * @param dst Target INT64 block of memory.
     * @param end Length of the FP64 target.
     * @param flag Atomic flag for conversion events the active method treats as an error.
     * @param method Active ConversionClampMethod.
     * @return A vector of shared futures representing the conversion tasks.
     */
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp64_to_int64(const float64_t* src, int64_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        static const auto supported = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512DQ);
        if (!supported) {
            throw ErrorHandling::CPUIllegalInstruction("AVX512DQ");
        }

        const auto t = IncludedCPUFeatures().threads;
        // NT stores gated on BOTH the runtime 64-byte alignment check and the
        // destination byte size clearing NtStoreMinBytes().
        const auto use_nt = reinterpret_cast<uintptr_t>(dst) % 64 == 0
            && end * sizeof(int64_t) > NtStoreMinBytes();

        auto compute = [src, dst, use_nt, &flag, method](const size_t start, const size_t end_l) {
            auto local_flag = false;
            auto i = start;

            // Scalar prologue: align dst to the 64-byte NT store boundary (8
            // elements of 8 bytes). Same scalar converter and local_flag as the tail.
            if (use_nt) {
                if (
                    const auto misalign = reinterpret_cast<uintptr_t>(dst + i) % 64;
                    misalign != 0
                ) {
                    const auto prologue_end = std::min(end_l, i + (64 - misalign) / sizeof(int64_t));
                    scalar_tail<float64_t, int64_t>(src, dst, i, prologue_end, local_flag, method);
                    i = prologue_end;
                }
            }

            for (; i + 8 <= end_l; i += 8) {
                const auto v = _mm512_loadu_pd(src + i);
                auto r = _mm512_cvttpd_epi64(v);

                // Bounds mirror scalar_convert float->int: (double)INT64_MAX rounds up to 2^63, and
                // since fp64 cannot represent INT64_MAX exactly, equality with 2^63 is also overflow.
                const auto nan = _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q);
                const auto ge = _mm512_cmp_pd_mask(v, _mm512_set1_pd(9223372036854775808.0), _CMP_GE_OQ);
                const auto lt = _mm512_cmp_pd_mask(v, _mm512_set1_pd(-9223372036854775808.0), _CMP_LT_OQ);
                if (
                    const auto bad = static_cast<unsigned>(nan | ge | lt);
                    bad != 0
                ) {
                    if (method != ConversionClampMethod::PERMISSIVE) {
                        local_flag = true;
                    }
                    r = _mm512_mask_mov_epi64(r, ge, _mm512_set1_epi64(std::numeric_limits<int64_t>::max()));
                    r = _mm512_mask_mov_epi64(r, lt, _mm512_set1_epi64(std::numeric_limits<int64_t>::min()));
                    r = _mm512_mask_mov_epi64(r, nan, _mm512_setzero_si512());
                }

                if (use_nt) {
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), r);
                } else {
                    _mm512_storeu_si512(dst + i, r);
                }
            }

            // Scalar tail — semantics live in scalar_tail (scalar_convert).
            scalar_tail<float64_t, int64_t>(src, dst, i, end_l, local_flag, method);

            if (use_nt) {
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
        const auto keep_local = KeepWorkLocal(end * sizeof(int64_t));
        const auto ccd_group = CcdGroupOfCurrentCpu();

        for (auto thread_id = size_t{0}; thread_id < t; ++thread_id) {
            const auto start = thread_id * chunk_size;
            const auto end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) {
                break;
            }
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { compute(start, end_worker); },
                    ccd_group,
                    Threading::TaskPriority::Normal
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { compute(start, end_worker); },
                    Threading::TaskPriority::Normal
                ).share());
            }
        }
        return futures;
    }
#else
    /**
     * @brief IllegalInstructionPresent - AVX512DQ support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_int64_to_fp64(const int64_t* src, float64_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512DQ");
    }
    /**
     * @brief IllegalInstructionPresent - AVX512DQ support not compiled.
     */
    std::vector<std::shared_future<void>> avx512_fp64_to_int64(const float64_t* src, int64_t* dst, const size_t& end, std::atomic<bool>& flag, const ConversionClampMethod method) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512DQ");
    }
#endif
}
