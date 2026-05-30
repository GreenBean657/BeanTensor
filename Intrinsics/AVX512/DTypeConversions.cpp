//
// Created by greenbean on 2026-05-30.
//

#include "DTypeConversions.h"
#include "Errors/logic.h"
#include <immintrin.h>
namespace BeanTensor::Intrinsics::detail {
#ifdef __AVX512BF16__
    std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const Casting::float32_t* src, Casting::bfloat16_t* dst, const size_t& end) {
        const size_t t = Hardware::CPU().threads;
        const auto* src_f = static_cast<const Casting::float32_t*>(src);

        // FIX: runtime 64-byte alignment check — enables NT stores which bypass cache
        const bool aligned = (reinterpret_cast<uintptr_t>(dst) % 64) == 0;

        auto compute = [src_f, dst, aligned](const size_t start, const size_t end_l) {
            std::size_t i = start;

            if (aligned) {
                // NT store path — bypasses cache
                for (; i + 32 <= end_l; i += 32) {
                    __m256bh lo = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_f + i));
                    __m256bh hi = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_f + i + 16));
                    const auto packed = _mm512_inserti64x4(
                        _mm512_castsi256_si512(reinterpret_cast<__m256i>(lo)),
                        reinterpret_cast<__m256i>(hi), 1);
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i), packed); // FIX: NT store
                }
                for (; i + 16 <= end_l; i += 16) {
                    __m256bh result = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_f + i));
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i),          // FIX: NT store
                                        reinterpret_cast<__m256i>(result));
                }
            } else {
                // Unaligned fallback — no NT stores
                for (; i + 32 <= end_l; i += 32) {
                    __m256bh lo = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_f + i));
                    __m256bh hi = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_f + i + 16));
                    const auto packed = _mm512_inserti64x4(
                        _mm512_castsi256_si512(reinterpret_cast<__m256i>(lo)),
                        reinterpret_cast<__m256i>(hi), 1);
                    _mm512_storeu_si512(dst + i, packed);
                }
                for (; i + 16 <= end_l; i += 16) {
                    __m256bh result = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_f + i));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                                        reinterpret_cast<__m256i>(result));
                }
            }

            for (; i < end_l; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &src_f[i], sizeof(bits));
                bits += 0x7FFF + ((bits >> 16) & 1);
                auto raw = static_cast<uint16_t>(bits >> 16);
                std::memcpy(&dst[i], &raw, sizeof(raw));
            }

            // FIX: sfence required after NT stores to ensure visibility to other threads/cores
            if (aligned) _mm_sfence();
        };

        std::vector<std::shared_future<void>> futures{};
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::get_cpu_thread_pool();
        const size_t chunk_size = ((end / t + 31) / 32) * 32;

        size_t actual_threads = 0;
        for (size_t tid = 0; tid < t; ++tid) {
            if (tid * chunk_size >= end) break;
            ++actual_threads;
        }

        auto worker = [compute](const size_t start, const size_t w_end) {
            compute(start, w_end);
        };
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_worker = std::min(start + chunk_size, end);
            if (start >= end_worker) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_worker); },
                Threading::ThreadPriority::Normal
            ).share());
        }
        return futures;
    }

    std::vector<std::shared_future<void>> avx512_bf16_to_fp32(const Casting::bfloat16_t* src, Casting::float32_t* dst, const size_t& end) {
        const size_t t = Hardware::CPU().threads;

        // FIX: runtime 64-byte alignment check for NT stores
        const bool aligned = (reinterpret_cast<uintptr_t>(dst) % 64) == 0;

        auto compute = [src, dst, aligned](const size_t start, const size_t end_l) {
            std::size_t i = start;

            if (aligned) {
                //added symmetric 32-wide unroll to match avx512_fp32_to_bf16 + NT stores
                for (; i + 32 <= end_l; i += 32) {
                    const __m256i lo_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                    const __m256i hi_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 16));
                    const __m512i lo_ext  = _mm512_cvtepu16_epi32(lo_bf16);
                    const __m512i hi_ext  = _mm512_cvtepu16_epi32(hi_bf16);
                    _mm512_stream_ps(dst + i,      _mm512_castsi512_ps(_mm512_slli_epi32(lo_ext, 16))); // FIX: NT store
                    _mm512_stream_ps(dst + i + 16, _mm512_castsi512_ps(_mm512_slli_epi32(hi_ext, 16))); // FIX: NT store
                }
                for (; i + 16 <= end_l; i += 16) {
                    const __m256i bf16_ints = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                    const __m512i extended  = _mm512_cvtepu16_epi32(bf16_ints);
                    _mm512_stream_ps(dst + i, _mm512_castsi512_ps(_mm512_slli_epi32(extended, 16))); // FIX: NT store
                }
            } else {
                //32-wide unroll, unaligned fallback
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
                dst[i] = std::bit_cast<Casting::float32_t>(bits);
            }

            // FIX: sfence after NT stores
            if (aligned) _mm_sfence();
        };

        std::vector<std::shared_future<void>> futures;
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::get_cpu_thread_pool();
        const size_t chunk_size = ((end / t + 31) / 32) * 32;

        size_t actual_threads = 0;
        for (size_t tid = 0; tid < t; ++tid) {
            if (tid * chunk_size >= end) break;
            ++actual_threads;
        }

        auto worker = [compute](const size_t start, const size_t end_l) {
            compute(start, end_l);
        };
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_v = std::min(start + chunk_size, end);
            if (start >= end) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_v); },
                Threading::ThreadPriority::Normal
            ).share());
        }
        return futures;
    }
#else
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const float* src, Casting::bfloat16_t* dst, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512BF");
    }
    std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const float* src, const size_t& end) {
        throw ErrorHandling::CPUIllegalInstruction("AVX512BF");
    }
#endif
}