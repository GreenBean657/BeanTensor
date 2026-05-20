#pragma once
#include "Helper/hardware.h"
#include "Helper/Broadcast/Broadcast.h"
#include "Helper/Casting/casting_detail.h"
#ifdef USE_HIP

#include <hip/hip_runtime.h>
#elif USE_CUDA
#include <cuda/cuda_runtime.h>
#endif
#include "Dtypes.h"
#include <immintrin.h>
#include <vector>
using namespace BeanTensor;
namespace BeanTensor::Tensors::detail {
    inline void avx512_fp32_to_bf16(const float* src, Casting::bfloat16_t* dst, const size_t n) {
        const size_t t = Hardware::CPU().threads;

        auto worker = [&](const size_t start, const size_t end) {
            std::size_t i = start;
            for ( ; i + 16 <= end; i += 16 ) {
                const __m512 vec = _mm512_loadu_ps(src + i);
                __m256bh result  = _mm512_cvtneps_pbh(vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), reinterpret_cast<__m256i>(result));
            }
            for ( ; i < end; ++i ) {
                uint32_t bits;
                std::memcpy(&bits, &src[i], sizeof(bits));
                bits += 0x7FFF + ((bits >> 16) & 1);
                auto raw = static_cast<uint16_t>(bits >> 16);
                std::memcpy(&dst[i], &raw, sizeof(raw));
            }
        };
        // TODO: Benchmark other modes
        if (n <= t * 64) {
            worker(0, n);
            return;
        }
        std::vector<std::thread> threads(t);
        const size_t chunk_size = (n + t - 1) / t;
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            size_t start = thread_id * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            threads[thread_id] = std::thread(worker, start, end);
        }
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            threads[thread_id].join();
        }
    }


    inline void unaccelerated_bf16_to_fp32(const Casting::bfloat16_t* src, float* dst, const size_t n) {

        const size_t t = Hardware::CPU().threads;
        auto worker = [&](const size_t start, const size_t end) {
            for (size_t pos = start; pos < end; ++pos) {
                uint16_t raw;
                std::memcpy(&raw, &src[pos], sizeof(raw));
                uint32_t bits = static_cast<uint32_t>(raw) << 16;
                dst[pos] = std::bit_cast<float>(bits);
            }
        };
        // TODO: Benchmark other modes
        if (n <= t * 64) {
            worker(0, n);
            return;
        }

        std::vector<std::thread> threads(t);
        const size_t chunk_size = (n + t - 1) / t;
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            size_t start = thread_id * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            threads[thread_id] = std::thread(worker, start, end);
        }
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            threads[thread_id].join();
        }
    }


    inline void avx512_bf16_to_fp32(const Casting::bfloat16_t* src, float* dst, const size_t n) {
        const size_t t = Hardware::CPU().threads;

        auto worker = [&](const size_t start, const size_t end) {
            std::size_t i = start;
            for ( ; i + 16 <= end; i += 16 ) {
                const __m256i bf16_ints = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                __m512i extended  = _mm512_cvtepu16_epi32(bf16_ints);
                const auto shifted   = _mm512_slli_epi32(extended, 16);
                _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(shifted));
            }
            for ( ; i < end; ++i ) {
                uint16_t raw;
                std::memcpy(&raw, &src[i], sizeof(raw));
                uint32_t bits = static_cast<uint32_t>(raw) << 16;
                dst[i] = std::bit_cast<float>(bits);
            }
        };
        if (n <= t * 64) {
            worker(0, n);
            return;
        }
        std::vector<std::thread> threads(t);
        const size_t chunk_size = (n + t - 1) / t;
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            size_t start = thread_id * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            threads[thread_id] = std::thread(worker, start, end);
        }
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            threads[thread_id].join();
        }
    }

    inline void launch_convert(const Casting::DType& origin, const Casting::DType& dest_dtype, void* source, void* destination, size_t& n) {
        if (Casting::dtype_is_int(origin)) {

        } else if (Casting::dtype_is_uint(origin)) {
            return;

        } else if (Casting::dtype_is_float(origin)) {
            if (dest_dtype == Casting::DType::Float32) {
                // Convert to float32
                if (origin == Casting::DType::BFloat16) {
                    const auto* src = static_cast<Casting::bfloat16_t*>(source);
                    auto* dst = static_cast<float*>(destination);
                    if (Hardware::CPU().f16c) {
                        Tensors::detail::avx512_bf16_to_fp32(src, dst, n);
                        return;
                    }
                    assert(false);
                    Tensors::detail::unaccelerated_bf16_to_fp32(src, dst, n);
                    return;
                }
            }
            if (dest_dtype == Casting::DType::BFloat16) {
                // Convert to float16
                if (origin == Casting::DType::Float32) {
                    const auto* src = static_cast<float*>(source);
                    auto* dst = static_cast<Casting::bfloat16_t*>(destination);
                    if (Hardware::CPU().f16c) {
                        Tensors::detail::avx512_fp32_to_bf16(src, dst, n);
                        return;
                    }
                    assert(false && "Non AVX512 cpu for BFLOAT16>FP32.");
                }
                assert(false && "Unsupported dtype for conversion");
            }
            assert(false && "Unsupported dtype for conversion");
        } else {
            assert(false && "Unsupported dtype for conversion");
        }

    }
}