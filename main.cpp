#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <immintrin.h>

#include "Tensor/tensor.h"

// ─────────────────────────────────────────────
//  Approach 1: memcpy (no alignment guarantee)
// ─────────────────────────────────────────────
void approach_memcpy(const float* accum, uint16_t* bf16, size_t n) {
    for (size_t i = 0; i + 16 <= n; i += 16) {
        __m512 vec;
        memcpy(&vec, accum + i, 16 * sizeof(float));
        __m256bh result = _mm512_cvtneps_pbh(vec);
        memcpy(bf16 + i, &result, 16 * sizeof(uint16_t));
    }
}
 
// ─────────────────────────────────────────────
//  Approach 2: aligned load/store
// ─────────────────────────────────────────────
void approach_aligned(const float* accum, uint16_t* bf162, size_t n) {
    alignas(64) float accum_align[128]{};
    for (size_t i = 0; i < n; i++)
        accum_align[i] = accum[i];
 
    for (size_t i = 0; i + 16 <= n; i += 16) {
        const __m512 vec = _mm512_load_ps(accum_align + i);
        __m256bh result  = _mm512_cvtneps_pbh(vec);
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(bf162 + i),
            reinterpret_cast<__m256i>(result)
        );
    }
}

void convert(const void* src, uint16_t* out, size_t n) {
    const auto* data = static_cast<const float*>(src);
    for (size_t i = 0; i + 16 <= n; i += 16) {
        __m512 vec      = _mm512_loadu_ps(data + i);   // no assumption needed
        __m256bh result = _mm512_cvtneps_pbh(vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i),
                            reinterpret_cast<__m256i>(result));
    }
}
 
// ─────────────────────────────────────────────
//  Approach 3: owned aligned allocation (ideal)
//  Fill directly into aligned buffer — no extra copy
// ─────────────────────────────────────────────
void approach_owned(std::mt19937& rng,
                    std::uniform_real_distribution<float>& dist,
                    float* accum_align,     // pre-aligned by caller
                    uint16_t* bf162,        // pre-aligned by caller
                    size_t n) {
    for (size_t i = 0; i < n; i++)
        accum_align[i] = dist(rng);
 
    for (size_t i = 0; i + 16 <= n; i += 16) {
        const __m512 vec = _mm512_load_ps(accum_align + i);
        __m256bh result  = _mm512_cvtneps_pbh(vec);
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(bf162 + i),
            reinterpret_cast<__m256i>(result)
        );
    }
}
 
// ─────────────────────────────────────────────
//  Benchmark runner
// ─────────────────────────────────────────────
template<typename Fn>
long long run_bench(Fn fn, int iterations) {
    // warmup
    for (int i = 0; i < 1000; i++) fn();
 
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
 
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}
 
int main2() {

    constexpr size_t N          = 128;
    constexpr int    ITERATIONS = 1'000'000;
 
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
 
    // shared input
    float accum[N];
    for (size_t i = 0; i < N; i++) accum[i] = dist(rng);
 
    // output buffers
    uint16_t bf16_1[N];
    alignas(64) uint16_t bf16_2[N]{};
    alignas(64) float    accum_owned[N]{};
    alignas(64) uint16_t bf16_3[N]{};
 
    // ── run benchmarks ──
    auto t1 = run_bench([&]() {
        approach_memcpy(accum, bf16_1, N);
    }, ITERATIONS);
 
    auto t2 = run_bench([&]() {
        approach_aligned(accum, bf16_2, N);
    }, ITERATIONS);
 
    // pre-fill once, outside the benchmark
    for (size_t i = 0; i < N; i++) accum_owned[i] = dist(rng);

    auto t3 = run_bench([&]() {
        for (size_t i = 0; i + 16 <= N; i += 16) {
            const __m512 vec = _mm512_load_ps(accum_owned + i);
            __m256bh result  = _mm512_cvtneps_pbh(vec);
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(bf16_3 + i),
                reinterpret_cast<__m256i>(result)
            );
        }
    }, ITERATIONS);

    auto t4 = run_bench([&]() {
        convert(accum, bf16_3, N);
    }, ITERATIONS);

    // ── results ──
    std::cout << "\n=== BF16 Conversion Benchmark (" << ITERATIONS << " iterations) ===\n\n";
    std::cout << "  Approach 1 (memcpy):              " << t1 << " us\n";
    std::cout << "  Approach 2 (aligned + extra copy):" << t2 << " us\n";
    std::cout << "  Approach 3 (owned aligned buffer):" << t3 << " us\n";
    std::cout << "  Approach 4 (convert):              " << t4 << " us\n";
    std::cout << "\n  Approach 1 vs 2: " << (t2 > t1 ? "+" : "") << (t2 - t1) << " us\n";
    std::cout << "  Approach 1 vs 3: " << (t3 > t1 ? "+" : "") << (t3 - t1) << " us\n";
    std::cout << "  Approach 2 vs 3: " << (t3 > t2 ? "+" : "") << (t3 - t2) << " us\n"
    << "  Approach 1 vs 4: " << (t4 > t1 ? "+" : "") << (t4 - t1) << " us\n";
 
    // sanity check: spot-check first value matches between approaches
    std::cout << "\n  Sanity (bf16[0]): "
              << "A1=" << bf16_1[0]
              << "  A2=" << bf16_2[0] << "\n\n";
 
    return 0;
}

using namespace BeanTensor;
int main() {
    std::cout << Hardware::cputostring() << "\n";
    Tensors::Tensor b = Tensors::make_tensor((size_t[]){2, 6}, 2, Casting::DType::Float32, Tensors::Device::CPU, false);
    Tensors::set_random(b, 0);
    std::cout << "=== Before conversion ===\n";
    std::cout << Tensors::print_tensor(b) << std::endl;
    std::cout << "Printed" << std::endl;
    sleep(2);
    Tensors::convert_dtype(b, Casting::DType::BFloat16);
    sleep(2);
    std::cout << "=== After conversion to BFloat16 ===\n" << std::endl;
    sleep(2);
    std::cout << Tensors::print_tensor(b);
    return 0;
}
 
