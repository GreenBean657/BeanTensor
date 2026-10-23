/**
 * @file matmul_tests.cpp
 * @author GreenBean
 * @brief Matmul tests - framework-free assertion suite.
 * Covers CPU GEMM across the full dtype family, batched rank-3 matmul, argument validation,
 * and GPU GEMM (rocBLAS/cuBLAS, packed dtypes, int8, integer fallback kernels) when a backend and device exist.
 * @version 0.1
 * @date 2026-10-18
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "Helpers/Errors.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float16_t;
using BeanTensor::float16x2_t;
using BeanTensor::bfloat16_t;
using BeanTensor::bfloat16x2_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::int4x2_t;
using DType = BeanTensor::DataTypes::DType;
namespace idet = BeanTensor::Intrinsics::detail;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool gpu_available() {
#if defined(USE_HIP)
    int32_t count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
#elif defined(USE_CUDA)
    int32_t count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

static bool near(const float32_t a, const float32_t b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - static_cast<float64_t>(b)) <= eps;
}

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

// --- Packed-dtype helpers (host-side, via the unaccel conversion pathway) ---

template<typename H>
static H narrow_lane(const float32_t v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return idet::unaccel_fp32_to_fp16_untracked(v);
    } else {
        return idet::unaccel_fp32_to_bf16_untracked(v);
    }
}

template<typename H>
static float32_t widen_lane(const H& v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return idet::unaccel_fp16_to_fp32_untracked(v);
    } else {
        return idet::unaccel_bf16_to_fp32(v);
    }
}

template<typename P, typename H>
static P pack_lanes(const float32_t lo, const float32_t hi) {
    const auto lo_bits = static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(lo)));
    const auto hi_bits = static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(narrow_lane<H>(hi)));
    return idet::bit_cast_memcpy<P>(lo_bits | (hi_bits << 16));
}

template<typename H, typename P>
static float32_t lane_lo_f32(const P& v) {
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    return widen_lane(idet::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu)));
}

template<typename H, typename P>
static float32_t lane_hi_f32(const P& v) {
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    return widen_lane(idet::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16)));
}

// Independent ijk-order reference with fp64 accumulation.
static void ref_gemm_f32(const std::vector<float32_t>& a, const std::vector<float32_t>& b,
                         std::vector<float32_t>& c, const size_t m, const size_t k, const size_t n) {
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            auto acc = 0.0;
            for (auto kk = size_t{0}; kk < k; ++kk) {
                acc += static_cast<float64_t>(a[i * k + kk]) * static_cast<float64_t>(b[kk * n + j]);
            }
            c[i * n + j] = static_cast<float32_t>(acc);
        }
    }
}

static void test_matmul_cpu_f32_exact() {
    // [1 2 3; 4 5 6] · [1 2; 3 4; 5 6] = [22 28; 49 64] — exact in fp32.
    Tensor a({2, 3}, DType::Float32);
    Tensor b({3, 2}, DType::Float32);
    auto v = 1.0f;
    for (auto i = size_t{0}; i < 2; ++i) {
        for (auto j = size_t{0}; j < 3; ++j) {
            a.set<float32_t>({i, j}, v);
            v += 1.0f;
        }
    }
    v = 1.0f;
    for (auto i = size_t{0}; i < 3; ++i) {
        for (auto j = size_t{0}; j < 2; ++j) {
            b.set<float32_t>({i, j}, v);
            v += 1.0f;
        }
    }
    auto c = a.matmul(b);
    c.sync();
    CHECK(near(c.at<float32_t>({0, 0}), 22.0f, 0.0));
    CHECK(near(c.at<float32_t>({0, 1}), 28.0f, 0.0));
    CHECK(near(c.at<float32_t>({1, 0}), 49.0f, 0.0));
    CHECK(near(c.at<float32_t>({1, 1}), 64.0f, 0.0));
    CHECK(c.shape_to_string() == "{2, 2}");
}

static void test_matmul_cpu_f64_identity() {
    Tensor a({3, 3}, DType::Float64);
    Tensor eye({3, 3}, DType::Float64);
    for (auto i = size_t{0}; i < 3; ++i) {
        for (auto j = size_t{0}; j < 3; ++j) {
            a.set<float64_t>({i, j}, static_cast<float64_t>(i * 3 + j) + 0.25);
            eye.set<float64_t>({i, j}, i == j ? 1.0 : 0.0);
        }
    }
    auto c = a.matmul(eye);
    c.sync();
    for (auto i = size_t{0}; i < 3; ++i) {
        for (auto j = size_t{0}; j < 3; ++j) {
            CHECK(near64(c.at<float64_t>({i, j}), static_cast<float64_t>(i * 3 + j) + 0.25, 0.0));
        }
    }
}

static void test_matmul_cpu_threaded_vs_reference() {
    // Large enough to engage the thread-pool row split (B·m·n·k > 32768) and
    // the AVX512/AVX2 row kernels where the CPU supports them.
    constexpr size_t m = 128;
    constexpr size_t k = 96;
    constexpr size_t n = 64;
    Tensor a({m, k}, DType::Float32);
    Tensor b({k, n}, DType::Float32);
    a.random(-1.0, 1.0, 42);
    b.random(-1.0, 1.0, 43);
    a.sync();
    b.sync();

    std::vector<float32_t> av(m * k);
    std::vector<float32_t> bv(k * n);
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto kk = size_t{0}; kk < k; ++kk) {
            av[i * k + kk] = a.at<float32_t>({i, kk});
        }
    }
    for (auto kk = size_t{0}; kk < k; ++kk) {
        for (auto j = size_t{0}; j < n; ++j) {
            bv[kk * n + j] = b.at<float32_t>({kk, j});
        }
    }
    std::vector<float32_t> ref(m * n);
    ref_gemm_f32(av, bv, ref, m, k, n);

    auto c = a.matmul(b);
    c.sync();
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(near(c.at<float32_t>({i, j}), ref[i * n + j], 1e-3));
        }
    }
}

static void test_matmul_cpu_f16_bf16() {
    // a = [1 2; 3 4], b = [1 3; 2 4] → c = [5 11; 11 25] — exact in fp16/bf16.
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        Tensor a({2, 2}, dtype);
        Tensor b({2, 2}, dtype);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto j = size_t{0}; j < 2; ++j) {
                const auto av = static_cast<float32_t>(i * 2 + j) + 1.0f;
                const auto bv = static_cast<float32_t>(j * 2 + i) + 1.0f;
                if (dtype == DType::Float16) {
                    a.set<float16_t>({i, j}, narrow_lane<float16_t>(av));
                    b.set<float16_t>({i, j}, narrow_lane<float16_t>(bv));
                } else {
                    a.set<bfloat16_t>({i, j}, narrow_lane<bfloat16_t>(av));
                    b.set<bfloat16_t>({i, j}, narrow_lane<bfloat16_t>(bv));
                }
            }
        }
        auto c = a.matmul(b);
        c.convert_dtype(DType::Float32);
        c.sync();
        CHECK(near(c.at<float32_t>({0, 0}), 5.0f, 0.26));
        CHECK(near(c.at<float32_t>({0, 1}), 11.0f, 0.26));
        CHECK(near(c.at<float32_t>({1, 0}), 11.0f, 0.26));
        CHECK(near(c.at<float32_t>({1, 1}), 25.0f, 0.26));
    }
}

static void test_matmul_cpu_packed() {
    // Per-lane semantics: lo lanes are a_lo·b_lo, hi lanes a_hi·b_hi.
    // a_lo = [1 2; 3 4], b_lo = [1 3; 2 4] → [5 11; 11 25]
    // a_hi = [2 4; 6 8], b_hi = b_lo      → [10 22; 22 50]
    for (const auto dtype : {DType::Float16x2, DType::BFloat16x2}) {
        Tensor a({2, 2}, dtype);
        Tensor b({2, 2}, dtype);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto j = size_t{0}; j < 2; ++j) {
                const auto a_lo = static_cast<float32_t>(i * 2 + j) + 1.0f;
                const auto a_hi = 2.0f * a_lo;
                const auto b_v = static_cast<float32_t>(j * 2 + i) + 1.0f;
                if (dtype == DType::Float16x2) {
                    a.set<float16x2_t>({i, j}, pack_lanes<float16x2_t, float16_t>(a_lo, a_hi));
                    b.set<float16x2_t>({i, j}, pack_lanes<float16x2_t, float16_t>(b_v, b_v));
                } else {
                    a.set<bfloat16x2_t>({i, j}, pack_lanes<bfloat16x2_t, bfloat16_t>(a_lo, a_hi));
                    b.set<bfloat16x2_t>({i, j}, pack_lanes<bfloat16x2_t, bfloat16_t>(b_v, b_v));
                }
            }
        }
        auto c = a.matmul(b);
        c.sync();
        constexpr float32_t expected_lo[4] = {5.0f, 11.0f, 11.0f, 25.0f};
        constexpr float32_t expected_hi[4] = {10.0f, 22.0f, 22.0f, 50.0f};
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto j = size_t{0}; j < 2; ++j) {
                const auto idx = i * 2 + j;
                if (dtype == DType::Float16x2) {
                    const auto v = c.at<float16x2_t>({i, j});
                    CHECK(near(lane_lo_f32<float16_t>(v), expected_lo[idx], 0.26));
                    CHECK(near(lane_hi_f32<float16_t>(v), expected_hi[idx], 0.26));
                } else {
                    const auto v = c.at<bfloat16x2_t>({i, j});
                    CHECK(near(lane_lo_f32<bfloat16_t>(v), expected_lo[idx], 0.51));
                    CHECK(near(lane_hi_f32<bfloat16_t>(v), expected_hi[idx], 0.51));
                }
            }
        }
    }
}

static void test_matmul_cpu_batched() {
    // batch 0: A·I = A; batch 1: [1 2; 3 4]·[2 0; 1 2] = [4 4; 10 8].
    Tensor a({2, 2, 2}, DType::Float32);
    Tensor b({2, 2, 2}, DType::Float32);
    for (auto i = size_t{0}; i < 2; ++i) {
        for (auto j = size_t{0}; j < 2; ++j) {
            const auto av = static_cast<float32_t>(i * 2 + j) + 1.0f;
            a.set<float32_t>({0, i, j}, av);
            a.set<float32_t>({1, i, j}, av);
        }
    }
    b.set<float32_t>({0, 0, 0}, 1.0f);
    b.set<float32_t>({0, 0, 1}, 0.0f);
    b.set<float32_t>({0, 1, 0}, 0.0f);
    b.set<float32_t>({0, 1, 1}, 1.0f);
    b.set<float32_t>({1, 0, 0}, 2.0f);
    b.set<float32_t>({1, 0, 1}, 0.0f);
    b.set<float32_t>({1, 1, 0}, 1.0f);
    b.set<float32_t>({1, 1, 1}, 2.0f);
    auto c = a.matmul(b);
    c.sync();
    CHECK(c.shape_to_string() == "{2, 2, 2}");
    CHECK(near(c.at<float32_t>({0, 0, 0}), 1.0f, 0.0));
    CHECK(near(c.at<float32_t>({0, 0, 1}), 2.0f, 0.0));
    CHECK(near(c.at<float32_t>({0, 1, 0}), 3.0f, 0.0));
    CHECK(near(c.at<float32_t>({0, 1, 1}), 4.0f, 0.0));
    CHECK(near(c.at<float32_t>({1, 0, 0}), 4.0f, 0.0));
    CHECK(near(c.at<float32_t>({1, 0, 1}), 4.0f, 0.0));
    CHECK(near(c.at<float32_t>({1, 1, 0}), 10.0f, 0.0));
    CHECK(near(c.at<float32_t>({1, 1, 1}), 8.0f, 0.0));
}

// fp64-accumulation reference for the half/packed SIMD-kernel tests.
static void ref_gemm_f64(const std::vector<float64_t>& a, const std::vector<float64_t>& b,
                         std::vector<float64_t>& c, const size_t m, const size_t k, const size_t n) {
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            auto acc = 0.0;
            for (auto kk = size_t{0}; kk < k; ++kk) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// Deterministic LCG values in [-1, 1] — Tensor::random does not cover the
// half/packed dtypes.
static float32_t lcg_value(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<float32_t>((state >> 8) % 2001u) / 1000.0f - 1.0f;
}

template<typename H>
static void fill_half(Tensor& t, std::vector<float64_t>& widened, const size_t rows, const size_t cols, const uint32_t seed) {
    auto state = seed;
    for (auto i = size_t{0}; i < rows; ++i) {
        for (auto j = size_t{0}; j < cols; ++j) {
            const auto narrowed = narrow_lane<H>(lcg_value(state));
            t.set<H>({i, j}, narrowed);
            widened[i * cols + j] = static_cast<float64_t>(widen_lane(narrowed));
        }
    }
}

static void test_matmul_cpu_f16_bf16_large() {
    // Odd extents exercise the odd-k tail, the masked column tails and the
    // single-row tail of the AVX512 half-dtype kernels (and of the scalar
    // fallback on CPUs without them). Big enough to engage the thread pool.
    constexpr size_t m = 67;
    constexpr size_t k = 51;
    constexpr size_t n = 29;
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        Tensor a({m, k}, dtype);
        Tensor b({k, n}, dtype);
        std::vector<float64_t> av(m * k);
        std::vector<float64_t> bv(k * n);
        if (dtype == DType::Float16) {
            fill_half<float16_t>(a, av, m, k, 11);
            fill_half<float16_t>(b, bv, k, n, 12);
        } else {
            fill_half<bfloat16_t>(a, av, m, k, 11);
            fill_half<bfloat16_t>(b, bv, k, n, 12);
        }
        std::vector<float64_t> ref(m * n);
        ref_gemm_f64(av, bv, ref, m, k, n);
        auto c = a.matmul(b);
        c.convert_dtype(DType::Float32);
        c.sync();
        // Output rounding dominates: one half/bf16 ulp of the result.
        const auto rel = dtype == DType::Float16 ? 0.006 : 0.025;
        for (auto i = size_t{0}; i < m; ++i) {
            for (auto j = size_t{0}; j < n; ++j) {
                const auto rv = ref[i * n + j];
                CHECK(near(c.at<float32_t>({i, j}), static_cast<float32_t>(rv), rel * std::fabs(rv) + 1e-3));
            }
        }
    }
}

template<typename P, typename H>
static void fill_packed(Tensor& t, std::vector<float64_t>& widened_lo, std::vector<float64_t>& widened_hi,
                        const size_t rows, const size_t cols, const uint32_t seed) {
    auto state = seed;
    for (auto i = size_t{0}; i < rows; ++i) {
        for (auto j = size_t{0}; j < cols; ++j) {
            const auto lo = narrow_lane<H>(lcg_value(state));
            const auto hi = narrow_lane<H>(lcg_value(state));
            const auto lo_bits = static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(lo));
            const auto hi_bits = static_cast<uint32_t>(idet::bit_cast_memcpy<uint16_t>(hi));
            t.set<P>({i, j}, idet::bit_cast_memcpy<P>(lo_bits | (hi_bits << 16)));
            widened_lo[i * cols + j] = static_cast<float64_t>(widen_lane(lo));
            widened_hi[i * cols + j] = static_cast<float64_t>(widen_lane(hi));
        }
    }
}

static void test_matmul_cpu_packed_large() {
    // Per-lane FP64 reference; odd k exercises the packed kernels' zero-pair
    // tail, n = 17 the masked 16-column tail, m = 129 the single-row tail.
    constexpr size_t m = 129;
    constexpr size_t k = 25;
    constexpr size_t n = 17;
    for (const auto dtype : {DType::Float16x2, DType::BFloat16x2}) {
        Tensor a({m, k}, dtype);
        Tensor b({k, n}, dtype);
        std::vector<float64_t> av_lo(m * k);
        std::vector<float64_t> av_hi(m * k);
        std::vector<float64_t> bv_lo(k * n);
        std::vector<float64_t> bv_hi(k * n);
        if (dtype == DType::Float16x2) {
            fill_packed<float16x2_t, float16_t>(a, av_lo, av_hi, m, k, 21);
            fill_packed<float16x2_t, float16_t>(b, bv_lo, bv_hi, k, n, 22);
        } else {
            fill_packed<bfloat16x2_t, bfloat16_t>(a, av_lo, av_hi, m, k, 21);
            fill_packed<bfloat16x2_t, bfloat16_t>(b, bv_lo, bv_hi, k, n, 22);
        }
        std::vector<float64_t> ref_lo(m * n);
        std::vector<float64_t> ref_hi(m * n);
        ref_gemm_f64(av_lo, bv_lo, ref_lo, m, k, n);
        ref_gemm_f64(av_hi, bv_hi, ref_hi, m, k, n);
        auto c = a.matmul(b);
        c.sync();
        const auto rel = dtype == DType::Float16x2 ? 0.006 : 0.025;
        for (auto i = size_t{0}; i < m; ++i) {
            for (auto j = size_t{0}; j < n; ++j) {
                const auto rv_lo = ref_lo[i * n + j];
                const auto rv_hi = ref_hi[i * n + j];
                if (dtype == DType::Float16x2) {
                    const auto v = c.at<float16x2_t>({i, j});
                    CHECK(near(lane_lo_f32<float16_t>(v), static_cast<float32_t>(rv_lo), rel * std::fabs(rv_lo) + 1e-3));
                    CHECK(near(lane_hi_f32<float16_t>(v), static_cast<float32_t>(rv_hi), rel * std::fabs(rv_hi) + 1e-3));
                } else {
                    const auto v = c.at<bfloat16x2_t>({i, j});
                    CHECK(near(lane_lo_f32<bfloat16_t>(v), static_cast<float32_t>(rv_lo), rel * std::fabs(rv_lo) + 1e-3));
                    CHECK(near(lane_hi_f32<bfloat16_t>(v), static_cast<float32_t>(rv_hi), rel * std::fabs(rv_hi) + 1e-3));
                }
            }
        }
    }
}

// --- Integer matmul: modular (two's-complement wrap) contract -------------
// Every pathway (scalar/AVX2/AVX512-VNNI CPU, BLAS/fallback GPU) accumulates
// with wrapping adds at >=32-bit width and truncates on store, so results are
// bit-exact against this reference.

template<typename T>
static void ref_gemm_int(const std::vector<T>& a, const std::vector<T>& b,
                         std::vector<T>& c, const size_t m, const size_t k, const size_t n) {
    using UT = std::make_unsigned_t<T>;
    using Acc = std::conditional_t<(sizeof(T) < 8), uint32_t, uint64_t>;
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            auto acc = Acc{0};
            for (auto kk = size_t{0}; kk < k; ++kk) {
                acc += static_cast<Acc>(static_cast<UT>(a[i * k + kk]) * static_cast<UT>(b[kk * n + j]));
            }
            c[i * n + j] = static_cast<T>(acc);
        }
    }
}

// Full-range deterministic integer values (high LCG bits, modular truncation).
template<typename T>
static T lcg_int(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(state >> 16));
}

template<typename T>
static void run_int_case(const DType dtype, const size_t m, const size_t k, const size_t n, const uint32_t seed) {
    Tensor a({m, k}, dtype);
    Tensor b({k, n}, dtype);
    std::vector<T> av(m * k);
    std::vector<T> bv(k * n);
    auto state = seed;
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto kk = size_t{0}; kk < k; ++kk) {
            av[i * k + kk] = lcg_int<T>(state);
            a.set<T>({i, kk}, av[i * k + kk]);
        }
    }
    for (auto kk = size_t{0}; kk < k; ++kk) {
        for (auto j = size_t{0}; j < n; ++j) {
            bv[kk * n + j] = lcg_int<T>(state);
            b.set<T>({kk, j}, bv[kk * n + j]);
        }
    }
    std::vector<T> ref(m * n);
    ref_gemm_int(av, bv, ref, m, k, n);
    auto c = a.matmul(b);
    c.sync();
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(c.at<T>({i, j}) == ref[i * n + j]);
        }
    }
}

static void test_matmul_cpu_int_small() {
    // Odd extents hit masked/tail paths of the SIMD kernels.
    run_int_case<int8_t>(DType::Int8, 3, 5, 7, 31);
    run_int_case<uint8_t>(DType::UInt8, 3, 5, 7, 32);
    run_int_case<int16_t>(DType::Int16, 3, 5, 7, 33);
    run_int_case<uint16_t>(DType::UInt16, 3, 5, 7, 34);
    run_int_case<int32_t>(DType::Int32, 3, 5, 7, 35);
    run_int_case<uint32_t>(DType::UInt32, 3, 5, 7, 36);
    run_int_case<int64_t>(DType::Int64, 3, 5, 7, 37);
    run_int_case<uint64_t>(DType::UInt64, 3, 5, 7, 38);
}

static void test_matmul_cpu_int_wrap() {
    // Deliberate overflow: int8 127·127·k wraps mod 2^8; every pathway must
    // produce the identical wrapped byte.
    constexpr size_t m = 4;
    constexpr size_t k = 40;
    constexpr size_t n = 4;
    Tensor a({m, k}, DType::Int8);
    Tensor b({k, n}, DType::Int8);
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto kk = size_t{0}; kk < k; ++kk) {
            a.set<int8_t>({i, kk}, static_cast<int8_t>(127));
        }
    }
    for (auto kk = size_t{0}; kk < k; ++kk) {
        for (auto j = size_t{0}; j < n; ++j) {
            b.set<int8_t>({kk, j}, static_cast<int8_t>(127));
        }
    }
    auto c = a.matmul(b);
    c.sync();
    // 40·127·127 = 645160; 645160 mod 256 = 216 → as int8: -40.
    const auto expected = static_cast<int8_t>(static_cast<uint8_t>((645160u) & 0xFFu));
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(c.at<int8_t>({i, j}) == expected);
        }
    }
}

static void test_matmul_cpu_int_large() {
    // 257×511×129: engages the thread pool and the SIMD main loops; odd k
    // exercises the pair tails.
    run_int_case<int8_t>(DType::Int8, 257, 129, 65, 41);
    run_int_case<uint8_t>(DType::UInt8, 257, 129, 65, 42);
    run_int_case<int16_t>(DType::Int16, 257, 129, 65, 43);
    run_int_case<uint16_t>(DType::UInt16, 257, 129, 65, 44);
    run_int_case<int32_t>(DType::Int32, 129, 255, 65, 45);
    run_int_case<uint32_t>(DType::UInt32, 129, 255, 65, 46);
    run_int_case<int64_t>(DType::Int64, 65, 129, 33, 47);
    run_int_case<uint64_t>(DType::UInt64, 65, 129, 33, 48);
}

static void test_matmul_cpu_int_batched() {
    // Rank-3 batched: [B, M, K] @ [B, K, N] over the pool's unit space.
    constexpr size_t batch = 4;
    constexpr size_t m = 16;
    constexpr size_t k = 24;
    constexpr size_t n = 8;
    Tensor a({batch, m, k}, DType::Int32);
    Tensor b({batch, k, n}, DType::Int32);
    std::vector<int32_t> av(batch * m * k);
    std::vector<int32_t> bv(batch * k * n);
    auto state = uint32_t{51};
    for (auto bi = size_t{0}; bi < batch; ++bi) {
        for (auto i = size_t{0}; i < m; ++i) {
            for (auto kk = size_t{0}; kk < k; ++kk) {
                av[bi * m * k + i * k + kk] = lcg_int<int32_t>(state);
                a.set<int32_t>({bi, i, kk}, av[bi * m * k + i * k + kk]);
            }
        }
        for (auto kk = size_t{0}; kk < k; ++kk) {
            for (auto j = size_t{0}; j < n; ++j) {
                bv[bi * k * n + kk * n + j] = lcg_int<int32_t>(state);
                b.set<int32_t>({bi, kk, j}, bv[bi * k * n + kk * n + j]);
            }
        }
    }
    auto c = a.matmul(b);
    c.sync();
    CHECK(c.shape_to_string() == "{4, 16, 8}");
    for (auto bi = size_t{0}; bi < batch; ++bi) {
        std::vector<int32_t> ref(m * n);
        ref_gemm_int(std::vector<int32_t>(av.begin() + static_cast<std::ptrdiff_t>(bi * m * k), av.begin() + static_cast<std::ptrdiff_t>((bi + 1) * m * k)),
                     std::vector<int32_t>(bv.begin() + static_cast<std::ptrdiff_t>(bi * k * n), bv.begin() + static_cast<std::ptrdiff_t>((bi + 1) * k * n)),
                     ref, m, k, n);
        for (auto i = size_t{0}; i < m; ++i) {
            for (auto j = size_t{0}; j < n; ++j) {
                CHECK(c.at<int32_t>({bi, i, j}) == ref[i * n + j]);
            }
        }
    }
}

// --- Int4x2 matmul: per-lane modular contract ------------------------------

static int8_t test_nib_lo(const int4x2_t v) {
    const auto nib = static_cast<int8_t>(idet::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
    return nib > 7 ? static_cast<int8_t>(nib - 16) : nib;
}

static int8_t test_nib_hi(const int4x2_t v) {
    const auto nib = static_cast<int8_t>(idet::bit_cast_memcpy<uint8_t>(v) >> 4);
    return nib > 7 ? static_cast<int8_t>(nib - 16) : nib;
}

static int4x2_t pack_i4(const int8_t lo, const int8_t hi) {
    return idet::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>((static_cast<uint8_t>(lo) & 0x0Fu) | ((static_cast<uint8_t>(hi) & 0x0Fu) << 4)));
}

static void ref_gemm_int4x2(const std::vector<int4x2_t>& a, const std::vector<int4x2_t>& b,
                            std::vector<int4x2_t>& c, const size_t m, const size_t k, const size_t n) {
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            auto acc_lo = uint32_t{0};
            auto acc_hi = uint32_t{0};
            for (auto kk = size_t{0}; kk < k; ++kk) {
                const auto av = a[i * k + kk];
                const auto bv = b[kk * n + j];
                acc_lo += static_cast<uint32_t>(static_cast<int32_t>(test_nib_lo(av)) * static_cast<int32_t>(test_nib_lo(bv)));
                acc_hi += static_cast<uint32_t>(static_cast<int32_t>(test_nib_hi(av)) * static_cast<int32_t>(test_nib_hi(bv)));
            }
            c[i * n + j] = idet::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>((acc_lo & 0x0Fu) | ((acc_hi & 0x0Fu) << 4)));
        }
    }
}

static int4x2_t lcg_int4x2(uint32_t& state) {
    const auto lo = lcg_int<int8_t>(state);
    const auto hi = lcg_int<int8_t>(state);
    return pack_i4(static_cast<int8_t>(lo % 16), static_cast<int8_t>(hi % 16));
}

static void run_int4x2_case(const size_t m, const size_t k, const size_t n, const uint32_t seed) {
    Tensor a({m, k}, DType::Int4x2);
    Tensor b({k, n}, DType::Int4x2);
    std::vector<int4x2_t> av(m * k);
    std::vector<int4x2_t> bv(k * n);
    auto state = seed;
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto kk = size_t{0}; kk < k; ++kk) {
            av[i * k + kk] = lcg_int4x2(state);
            a.set<int4x2_t>({i, kk}, av[i * k + kk]);
        }
    }
    for (auto kk = size_t{0}; kk < k; ++kk) {
        for (auto j = size_t{0}; j < n; ++j) {
            bv[kk * n + j] = lcg_int4x2(state);
            b.set<int4x2_t>({kk, j}, bv[kk * n + j]);
        }
    }
    std::vector<int4x2_t> ref(m * n);
    ref_gemm_int4x2(av, bv, ref, m, k, n);
    auto c = a.matmul(b);
    c.sync();
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            const auto got = idet::bit_cast_memcpy<uint8_t>(c.at<int4x2_t>({i, j}));
            const auto want = idet::bit_cast_memcpy<uint8_t>(ref[i * n + j]);
            CHECK(got == want);
        }
    }
}

static void test_matmul_cpu_int4x2() {
    run_int4x2_case(3, 5, 7, 71);       // small, odd everything
    run_int4x2_case(129, 65, 33, 72);   // threaded + SIMD main loops + tails
}

static void test_matmul_argument_validation() {
    namespace EH = BeanTensor::ErrorHandling;
    bool threw = false;
    try {
        Tensor vec({4}, DType::Float32);
        Tensor mat({4, 4}, DType::Float32);
        auto c = vec.matmul(mat);
        c.sync();
    } catch (const EH::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Tensor a({2, 3}, DType::Float32);
        Tensor b({4, 2}, DType::Float32); // inner dims 3 vs 4
        auto c = a.matmul(b);
        c.sync();
    } catch (const EH::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Tensor a({2, 2, 2}, DType::Float32);
        Tensor b({3, 2, 2}, DType::Float32); // batch extents 2 vs 3
        auto c = a.matmul(b);
        c.sync();
    } catch (const EH::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Tensor a({2, 2}, DType::Float32);
        Tensor b({2, 2, 2}, DType::Float32); // rank 2 vs 3
        auto c = a.matmul(b);
        c.sync();
    } catch (const EH::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Tensor a({2, 2}, DType::Float32);
        Tensor b({2, 2}, DType::Float64);
        auto c = a.matmul(b);
        c.sync();
    } catch (const EH::MismatchedDTypes&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        Tensor a({2, 2}, DType::Float32);
        Tensor b({2, 2}, DType::Int32); // dtype mismatch still rejects
        auto c = a.matmul(b);
        c.sync();
    } catch (const EH::MismatchedDTypes&) {
        threw = true;
    }
    CHECK(threw);
}

#if defined(USE_HIP) || defined(USE_CUDA)
static void test_matmul_gpu_f32() {
    Tensor a_cpu({64, 48}, DType::Float32);
    Tensor b_cpu({48, 32}, DType::Float32);
    a_cpu.random(-1.0, 1.0, 7);
    b_cpu.random(-1.0, 1.0, 8);
    a_cpu.sync();
    b_cpu.sync();
    auto c_cpu = a_cpu.matmul(b_cpu);
    c_cpu.sync();

    auto a_gpu = a_cpu.clone(true);
    auto b_gpu = b_cpu.clone(true);
    a_gpu.to(Device::GPU);
    b_gpu.to(Device::GPU);
    a_gpu.sync();
    b_gpu.sync();
    auto c_gpu = a_gpu.matmul(b_gpu);
    c_gpu.sync();

    for (auto i = size_t{0}; i < 64; ++i) {
        for (auto j = size_t{0}; j < 32; ++j) {
            CHECK(near(c_gpu.at<float32_t>({i, j}), c_cpu.at<float32_t>({i, j}), 1e-3));
        }
    }
}

static void test_matmul_gpu_f16_bf16() {
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        Tensor a({2, 2}, dtype, Device::GPU);
        Tensor b({2, 2}, dtype, Device::GPU);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto j = size_t{0}; j < 2; ++j) {
                const auto av = static_cast<float32_t>(i * 2 + j) + 1.0f;
                const auto bv = static_cast<float32_t>(j * 2 + i) + 1.0f;
                if (dtype == DType::Float16) {
                    a.set<float16_t>({i, j}, narrow_lane<float16_t>(av));
                    b.set<float16_t>({i, j}, narrow_lane<float16_t>(bv));
                } else {
                    a.set<bfloat16_t>({i, j}, narrow_lane<bfloat16_t>(av));
                    b.set<bfloat16_t>({i, j}, narrow_lane<bfloat16_t>(bv));
                }
            }
        }
        auto c = a.matmul(b);
        c.convert_dtype(DType::Float32);
        c.sync();
        // a = [1 2; 3 4], b = [1 3; 2 4] → c = [5 11; 11 25]
        CHECK(near(c.at<float32_t>({0, 0}), 5.0f, 0.26));
        CHECK(near(c.at<float32_t>({0, 1}), 11.0f, 0.26));
        CHECK(near(c.at<float32_t>({1, 0}), 11.0f, 0.26));
        CHECK(near(c.at<float32_t>({1, 1}), 25.0f, 0.26));
    }
}

static void test_matmul_gpu_packed() {
    for (const auto dtype : {DType::Float16x2, DType::BFloat16x2}) {
        Tensor a({2, 2}, dtype, Device::GPU);
        Tensor b({2, 2}, dtype, Device::GPU);
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto j = size_t{0}; j < 2; ++j) {
                const auto a_lo = static_cast<float32_t>(i * 2 + j) + 1.0f;
                const auto a_hi = 2.0f * a_lo;
                const auto b_v = static_cast<float32_t>(j * 2 + i) + 1.0f;
                if (dtype == DType::Float16x2) {
                    a.set<float16x2_t>({i, j}, pack_lanes<float16x2_t, float16_t>(a_lo, a_hi));
                    b.set<float16x2_t>({i, j}, pack_lanes<float16x2_t, float16_t>(b_v, b_v));
                } else {
                    a.set<bfloat16x2_t>({i, j}, pack_lanes<bfloat16x2_t, bfloat16_t>(a_lo, a_hi));
                    b.set<bfloat16x2_t>({i, j}, pack_lanes<bfloat16x2_t, bfloat16_t>(b_v, b_v));
                }
            }
        }
        auto c = a.matmul(b);
        c.sync();
        constexpr float32_t expected_lo[4] = {5.0f, 11.0f, 11.0f, 25.0f};
        constexpr float32_t expected_hi[4] = {10.0f, 22.0f, 22.0f, 50.0f};
        for (auto i = size_t{0}; i < 2; ++i) {
            for (auto j = size_t{0}; j < 2; ++j) {
                const auto idx = i * 2 + j;
                if (dtype == DType::Float16x2) {
                    const auto v = c.at<float16x2_t>({i, j});
                    CHECK(near(lane_lo_f32<float16_t>(v), expected_lo[idx], 0.26));
                    CHECK(near(lane_hi_f32<float16_t>(v), expected_hi[idx], 0.26));
                } else {
                    const auto v = c.at<bfloat16x2_t>({i, j});
                    CHECK(near(lane_lo_f32<bfloat16_t>(v), expected_lo[idx], 0.51));
                    CHECK(near(lane_hi_f32<bfloat16_t>(v), expected_hi[idx], 0.51));
                }
            }
        }
    }
}

static void test_matmul_gpu_batched() {
    Tensor a_cpu({4, 16, 24}, DType::Float32);
    Tensor b_cpu({4, 24, 8}, DType::Float32);
    a_cpu.random(-1.0, 1.0, 11);
    b_cpu.random(-1.0, 1.0, 12);
    a_cpu.sync();
    b_cpu.sync();
    auto c_cpu = a_cpu.matmul(b_cpu);
    c_cpu.sync();

    auto a_gpu = a_cpu.clone(true);
    auto b_gpu = b_cpu.clone(true);
    a_gpu.to(Device::GPU);
    b_gpu.to(Device::GPU);
    a_gpu.sync();
    b_gpu.sync();
    auto c_gpu = a_gpu.matmul(b_gpu);
    c_gpu.sync();
    CHECK(c_gpu.shape_to_string() == "{4, 16, 8}");

    for (auto bi = size_t{0}; bi < 4; ++bi) {
        for (auto i = size_t{0}; i < 16; ++i) {
            for (auto j = size_t{0}; j < 8; ++j) {
                CHECK(near(c_gpu.at<float32_t>({bi, i, j}), c_cpu.at<float32_t>({bi, i, j}), 1e-3));
            }
        }
    }
}

// GPU integer matmul must be bit-exact with the CPU modular result: int8
// rides the BLAS i8->i32 path (truncate on narrow), the other integer dtypes
// run the tiled fallback kernel.
template<typename T>
static void run_gpu_int_case(const DType dtype, const size_t m, const size_t k, const size_t n, const uint32_t seed) {
    Tensor a_cpu({m, k}, dtype);
    Tensor b_cpu({k, n}, dtype);
    std::vector<T> av(m * k);
    std::vector<T> bv(k * n);
    auto state = seed;
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto kk = size_t{0}; kk < k; ++kk) {
            av[i * k + kk] = lcg_int<T>(state);
            a_cpu.set<T>({i, kk}, av[i * k + kk]);
        }
    }
    for (auto kk = size_t{0}; kk < k; ++kk) {
        for (auto j = size_t{0}; j < n; ++j) {
            bv[kk * n + j] = lcg_int<T>(state);
            b_cpu.set<T>({kk, j}, bv[kk * n + j]);
        }
    }
    std::vector<T> ref(m * n);
    ref_gemm_int(av, bv, ref, m, k, n);

    auto a_gpu = a_cpu.clone(true);
    auto b_gpu = b_cpu.clone(true);
    a_gpu.to(Device::GPU);
    b_gpu.to(Device::GPU);
    a_gpu.sync();
    b_gpu.sync();
    auto c_gpu = a_gpu.matmul(b_gpu);
    c_gpu.sync();

    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(c_gpu.at<T>({i, j}) == ref[i * n + j]);
        }
    }
}

static void test_matmul_gpu_int() {
    run_gpu_int_case<int8_t>(DType::Int8, 33, 65, 17, 61);   // fallback kernel
    run_gpu_int_case<uint8_t>(DType::UInt8, 33, 65, 17, 62); // fallback kernel
    run_gpu_int_case<int16_t>(DType::Int16, 33, 65, 17, 63);
    run_gpu_int_case<uint16_t>(DType::UInt16, 33, 65, 17, 64);
    run_gpu_int_case<int32_t>(DType::Int32, 33, 65, 17, 65);
    run_gpu_int_case<uint32_t>(DType::UInt32, 33, 65, 17, 66);
    run_gpu_int_case<int64_t>(DType::Int64, 17, 33, 9, 67);
    run_gpu_int_case<uint64_t>(DType::UInt64, 17, 33, 9, 68);
}

static void test_matmul_gpu_int4x2() {
    constexpr size_t m = 33;
    constexpr size_t k = 65;
    constexpr size_t n = 17;
    Tensor a_cpu({m, k}, DType::Int4x2);
    Tensor b_cpu({k, n}, DType::Int4x2);
    std::vector<int4x2_t> av(m * k);
    std::vector<int4x2_t> bv(k * n);
    auto state = uint32_t{81};
    for (auto i = size_t{0}; i < m; ++i) {
        for (auto kk = size_t{0}; kk < k; ++kk) {
            av[i * k + kk] = lcg_int4x2(state);
            a_cpu.set<int4x2_t>({i, kk}, av[i * k + kk]);
        }
    }
    for (auto kk = size_t{0}; kk < k; ++kk) {
        for (auto j = size_t{0}; j < n; ++j) {
            bv[kk * n + j] = lcg_int4x2(state);
            b_cpu.set<int4x2_t>({kk, j}, bv[kk * n + j]);
        }
    }
    std::vector<int4x2_t> ref(m * n);
    ref_gemm_int4x2(av, bv, ref, m, k, n);

    auto a_gpu = a_cpu.clone(true);
    auto b_gpu = b_cpu.clone(true);
    a_gpu.to(Device::GPU);
    b_gpu.to(Device::GPU);
    a_gpu.sync();
    b_gpu.sync();
    auto c_gpu = a_gpu.matmul(b_gpu);
    c_gpu.sync();

    for (auto i = size_t{0}; i < m; ++i) {
        for (auto j = size_t{0}; j < n; ++j) {
            CHECK(idet::bit_cast_memcpy<uint8_t>(c_gpu.at<int4x2_t>({i, j})) == idet::bit_cast_memcpy<uint8_t>(ref[i * n + j]));
        }
    }
}

static void test_matmul_device_mismatch() {
    bool threw = false;
    try {
        Tensor a({2, 2}, DType::Float32, Device::CPU);
        Tensor b({2, 2}, DType::Float32, Device::GPU);
        auto c = a.matmul(b);
        c.sync();
    } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
        threw = true;
    }
    CHECK(threw);
}
#endif

int main() {
    test_matmul_cpu_f32_exact();
    test_matmul_cpu_f64_identity();
    test_matmul_cpu_threaded_vs_reference();
    test_matmul_cpu_f16_bf16();
    test_matmul_cpu_f16_bf16_large();
    test_matmul_cpu_packed();
    test_matmul_cpu_packed_large();
    test_matmul_cpu_batched();
    test_matmul_cpu_int_small();
    test_matmul_cpu_int_wrap();
    test_matmul_cpu_int_large();
    test_matmul_cpu_int_batched();
    test_matmul_cpu_int4x2();
    test_matmul_argument_validation();
#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        test_matmul_gpu_f32();
        test_matmul_gpu_f16_bf16();
        test_matmul_gpu_packed();
        test_matmul_gpu_batched();
        test_matmul_gpu_int();
        test_matmul_gpu_int4x2();
        test_matmul_device_mismatch();
    } else {
        std::cout << "GPU not available — skipping GPU matmul tests\n";
    }
#endif

    if (failures == 0) {
        std::cout << "All matmul tests passed\n";
    } else {
        std::cerr << failures << " matmul test(s) failed\n";
    }
    return failures == 0 ? 0 : 1;
}
