/**
 * @file complex_ops_tests.cpp
 * @author GreenBean
 * @brief Linear algebra and activation ops tests - framework-free assertion suite.
 * Covers bias_add, relu and softmax_ce across the full dtype family on CPU (SIMD tiers, FP32-compute pathway,
 * integer wrap semantics, int4x2 per-nibble), threaded paths, argument validation,
 * and the GPU kernels when a backend and device exist.
 * @version 0.1
 * @date 2026-10-18
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

// --- Packed/half-dtype helpers (host-side, via the unaccel conversion pathway) ---

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

// --- int4x2 helpers ---

static int32_t int4_lo(const int4x2_t v) {
    const auto n = static_cast<int32_t>(idet::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
    return n >= 8 ? n - 16 : n;
}

static int32_t int4_hi(const int4x2_t v) {
    const int32_t n = idet::bit_cast_memcpy<uint8_t>(v) >> 4;
    return n >= 8 ? n - 16 : n;
}

static int4x2_t pack_int4(const int32_t lo, const int32_t hi) {
    return idet::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
}

// --- fp64 references ---

static float64_t ref_row_loss(const std::vector<float64_t>& row, const size_t target) {
    auto m = std::numeric_limits<float64_t>::lowest();
    for (const auto x : row) {
        if (x > m) {
            m = x;
        }
    }
    auto s = 0.0;
    for (const auto x : row) {
        s += std::exp(x - m);
    }
    return m + std::log(s) - row[target];
}

// --------------------------- bias_add CPU ---------------------------

static void test_bias_add_cpu_f32() {
    Tensor t({2, 3}, DType::Float32);
    Tensor bias({3}, DType::Float32);
    for (auto c = size_t{0}; c < 3; ++c) {
        bias.set<float32_t>({c}, static_cast<float32_t>(c) + 0.5f);
    }
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            t.set<float32_t>({r, c}, static_cast<float32_t>(r * 3 + c));
        }
    }
    auto out = t.bias_add(bias);
    out.sync();
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 3; ++c) {
            const auto want = static_cast<float32_t>(r * 3 + c) + (static_cast<float32_t>(c) + 0.5f);
            CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({r, c})) == idet::bit_cast_memcpy<uint32_t>(want));
        }
    }
}

static void test_bias_add_cpu_f64_rank3() {
    Tensor t({2, 3, 4}, DType::Float64);
    Tensor bias({4}, DType::Float64);
    for (auto c = size_t{0}; c < 4; ++c) {
        bias.set<float64_t>({c}, -static_cast<float64_t>(c));
    }
    for (auto i = size_t{0}; i < 24; ++i) {
        t.set<float64_t>({i / 12, i / 4 % 3, i % 4}, static_cast<float64_t>(i) * 0.25);
    }
    auto out = t.bias_add(bias);
    out.sync();
    for (auto i = size_t{0}; i < 24; ++i) {
        const auto want = static_cast<float64_t>(i) * 0.25 - static_cast<float64_t>(i % 4);
        CHECK(idet::bit_cast_memcpy<uint64_t>(out.at<float64_t>({i / 12, i / 4 % 3, i % 4})) == idet::bit_cast_memcpy<uint64_t>(want));
    }
}

static void test_bias_add_cpu_threaded() {
    // 96 x 4103 = 393888 elements > threads*1024: exercises the threaded path
    // with chunk boundaries that land mid-row for the AVX kernels.
    constexpr size_t rows = 96;
    constexpr size_t cols = 4103;
    Tensor t({rows, cols}, DType::Float32);
    Tensor bias({cols}, DType::Float32);
    auto state = uint32_t{7};
    auto lcg = [&] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float32_t>(static_cast<int32_t>(state >> 8) % 2000 - 1000) * 0.5f;
    };
    std::vector<float32_t> bv(cols);
    for (auto c = size_t{0}; c < cols; ++c) {
        bv[c] = lcg();
        bias.set<float32_t>({c}, bv[c]);
    }
    std::vector<float32_t> tv(rows * cols);
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        tv[i] = lcg();
        t.set<float32_t>({i / cols, i % cols}, tv[i]);
    }
    auto out = t.bias_add(bias);
    out.sync();
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        const auto want = tv[i] + bv[i % cols];
        CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({i / cols, i % cols})) == idet::bit_cast_memcpy<uint32_t>(want));
    }
}

static void test_bias_add_cpu_halfs() {
    constexpr size_t cols = 7;
    Tensor t16({3, cols}, DType::Float16);
    Tensor b16({cols}, DType::Float16);
    Tensor tbf({3, cols}, DType::BFloat16);
    Tensor bbf({cols}, DType::BFloat16);
    for (auto i = size_t{0}; i < 3 * cols; ++i) {
        const auto v = static_cast<float32_t>(static_cast<int32_t>(i) - 10) * 0.75f;
        t16.set<float16_t>({i / cols, i % cols}, narrow_lane<float16_t>(v));
        tbf.set<bfloat16_t>({i / cols, i % cols}, narrow_lane<bfloat16_t>(v));
    }
    for (auto c = size_t{0}; c < cols; ++c) {
        const auto v = static_cast<float32_t>(c) * 1.25f - 3.0f;
        b16.set<float16_t>({c}, narrow_lane<float16_t>(v));
        bbf.set<bfloat16_t>({c}, narrow_lane<bfloat16_t>(v));
    }
    auto o16 = t16.bias_add(b16);
    auto obf = tbf.bias_add(bbf);
    o16.sync();
    obf.sync();
    for (auto i = size_t{0}; i < 3 * cols; ++i) {
        const auto iv = static_cast<float32_t>(static_cast<int32_t>(i) - 10) * 0.75f;
        const auto bv = static_cast<float32_t>(i % cols) * 1.25f - 3.0f;
        // Reference: widen the stored halves, add in fp32, narrow RNE.
        const auto want16 = narrow_lane<float16_t>(widen_lane(narrow_lane<float16_t>(iv)) + widen_lane(narrow_lane<float16_t>(bv)));
        const auto wantbf = narrow_lane<bfloat16_t>(widen_lane(narrow_lane<bfloat16_t>(iv)) + widen_lane(narrow_lane<bfloat16_t>(bv)));
        CHECK(idet::bit_cast_memcpy<uint16_t>(o16.at<float16_t>({i / cols, i % cols})) == idet::bit_cast_memcpy<uint16_t>(want16));
        CHECK(idet::bit_cast_memcpy<uint16_t>(obf.at<bfloat16_t>({i / cols, i % cols})) == idet::bit_cast_memcpy<uint16_t>(wantbf));
    }
}

static void test_bias_add_cpu_packed() {
    constexpr size_t cols = 5;
    Tensor t({2, cols}, DType::Float16x2);
    Tensor bias({cols}, DType::Float16x2);
    for (auto i = size_t{0}; i < 2 * cols; ++i) {
        const auto lo = static_cast<float32_t>(static_cast<int32_t>(i) - 4);
        const auto hi = -lo * 0.5f;
        t.set<float16x2_t>({i / cols, i % cols}, pack_lanes<float16x2_t, float16_t>(lo, hi));
    }
    for (auto c = size_t{0}; c < cols; ++c) {
        bias.set<float16x2_t>({c}, pack_lanes<float16x2_t, float16_t>(static_cast<float32_t>(c) * 0.5f, -static_cast<float32_t>(c)));
    }
    auto out = t.bias_add(bias);
    out.sync();
    for (auto i = size_t{0}; i < 2 * cols; ++i) {
        const auto lo = static_cast<float32_t>(static_cast<int32_t>(i) - 4);
        const auto hi = -lo * 0.5f;
        const auto c = i % cols;
        const auto want = pack_lanes<float16x2_t, float16_t>(
            widen_lane(narrow_lane<float16_t>(lo)) + widen_lane(narrow_lane<float16_t>(static_cast<float32_t>(c) * 0.5f)),
            widen_lane(narrow_lane<float16_t>(hi)) + widen_lane(narrow_lane<float16_t>(-static_cast<float32_t>(c))));
        CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float16x2_t>({i / cols, c})) == idet::bit_cast_memcpy<uint32_t>(want));
    }
}

static void test_bias_add_cpu_ints() {
    // int8 wrap: 100 + 100 wraps to -56 (modular uint8 arithmetic).
    Tensor t({2, 2}, DType::Int8);
    Tensor bias({2}, DType::Int8);
    t.set<int8_t>({0, 0}, int8_t{100});
    t.set<int8_t>({0, 1}, int8_t{-100});
    t.set<int8_t>({1, 0}, int8_t{127});
    t.set<int8_t>({1, 1}, int8_t{-128});
    bias.set<int8_t>({0}, int8_t{100});
    bias.set<int8_t>({1}, int8_t{-100});
    auto out = t.bias_add(bias);
    out.sync();
    CHECK(out.at<int8_t>({0, 0}) == int8_t{-56});
    CHECK(out.at<int8_t>({0, 1}) == int8_t{56});
    CHECK(out.at<int8_t>({1, 0}) == int8_t{-29});
    CHECK(out.at<int8_t>({1, 1}) == int8_t{28});

    Tensor tu({1, 3}, DType::UInt64);
    Tensor bu({3}, DType::UInt64);
    tu.set<uint64_t>({0, 0}, uint64_t{0});
    tu.set<uint64_t>({0, 1}, ~uint64_t{0});
    tu.set<uint64_t>({0, 2}, uint64_t{42});
    bu.set<uint64_t>({0}, uint64_t{5});
    bu.set<uint64_t>({1}, uint64_t{5});
    bu.set<uint64_t>({2}, uint64_t{0});
    auto outu = tu.bias_add(bu);
    outu.sync();
    CHECK(outu.at<uint64_t>({0, 0}) == uint64_t{5});
    CHECK(outu.at<uint64_t>({0, 1}) == uint64_t{4}); // wraps
    CHECK(outu.at<uint64_t>({0, 2}) == uint64_t{42});
}

static void test_bias_add_cpu_int4x2() {
    Tensor t({1, 3}, DType::Int4x2);
    Tensor bias({3}, DType::Int4x2);
    t.set<int4x2_t>({0, 0}, pack_int4(7, -8));
    t.set<int4x2_t>({0, 1}, pack_int4(0, 1));
    t.set<int4x2_t>({0, 2}, pack_int4(-3, 5));
    bias.set<int4x2_t>({0}, pack_int4(1, -1));
    bias.set<int4x2_t>({1}, pack_int4(7, 7));
    bias.set<int4x2_t>({2}, pack_int4(-8, -8));
    auto out = t.bias_add(bias);
    out.sync();
    // per-nibble modular wrap: 7+1=-8, -8-1=7, 0+7=7, 1+7=-8, -3-8=5, 5-8=-3
    CHECK(idet::bit_cast_memcpy<uint8_t>(out.at<int4x2_t>({0, 0})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(-8, 7)));
    CHECK(idet::bit_cast_memcpy<uint8_t>(out.at<int4x2_t>({0, 1})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(7, -8)));
    CHECK(idet::bit_cast_memcpy<uint8_t>(out.at<int4x2_t>({0, 2})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(5, -3)));
}

// --------------------------- relu CPU ---------------------------

static void test_relu_cpu_f32() {
    Tensor t({1, 5}, DType::Float32);
    t.set<float32_t>({0, 0}, -2.5f);
    t.set<float32_t>({0, 1}, 3.5f);
    t.set<float32_t>({0, 2}, -0.0f);
    t.set<float32_t>({0, 3}, std::numeric_limits<float32_t>::quiet_NaN());
    t.set<float32_t>({0, 4}, 0.0f);
    auto out = t.relu();
    out.sync();
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 0})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 1})) == idet::bit_cast_memcpy<uint32_t>(3.5f));
    // -0.0 survives (sign bit preserved)
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 2})) == 0x80000000u);
    // NaN propagates
    CHECK(std::isnan(out.at<float32_t>({0, 3})));
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 4})) == 0u);
}

static void test_relu_cpu_threaded() {
    // 70000 elements > threads*1024: exercises the threaded + AVX path.
    constexpr size_t n = 70000;
    Tensor t({n}, DType::Float32);
    for (auto i = size_t{0}; i < n; ++i) {
        t.set<float32_t>({i}, static_cast<float32_t>(static_cast<int32_t>(i % 2001) - 1000) * 0.5f);
    }
    auto out = t.relu();
    out.sync();
    for (auto i = size_t{0}; i < n; ++i) {
        const auto v = static_cast<float32_t>(static_cast<int32_t>(i % 2001) - 1000) * 0.5f;
        const auto want = v < 0.0f ? 0.0f : v;
        CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({i})) == idet::bit_cast_memcpy<uint32_t>(want));
    }
}

static void test_relu_cpu_f64() {
    Tensor t({1, 4}, DType::Float64);
    t.set<float64_t>({0, 0}, -1.0e300);
    t.set<float64_t>({0, 1}, 1.0e-300);
    t.set<float64_t>({0, 2}, std::numeric_limits<float64_t>::quiet_NaN());
    t.set<float64_t>({0, 3}, -0.0);
    auto out = t.relu();
    out.sync();
    CHECK(idet::bit_cast_memcpy<uint64_t>(out.at<float64_t>({0, 0})) == 0ull);
    CHECK(idet::bit_cast_memcpy<uint64_t>(out.at<float64_t>({0, 1})) == idet::bit_cast_memcpy<uint64_t>(1.0e-300));
    CHECK(std::isnan(out.at<float64_t>({0, 2})));
    CHECK(idet::bit_cast_memcpy<uint64_t>(out.at<float64_t>({0, 3})) == uint64_t{1} << 63);
}

static void test_relu_cpu_halfs_packed() {
    Tensor t({1, 3}, DType::Float16);
    t.set<float16_t>({0, 0}, narrow_lane<float16_t>(-1.5f));
    t.set<float16_t>({0, 1}, narrow_lane<float16_t>(2.5f));
    t.set<float16_t>({0, 2}, narrow_lane<float16_t>(std::numeric_limits<float32_t>::quiet_NaN()));
    auto out = t.relu();
    out.sync();
    CHECK(idet::bit_cast_memcpy<uint32_t>(widen_lane(out.at<float16_t>({0, 0}))) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(widen_lane(out.at<float16_t>({0, 1}))) == idet::bit_cast_memcpy<uint32_t>(2.5f));
    CHECK(std::isnan(widen_lane(out.at<float16_t>({0, 2}))));

    Tensor tb({1, 2}, DType::BFloat16);
    tb.set<bfloat16_t>({0, 0}, narrow_lane<bfloat16_t>(-4.0f));
    tb.set<bfloat16_t>({0, 1}, narrow_lane<bfloat16_t>(8.0f));
    auto outb = tb.relu();
    outb.sync();
    CHECK(idet::bit_cast_memcpy<uint32_t>(widen_lane(outb.at<bfloat16_t>({0, 0}))) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(widen_lane(outb.at<bfloat16_t>({0, 1}))) == idet::bit_cast_memcpy<uint32_t>(8.0f));

    Tensor tp({1, 2}, DType::BFloat16x2);
    tp.set<bfloat16x2_t>({0, 0}, pack_lanes<bfloat16x2_t, bfloat16_t>(-1.0f, 2.0f));
    tp.set<bfloat16x2_t>({0, 1}, pack_lanes<bfloat16x2_t, bfloat16_t>(3.0f, -4.0f));
    auto outp = tp.relu();
    outp.sync();
    CHECK(idet::bit_cast_memcpy<uint32_t>(lane_lo_f32<bfloat16_t>(outp.at<bfloat16x2_t>({0, 0}))) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(lane_hi_f32<bfloat16_t>(outp.at<bfloat16x2_t>({0, 0}))) == idet::bit_cast_memcpy<uint32_t>(2.0f));
    CHECK(idet::bit_cast_memcpy<uint32_t>(lane_lo_f32<bfloat16_t>(outp.at<bfloat16x2_t>({0, 1}))) == idet::bit_cast_memcpy<uint32_t>(3.0f));
    CHECK(idet::bit_cast_memcpy<uint32_t>(lane_hi_f32<bfloat16_t>(outp.at<bfloat16x2_t>({0, 1}))) == 0u);
}

static void test_relu_cpu_ints() {
    Tensor t({1, 4}, DType::Int16);
    t.set<int16_t>({0, 0}, int16_t{-7});
    t.set<int16_t>({0, 1}, int16_t{9});
    t.set<int16_t>({0, 2}, int16_t{-32768});
    t.set<int16_t>({0, 3}, int16_t{0});
    auto out = t.relu();
    out.sync();
    CHECK(out.at<int16_t>({0, 0}) == int16_t{0});
    CHECK(out.at<int16_t>({0, 1}) == int16_t{9});
    CHECK(out.at<int16_t>({0, 2}) == int16_t{0});
    CHECK(out.at<int16_t>({0, 3}) == int16_t{0});

    // Unsigned dtypes copy through unchanged.
    Tensor tu({1, 2}, DType::UInt8);
    tu.set<uint8_t>({0, 0}, uint8_t{0});
    tu.set<uint8_t>({0, 1}, uint8_t{250});
    auto outu = tu.relu();
    outu.sync();
    CHECK(outu.at<uint8_t>({0, 0}) == uint8_t{0});
    CHECK(outu.at<uint8_t>({0, 1}) == uint8_t{250});

    Tensor t4({1, 2}, DType::Int4x2);
    t4.set<int4x2_t>({0, 0}, pack_int4(-8, 7));
    t4.set<int4x2_t>({0, 1}, pack_int4(3, -1));
    auto out4 = t4.relu();
    out4.sync();
    CHECK(idet::bit_cast_memcpy<uint8_t>(out4.at<int4x2_t>({0, 0})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(0, 7)));
    CHECK(idet::bit_cast_memcpy<uint8_t>(out4.at<int4x2_t>({0, 1})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(3, 0)));
}

// --------------------------- softmax_ce CPU ---------------------------

static void test_softmax_ce_cpu_f32() {
    // rows: [1,2,3] target 2 -> ln(e^-2 + e^-1 + 1); [1,1,1] target 0 -> ln(3).
    Tensor t({2, 3}, DType::Float32);
    Tensor targets({2}, DType::Int32);
    t.set<float32_t>({0, 0}, 1.0f);
    t.set<float32_t>({0, 1}, 2.0f);
    t.set<float32_t>({0, 2}, 3.0f);
    t.set<float32_t>({1, 0}, 1.0f);
    t.set<float32_t>({1, 1}, 1.0f);
    t.set<float32_t>({1, 2}, 1.0f);
    targets.set<int32_t>({0}, 2);
    targets.set<int32_t>({1}, 0);
    auto out = t.softmax_ce(targets);
    out.sync();
    const auto ref = (ref_row_loss({1.0, 2.0, 3.0}, 2) + ref_row_loss({1.0, 1.0, 1.0}, 0)) / 2.0;
    CHECK(near(out.at<float32_t>({0}), static_cast<float32_t>(ref), 1e-5));
}

static void test_softmax_ce_cpu_f64() {
    Tensor t({2, 3}, DType::Float64);
    Tensor targets({2}, DType::UInt64);
    t.set<float64_t>({0, 0}, -1.5);
    t.set<float64_t>({0, 1}, 0.25);
    t.set<float64_t>({0, 2}, 4.75);
    t.set<float64_t>({1, 0}, 100.0);
    t.set<float64_t>({1, 1}, 101.0);
    t.set<float64_t>({1, 2}, 99.0);
    targets.set<uint64_t>({0}, 2);
    targets.set<uint64_t>({1}, 1);
    auto out = t.softmax_ce(targets);
    out.sync();
    const auto ref = (ref_row_loss({-1.5, 0.25, 4.75}, 2) + ref_row_loss({100.0, 101.0, 99.0}, 1)) / 2.0;
    CHECK(near64(out.at<float64_t>({0}), ref, 1e-12));
}

static void test_softmax_ce_cpu_threaded() {
    // 300 rows > threads*8: exercises the threaded row launcher.
    constexpr size_t rows = 300;
    constexpr size_t cols = 17;
    Tensor t({rows, cols}, DType::Float32);
    Tensor targets({rows}, DType::Int32);
    auto ref_total = 0.0;
    for (auto r = size_t{0}; r < rows; ++r) {
        std::vector<float64_t> row(cols);
        for (auto c = size_t{0}; c < cols; ++c) {
            const auto v = static_cast<float32_t>(static_cast<int32_t>((r * cols + c) % 31) - 15) * 0.5f;
            row[c] = static_cast<float64_t>(v);
            t.set<float32_t>({r, c}, v);
        }
        const auto target = r % cols;
        targets.set<int32_t>({r}, static_cast<int32_t>(target));
        ref_total += ref_row_loss(row, target);
    }
    auto out = t.softmax_ce(targets);
    out.sync();
    CHECK(near(out.at<float32_t>({0}), static_cast<float32_t>(ref_total / static_cast<float64_t>(rows)), 1e-3));
}

static void test_softmax_ce_cpu_halfs() {
    Tensor t({2, 4}, DType::Float16);
    Tensor targets({2}, DType::Int16);
    constexpr float32_t vals[2][4] = {{0.5f, -1.0f, 2.0f, 0.25f}, {3.0f, 1.0f, -2.0f, 0.0f}};
    for (auto r = size_t{0}; r < 2; ++r) {
        for (auto c = size_t{0}; c < 4; ++c) {
            t.set<float16_t>({r, c}, narrow_lane<float16_t>(vals[r][c]));
        }
    }
    targets.set<int16_t>({0}, 2);
    targets.set<int16_t>({1}, 0);
    auto out = t.softmax_ce(targets);
    out.sync();
    // fp32-compute reference over the widened stored halves.
    auto ref = 0.0;
    for (auto r = size_t{0}; r < 2; ++r) {
        std::vector<float64_t> row(4);
        for (auto c = size_t{0}; c < 4; ++c) {
            row[c] = static_cast<float64_t>(widen_lane(narrow_lane<float16_t>(vals[r][c])));
        }
        ref += ref_row_loss(row, r == 0 ? 2 : 0);
    }
    ref /= 2.0;
    CHECK(near(widen_lane(out.at<float16_t>({0})), static_cast<float32_t>(ref), 1e-2));
}

static void test_softmax_ce_cpu_packed() {
    // Packed logits: each element's (lo, hi) lanes are consecutive classes, so
    // a {2, 2} Float16x2 tensor has 4 classes per row.
    Tensor t({2, 2}, DType::Float16x2);
    Tensor targets({2}, DType::Int32);
    t.set<float16x2_t>({0, 0}, pack_lanes<float16x2_t, float16_t>(1.0f, 3.0f));
    t.set<float16x2_t>({0, 1}, pack_lanes<float16x2_t, float16_t>(-2.0f, 0.0f));
    t.set<float16x2_t>({1, 0}, pack_lanes<float16x2_t, float16_t>(0.0f, 0.0f));
    t.set<float16x2_t>({1, 1}, pack_lanes<float16x2_t, float16_t>(0.0f, 0.0f));
    targets.set<int32_t>({0}, 1); // class 1 = hi lane of element 0
    targets.set<int32_t>({1}, 3);
    auto out = t.softmax_ce(targets);
    out.sync();
    const auto ref = (ref_row_loss({1.0, 3.0, -2.0, 0.0}, 1) + ref_row_loss({0.0, 0.0, 0.0, 0.0}, 3)) / 2.0;
    const auto got_lo = lane_lo_f32<float16_t>(out.at<float16x2_t>({0}));
    const auto got_hi = lane_hi_f32<float16_t>(out.at<float16x2_t>({0}));
    CHECK(near(got_lo, static_cast<float32_t>(ref), 1e-2));
    CHECK(idet::bit_cast_memcpy<uint32_t>(got_lo) == idet::bit_cast_memcpy<uint32_t>(got_hi)); // the mean is duplicated into both lanes
}

static void test_softmax_ce_cpu_ints() {
    // Integer logits compute in fp64; the mean narrows clamp+truncate.
    Tensor t({2, 2}, DType::Int32);
    Tensor targets({2}, DType::Int8);
    t.set<int32_t>({0, 0}, 0);
    t.set<int32_t>({0, 1}, 10);
    t.set<int32_t>({1, 0}, 0);
    t.set<int32_t>({1, 1}, 10);
    targets.set<int8_t>({0}, 0);
    targets.set<int8_t>({1}, 0);
    auto out = t.softmax_ce(targets);
    out.sync();
    // loss per row = 10 + ln(1 + e^-10) - 0 ~= 10.0000454; mean truncates to 10.
    CHECK(out.at<int32_t>({0}) == 10);

    // int4x2 logits: nibbles unpack as consecutive classes.
    Tensor t4({1, 2}, DType::Int4x2);
    Tensor targets4({1}, DType::Int4x2);
    t4.set<int4x2_t>({0, 0}, pack_int4(1, 3));
    t4.set<int4x2_t>({0, 1}, pack_int4(2, 0));
    targets4.set<int4x2_t>({0}, pack_int4(1, 0)); // low nibble = target
    auto out4 = t4.softmax_ce(targets4);
    out4.sync();
    // loss = ln(e^-2 + 1 + e^-1 + e^-3) ~= 0.4401 -> truncates to 0 in [-8, 7].
    CHECK(int4_lo(out4.at<int4x2_t>({0})) == 0);
    CHECK(int4_hi(out4.at<int4x2_t>({0})) == 0);
}

static void test_softmax_ce_nan() {
    Tensor t({2, 3}, DType::Float32);
    Tensor targets({2}, DType::Int32);
    t.set<float32_t>({0, 0}, std::numeric_limits<float32_t>::quiet_NaN());
    t.set<float32_t>({0, 1}, 1.0f);
    t.set<float32_t>({0, 2}, 2.0f);
    t.set<float32_t>({1, 0}, 1.0f);
    t.set<float32_t>({1, 1}, 1.0f);
    t.set<float32_t>({1, 2}, 1.0f);
    targets.set<int32_t>({0}, 0);
    targets.set<int32_t>({1}, 0);
    auto out = t.softmax_ce(targets);
    out.sync();
    CHECK(std::isnan(out.at<float32_t>({0})));
}

static void test_softmax_ce_target_oob() {
    Tensor t({1, 3}, DType::Float32);
    Tensor targets({1}, DType::Int32);
    t.set<float32_t>({0, 0}, 1.0f);
    t.set<float32_t>({0, 1}, 2.0f);
    t.set<float32_t>({0, 2}, 3.0f);
    targets.set<int32_t>({0}, 3); // out of range: classes are [0, 3)
    bool threw = false;
    try {
        auto out = t.softmax_ce(targets);
        out.sync();
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_ops_argument_validation() {
    {
        // bias length must match the last dimension.
        Tensor t({2, 3}, DType::Float32);
        Tensor bias({4}, DType::Float32);
        bool threw = false;
        try {
            auto out = t.bias_add(bias);
            out.sync();
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // bias must be rank-1.
        Tensor t({2, 3}, DType::Float32);
        Tensor bias({1, 3}, DType::Float32);
        bool threw = false;
        try {
            auto out = t.bias_add(bias);
            out.sync();
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // dtype mismatch between tensor and bias.
        Tensor t({2, 3}, DType::Float32);
        Tensor bias({3}, DType::Float64);
        bool threw = false;
        try {
            auto out = t.bias_add(bias);
            out.sync();
        } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // softmax_ce requires rank-2 logits.
        Tensor t({6}, DType::Float32);
        Tensor targets({6}, DType::Int32);
        bool threw = false;
        try {
            auto out = t.softmax_ce(targets);
            out.sync();
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // softmax_ce targets must be an integer dtype.
        Tensor t({2, 3}, DType::Float32);
        Tensor targets({2}, DType::Float32);
        bool threw = false;
        try {
            auto out = t.softmax_ce(targets);
            out.sync();
        } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // softmax_ce targets length must match the row count.
        Tensor t({2, 3}, DType::Float32);
        Tensor targets({3}, DType::Int32);
        bool threw = false;
        try {
            auto out = t.softmax_ce(targets);
            out.sync();
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// --------------------------- GPU ---------------------------

#if defined(USE_HIP) || defined(USE_CUDA)
static void test_bias_add_gpu_f32() {
    constexpr size_t rows = 33;
    constexpr size_t cols = 259;
    Tensor t_cpu({rows, cols}, DType::Float32);
    Tensor b_cpu({cols}, DType::Float32);
    auto state = uint32_t{19};
    auto lcg = [&] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float32_t>(static_cast<int32_t>(state >> 8) % 2000 - 1000) * 0.25f;
    };
    std::vector<float32_t> bv(cols);
    for (auto c = size_t{0}; c < cols; ++c) {
        bv[c] = lcg();
        b_cpu.set<float32_t>({c}, bv[c]);
    }
    std::vector<float32_t> tv(rows * cols);
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        tv[i] = lcg();
        t_cpu.set<float32_t>({i / cols, i % cols}, tv[i]);
    }
    auto t = t_cpu.clone(true);
    auto bias = b_cpu.clone(true);
    t.to(Device::GPU);
    bias.to(Device::GPU);
    t.sync();
    bias.sync();
    auto out = t.bias_add(bias);
    out.sync();
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        const auto want = tv[i] + bv[i % cols];
        CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({i / cols, i % cols})) == idet::bit_cast_memcpy<uint32_t>(want));
    }
}

static void test_bias_add_gpu_ints() {
    Tensor t_cpu({1, 4}, DType::Int8);
    Tensor b_cpu({4}, DType::Int8);
    t_cpu.set<int8_t>({0, 0}, int8_t{100});
    t_cpu.set<int8_t>({0, 1}, int8_t{-100});
    t_cpu.set<int8_t>({0, 2}, int8_t{127});
    t_cpu.set<int8_t>({0, 3}, int8_t{-128});
    b_cpu.set<int8_t>({0}, int8_t{100});
    b_cpu.set<int8_t>({1}, int8_t{-100});
    b_cpu.set<int8_t>({2}, int8_t{100});
    b_cpu.set<int8_t>({3}, int8_t{-100});
    auto t = t_cpu.clone(true);
    auto bias = b_cpu.clone(true);
    t.to(Device::GPU);
    bias.to(Device::GPU);
    t.sync();
    bias.sync();
    auto out = t.bias_add(bias);
    out.sync();
    CHECK(out.at<int8_t>({0, 0}) == int8_t{-56});
    CHECK(out.at<int8_t>({0, 1}) == int8_t{56});
    CHECK(out.at<int8_t>({0, 2}) == int8_t{-29});
    CHECK(out.at<int8_t>({0, 3}) == int8_t{28});
}

static void test_bias_add_gpu_halfs_packed() {
    Tensor t_cpu({2, 5}, DType::Float16);
    Tensor b_cpu({5}, DType::Float16);
    for (auto i = size_t{0}; i < 10; ++i) {
        t_cpu.set<float16_t>({i / 5, i % 5}, narrow_lane<float16_t>(static_cast<float32_t>(static_cast<int32_t>(i) - 5) * 0.5f));
    }
    for (auto c = size_t{0}; c < 5; ++c) {
        b_cpu.set<float16_t>({c}, narrow_lane<float16_t>(static_cast<float32_t>(c) - 2.0f));
    }
    auto t = t_cpu.clone(true);
    auto bias = b_cpu.clone(true);
    t.to(Device::GPU);
    bias.to(Device::GPU);
    t.sync();
    bias.sync();
    auto out = t.bias_add(bias);
    out.sync();
    for (auto i = size_t{0}; i < 10; ++i) {
        const auto iv = static_cast<float32_t>(static_cast<int32_t>(i) - 5) * 0.5f;
        const auto bv = static_cast<float32_t>(i % 5) - 2.0f;
        const auto want = narrow_lane<float16_t>(widen_lane(narrow_lane<float16_t>(iv)) + widen_lane(narrow_lane<float16_t>(bv)));
        CHECK(idet::bit_cast_memcpy<uint16_t>(out.at<float16_t>({i / 5, i % 5})) == idet::bit_cast_memcpy<uint16_t>(want));
    }

    Tensor tp_cpu({1, 3}, DType::BFloat16x2);
    Tensor bp_cpu({3}, DType::BFloat16x2);
    for (auto c = size_t{0}; c < 3; ++c) {
        tp_cpu.set<bfloat16x2_t>({0, c}, pack_lanes<bfloat16x2_t, bfloat16_t>(static_cast<float32_t>(c), -static_cast<float32_t>(c)));
        bp_cpu.set<bfloat16x2_t>({c}, pack_lanes<bfloat16x2_t, bfloat16_t>(1.0f, -1.0f));
    }
    auto tp = tp_cpu.clone(true);
    auto bp = bp_cpu.clone(true);
    tp.to(Device::GPU);
    bp.to(Device::GPU);
    tp.sync();
    bp.sync();
    auto outp = tp.bias_add(bp);
    outp.sync();
    for (auto c = size_t{0}; c < 3; ++c) {
        CHECK(near(lane_lo_f32<bfloat16_t>(outp.at<bfloat16x2_t>({0, c})), static_cast<float32_t>(c) + 1.0f, 1e-2));
        CHECK(near(lane_hi_f32<bfloat16_t>(outp.at<bfloat16x2_t>({0, c})), -static_cast<float32_t>(c) - 1.0f, 1e-2));
    }
}

static void test_bias_add_gpu_int4x2() {
    Tensor t_cpu({1, 3}, DType::Int4x2);
    Tensor b_cpu({3}, DType::Int4x2);
    t_cpu.set<int4x2_t>({0, 0}, pack_int4(7, -8));
    t_cpu.set<int4x2_t>({0, 1}, pack_int4(0, 1));
    t_cpu.set<int4x2_t>({0, 2}, pack_int4(-3, 5));
    b_cpu.set<int4x2_t>({0}, pack_int4(1, -1));
    b_cpu.set<int4x2_t>({1}, pack_int4(7, 7));
    b_cpu.set<int4x2_t>({2}, pack_int4(-8, -8));
    auto t = t_cpu.clone(true);
    auto bias = b_cpu.clone(true);
    t.to(Device::GPU);
    bias.to(Device::GPU);
    t.sync();
    bias.sync();
    auto out = t.bias_add(bias);
    out.sync();
    CHECK(idet::bit_cast_memcpy<uint8_t>(out.at<int4x2_t>({0, 0})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(-8, 7)));
    CHECK(idet::bit_cast_memcpy<uint8_t>(out.at<int4x2_t>({0, 1})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(7, -8)));
    CHECK(idet::bit_cast_memcpy<uint8_t>(out.at<int4x2_t>({0, 2})) == idet::bit_cast_memcpy<uint8_t>(pack_int4(5, -3)));
}

static void test_relu_gpu() {
    Tensor t_cpu({1, 5}, DType::Float32);
    t_cpu.set<float32_t>({0, 0}, -2.5f);
    t_cpu.set<float32_t>({0, 1}, 3.5f);
    t_cpu.set<float32_t>({0, 2}, -0.0f);
    t_cpu.set<float32_t>({0, 3}, std::numeric_limits<float32_t>::quiet_NaN());
    t_cpu.set<float32_t>({0, 4}, -1.0e30f);
    auto t = t_cpu.clone(true);
    t.to(Device::GPU);
    t.sync();
    auto out = t.relu();
    out.sync();
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 0})) == 0u);
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 1})) == idet::bit_cast_memcpy<uint32_t>(3.5f));
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 2})) == 0x80000000u);
    CHECK(std::isnan(out.at<float32_t>({0, 3})));
    CHECK(idet::bit_cast_memcpy<uint32_t>(out.at<float32_t>({0, 4})) == 0u);

    Tensor ti_cpu({1, 3}, DType::Int32);
    ti_cpu.set<int32_t>({0, 0}, -5);
    ti_cpu.set<int32_t>({0, 1}, 0);
    ti_cpu.set<int32_t>({0, 2}, 7);
    auto ti = ti_cpu.clone(true);
    ti.to(Device::GPU);
    ti.sync();
    auto outi = ti.relu();
    outi.sync();
    CHECK(outi.at<int32_t>({0, 0}) == 0);
    CHECK(outi.at<int32_t>({0, 1}) == 0);
    CHECK(outi.at<int32_t>({0, 2}) == 7);
}

static void test_softmax_ce_gpu() {
    constexpr size_t rows = 64;
    constexpr size_t cols = 33;
    Tensor t_cpu({rows, cols}, DType::Float32);
    Tensor targets_cpu({rows}, DType::Int32);
    auto ref_total = 0.0;
    for (auto r = size_t{0}; r < rows; ++r) {
        std::vector<float64_t> row(cols);
        for (auto c = size_t{0}; c < cols; ++c) {
            const auto v = static_cast<float32_t>(static_cast<int32_t>((r * cols + c) % 41) - 20) * 0.25f;
            row[c] = static_cast<float64_t>(v);
            t_cpu.set<float32_t>({r, c}, v);
        }
        const auto target = r * 7 % cols;
        targets_cpu.set<int32_t>({r}, static_cast<int32_t>(target));
        ref_total += ref_row_loss(row, target);
    }
    auto t = t_cpu.clone(true);
    auto targets = targets_cpu.clone(true);
    t.to(Device::GPU);
    targets.to(Device::GPU);
    t.sync();
    targets.sync();
    auto out = t.softmax_ce(targets);
    out.sync();
    CHECK(near(out.at<float32_t>({0}), static_cast<float32_t>(ref_total / static_cast<float64_t>(rows)), 1e-3));
}

static void test_softmax_ce_gpu_packed() {
    Tensor t_cpu({2, 2}, DType::Float16x2);
    Tensor targets_cpu({2}, DType::Int32);
    t_cpu.set<float16x2_t>({0, 0}, pack_lanes<float16x2_t, float16_t>(1.0f, 3.0f));
    t_cpu.set<float16x2_t>({0, 1}, pack_lanes<float16x2_t, float16_t>(-2.0f, 0.0f));
    t_cpu.set<float16x2_t>({1, 0}, pack_lanes<float16x2_t, float16_t>(0.0f, 0.0f));
    t_cpu.set<float16x2_t>({1, 1}, pack_lanes<float16x2_t, float16_t>(0.0f, 0.0f));
    targets_cpu.set<int32_t>({0}, 1);
    targets_cpu.set<int32_t>({1}, 3);
    auto t = t_cpu.clone(true);
    auto targets = targets_cpu.clone(true);
    t.to(Device::GPU);
    targets.to(Device::GPU);
    t.sync();
    targets.sync();
    auto out = t.softmax_ce(targets);
    out.sync();
    const auto ref = (ref_row_loss({1.0, 3.0, -2.0, 0.0}, 1) + ref_row_loss({0.0, 0.0, 0.0, 0.0}, 3)) / 2.0;
    const auto got = lane_lo_f32<float16_t>(out.at<float16x2_t>({0}));
    CHECK(near(got, static_cast<float32_t>(ref), 1e-2));
}

static void test_softmax_ce_gpu_target_oob() {
    Tensor t_cpu({1, 3}, DType::Float32);
    Tensor targets_cpu({1}, DType::Int32);
    t_cpu.set<float32_t>({0, 0}, 1.0f);
    t_cpu.set<float32_t>({0, 1}, 2.0f);
    t_cpu.set<float32_t>({0, 2}, 3.0f);
    targets_cpu.set<int32_t>({0}, 3);
    auto t = t_cpu.clone(true);
    auto targets = targets_cpu.clone(true);
    t.to(Device::GPU);
    targets.to(Device::GPU);
    t.sync();
    targets.sync();
    bool threw = false;
    try {
        auto out = t.softmax_ce(targets);
        out.sync();
    } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_ops_device_mismatch() {
    bool threw = false;
    try {
        Tensor t({2, 2}, DType::Float32, Device::CPU);
        Tensor bias({2}, DType::Float32, Device::GPU);
        auto out = t.bias_add(bias);
        out.sync();
    } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
        threw = true;
    }
    CHECK(threw);
}
#endif

int main() {
    test_bias_add_cpu_f32();
    test_bias_add_cpu_f64_rank3();
    test_bias_add_cpu_threaded();
    test_bias_add_cpu_halfs();
    test_bias_add_cpu_packed();
    test_bias_add_cpu_ints();
    test_bias_add_cpu_int4x2();
    test_relu_cpu_f32();
    test_relu_cpu_threaded();
    test_relu_cpu_f64();
    test_relu_cpu_halfs_packed();
    test_relu_cpu_ints();
    test_softmax_ce_cpu_f32();
    test_softmax_ce_cpu_f64();
    test_softmax_ce_cpu_threaded();
    test_softmax_ce_cpu_halfs();
    test_softmax_ce_cpu_packed();
    test_softmax_ce_cpu_ints();
    test_softmax_ce_nan();
    test_softmax_ce_target_oob();
    test_ops_argument_validation();
#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        test_bias_add_gpu_f32();
        test_bias_add_gpu_ints();
        test_bias_add_gpu_halfs_packed();
        test_bias_add_gpu_int4x2();
        test_relu_gpu();
        test_softmax_ce_gpu();
        test_softmax_ce_gpu_packed();
        test_softmax_ce_gpu_target_oob();
        test_ops_device_mismatch();
    } else {
        std::cout << "GPU not available — skipping GPU linear-algebra/activation tests\n";
    }
#endif

    if (failures == 0) {
        std::cout << "All linear-algebra/activation tests passed\n";
    } else {
        std::cerr << failures << " linear-algebra/activation test(s) failed\n";
    }
    return failures == 0 ? 0 : 1;
}
