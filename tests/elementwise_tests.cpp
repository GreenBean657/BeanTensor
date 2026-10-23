/**
 * @file elementwise_tests.cpp
 * @author GreenBean
 * @brief Elementwise arithmetic/broadcasting tests - framework-free assertion suite.
 * Covers add/sub/mul/div/pow (tensor-tensor with NumPy-style broadcasting and
 * tensor-scalar), exp/log, across the full dtype family on CPU: arithmetic
 * correctness, every broadcasting case, scalar narrowing into non-float dtypes,
 * division-by-zero and pow/exp/log edge semantics, and shape/dtype errors.
 * Also covers the GPU (CUDA/HIP) paths for every op — arithmetic, broadcasting,
 * scalar narrowing, edge semantics — and asserts bit-exact CPU<->GPU parity
 * across all 15 dtypes for add/sub/mul/div (tensor-tensor, broadcast, and
 * tensor-scalar), which is the first exercise of the device-side parity claim.
 * Also covers the runtime-gated AVX2/AVX512 contiguous kernels: direct kernel
 * parity against the scalar applies (sentinel + random values, odd sizes,
 * bit-exact), end-to-end dispatch checks for the packed and unsigned paths,
 * and phase 2's additions: tensor-scalar splat kernels (plain and
 * scalar-on-the-left) exercised directly and end to end, row-decomposition
 * broadcast dispatch, and the per-nibble Int4x2 kernels.
 * @version 0.4
 * @date 2026-10-23
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Math/Elementwise/ElementwiseKernels.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::ElementwiseOp;
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

// Runtime GPU presence check, matching tests/basic_ops_tests.cpp: on GPU
// builds query the backend; on CPU-only builds there is never a device.
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

// --- Half/packed helpers (host-side, via the unaccel conversion pathway) ---

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

template<typename H>
static uint16_t lane_bits(const H& v) {
    return idet::bit_cast_memcpy<uint16_t>(v);
}

template<typename P, typename H>
static P pack_lanes(const float32_t lo, const float32_t hi) {
    const auto lo_bits = static_cast<uint32_t>(lane_bits(narrow_lane<H>(lo)));
    const auto hi_bits = static_cast<uint32_t>(lane_bits(narrow_lane<H>(hi)));
    return idet::bit_cast_memcpy<P>(lo_bits | (hi_bits << 16));
}

template<typename P>
static float32_t unpack_lo(const P& v) {
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    if constexpr (std::is_same_v<P, float16x2_t>) {
        return widen_lane(idet::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(bits & 0xFFFFu)));
    } else {
        return widen_lane(idet::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(bits & 0xFFFFu)));
    }
}

template<typename P>
static float32_t unpack_hi(const P& v) {
    const auto bits = idet::bit_cast_memcpy<uint32_t>(v);
    if constexpr (std::is_same_v<P, float16x2_t>) {
        return widen_lane(idet::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(bits >> 16)));
    } else {
        return widen_lane(idet::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(bits >> 16)));
    }
}

// --- int4x2 helpers (packed signed 4-bit lanes) ---

static int4x2_t i4x2(const int8_t lo, const int8_t hi) {
    const auto lo_bits = static_cast<uint8_t>(lo) & 0x0Fu;
    const auto hi_bits = static_cast<uint8_t>(hi) & 0x0Fu;
    return idet::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>(lo_bits | static_cast<uint8_t>(hi_bits << 4)));
}

static int8_t i4x2_lo(const int4x2_t v) {
    const auto n = static_cast<int8_t>(idet::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
    return n > 7 ? static_cast<int8_t>(n - 16) : n;
}

static int8_t i4x2_hi(const int4x2_t v) {
    const auto n = static_cast<int8_t>(idet::bit_cast_memcpy<uint8_t>(v) >> 4);
    return n > 7 ? static_cast<int8_t>(n - 16) : n;
}

// --- fp32/fp64 same-shape arithmetic ---

static void test_arith_f32_f64() {
    auto a = Tensor({4}, DType::Float32, Device::CPU);
    auto b = Tensor({4}, DType::Float32, Device::CPU);
    const auto av = std::vector{1.5f, -2.25f, 0.0f, 100.0f};
    const auto bv = std::vector{0.25f, 4.0f, -3.0f, 0.5f};
    for (size_t i = 0; i < 4; ++i) {
        a.set<float32_t>({i}, av[i]);
        b.set<float32_t>({i}, bv[i]);
    }
    auto add = a.add(b);
    auto sub = a.sub(b);
    auto mul = a.mul(b);
    auto div = a.div(b);
    auto powr = a.pow(b);
    for (size_t i = 0; i < 4; ++i) {
        CHECK(near(add.at<float32_t>({i}), av[i] + bv[i], 1e-6));
        CHECK(near(sub.at<float32_t>({i}), av[i] - bv[i], 1e-6));
        CHECK(near(mul.at<float32_t>({i}), av[i] * bv[i], 1e-4));
        CHECK(near(div.at<float32_t>({i}), av[i] / bv[i], 1e-4));
        if (i == 2) {
            // 0^-3 is +inf; fabs(inf - inf) is NaN, so check it explicitly.
            CHECK(std::isinf(static_cast<float64_t>(powr.at<float32_t>({i}))));
        } else {
            CHECK(near(powr.at<float32_t>({i}), std::pow(av[i], bv[i]), 1e-3));
        }
        CHECK(add.get_dtype() == DType::Float32);
    }
    // Scalar overloads agree with direct scalar math.
    for (size_t i = 0; i < 4; ++i) {
        CHECK(near(a.add(0.25).at<float32_t>({i}), av[i] + 0.25f, 1e-6));
        CHECK(near(a.sub(4.0).at<float32_t>({i}), av[i] - 4.0f, 1e-6));
        CHECK(near(a.mul(-3.0).at<float32_t>({i}), a.at<float32_t>({i}) * -3.0f, 1e-4));
        CHECK(near(a.div(0.5).at<float32_t>({i}), a.at<float32_t>({i}) / 0.5f, 1e-4));
        CHECK(near(a.pow(2.0).at<float32_t>({i}), a.at<float32_t>({i}) * a.at<float32_t>({i}), 1e-3));
    }
    // fp64 computes in the storage width.
    auto a64 = Tensor({2}, DType::Float64, Device::CPU);
    auto b64 = Tensor({2}, DType::Float64, Device::CPU);
    a64.set<float64_t>({0}, 1.0e150);
    a64.set<float64_t>({1}, -7.0);
    b64.set<float64_t>({0}, 1.0e150);
    b64.set<float64_t>({1}, 2.0);
    auto mul64 = a64.mul(b64);
    CHECK(near64(mul64.at<float64_t>({0}), 1.0e300, 1e285));
    CHECK(near64(mul64.at<float64_t>({1}), -14.0, 1e-12));
    CHECK(mul64.get_dtype() == DType::Float64);
    CHECK(near64(a64.add(0.5).at<float64_t>({1}), -6.5, 1e-12));
}

// --- integer arithmetic: wrap, truncating div, div-by-zero, min/-1 ---

static void test_arith_ints() {
    // int8 modular wrap on add/sub/mul.
    auto a = Tensor({3}, DType::Int8, Device::CPU);
    a.set<int8_t>({0}, int8_t{127});
    a.set<int8_t>({1}, int8_t{-128});
    a.set<int8_t>({2}, int8_t{100});
    auto add = a.add(int8_t{1});
    CHECK(add.at<int8_t>({0}) == int8_t{-128}); // 127 + 1 wraps
    CHECK(add.at<int8_t>({1}) == int8_t{-127});
    CHECK(add.at<int8_t>({2}) == int8_t{101});
    auto sub = a.sub(int8_t{1});
    CHECK(sub.at<int8_t>({1}) == int8_t{127}); // -128 - 1 wraps
    auto mul = a.mul(int8_t{2});
    CHECK(mul.at<int8_t>({2}) == int8_t{-56}); // 200 wraps through the unsigned twin
    // int32 div truncates toward zero; div-by-zero yields 0.
    auto d = Tensor({3}, DType::Int32, Device::CPU);
    auto e = Tensor({3}, DType::Int32, Device::CPU);
    d.set<int32_t>({0}, -7);
    d.set<int32_t>({1}, 7);
    d.set<int32_t>({2}, -7);
    e.set<int32_t>({0}, 2);
    e.set<int32_t>({1}, -2);
    e.set<int32_t>({2}, 0);
    auto div = d.div(e);
    CHECK(div.at<int32_t>({0}) == -3); // truncation toward zero, not floor
    CHECK(div.at<int32_t>({1}) == -3);
    CHECK(div.at<int32_t>({2}) == 0);  // division by zero yields 0
    // min / -1 yields min (wrap), never UB — int8 and int32.
    auto mn8 = Tensor({1}, DType::Int8, Device::CPU);
    auto neg8 = Tensor({1}, DType::Int8, Device::CPU);
    mn8.set<int8_t>({0}, int8_t{-128});
    neg8.set<int8_t>({0}, int8_t{-1});
    CHECK(mn8.div(neg8).at<int8_t>({0}) == int8_t{-128});
    auto mn32 = Tensor({1}, DType::Int32, Device::CPU);
    auto neg32 = Tensor({1}, DType::Int32, Device::CPU);
    mn32.set<int32_t>({0}, std::numeric_limits<int32_t>::min());
    neg32.set<int32_t>({0}, -1);
    CHECK(mn32.div(neg32).at<int32_t>({0}) == std::numeric_limits<int32_t>::min());
    // Unsigned kinds: plain division and div-by-zero -> 0.
    auto u = Tensor({2}, DType::UInt8, Device::CPU);
    auto v = Tensor({2}, DType::UInt8, Device::CPU);
    u.set<uint8_t>({0}, uint8_t{200});
    u.set<uint8_t>({1}, uint8_t{5});
    v.set<uint8_t>({0}, uint8_t{3});
    v.set<uint8_t>({1}, uint8_t{0});
    auto udiv = u.div(v);
    CHECK(udiv.at<uint8_t>({0}) == uint8_t{66});
    CHECK(udiv.at<uint8_t>({1}) == uint8_t{0});
    // uint16 add wraps; int64 mul.
    auto w = Tensor({1}, DType::UInt16, Device::CPU);
    w.set<uint16_t>({0}, uint16_t{65535});
    CHECK(w.add(uint16_t{1}).at<uint16_t>({0}) == uint16_t{0});
    auto big = Tensor({1}, DType::Int64, Device::CPU);
    big.set<int64_t>({0}, 1000000);
    CHECK(big.mul(int64_t{1000000}).at<int64_t>({0}) == int64_t{1000000000000});
    // Scalar overloads on integers use the same wrap contract.
    CHECK(mn8.add(1.0).at<int8_t>({0}) == int8_t{-127});
    CHECK(w.mul(2.0).at<uint16_t>({0}) == uint16_t{65534});
}

// --- int4x2 per-nibble arithmetic ---

static void test_arith_int4x2() {
    auto a = Tensor({2}, DType::Int4x2, Device::CPU);
    auto b = Tensor({2}, DType::Int4x2, Device::CPU);
    a.set<int4x2_t>({0}, i4x2(int8_t{7}, int8_t{-8}));
    a.set<int4x2_t>({1}, i4x2(int8_t{-3}, int8_t{5}));
    b.set<int4x2_t>({0}, i4x2(int8_t{1}, int8_t{-1}));
    b.set<int4x2_t>({1}, i4x2(int8_t{0}, int8_t{0}));
    // add wraps per nibble: 7+1 = 8 -> -8; -8 + -1 = -9 -> 7.
    auto add = a.add(b);
    CHECK(i4x2_lo(add.at<int4x2_t>({0})) == int8_t{-8});
    CHECK(i4x2_hi(add.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_lo(add.at<int4x2_t>({1})) == int8_t{-3});
    // sub: -3 - 0 = -3; 5 - 0 = 5.
    auto sub = a.sub(b);
    CHECK(i4x2_lo(sub.at<int4x2_t>({1})) == int8_t{-3});
    CHECK(i4x2_hi(sub.at<int4x2_t>({1})) == int8_t{5});
    // mul wraps per nibble: 7*1 = 7; -8*-1 = 8 -> -8.
    auto mul = a.mul(b);
    CHECK(i4x2_lo(mul.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_hi(mul.at<int4x2_t>({0})) == int8_t{-8});
    // div truncates per nibble: 7/1 = 7, and -8/-1 = 8 wraps back to -8.
    auto div = a.div(b);
    CHECK(i4x2_lo(div.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_hi(div.at<int4x2_t>({0})) == int8_t{-8});
    // div-by-zero nibble yields 0: (-3, 5) / (0, 0) -> (0, 0).
    CHECK(i4x2_lo(div.at<int4x2_t>({1})) == int8_t{0});
    CHECK(i4x2_hi(div.at<int4x2_t>({1})) == int8_t{0});
    // Scalar op: the scalar narrows into BOTH nibbles (2 -> (2, 2)).
    auto sc = a.add(2.0);
    CHECK(i4x2_lo(sc.at<int4x2_t>({1})) == int8_t{-1});
    CHECK(i4x2_hi(sc.at<int4x2_t>({1})) == int8_t{7});
}

// --- broadcasting ---

// {2,3} fp32 tensor: row 0 = 1,2,3; row 1 = 11,12,13.
static Tensor make_f32_bias() {
    auto t = Tensor({2, 3}, DType::Float32, Device::CPU);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            t.set<float32_t>({r, c}, static_cast<float32_t>(r * 10 + c + 1));
        }
    }
    return t;
}

static void test_broadcast() {
    // Same shape.
    auto a = make_f32_bias();
    auto b = make_f32_bias();
    auto same = a.add(b);
    CHECK(same.is_same_shape(Tensor({2, 3}, DType::Float32)));
    CHECK(near(same.at<float32_t>({1, 2}), 26.0f, 1e-6)); // 13 + 13
    // Per-row bias: [2,3] + [1,3].
    auto bias = Tensor({1, 3}, DType::Float32, Device::CPU);
    bias.set<float32_t>({0, 0}, 10.0f);
    bias.set<float32_t>({0, 1}, 20.0f);
    bias.set<float32_t>({0, 2}, 30.0f);
    auto rb = a.add(bias);
    CHECK(rb.is_same_shape(Tensor({2, 3}, DType::Float32)));
    CHECK(near(rb.at<float32_t>({0, 0}), 11.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({0, 2}), 33.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({1, 0}), 21.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({1, 2}), 43.0f, 1e-6));
    // Per-column: [2,3] + [2,1].
    auto col = Tensor({2, 1}, DType::Float32, Device::CPU);
    col.set<float32_t>({0, 0}, 100.0f);
    col.set<float32_t>({1, 0}, 200.0f);
    auto cb = a.add(col);
    CHECK(near(cb.at<float32_t>({0, 2}), 103.0f, 1e-6));
    CHECK(near(cb.at<float32_t>({1, 2}), 213.0f, 1e-6));
    // Right-aligned rank promotion: [2,3] + [3].
    auto r3 = Tensor({3}, DType::Float32, Device::CPU);
    r3.set<float32_t>({0}, 7.0f);
    r3.set<float32_t>({1}, 8.0f);
    r3.set<float32_t>({2}, 9.0f);
    auto pr = a.add(r3);
    CHECK(near(pr.at<float32_t>({0, 0}), 8.0f, 1e-6));
    CHECK(near(pr.at<float32_t>({1, 2}), 22.0f, 1e-6));
    // [1] against [2,3]: the single element lands everywhere.
    auto one = Tensor({1}, DType::Float32, Device::CPU);
    one.set<float32_t>({0}, 5.0f);
    auto o = a.add(one);
    CHECK(near(o.at<float32_t>({0, 0}), 6.0f, 1e-6));
    CHECK(near(o.at<float32_t>({1, 2}), 18.0f, 1e-6));
    // [1,1] against [2,3].
    auto sq = Tensor({1, 1}, DType::Float32, Device::CPU);
    sq.set<float32_t>({0, 0}, 5.0f);
    CHECK(near(a.add(sq).at<float32_t>({1, 1}), 17.0f, 1e-6));
    // Rank-3 with a rank-2 right operand: [2,2,3] + [2,3] -> [2,2,3].
    auto c3 = Tensor({2, 2, 3}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < 12; ++i) {
        c3.set<float32_t>({i / 6, i / 3 % 2, i % 3}, static_cast<float32_t>(i + 1));
    }
    auto base = Tensor({2, 3}, DType::Float32, Device::CPU);
    base.set<float32_t>({0, 0}, 1.0f);
    base.set<float32_t>({0, 1}, 2.0f);
    base.set<float32_t>({0, 2}, 3.0f);
    base.set<float32_t>({1, 0}, 4.0f);
    base.set<float32_t>({1, 1}, 5.0f);
    base.set<float32_t>({1, 2}, 6.0f);
    auto c3r = c3.add(base);
    CHECK(c3r.is_same_shape(Tensor({2, 2, 3}, DType::Float32)));
    CHECK(near(c3r.at<float32_t>({0, 0, 0}), 2.0f, 1e-6));   // 1 + base[0,0]=1
    CHECK(near(c3r.at<float32_t>({0, 1, 0}), 8.0f, 1e-6));   // 4 + base[1,0]=4
    CHECK(near(c3r.at<float32_t>({1, 1, 2}), 18.0f, 1e-6));  // 12 + base[1,2]=6
    // Broadcasting works for every op, both directions of materialization.
    auto inv = bias.sub(a); // [1,3] - [2,3] -> [2,3]
    CHECK(near(inv.at<float32_t>({0, 0}), 9.0f, 1e-6));
    CHECK(near(inv.at<float32_t>({1, 0}), -1.0f, 1e-6));
    auto mulb = a.mul(col);
    CHECK(near(mulb.at<float32_t>({1, 2}), 2600.0f, 1e-4));
    auto divb = col.div(a);
    CHECK(near(divb.at<float32_t>({1, 0}), 18.1818f, 1e-3)); // 200 / 11
}

// --- broadcasting errors ---

static void test_broadcast_errors() {
    auto a = make_f32_bias();
    // Non-broadcastable inner dim.
    auto bad_cols = Tensor({2, 4}, DType::Float32, Device::CPU);
    auto threw = false;
    try {
        auto r = a.add(bad_cols);
        static_cast<void>(r);
    } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);
    // Non-broadcastable rank-1 pair.
    threw = false;
    try {
        auto r = Tensor({2}, DType::Float32, Device::CPU).add(Tensor({3}, DType::Float32, Device::CPU));
        static_cast<void>(r);
    } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);
    // Rank-3 against an incompatible rank-2.
    threw = false;
    try {
        auto r = Tensor({2, 2, 3}, DType::Float32, Device::CPU).add(Tensor({3, 3}, DType::Float32, Device::CPU));
        static_cast<void>(r);
    } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);
    // Dtype mismatch: no implicit promotion.
    threw = false;
    try {
        auto r = a.add(Tensor({2, 3}, DType::Float64, Device::CPU));
        static_cast<void>(r);
    } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
        threw = true;
    }
    CHECK(threw);
    // Non-contiguous operand is rejected.
    auto base = make_f32_bias();
    auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
    auto view = base.slice(ranges); // inner box: non-contiguous view
    threw = false;
    try {
        auto r = view.add(Tensor({2, 2}, DType::Float32, Device::CPU));
        static_cast<void>(r);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

// --- scalar narrowing into non-float dtypes (mixed-dtype conversion rules) ---

static void test_scalar_narrowing() {
    // Float scalar truncates toward zero into integers.
    auto i = Tensor({2}, DType::Int32, Device::CPU);
    i.set<int32_t>({0}, 10);
    i.set<int32_t>({1}, -10);
    CHECK(i.add(2.9).at<int32_t>({0}) == 12);
    CHECK(i.add(-2.9).at<int32_t>({1}) == -12);
    // Out-of-range scalar clamps to the dtype range.
    auto big = Tensor({1}, DType::Int32, Device::CPU);
    big.set<int32_t>({0}, 0);
    CHECK(big.add(1e300).at<int32_t>({0}) == std::numeric_limits<int32_t>::max());
    auto small = Tensor({1}, DType::Int8, Device::CPU);
    small.set<int8_t>({0}, 0);
    CHECK(small.add(1000.0).at<int8_t>({0}) == int8_t{127});
    // NaN scalar narrows to 0 for integer kinds.
    auto n = Tensor({1}, DType::Int32, Device::CPU);
    n.set<int32_t>({0}, 5);
    CHECK(n.add(std::numeric_limits<float64_t>::quiet_NaN()).at<int32_t>({0}) == 5);
    // Unsigned kinds clamp negative scalars to 0.
    auto u = Tensor({1}, DType::UInt8, Device::CPU);
    u.set<uint8_t>({0}, uint8_t{3});
    CHECK(u.add(-5.0).at<uint8_t>({0}) == uint8_t{3});
    // int4x2 clamps the scalar to [-8, 7] per nibble, both nibbles.
    auto q = Tensor({1}, DType::Int4x2, Device::CPU);
    q.set<int4x2_t>({0}, i4x2(int8_t{0}, int8_t{0}));
    auto q7 = q.add(20.0);
    CHECK(i4x2_lo(q7.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_hi(q7.at<int4x2_t>({0})) == int8_t{7});
    // The scalar narrows before the op: -20 clamps to -8, so 0-(-8) = 8 wraps to -8.
    auto qm8 = q.sub(-20.0);
    CHECK(i4x2_lo(qm8.at<int4x2_t>({0})) == int8_t{-8});
    CHECK(i4x2_hi(qm8.at<int4x2_t>({0})) == int8_t{-8});
    // Packed dtypes duplicate the narrowed scalar into both lanes.
    auto p = Tensor({1}, DType::Float16x2, Device::CPU);
    p.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 2.0f));
    auto p2 = p.add(0.5);
    CHECK(near(unpack_lo(p2.at<float16x2_t>({0})), 1.5f, 1e-3));
    CHECK(near(unpack_hi(p2.at<float16x2_t>({0})), 2.5f, 1e-3));
}

// --- division/pow/exp/log edge semantics ---

static void test_edge_semantics() {
    // Float division by zero is IEEE.
    auto z = Tensor({3}, DType::Float32, Device::CPU);
    z.set<float32_t>({0}, 1.0f);
    z.set<float32_t>({1}, -1.0f);
    z.set<float32_t>({2}, 0.0f);
    auto zero = Tensor({3}, DType::Float32, Device::CPU);
    zero.set_zero();
    auto r = z.div(zero);
    CHECK(std::isinf(static_cast<float64_t>(r.at<float32_t>({0}))));
    CHECK(r.at<float32_t>({0}) > 0.0f);
    CHECK(std::isinf(static_cast<float64_t>(r.at<float32_t>({1}))));
    CHECK(r.at<float32_t>({1}) < 0.0f);
    CHECK(std::isnan(static_cast<float64_t>(r.at<float32_t>({2}))));
    // Scalar zero divisor, fp64.
    auto z64 = Tensor({1}, DType::Float64, Device::CPU);
    z64.set<float64_t>({0}, 1.0);
    CHECK(std::isinf(z64.div(0.0).at<float64_t>({0})));
    // pow edges on fp32.
    auto b = Tensor({4}, DType::Float32, Device::CPU);
    b.set<float32_t>({0}, 2.0f);
    b.set<float32_t>({1}, -2.0f);
    b.set<float32_t>({2}, 0.0f);
    b.set<float32_t>({3}, 4.0f);
    CHECK(near(b.pow(10.0).at<float32_t>({0}), 1024.0f, 1e-3));
    CHECK(near(b.pow(3.0).at<float32_t>({1}), -8.0f, 1e-6));
    CHECK(near(b.pow(0.0).at<float32_t>({2}), 1.0f, 1e-6)); // 0^0 = 1 (IEEE pow)
    CHECK(std::isnan(static_cast<float64_t>(b.pow(0.5).at<float32_t>({1})))); // negative base, fractional exp
    CHECK(near(b.pow(-1.0).at<float32_t>({3}), 0.25f, 1e-6));
    // Integer pow is rejected (tensor-tensor and scalar).
    auto ints = Tensor({2}, DType::Int32, Device::CPU);
    ints.set<int32_t>({0}, 2);
    ints.set<int32_t>({1}, 3);
    auto threw = false;
    try {
        auto r2 = ints.pow(ints);
        static_cast<void>(r2);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        auto r2 = ints.pow(2.0);
        static_cast<void>(r2);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
    // exp: exp(0) = 1, overflow -> +inf.
    auto e = Tensor({3}, DType::Float32, Device::CPU);
    e.set<float32_t>({0}, 0.0f);
    e.set<float32_t>({1}, 1.0f);
    e.set<float32_t>({2}, 200.0f);
    auto ex = e.exp();
    CHECK(near(ex.at<float32_t>({0}), 1.0f, 1e-6));
    CHECK(near(ex.at<float32_t>({1}), 2.7182817f, 1e-5));
    CHECK(std::isinf(static_cast<float64_t>(ex.at<float32_t>({2}))));
    // log: log(1) = 0, log(0) = -inf, log(negative) = NaN.
    auto l = Tensor({3}, DType::Float32, Device::CPU);
    l.set<float32_t>({0}, 1.0f);
    l.set<float32_t>({1}, 0.0f);
    l.set<float32_t>({2}, -4.0f);
    auto lg = l.log();
    CHECK(near(lg.at<float32_t>({0}), 0.0f, 1e-6));
    CHECK(std::isinf(static_cast<float64_t>(lg.at<float32_t>({1}))));
    CHECK(lg.at<float32_t>({1}) < 0.0f);
    CHECK(std::isnan(static_cast<float64_t>(lg.at<float32_t>({2}))));
    // Integer exp/log are rejected.
    threw = false;
    try {
        auto r2 = ints.exp();
        static_cast<void>(r2);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        auto r2 = ints.log();
        static_cast<void>(r2);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

// --- half and packed dtypes ---

static void test_half_and_packed() {
    // fp16: widen-compute-narrow in fp32 (scalars arrive as float64_t and
    // narrow into the tensor dtype through the same PERMISSIVE contract).
    auto h = Tensor({2}, DType::Float16, Device::CPU);
    h.set<float16_t>({0}, narrow_lane<float16_t>(1.5f));
    h.set<float16_t>({1}, narrow_lane<float16_t>(-2.25f));
    auto h2 = h.add(0.25);
    CHECK(near(widen_lane(h2.at<float16_t>({0})), 1.75f, 1e-3));
    CHECK(near(widen_lane(h2.at<float16_t>({1})), -2.0f, 1e-3));
    auto hm = h.mul(2.0);
    CHECK(near(widen_lane(hm.at<float16_t>({0})), 3.0f, 1e-3));
    // fp16 overflow clamps to +/-65504 (the unaccel pathway never yields Inf from finite input).
    auto big = Tensor({1}, DType::Float16, Device::CPU);
    big.set<float16_t>({0}, narrow_lane<float16_t>(300.0f));
    auto bm = big.mul(300.0);
    CHECK(near(widen_lane(bm.at<float16_t>({0})), 65504.0f, 0.0));
    // bf16 multiply with an fp32-subnormal result flushes to zero.
    auto s = Tensor({1}, DType::BFloat16, Device::CPU);
    s.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(1.0e-20f));
    auto sm = s.mul(1.0e-20);
    CHECK(near(widen_lane(sm.at<bfloat16_t>({0})), 0.0f, 0.0));
    // exp/log on fp16 go through fp32.
    auto he = Tensor({1}, DType::Float16, Device::CPU);
    he.set<float16_t>({0}, narrow_lane<float16_t>(0.0f));
    CHECK(near(widen_lane(he.exp().at<float16_t>({0})), 1.0f, 1e-3));
    auto hl = Tensor({1}, DType::Float16, Device::CPU);
    hl.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
    CHECK(near(widen_lane(hl.log().at<float16_t>({0})), 0.0f, 1e-3));
    // Packed x2: per-lane compute.
    auto pk = Tensor({2}, DType::Float16x2, Device::CPU);
    pk.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 2.0f));
    pk.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(-3.0f, 4.0f));
    auto pkb = Tensor({2}, DType::Float16x2, Device::CPU);
    pkb.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(0.5f, 0.25f));
    pkb.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(0.5f, 0.25f));
    auto pk2 = pk.add(pkb);
    CHECK(near(unpack_lo(pk2.at<float16x2_t>({0})), 1.5f, 1e-3));
    CHECK(near(unpack_hi(pk2.at<float16x2_t>({0})), 2.25f, 1e-3));
    CHECK(near(unpack_lo(pk2.at<float16x2_t>({1})), -2.5f, 1e-3));
    CHECK(near(unpack_hi(pk2.at<float16x2_t>({1})), 4.25f, 1e-3));
    auto pkm2 = Tensor({2}, DType::Float16x2, Device::CPU);
    pkm2.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(2.0f, -2.0f));
    pkm2.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(2.0f, -2.0f));
    auto pkm = pk.mul(pkm2);
    CHECK(near(unpack_lo(pkm.at<float16x2_t>({0})), 2.0f, 1e-3));
    CHECK(near(unpack_hi(pkm.at<float16x2_t>({0})), -4.0f, 1e-3));
    // bf16x2 packed lanes.
    auto pb = Tensor({1}, DType::BFloat16x2, Device::CPU);
    pb.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(8.0f, 1.0f));
    auto pbd = Tensor({1}, DType::BFloat16x2, Device::CPU);
    pbd.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(2.0f, 4.0f));
    auto pb2 = pb.div(pbd);
    CHECK(near(unpack_lo(pb2.at<bfloat16x2_t>({0})), 4.0f, 1e-2));
    CHECK(near(unpack_hi(pb2.at<bfloat16x2_t>({0})), 0.25f, 1e-2));
    // Packed unary: exp(0, 0) -> (1, 1); log(1, 1) -> (0, 0).
    auto pz = Tensor({1}, DType::BFloat16x2, Device::CPU);
    pz.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(0.0f, 0.0f));
    auto pze = pz.exp();
    CHECK(near(unpack_lo(pze.at<bfloat16x2_t>({0})), 1.0f, 1e-2));
    CHECK(near(unpack_hi(pze.at<bfloat16x2_t>({0})), 1.0f, 1e-2));
}

// --- large tensor through the thread pool ---

static void test_threaded_path() {
    // 100000 elements: well above the inline threshold, so the pool chunks it.
    auto a = Tensor({100000}, DType::Float32, Device::CPU);
    a.set_one();
    auto b = Tensor({100000}, DType::Float32, Device::CPU);
    b.set_one();
    auto c = a.add(b);
    auto d = a.mul(b);
    for (const auto i : {size_t{0}, size_t{49999}, size_t{99999}}) {
        CHECK(near(c.at<float32_t>({i}), 2.0f, 1e-6));
        CHECK(near(d.at<float32_t>({i}), 1.0f, 1e-6));
    }
    auto s = a.mul(3.0);
    CHECK(near(s.at<float32_t>({12345}), 3.0f, 1e-6));
}

// ==================== GPU paths ====================
// Guarded identically to tests/basic_ops_tests.cpp: the section runs only when
// a backend is compiled in AND a device is present; CPU-only builds skip it.

// --- fp32/fp64 arithmetic on the GPU ---

static void test_gpu_arith_f32_f64() {
    {
        auto a = Tensor({4}, DType::Float32, Device::GPU);
        auto b = Tensor({4}, DType::Float32, Device::GPU);
        const auto av = std::vector{1.5f, -2.25f, 0.0f, 100.0f};
        const auto bv = std::vector{0.25f, 4.0f, -3.0f, 0.5f};
        for (size_t i = 0; i < 4; ++i) {
            a.set<float32_t>({i}, av[i]);
            b.set<float32_t>({i}, bv[i]);
        }
        auto add = a.add(b);
        auto sub = a.sub(b);
        auto mul = a.mul(b);
        auto div = a.div(b);
        auto powr = a.pow(b);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(near(add.at<float32_t>({i}), av[i] + bv[i], 1e-6));
            CHECK(near(sub.at<float32_t>({i}), av[i] - bv[i], 1e-6));
            CHECK(near(mul.at<float32_t>({i}), av[i] * bv[i], 1e-4));
            CHECK(near(div.at<float32_t>({i}), av[i] / bv[i], 1e-4));
            if (i == 2) {
                // 0^-3 is +inf; fabs(inf - inf) is NaN, so check it explicitly.
                CHECK(std::isinf(static_cast<float64_t>(powr.at<float32_t>({i}))));
            } else {
                CHECK(near(powr.at<float32_t>({i}), std::pow(av[i], bv[i]), 1e-3));
            }
        }
        // Scalar overloads agree with direct scalar math.
        for (size_t i = 0; i < 4; ++i) {
            CHECK(near(a.add(0.25).at<float32_t>({i}), av[i] + 0.25f, 1e-6));
            CHECK(near(a.sub(4.0).at<float32_t>({i}), av[i] - 4.0f, 1e-6));
            CHECK(near(a.mul(-3.0).at<float32_t>({i}), av[i] * -3.0f, 1e-4));
            CHECK(near(a.div(0.5).at<float32_t>({i}), av[i] / 0.5f, 1e-4));
            CHECK(near(a.pow(2.0).at<float32_t>({i}), av[i] * av[i], 1e-3));
        }
        // exp/log smoke: same edge semantics as the CPU backend libm.
        auto ex = a.exp();
        CHECK(near(ex.at<float32_t>({0}), 4.481689f, 1e-5)); // exp(1.5)
        auto ebig = Tensor({1}, DType::Float32, Device::GPU);
        ebig.set<float32_t>({0}, 200.0f);
        CHECK(std::isinf(static_cast<float64_t>(ebig.exp().at<float32_t>({0}))));
        auto lg = b.log();
        CHECK(near(lg.at<float32_t>({1}), std::log(4.0f), 1e-6));
        auto lzero = Tensor({1}, DType::Float32, Device::GPU);
        lzero.set_zero();
        auto lz = lzero.log();
        CHECK(std::isinf(static_cast<float64_t>(lz.at<float32_t>({0}))));
        CHECK(lz.at<float32_t>({0}) < 0.0f);
    }
    {
        // fp64 computes in the storage width on the GPU too.
        auto a64 = Tensor({2}, DType::Float64, Device::GPU);
        auto b64 = Tensor({2}, DType::Float64, Device::GPU);
        a64.set<float64_t>({0}, 1.0e150);
        a64.set<float64_t>({1}, -7.0);
        b64.set<float64_t>({0}, 1.0e150);
        b64.set<float64_t>({1}, 2.0);
        auto mul64 = a64.mul(b64);
        CHECK(near64(mul64.at<float64_t>({0}), 1.0e300, 1e285));
        CHECK(near64(mul64.at<float64_t>({1}), -14.0, 1e-12));
        CHECK(near64(a64.add(0.5).at<float64_t>({1}), -6.5, 1e-12));
        CHECK(near64(a64.div(b64).at<float64_t>({1}), -3.5, 1e-12));
    }
}

// --- integer arithmetic on the GPU: wrap, truncating div, div-by-zero, min/-1 ---

static void test_gpu_arith_ints() {
    // int8 modular wrap on add/sub/mul.
    auto a = Tensor({3}, DType::Int8, Device::GPU);
    a.set<int8_t>({0}, int8_t{127});
    a.set<int8_t>({1}, int8_t{-128});
    a.set<int8_t>({2}, int8_t{100});
    CHECK(a.add(int8_t{1}).at<int8_t>({0}) == int8_t{-128}); // 127 + 1 wraps
    CHECK(a.add(int8_t{1}).at<int8_t>({1}) == int8_t{-127});
    CHECK(a.sub(int8_t{1}).at<int8_t>({1}) == int8_t{127});  // -128 - 1 wraps
    CHECK(a.mul(int8_t{2}).at<int8_t>({2}) == int8_t{-56});  // 200 wraps through the unsigned twin
    // int32 div truncates toward zero; div-by-zero yields 0.
    auto d = Tensor({3}, DType::Int32, Device::GPU);
    auto e = Tensor({3}, DType::Int32, Device::GPU);
    d.set<int32_t>({0}, -7);
    d.set<int32_t>({1}, 7);
    d.set<int32_t>({2}, -7);
    e.set<int32_t>({0}, 2);
    e.set<int32_t>({1}, -2);
    e.set<int32_t>({2}, 0);
    auto div = d.div(e);
    CHECK(div.at<int32_t>({0}) == -3); // truncation toward zero, not floor
    CHECK(div.at<int32_t>({1}) == -3);
    CHECK(div.at<int32_t>({2}) == 0);  // division by zero yields 0
    // min / -1 yields min (wrap), never UB — int8, int32, int64.
    auto mn8 = Tensor({1}, DType::Int8, Device::GPU);
    auto neg8 = Tensor({1}, DType::Int8, Device::GPU);
    mn8.set<int8_t>({0}, int8_t{-128});
    neg8.set<int8_t>({0}, int8_t{-1});
    CHECK(mn8.div(neg8).at<int8_t>({0}) == int8_t{-128});
    auto mn32 = Tensor({1}, DType::Int32, Device::GPU);
    auto neg32 = Tensor({1}, DType::Int32, Device::GPU);
    mn32.set<int32_t>({0}, std::numeric_limits<int32_t>::min());
    neg32.set<int32_t>({0}, -1);
    CHECK(mn32.div(neg32).at<int32_t>({0}) == std::numeric_limits<int32_t>::min());
    auto mn64 = Tensor({1}, DType::Int64, Device::GPU);
    auto neg64 = Tensor({1}, DType::Int64, Device::GPU);
    mn64.set<int64_t>({0}, std::numeric_limits<int64_t>::min());
    neg64.set<int64_t>({0}, -1);
    CHECK(mn64.div(neg64).at<int64_t>({0}) == std::numeric_limits<int64_t>::min());
    // Unsigned kinds: plain division and div-by-zero -> 0.
    auto u = Tensor({2}, DType::UInt8, Device::GPU);
    auto v = Tensor({2}, DType::UInt8, Device::GPU);
    u.set<uint8_t>({0}, uint8_t{200});
    u.set<uint8_t>({1}, uint8_t{5});
    v.set<uint8_t>({0}, uint8_t{3});
    v.set<uint8_t>({1}, uint8_t{0});
    auto udiv = u.div(v);
    CHECK(udiv.at<uint8_t>({0}) == uint8_t{66});
    CHECK(udiv.at<uint8_t>({1}) == uint8_t{0});
    // uint16 add wraps; int64 mul.
    auto w = Tensor({1}, DType::UInt16, Device::GPU);
    w.set<uint16_t>({0}, uint16_t{65535});
    CHECK(w.add(uint16_t{1}).at<uint16_t>({0}) == uint16_t{0});
    auto big = Tensor({1}, DType::Int64, Device::GPU);
    big.set<int64_t>({0}, 1000000);
    CHECK(big.mul(int64_t{1000000}).at<int64_t>({0}) == int64_t{1000000000000});
    // Scalar overloads on integers use the same wrap contract.
    CHECK(mn8.add(1.0).at<int8_t>({0}) == int8_t{-127});
    CHECK(w.mul(2.0).at<uint16_t>({0}) == uint16_t{65534});
}

// --- int4x2 per-nibble arithmetic on the GPU ---

static void test_gpu_arith_int4x2() {
    auto a = Tensor({2}, DType::Int4x2, Device::GPU);
    auto b = Tensor({2}, DType::Int4x2, Device::GPU);
    a.set<int4x2_t>({0}, i4x2(int8_t{7}, int8_t{-8}));
    a.set<int4x2_t>({1}, i4x2(int8_t{-3}, int8_t{5}));
    b.set<int4x2_t>({0}, i4x2(int8_t{1}, int8_t{-1}));
    b.set<int4x2_t>({1}, i4x2(int8_t{0}, int8_t{0}));
    // add wraps per nibble: 7+1 = 8 -> -8; -8 + -1 = -9 -> 7.
    auto add = a.add(b);
    CHECK(i4x2_lo(add.at<int4x2_t>({0})) == int8_t{-8});
    CHECK(i4x2_hi(add.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_lo(add.at<int4x2_t>({1})) == int8_t{-3});
    // mul wraps per nibble: 7*1 = 7; -8*-1 = 8 -> -8.
    auto mul = a.mul(b);
    CHECK(i4x2_lo(mul.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_hi(mul.at<int4x2_t>({0})) == int8_t{-8});
    // div truncates per nibble: 7/1 = 7, -8/-1 wraps back to -8, and the
    // div-by-zero nibble yields 0: (-3, 5) / (0, 0) -> (0, 0).
    auto div = a.div(b);
    CHECK(i4x2_lo(div.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_hi(div.at<int4x2_t>({0})) == int8_t{-8});
    CHECK(i4x2_lo(div.at<int4x2_t>({1})) == int8_t{0});
    CHECK(i4x2_hi(div.at<int4x2_t>({1})) == int8_t{0});
    // Scalar op: the scalar narrows into BOTH nibbles (2 -> (2, 2)).
    auto sc = a.add(2.0);
    CHECK(i4x2_lo(sc.at<int4x2_t>({1})) == int8_t{-1});
    CHECK(i4x2_hi(sc.at<int4x2_t>({1})) == int8_t{7});
}

// --- half and packed dtypes on the GPU ---

static void test_gpu_half_and_packed() {
    // fp16: widen-compute-narrow in fp32, overflow clamps to +/-65504.
    auto h = Tensor({2}, DType::Float16, Device::GPU);
    h.set<float16_t>({0}, narrow_lane<float16_t>(1.5f));
    h.set<float16_t>({1}, narrow_lane<float16_t>(-2.25f));
    auto h2 = h.add(0.25);
    CHECK(near(widen_lane(h2.at<float16_t>({0})), 1.75f, 1e-3));
    CHECK(near(widen_lane(h2.at<float16_t>({1})), -2.0f, 1e-3));
    CHECK(near(widen_lane(h.mul(2.0).at<float16_t>({0})), 3.0f, 1e-3));
    auto big = Tensor({1}, DType::Float16, Device::GPU);
    big.set<float16_t>({0}, narrow_lane<float16_t>(300.0f));
    CHECK(near(widen_lane(big.mul(300.0).at<float16_t>({0})), 65504.0f, 0.0));
    // bf16 multiply with an fp32-subnormal result flushes to zero.
    auto s = Tensor({1}, DType::BFloat16, Device::GPU);
    s.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(1.0e-20f));
    CHECK(near(widen_lane(s.mul(1.0e-20).at<bfloat16_t>({0})), 0.0f, 0.0));
    // Packed x2: per-lane compute.
    auto pk = Tensor({2}, DType::Float16x2, Device::GPU);
    pk.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 2.0f));
    pk.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(-3.0f, 4.0f));
    auto pkb = Tensor({2}, DType::Float16x2, Device::GPU);
    pkb.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(0.5f, 0.25f));
    pkb.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(0.5f, 0.25f));
    auto pk2 = pk.add(pkb);
    CHECK(near(unpack_lo(pk2.at<float16x2_t>({0})), 1.5f, 1e-3));
    CHECK(near(unpack_hi(pk2.at<float16x2_t>({0})), 2.25f, 1e-3));
    CHECK(near(unpack_lo(pk2.at<float16x2_t>({1})), -2.5f, 1e-3));
    CHECK(near(unpack_hi(pk2.at<float16x2_t>({1})), 4.25f, 1e-3));
    auto pkm2 = Tensor({2}, DType::Float16x2, Device::GPU);
    pkm2.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(2.0f, -2.0f));
    pkm2.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(2.0f, -2.0f));
    auto pkm = pk.mul(pkm2);
    CHECK(near(unpack_lo(pkm.at<float16x2_t>({0})), 2.0f, 1e-3));
    CHECK(near(unpack_hi(pkm.at<float16x2_t>({0})), -4.0f, 1e-3));
    // bf16x2 packed lanes.
    auto pb = Tensor({1}, DType::BFloat16x2, Device::GPU);
    pb.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(8.0f, 1.0f));
    auto pbd = Tensor({1}, DType::BFloat16x2, Device::GPU);
    pbd.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(2.0f, 4.0f));
    auto pb2 = pb.div(pbd);
    CHECK(near(unpack_lo(pb2.at<bfloat16x2_t>({0})), 4.0f, 1e-2));
    CHECK(near(unpack_hi(pb2.at<bfloat16x2_t>({0})), 0.25f, 1e-2));
    // Packed unary: exp(0, 0) -> (1, 1); log(1, 1) -> (0, 0).
    auto pz = Tensor({1}, DType::BFloat16x2, Device::GPU);
    pz.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(0.0f, 0.0f));
    auto pze = pz.exp();
    CHECK(near(unpack_lo(pze.at<bfloat16x2_t>({0})), 1.0f, 1e-2));
    CHECK(near(unpack_hi(pze.at<bfloat16x2_t>({0})), 1.0f, 1e-2));
}

// --- broadcasting on the GPU ---

// GPU twin of make_f32_bias: {2,3} fp32, row 0 = 1,2,3; row 1 = 11,12,13.
static Tensor make_f32_bias_gpu() {
    auto cpu = make_f32_bias();
    cpu.to(Device::GPU);
    return cpu;
}

static void test_gpu_broadcast() {
    // Same shape.
    auto a = make_f32_bias_gpu();
    auto b = make_f32_bias_gpu();
    auto same = a.add(b);
    CHECK(same.is_same_shape(Tensor({2, 3}, DType::Float32)));
    CHECK(near(same.at<float32_t>({1, 2}), 26.0f, 1e-6)); // 13 + 13
    // Per-row bias: [2,3] + [1,3].
    auto bias = Tensor({1, 3}, DType::Float32, Device::GPU);
    bias.set<float32_t>({0, 0}, 10.0f);
    bias.set<float32_t>({0, 1}, 20.0f);
    bias.set<float32_t>({0, 2}, 30.0f);
    auto rb = a.add(bias);
    CHECK(near(rb.at<float32_t>({0, 0}), 11.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({0, 2}), 33.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({1, 0}), 21.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({1, 2}), 43.0f, 1e-6));
    // Per-column: [2,3] * [2,1].
    auto col = Tensor({2, 1}, DType::Float32, Device::GPU);
    col.set<float32_t>({0, 0}, 100.0f);
    col.set<float32_t>({1, 0}, 200.0f);
    CHECK(near(a.mul(col).at<float32_t>({1, 2}), 2600.0f, 1e-4));
    // Right-aligned rank promotion: [2,3] + [3].
    auto r3 = Tensor({3}, DType::Float32, Device::GPU);
    r3.set<float32_t>({0}, 7.0f);
    r3.set<float32_t>({1}, 8.0f);
    r3.set<float32_t>({2}, 9.0f);
    auto pr = a.add(r3);
    CHECK(near(pr.at<float32_t>({0, 0}), 8.0f, 1e-6));
    CHECK(near(pr.at<float32_t>({1, 2}), 22.0f, 1e-6));
    // [1,1] against [2,3].
    auto sq = Tensor({1, 1}, DType::Float32, Device::GPU);
    sq.set<float32_t>({0, 0}, 5.0f);
    CHECK(near(a.add(sq).at<float32_t>({1, 1}), 17.0f, 1e-6));
    // Both directions of materialization.
    auto inv = bias.sub(a); // [1,3] - [2,3] -> [2,3]
    CHECK(near(inv.at<float32_t>({0, 0}), 9.0f, 1e-6));
    CHECK(near(inv.at<float32_t>({1, 0}), -1.0f, 1e-6));
    CHECK(near(col.div(a).at<float32_t>({1, 0}), 18.1818f, 1e-3)); // 200 / 11
    // Integer broadcasting on the GPU: [2,3] + [3] on int32.
    auto ai = Tensor({2, 3}, DType::Int32, Device::GPU);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            ai.set<int32_t>({r, c}, static_cast<int32_t>(r * 10 + c + 1));
        }
    }
    auto bi = Tensor({3}, DType::Int32, Device::GPU);
    bi.set<int32_t>({0}, 1);
    bi.set<int32_t>({1}, 2);
    bi.set<int32_t>({2}, 3);
    auto ri = ai.mul(bi);
    CHECK(ri.at<int32_t>({0, 0}) == 1);
    CHECK(ri.at<int32_t>({0, 2}) == 9);
    CHECK(ri.at<int32_t>({1, 0}) == 11);
    CHECK(ri.at<int32_t>({1, 2}) == 39);
}

// --- scalar narrowing into non-float dtypes on the GPU ---

static void test_gpu_scalar_narrowing() {
    // Float scalar truncates toward zero into integers.
    auto i = Tensor({2}, DType::Int32, Device::GPU);
    i.set<int32_t>({0}, 10);
    i.set<int32_t>({1}, -10);
    CHECK(i.add(2.9).at<int32_t>({0}) == 12);
    CHECK(i.add(-2.9).at<int32_t>({1}) == -12);
    // Out-of-range scalar clamps to the dtype range.
    auto big = Tensor({1}, DType::Int32, Device::GPU);
    big.set<int32_t>({0}, 0);
    CHECK(big.add(1e300).at<int32_t>({0}) == std::numeric_limits<int32_t>::max());
    auto small = Tensor({1}, DType::Int8, Device::GPU);
    small.set<int8_t>({0}, 0);
    CHECK(small.add(1000.0).at<int8_t>({0}) == int8_t{127});
    // NaN scalar narrows to 0 for integer kinds.
    auto n = Tensor({1}, DType::Int32, Device::GPU);
    n.set<int32_t>({0}, 5);
    CHECK(n.add(std::numeric_limits<float64_t>::quiet_NaN()).at<int32_t>({0}) == 5);
    // Unsigned kinds clamp negative scalars to 0.
    auto u = Tensor({1}, DType::UInt8, Device::GPU);
    u.set<uint8_t>({0}, uint8_t{3});
    CHECK(u.add(-5.0).at<uint8_t>({0}) == uint8_t{3});
    // int4x2 clamps the scalar to [-8, 7] per nibble, both nibbles.
    auto q = Tensor({1}, DType::Int4x2, Device::GPU);
    q.set<int4x2_t>({0}, i4x2(int8_t{0}, int8_t{0}));
    auto q7 = q.add(20.0);
    CHECK(i4x2_lo(q7.at<int4x2_t>({0})) == int8_t{7});
    CHECK(i4x2_hi(q7.at<int4x2_t>({0})) == int8_t{7});
    // The scalar narrows before the op: -20 clamps to -8, so 0-(-8) = 8 wraps to -8.
    auto qm8 = q.sub(-20.0);
    CHECK(i4x2_lo(qm8.at<int4x2_t>({0})) == int8_t{-8});
    CHECK(i4x2_hi(qm8.at<int4x2_t>({0})) == int8_t{-8});
    // Packed dtypes duplicate the narrowed scalar into both lanes.
    auto p = Tensor({1}, DType::Float16x2, Device::GPU);
    p.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 2.0f));
    auto p2 = p.add(0.5);
    CHECK(near(unpack_lo(p2.at<float16x2_t>({0})), 1.5f, 1e-3));
    CHECK(near(unpack_hi(p2.at<float16x2_t>({0})), 2.5f, 1e-3));
}

// --- division edge semantics on the GPU ---

static void test_gpu_edge_semantics() {
    // Float division by zero is IEEE on the GPU too.
    auto z = Tensor({3}, DType::Float32, Device::GPU);
    z.set<float32_t>({0}, 1.0f);
    z.set<float32_t>({1}, -1.0f);
    z.set<float32_t>({2}, 0.0f);
    auto zero = Tensor({3}, DType::Float32, Device::GPU);
    zero.set_zero();
    auto r = z.div(zero);
    CHECK(std::isinf(static_cast<float64_t>(r.at<float32_t>({0}))));
    CHECK(r.at<float32_t>({0}) > 0.0f);
    CHECK(std::isinf(static_cast<float64_t>(r.at<float32_t>({1}))));
    CHECK(r.at<float32_t>({1}) < 0.0f);
    CHECK(std::isnan(static_cast<float64_t>(r.at<float32_t>({2}))));
    // Scalar zero divisor, fp64.
    auto z64 = Tensor({1}, DType::Float64, Device::GPU);
    z64.set<float64_t>({0}, 1.0);
    CHECK(std::isinf(z64.div(0.0).at<float64_t>({0})));
    // Integer pow is rejected on the GPU (tensor-tensor and scalar).
    auto ints = Tensor({2}, DType::Int32, Device::GPU);
    ints.set<int32_t>({0}, 2);
    ints.set<int32_t>({1}, 3);
    auto threw = false;
    try {
        auto r2 = ints.pow(ints);
        static_cast<void>(r2);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        auto r2 = ints.pow(2.0);
        static_cast<void>(r2);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

// ==================== CPU<->GPU bit-exact parity ====================
// The AGENTS.md parity invariant: GPU kernels must match the CPU dispatcher
// bit-for-bit. These tests feed identical input bytes through both backends
// and compare result BITS (never approximate equality). Float inputs keep
// divisors non-zero: 0/0 NaN payloads are ISA-defined and out of scope, while
// every other edge (inf, wrap, truncation, clamp, flush) has a fixed bit
// pattern both sides must produce.

static size_t numel_of(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (const auto d : shape) {
        n *= d;
    }
    return n;
}

// Flat-index accessor for the rank-1/rank-2 shapes used by the parity tests.
template<typename T>
static void set_flat(Tensor& t, const std::vector<size_t>& shape, const size_t flat, const T& v) {
    if (shape.size() == 1) {
        t.set<T>({flat}, v);
    } else {
        t.set<T>({flat / shape[1], flat % shape[1]}, v);
    }
}

template<typename T>
static T get_flat(Tensor& t, const std::vector<size_t>& shape, const size_t flat) {
    if (shape.size() == 1) {
        return t.at<T>({flat});
    }
    return t.at<T>({flat / shape[1], flat % shape[1]});
}

static uint32_t lcg_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

// -Werror=float-equal: zero tests go through the bits.
static bool float_is_zero(const float32_t v) {
    return idet::bit_cast_memcpy<uint32_t>(v) << 1 == 0u;
}

template<typename F>
static F coerce_float(const float32_t v) {
    if constexpr (std::is_same_v<F, float16_t>) {
        return narrow_lane<float16_t>(v);
    } else if constexpr (std::is_same_v<F, bfloat16_t>) {
        return narrow_lane<bfloat16_t>(v);
    } else {
        return static_cast<F>(v);
    }
}

template<typename F>
static float32_t widen_value(const F& v) {
    if constexpr (std::is_same_v<F, float16_t> || std::is_same_v<F, bfloat16_t>) {
        return widen_lane(v);
    } else {
        return static_cast<float32_t>(v);
    }
}

template<typename F>
static void fill_floats_tensor(Tensor& t, const std::vector<size_t>& shape, const bool divisor) {
    uint32_t s = divisor ? 0x9E3779B9u : 0x85EBCA6Bu;
    const auto n = numel_of(shape);
    for (size_t i = 0; i < n; ++i) {
        auto v = static_cast<float32_t>(static_cast<int32_t>(lcg_next(s) % 40001u) - 20000) / 37.0f;
        if (i % 7u == 3u) {
            v *= 1.0e3f;
        }
        if (i % 11u == 5u) {
            v *= 1.0e-2f;
        }
        if (i == 5u || i == 6u) {
            v = 1.0e-20f; // bf16 flush-to-zero / fp16 underflow probe on mul
        }
        if (i == 7u) {
            v = 1.0e-6f; // fp16-subnormal narrowing probe
        }
        auto cv = coerce_float<F>(v);
        if (divisor && float_is_zero(widen_value<F>(cv))) {
            cv = coerce_float<F>(1.0f);
        }
        set_flat<F>(t, shape, i, cv);
    }
}

template<typename P, typename H>
static void fill_packed_tensor(Tensor& t, const std::vector<size_t>& shape, const bool divisor) {
    uint32_t s = divisor ? 0xC2B2AE35u : 0x27D4EB4Fu;
    const auto n = numel_of(shape);
    for (size_t i = 0; i < n; ++i) {
        auto lo = static_cast<float32_t>(static_cast<int32_t>(lcg_next(s) % 40001u) - 20000) / 37.0f;
        auto hi = static_cast<float32_t>(static_cast<int32_t>(lcg_next(s) % 40001u) - 20000) / 53.0f;
        if (i == 5u || i == 6u) {
            lo = 1.0e-20f;
            hi = 1.0e-20f;
        }
        if (i == 7u) {
            lo = 1.0e-6f;
            hi = 1.0e-6f;
        }
        auto clo = coerce_float<H>(lo);
        auto chi = coerce_float<H>(hi);
        if (divisor) {
            if (float_is_zero(widen_value<H>(clo))) {
                clo = coerce_float<H>(1.0f);
            }
            if (float_is_zero(widen_value<H>(chi))) {
                chi = coerce_float<H>(1.0f);
            }
        }
        set_flat<P>(t, shape, i, pack_lanes<P, H>(widen_value<H>(clo), widen_value<H>(chi)));
    }
}

template<typename I>
static I int_draw(uint32_t& s) {
    if constexpr (sizeof(I) > 4u) {
        const auto wide = (static_cast<uint64_t>(lcg_next(s)) << 32) | static_cast<uint64_t>(lcg_next(s));
        return static_cast<I>(wide);
    } else {
        return static_cast<I>(lcg_next(s));
    }
}

template<typename I>
static void fill_ints_tensor(Tensor& t, const std::vector<size_t>& shape, const bool divisor) {
    uint32_t s = divisor ? 0x165667B1u : 0xD3A2646Cu;
    const auto n = numel_of(shape);
    for (size_t i = 0; i < n; ++i) {
        auto v = int_draw<I>(s);
        if (divisor && v == I{0} && i % 4u != 0u) {
            v = I{1}; // keep every fourth divisor at zero: div-by-zero -> 0
        }
        set_flat<I>(t, shape, i, v);
    }
    // Edge sentinels, identical on both devices: div-by-zero, truncation
    // toward zero (-7/3, 7/-3), and min/-1 -> min (wrap).
    if (n > 0) {
        set_flat<I>(t, shape, 0, I{0});
    }
    if (n > 1) {
        set_flat<I>(t, shape, 1, divisor ? I{2} : I{7});
    }
    if (n > 2) {
        set_flat<I>(t, shape, 2, divisor ? static_cast<I>(-1) : std::numeric_limits<I>::min());
    }
    if (n > 3) {
        set_flat<I>(t, shape, 3, divisor ? I{3} : static_cast<I>(-7));
    }
    if (n > 4 && divisor) {
        set_flat<I>(t, shape, 4, static_cast<I>(-3));
    }
}

static int8_t nibble_draw(uint32_t& s) {
    return static_cast<int8_t>(static_cast<int32_t>(lcg_next(s) % 16u) - 8);
}

static void fill_int4x2_tensor(Tensor& t, const std::vector<size_t>& shape, const bool divisor) {
    uint32_t s = divisor ? 0x85EBCA77u : 0x9E3779F9u;
    const auto n = numel_of(shape);
    for (size_t i = 0; i < n; ++i) {
        auto lo = nibble_draw(s);
        auto hi = nibble_draw(s);
        if (divisor) {
            if (lo == int8_t{0} && i % 4u != 0u) {
                lo = int8_t{1};
            }
            if (hi == int8_t{0} && i % 4u != 0u) {
                hi = int8_t{1};
            }
        }
        set_flat<int4x2_t>(t, shape, i, i4x2(lo, hi));
    }
    if (n > 0) {
        set_flat<int4x2_t>(t, shape, 0, i4x2(int8_t{0}, int8_t{0}));
    }
    if (n > 1) {
        set_flat<int4x2_t>(t, shape, 1, i4x2(int8_t{7}, int8_t{-7}));
    }
    if (n > 2) {
        set_flat<int4x2_t>(t, shape, 2, divisor ? i4x2(int8_t{-1}, int8_t{-1}) : i4x2(int8_t{-8}, int8_t{-8}));
    }
    if (n > 3 && !divisor) {
        set_flat<int4x2_t>(t, shape, 3, i4x2(int8_t{-7}, int8_t{7}));
    }
}

// Deterministic per-dtype fill: `divisor` tensors keep the float divisors
// non-zero and plant integer zero divisors/min patterns; `!divisor` tensors
// plant the matching numerators.
static void fill_role(const DType dt, Tensor& t, const std::vector<size_t>& shape, const bool divisor) {
    switch (dt) {
        case DType::Float32:
            fill_floats_tensor<float32_t>(t, shape, divisor);
            break;
        case DType::Float64:
            fill_floats_tensor<float64_t>(t, shape, divisor);
            break;
        case DType::Float16:
            fill_floats_tensor<float16_t>(t, shape, divisor);
            break;
        case DType::BFloat16:
            fill_floats_tensor<bfloat16_t>(t, shape, divisor);
            break;
        case DType::Float16x2:
            fill_packed_tensor<float16x2_t, float16_t>(t, shape, divisor);
            break;
        case DType::BFloat16x2:
            fill_packed_tensor<bfloat16x2_t, bfloat16_t>(t, shape, divisor);
            break;
        case DType::Int64:
            fill_ints_tensor<int64_t>(t, shape, divisor);
            break;
        case DType::Int32:
            fill_ints_tensor<int32_t>(t, shape, divisor);
            break;
        case DType::Int16:
            fill_ints_tensor<int16_t>(t, shape, divisor);
            break;
        case DType::Int8:
            fill_ints_tensor<int8_t>(t, shape, divisor);
            break;
        case DType::UInt64:
            fill_ints_tensor<uint64_t>(t, shape, divisor);
            break;
        case DType::UInt32:
            fill_ints_tensor<uint32_t>(t, shape, divisor);
            break;
        case DType::UInt16:
            fill_ints_tensor<uint16_t>(t, shape, divisor);
            break;
        case DType::UInt8:
            fill_ints_tensor<uint8_t>(t, shape, divisor);
            break;
        case DType::Int4x2:
            fill_int4x2_tensor(t, shape, divisor);
            break;
        [[unlikely]] default:
            throw BeanTensor::ErrorHandling::NotImplemented();
    }
}

static const std::vector<DType>& all_dtypes() {
    static const auto dtypes = std::vector{
        DType::BFloat16, DType::BFloat16x2, DType::Float16, DType::Float16x2,
        DType::Float32, DType::Float64, DType::Int64, DType::Int32, DType::Int16, DType::Int8,
        DType::UInt64, DType::UInt32, DType::UInt16, DType::UInt8, DType::Int4x2
    };
    return dtypes;
}

static bool is_float_dtype(const DType dt) {
    switch (dt) {
        case DType::BFloat16:
        case DType::BFloat16x2:
        case DType::Float16:
        case DType::Float16x2:
        case DType::Float32:
        case DType::Float64:
            return true;
        case DType::Int64:
        case DType::Int32:
        case DType::Int16:
        case DType::Int8:
        case DType::UInt64:
        case DType::UInt32:
        case DType::UInt16:
        case DType::UInt8:
        case DType::Int4x2:
            return false;
        [[unlikely]] default:
            throw BeanTensor::ErrorHandling::NotImplemented();
    }
}

static const char* op_name(const ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add: return "add";
        case ElementwiseOp::Sub: return "sub";
        case ElementwiseOp::Mul: return "mul";
        case ElementwiseOp::Div: return "div";
        [[unlikely]] default: return "pow";
    }
}

static Tensor apply_op(const Tensor& x, const Tensor& y, const ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add: return x.add(y);
        case ElementwiseOp::Sub: return x.sub(y);
        case ElementwiseOp::Mul: return x.mul(y);
        case ElementwiseOp::Div: return x.div(y);
        [[unlikely]] default: throw BeanTensor::ErrorHandling::NotImplemented();
    }
}

static Tensor apply_scalar_op(const Tensor& x, const float64_t s, const ElementwiseOp op) {
    switch (op) {
        case ElementwiseOp::Add: return x.add(s);
        case ElementwiseOp::Sub: return x.sub(s);
        case ElementwiseOp::Mul: return x.mul(s);
        case ElementwiseOp::Div: return x.div(s);
        [[unlikely]] default: throw BeanTensor::ErrorHandling::NotImplemented();
    }
}

// Bit-exact per-element comparison (floats compare through their bit pattern,
// so NaN payloads and signed zeros must match too).
static bool elem_bits_equal(const DType dt, Tensor& x, Tensor& y,
                            const std::vector<size_t>& shape, const size_t i) {
    switch (dt) {
        case DType::Float32:
            return idet::bit_cast_memcpy<uint32_t>(get_flat<float32_t>(x, shape, i)) ==
                   idet::bit_cast_memcpy<uint32_t>(get_flat<float32_t>(y, shape, i));
        case DType::Float64:
            return idet::bit_cast_memcpy<uint64_t>(get_flat<float64_t>(x, shape, i)) ==
                   idet::bit_cast_memcpy<uint64_t>(get_flat<float64_t>(y, shape, i));
        case DType::Float16:
            return lane_bits(get_flat<float16_t>(x, shape, i)) == lane_bits(get_flat<float16_t>(y, shape, i));
        case DType::BFloat16:
            return lane_bits(get_flat<bfloat16_t>(x, shape, i)) == lane_bits(get_flat<bfloat16_t>(y, shape, i));
        case DType::Float16x2:
            return idet::bit_cast_memcpy<uint32_t>(get_flat<float16x2_t>(x, shape, i)) ==
                   idet::bit_cast_memcpy<uint32_t>(get_flat<float16x2_t>(y, shape, i));
        case DType::BFloat16x2:
            return idet::bit_cast_memcpy<uint32_t>(get_flat<bfloat16x2_t>(x, shape, i)) ==
                   idet::bit_cast_memcpy<uint32_t>(get_flat<bfloat16x2_t>(y, shape, i));
        case DType::Int64:
            return get_flat<int64_t>(x, shape, i) == get_flat<int64_t>(y, shape, i);
        case DType::Int32:
            return get_flat<int32_t>(x, shape, i) == get_flat<int32_t>(y, shape, i);
        case DType::Int16:
            return get_flat<int16_t>(x, shape, i) == get_flat<int16_t>(y, shape, i);
        case DType::Int8:
            return get_flat<int8_t>(x, shape, i) == get_flat<int8_t>(y, shape, i);
        case DType::UInt64:
            return get_flat<uint64_t>(x, shape, i) == get_flat<uint64_t>(y, shape, i);
        case DType::UInt32:
            return get_flat<uint32_t>(x, shape, i) == get_flat<uint32_t>(y, shape, i);
        case DType::UInt16:
            return get_flat<uint16_t>(x, shape, i) == get_flat<uint16_t>(y, shape, i);
        case DType::UInt8:
            return get_flat<uint8_t>(x, shape, i) == get_flat<uint8_t>(y, shape, i);
        case DType::Int4x2:
            return idet::bit_cast_memcpy<uint8_t>(get_flat<int4x2_t>(x, shape, i)) ==
                   idet::bit_cast_memcpy<uint8_t>(get_flat<int4x2_t>(y, shape, i));
        [[unlikely]] default:
            return false;
    }
}

static const std::vector<ElementwiseOp>& parity_ops() {
    static const auto ops = std::vector{
        ElementwiseOp::Add, ElementwiseOp::Sub, ElementwiseOp::Mul, ElementwiseOp::Div
    };
    return ops;
}

// Runs one CPU<->GPU pair through every element and reports the first
// divergence with full context (a bare CHECK would print an unreadable expr).
static void parity_compare(const char* case_name, const DType dt, const ElementwiseOp op,
                           const std::vector<size_t>& out_shape, Tensor& r_cpu, Tensor& r_gpu) {
    r_gpu.sync();
    if (!r_cpu.is_same_shape(r_gpu)) {
        ++failures;
        std::cerr << "PARITY FAIL [" << case_name << "] dtype=" << BeanTensor::DataTypes::dtype_to_string(dt)
                  << " op=" << op_name(op) << ": shape mismatch\n";
        return;
    }
    CHECK(r_cpu.get_dtype() == dt);
    CHECK(r_gpu.get_dtype() == dt);
    const auto n = numel_of(out_shape);
    for (size_t i = 0; i < n; ++i) {
        if (!elem_bits_equal(dt, r_cpu, r_gpu, out_shape, i)) {
            ++failures;
            std::cerr << "PARITY FAIL [" << case_name << "] dtype=" << BeanTensor::DataTypes::dtype_to_string(dt)
                      << " op=" << op_name(op) << " elem=" << i << "\n";
            return;
        }
    }
}

// Same-shape parity: all 15 dtypes x add/sub/mul/div.
static void test_gpu_cpu_parity_exact() {
    constexpr size_t n = 64;
    const auto shape = std::vector{n};
    for (const auto dt : all_dtypes()) {
        auto a_cpu = Tensor(shape, dt, Device::CPU);
        auto b_cpu = Tensor(shape, dt, Device::CPU);
        fill_role(dt, a_cpu, shape, false);
        fill_role(dt, b_cpu, shape, true);
        auto a_gpu = a_cpu.clone();
        a_gpu.to(Device::GPU);
        auto b_gpu = b_cpu.clone();
        b_gpu.to(Device::GPU);
        for (const auto op : parity_ops()) {
            auto r_cpu = apply_op(a_cpu, b_cpu, op);
            auto r_gpu = apply_op(a_gpu, b_gpu, op);
            r_gpu.to(Device::CPU); // one bulk D2H copy instead of per-element reads
            parity_compare("exact", dt, op, shape, r_cpu, r_gpu);
        }
    }
}

// Broadcast parity: every dtype through both the exact plan and the
// broadcast plan (both operand orders), including a broadcast div.
static void test_gpu_cpu_parity_broadcast() {
    struct BcastCase {
        std::vector<size_t> a_shape;
        std::vector<size_t> b_shape;
        ElementwiseOp op;
        const char* name;
    };
    const auto cases = std::vector<BcastCase>{
        {.a_shape = {2, 3}, .b_shape = {3}, .op = ElementwiseOp::Add, .name = "a{2,3}+b{3}"},
        {.a_shape = {2, 3}, .b_shape = {1, 3}, .op = ElementwiseOp::Sub, .name = "a{2,3}-b{1,3}"},
        {.a_shape = {2, 3}, .b_shape = {2, 1}, .op = ElementwiseOp::Mul, .name = "a{2,3}*b{2,1}"},
        {.a_shape = {2, 3}, .b_shape = {1, 3}, .op = ElementwiseOp::Div, .name = "a{2,3}/b{1,3}"},
        {.a_shape = {1, 3}, .b_shape = {2, 3}, .op = ElementwiseOp::Add, .name = "a{1,3}+b{2,3}"},
        {.a_shape = {2, 3}, .b_shape = {1, 1}, .op = ElementwiseOp::Add, .name = "a{2,3}+b{1,1}"},
    };
    const auto out_shape = std::vector<size_t>{2, 3};
    for (const auto dt : all_dtypes()) {
        for (const auto& [a_shape, b_shape, op, name] : cases) {
            auto a_cpu = Tensor(a_shape, dt, Device::CPU);
            auto b_cpu = Tensor(b_shape, dt, Device::CPU);
            fill_role(dt, a_cpu, a_shape, false);
            fill_role(dt, b_cpu, b_shape, op == ElementwiseOp::Div);
            auto a_gpu = a_cpu.clone();
            a_gpu.to(Device::GPU);
            auto b_gpu = b_cpu.clone();
            b_gpu.to(Device::GPU);
            auto r_cpu = apply_op(a_cpu, b_cpu, op);
            auto r_gpu = apply_op(a_gpu, b_gpu, op);
            r_gpu.to(Device::CPU);
            parity_compare(name, dt, op, out_shape, r_cpu, r_gpu);
        }
    }
}

// Tensor-scalar parity: the scalar narrows on the host through the same
// PERMISSIVE contract on both backends, then the op applies on-device.
static void test_gpu_cpu_parity_scalar() {
    const auto float_scalars = std::vector{0.5, -2.9, 7.0, 1.0e-6};
    // NaN narrows to 0 for integer kinds (never a float NaN payload issue);
    // 20.0 clamps per-nibble for int4x2 and in-range elsewhere.
    const auto int_scalars = std::vector{
        0.5, -2.9, 7.0, std::numeric_limits<float64_t>::quiet_NaN(), 20.0
    };
    constexpr size_t n = 32;
    const auto shape = std::vector{n};
    for (const auto dt : all_dtypes()) {
        const auto& scalars = is_float_dtype(dt) ? float_scalars : int_scalars;
        auto a_cpu = Tensor(shape, dt, Device::CPU);
        fill_role(dt, a_cpu, shape, false);
        auto a_gpu = a_cpu.clone();
        a_gpu.to(Device::GPU);
        for (const auto op : parity_ops()) {
            for (const auto s : scalars) {
                auto r_cpu = apply_scalar_op(a_cpu, s, op);
                auto r_gpu = apply_scalar_op(a_gpu, s, op);
                r_gpu.to(Device::CPU);
                parity_compare("scalar", dt, op, shape, r_cpu, r_gpu);
            }
        }
    }
}

// Grid-stride parity: more elements than one wave (256 threads x 1024
// blocks) so the GPU kernels loop; catches per-wave indexing bugs.
static void test_gpu_cpu_parity_large() {
    constexpr size_t n = 300000;
    const auto shape = std::vector{n};
    for (const auto dt : {DType::Float32, DType::Int32}) {
        auto a_cpu = Tensor(shape, dt, Device::CPU);
        auto b_cpu = Tensor(shape, dt, Device::CPU);
        fill_role(dt, a_cpu, shape, false);
        fill_role(dt, b_cpu, shape, true);
        auto a_gpu = a_cpu.clone();
        a_gpu.to(Device::GPU);
        auto b_gpu = b_cpu.clone();
        b_gpu.to(Device::GPU);
        for (const auto op : parity_ops()) {
            auto r_cpu = apply_op(a_cpu, b_cpu, op);
            auto r_gpu = apply_op(a_gpu, b_gpu, op);
            r_gpu.to(Device::CPU);
            parity_compare("large", dt, op, shape, r_cpu, r_gpu);
        }
    }
}

// ==================== AVX2/AVX512 contiguous kernel parity ====================
// Direct kernel tests, in the style of tests/tensor_tests.cpp's probe pattern:
// the kernels throw ErrorHandling::CPUIllegalInstruction when the CPU lacks the
// feature (legitimate skip — the answer is authoritative because it comes from
// the kernel TUs, which carry the per-file -m flags; this TU's own
// IncludedCPUFeatures() copy is blind for AVX512). When a kernel runs, its
// output must BIT-MATCH the scalar applies in ElementwiseCPU.cpp
// (apply_native/apply_half/apply_int) on sentinel + random values, at odd
// sizes exercising vector tails.

namespace ew = BeanTensor::Math::Elementwise;

// Deterministic xorshift64 so the sentinel mix is reproducible.
static uint64_t simd_rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t simd_rand() {
    simd_rng_state ^= simd_rng_state >> 12;
    simd_rng_state ^= simd_rng_state << 25;
    simd_rng_state ^= simd_rng_state >> 27;
    return simd_rng_state * 0x2545F4914F6CDD1Dull;
}

// Odd sizes exercise the vector tails; 65536 clears every AVX512 main-loop
// width with a remainder.
static const std::vector<size_t>& simd_sizes() {
    static const auto sizes = std::vector<size_t>{1, 7, 31, 37, 65536};
    return sizes;
}

static const std::vector<uint64_t>& simd_f32_pool() {
    static const auto pool = std::vector<uint64_t>{
        0x00000000ull, 0x80000000ull, 0x7F800000ull, 0xFF800000ull,
        0x7FC00000ull, 0xFFC00000ull, 0x7F800001ull, 0xFF800001ull,
        0x7F80FFFFull, 0x00000001ull, 0x80000001ull, 0x007FFFFFull,
        0x807FFFFFull, 0x00800000ull, 0x7F7FFFFFull, 0x477FE000ull,
        0x477FE001ull, 0x477FF000ull, 0x469C4000ull, 0xC69C4000ull,
        0x3F800000ull, 0xBF800000ull, 0x3F000000ull, 0xB7000001ull,
    };
    return pool;
}

static const std::vector<uint64_t>& simd_f64_pool() {
    static const auto pool = std::vector<uint64_t>{
        0x0000000000000000ull, 0x8000000000000000ull,
        0x7FF0000000000000ull, 0xFFF0000000000000ull,
        0x7FF8000000000000ull, 0xFFF8000000000000ull,
        0x7FF0000000000001ull, 0x7FFFFFFFFFFFFFFFull,
        0x0000000000000001ull, 0x8000000000000001ull,
        0x000FFFFFFFFFFFFFull, 0x0010000000000000ull,
        0x7FEFFFFFFFFFFFFFull, 0x3FF0000000000000ull,
        0xBFF0000000000000ull, 0x3FE0000000000000ull,
        0x40F86A0000000000ull, 0xC0F86A0000000000ull,
    };
    return pool;
}

static const std::vector<uint64_t>& simd_int_pool() {
    static const auto pool = std::vector<uint64_t>{
        0x0000000000000000ull, 0x0000000000000001ull,
        0xFFFFFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull,
        0x8000000000000000ull, 0x5555555555555555ull,
        0xAAAAAAAAAAAAAAAAull, 0x00000000FFFFFFFFull,
        0xFFFFFFFF00000000ull, 0x0001000100010001ull,
    };
    return pool;
}

static const std::vector<uint64_t>& simd_f16_pool() {
    static const auto pool = std::vector<uint64_t>{
        0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x7DFF,
        0x03FF, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00, 0x3800,
        0x1400, 0x0C01, 0x7CFF, 0x3555,
    };
    return pool;
}

static const std::vector<uint64_t>& simd_bf16_pool() {
    static const auto pool = std::vector<uint64_t>{
        0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x7FFF,
        0x007F, 0x8080, 0x0080, 0x7F7F, 0x3F80, 0xBF80, 0x3F00, 0x0081,
        0x0040, 0x7F01, 0x0001, 0x3555,
    };
    return pool;
}

template<typename T>
static T bits_to(const uint64_t bits) {
    if constexpr (std::is_same_v<T, float32_t>) {
        return idet::bit_cast_memcpy<float32_t>(static_cast<uint32_t>(bits));
    } else if constexpr (std::is_same_v<T, float64_t>) {
        return idet::bit_cast_memcpy<float64_t>(bits);
    } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
        return idet::bit_cast_memcpy<T>(static_cast<uint16_t>(bits));
    } else {
        return static_cast<T>(bits);
    }
}

// Scalar reference mirroring apply_native (fp32/fp64) and apply_int's
// unsigned-twin wrap arithmetic (integers) — the exact contracts the kernels hold.
template<typename T, ElementwiseOp Op>
static T simd_ref_op(const T a, const T b) {
    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
        if constexpr (Op == ElementwiseOp::Add) { return a + b; }
        else if constexpr (Op == ElementwiseOp::Sub) { return a - b; }
        else if constexpr (Op == ElementwiseOp::Mul) { return a * b; }
        else { return a / b; }
    } else {
        using U = std::make_unsigned_t<T>;
        if constexpr (Op == ElementwiseOp::Add) { return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b))); }
        else if constexpr (Op == ElementwiseOp::Sub) { return static_cast<T>(static_cast<U>(static_cast<U>(a) - static_cast<U>(b))); }
        else { return static_cast<T>(static_cast<U>(static_cast<U>(a) * static_cast<U>(b))); }
    }
}

// Scalar reference mirroring apply_half (widen/apply/narrow through the unaccel
// pathway) for one flat 16-bit lane.
template<typename H, ElementwiseOp Op>
static uint16_t simd_ref_lane_op(const uint16_t x, const uint16_t y) {
    const auto wx = widen_lane(idet::bit_cast_memcpy<H>(x));
    const auto wy = widen_lane(idet::bit_cast_memcpy<H>(y));
    float32_t w;
    if constexpr (Op == ElementwiseOp::Add) { w = wx + wy; }
    else if constexpr (Op == ElementwiseOp::Sub) { w = wx - wy; }
    else if constexpr (Op == ElementwiseOp::Mul) { w = wx * wy; }
    else { w = wx / wy; }
    return lane_bits(narrow_lane<H>(w));
}

// NaN INPUT values are pinned to +-Inf (sign preserved): IEEE arithmetic
// propagates NaN payloads in operand-order-dependent ways (a compiler may swap
// the operands of a commutative FP op, and x86 propagates the FIRST source's
// NaN), which is about codegen, not kernel semantics. NaN payload parity
// between the kernels and the scalar path is covered bitwise — with NaN-laden
// inputs — by test_simd_nan_parity_e2e below, which compares the two paths
// inside the compiled core library instead of against a replica reference.
static uint16_t no_nan_lane(const uint16_t bits, const uint16_t exp_mask, const uint16_t mant_mask) {
    if ((bits & exp_mask) == exp_mask && (bits & mant_mask) != 0) {
        return static_cast<uint16_t>((bits & 0x8000u) | exp_mask);
    }
    return bits;
}

template<typename T>
static T simd_no_nan(const T v) {
    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
        if (std::isnan(v)) {
            return std::copysign(std::numeric_limits<T>::infinity(), v);
        }
    }
    return v;
}

// Periodic sentinels on a (so every SIMD block sees every sentinel) mixed with
// full-range randoms; b mirrors with a different phase. Deterministic.
template<typename T>
static void simd_fill_pair(std::vector<T>& a, std::vector<T>& b, const std::vector<uint64_t>& pool) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = simd_no_nan((i % 2 == 0) ? bits_to<T>(pool[i % pool.size()]) : bits_to<T>(simd_rand()));
        b[i] = simd_no_nan((i % 3 == 0) ? bits_to<T>(pool[(i + 5) % pool.size()]) : bits_to<T>(simd_rand()));
    }
}

template<typename T>
static void check_bitmatch(const char* const label, const std::vector<T>& got, const std::vector<T>& ref) {
    if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(T)) == 0) {
        return;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &ref[i], sizeof(T)) != 0) {
            auto got_bits = uint64_t{0};
            auto ref_bits = uint64_t{0};
            std::memcpy(&got_bits, &got[i], sizeof(T));
            std::memcpy(&ref_bits, &ref[i], sizeof(T));
            ++failures;
            std::cerr << "FAIL: " << label << " bit mismatch at [" << i << "] of " << got.size()
                      << ": got=0x" << std::hex << got_bits << " ref=0x" << ref_bits << std::dec << "\n";
            return;
        }
    }
}

// One (dtype, op) case against both ISAs' kernels: run the kernel over each
// test size and bit-compare against the scalar reference. A thrown
// CPUIllegalInstruction marks that ISA skipped (the CPU lacks the feature).
template<typename T, ElementwiseOp Op>
static void simd_native_case(const char* const label,
                             void(*const k512)(T*, const T*, const T*, size_t, size_t),
                             void(*const k2)(T*, const T*, const T*, size_t, size_t),
                             const std::vector<uint64_t>& pool,
                             bool& skipped512,
                             bool& skipped2) {
    for (const size_t n : simd_sizes()) {
        auto a = std::vector<T>(n);
        auto b = std::vector<T>(n);
        simd_fill_pair(a, b, pool);
        auto ref = std::vector<T>(n);
        for (size_t i = 0; i < n; ++i) {
            ref[i] = simd_ref_op<T, Op>(a[i], b[i]);
        }
        auto got = std::vector<T>(n);
        try {
            k512(got.data(), a.data(), b.data(), 0, n);
            check_bitmatch((std::string(label) + " [avx512]").c_str(), got, ref);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            skipped512 = true;
        }
        try {
            k2(got.data(), a.data(), b.data(), 0, n);
            check_bitmatch((std::string(label) + " [avx2]").c_str(), got, ref);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            skipped2 = true;
        }
    }
}

// One (half kind, op) case against both ISAs' flat-lane kernels.
template<typename H, ElementwiseOp Op>
static void simd_half_case(const char* const label,
                           void(*const k512)(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t),
                           void(*const k2)(uint16_t*, const uint16_t*, const uint16_t*, size_t, size_t),
                           const std::vector<uint64_t>& pool,
                           bool& skipped512,
                           bool& skipped2) {
    const auto exp_mask = std::is_same_v<H, float16_t> ? uint16_t{0x7C00} : uint16_t{0x7F80};
    const auto mant_mask = std::is_same_v<H, float16_t> ? uint16_t{0x03FF} : uint16_t{0x007F};
    for (const size_t n : simd_sizes()) {
        auto a = std::vector<uint16_t>(n);
        auto b = std::vector<uint16_t>(n);
        for (size_t i = 0; i < n; ++i) {
            const auto a_bits = (i % 2 == 0) ? static_cast<uint16_t>(pool[i % pool.size()]) : static_cast<uint16_t>(simd_rand());
            const auto b_bits = (i % 3 == 0) ? static_cast<uint16_t>(pool[(i + 5) % pool.size()]) : static_cast<uint16_t>(simd_rand());
            a[i] = no_nan_lane(a_bits, exp_mask, mant_mask);
            b[i] = no_nan_lane(b_bits, exp_mask, mant_mask);
        }
        auto ref = std::vector<uint16_t>(n);
        for (size_t i = 0; i < n; ++i) {
            ref[i] = simd_ref_lane_op<H, Op>(a[i], b[i]);
        }
        auto got = std::vector<uint16_t>(n);
        try {
            k512(got.data(), a.data(), b.data(), 0, n);
            check_bitmatch((std::string(label) + " [avx512]").c_str(), got, ref);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            skipped512 = true;
        }
        try {
            k2(got.data(), a.data(), b.data(), 0, n);
            check_bitmatch((std::string(label) + " [avx2]").c_str(), got, ref);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            skipped2 = true;
        }
    }
}

static void test_simd_kernel_parity() {
    auto skipped512 = false;
    auto skipped2 = false;

    // fp32/fp64: add/sub/mul/div, native IEEE.
    simd_native_case<float32_t, ElementwiseOp::Add>("f32 add", &ew::add_f32_range_avx512, &ew::add_f32_range_avx2, simd_f32_pool(), skipped512, skipped2);
    simd_native_case<float32_t, ElementwiseOp::Sub>("f32 sub", &ew::sub_f32_range_avx512, &ew::sub_f32_range_avx2, simd_f32_pool(), skipped512, skipped2);
    simd_native_case<float32_t, ElementwiseOp::Mul>("f32 mul", &ew::mul_f32_range_avx512, &ew::mul_f32_range_avx2, simd_f32_pool(), skipped512, skipped2);
    simd_native_case<float32_t, ElementwiseOp::Div>("f32 div", &ew::div_f32_range_avx512, &ew::div_f32_range_avx2, simd_f32_pool(), skipped512, skipped2);
    simd_native_case<float64_t, ElementwiseOp::Add>("f64 add", &ew::add_f64_range_avx512, &ew::add_f64_range_avx2, simd_f64_pool(), skipped512, skipped2);
    simd_native_case<float64_t, ElementwiseOp::Sub>("f64 sub", &ew::sub_f64_range_avx512, &ew::sub_f64_range_avx2, simd_f64_pool(), skipped512, skipped2);
    simd_native_case<float64_t, ElementwiseOp::Mul>("f64 mul", &ew::mul_f64_range_avx512, &ew::mul_f64_range_avx2, simd_f64_pool(), skipped512, skipped2);
    simd_native_case<float64_t, ElementwiseOp::Div>("f64 div", &ew::div_f64_range_avx512, &ew::div_f64_range_avx2, simd_f64_pool(), skipped512, skipped2);
    // Integers: wrap add/sub/mul through the signed-width kernels.
    simd_native_case<int8_t, ElementwiseOp::Add>("i8 add", &ew::add_i8_range_avx512, &ew::add_i8_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int8_t, ElementwiseOp::Sub>("i8 sub", &ew::sub_i8_range_avx512, &ew::sub_i8_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int8_t, ElementwiseOp::Mul>("i8 mul", &ew::mul_i8_range_avx512, &ew::mul_i8_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int16_t, ElementwiseOp::Add>("i16 add", &ew::add_i16_range_avx512, &ew::add_i16_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int16_t, ElementwiseOp::Sub>("i16 sub", &ew::sub_i16_range_avx512, &ew::sub_i16_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int16_t, ElementwiseOp::Mul>("i16 mul", &ew::mul_i16_range_avx512, &ew::mul_i16_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int32_t, ElementwiseOp::Add>("i32 add", &ew::add_i32_range_avx512, &ew::add_i32_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int32_t, ElementwiseOp::Sub>("i32 sub", &ew::sub_i32_range_avx512, &ew::sub_i32_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int32_t, ElementwiseOp::Mul>("i32 mul", &ew::mul_i32_range_avx512, &ew::mul_i32_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int64_t, ElementwiseOp::Add>("i64 add", &ew::add_i64_range_avx512, &ew::add_i64_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int64_t, ElementwiseOp::Sub>("i64 sub", &ew::sub_i64_range_avx512, &ew::sub_i64_range_avx2, simd_int_pool(), skipped512, skipped2);
    simd_native_case<int64_t, ElementwiseOp::Mul>("i64 mul", &ew::mul_i64_range_avx512, &ew::mul_i64_range_avx2, simd_int_pool(), skipped512, skipped2);
    // Half lanes (fp16/bf16 AND their packed twins share these kernels).
    simd_half_case<float16_t, ElementwiseOp::Add>("f16 add", &ew::add_f16_range_avx512, &ew::add_f16_range_avx2, simd_f16_pool(), skipped512, skipped2);
    simd_half_case<float16_t, ElementwiseOp::Sub>("f16 sub", &ew::sub_f16_range_avx512, &ew::sub_f16_range_avx2, simd_f16_pool(), skipped512, skipped2);
    simd_half_case<float16_t, ElementwiseOp::Mul>("f16 mul", &ew::mul_f16_range_avx512, &ew::mul_f16_range_avx2, simd_f16_pool(), skipped512, skipped2);
    simd_half_case<float16_t, ElementwiseOp::Div>("f16 div", &ew::div_f16_range_avx512, &ew::div_f16_range_avx2, simd_f16_pool(), skipped512, skipped2);
    simd_half_case<bfloat16_t, ElementwiseOp::Add>("bf16 add", &ew::add_bf16_range_avx512, &ew::add_bf16_range_avx2, simd_bf16_pool(), skipped512, skipped2);
    simd_half_case<bfloat16_t, ElementwiseOp::Sub>("bf16 sub", &ew::sub_bf16_range_avx512, &ew::sub_bf16_range_avx2, simd_bf16_pool(), skipped512, skipped2);
    simd_half_case<bfloat16_t, ElementwiseOp::Mul>("bf16 mul", &ew::mul_bf16_range_avx512, &ew::mul_bf16_range_avx2, simd_bf16_pool(), skipped512, skipped2);
    simd_half_case<bfloat16_t, ElementwiseOp::Div>("bf16 div", &ew::div_bf16_range_avx512, &ew::div_bf16_range_avx2, simd_bf16_pool(), skipped512, skipped2);

    // On a CPU that reports the feature, the kernels MUST have run — a silent
    // skip on a capable machine is a failure, not a skip. The probes are
    // authoritative: they are defined in the kernel TUs (per-file -m flags).
    if (ew::elementwise_avx512_supported()) {
        CHECK(!skipped512);
        std::cout << "AVX512 elementwise kernels executed and bit-matched the scalar path\n";
    } else {
        std::cout << "SKIP: AVX512 elementwise kernels (CPU lacks the feature)\n";
    }
    if (ew::elementwise_avx2_supported()) {
        CHECK(!skipped2);
        std::cout << "AVX2 elementwise kernels executed and bit-matched the scalar path\n";
    } else {
        std::cout << "SKIP: AVX2 elementwise kernels (CPU lacks the feature)\n";
    }
}

// End-to-end dispatch checks: same-shape contiguous CPU tensors route through
// the SIMD contiguous path (plan.exact) and bit-match the scalar references.

template<typename T, ElementwiseOp Op>
static void check_contig_op_bits(const char* const opname, const size_t n, Tensor& r, Tensor& a, Tensor& b) {
    const auto shape = std::vector{n};
    for (size_t i = 0; i < n; ++i) {
        const auto rv = get_flat<T>(r, shape, i);
        const auto ev = simd_ref_op<T, Op>(get_flat<T>(a, shape, i), get_flat<T>(b, shape, i));
        bool same;
        if constexpr (std::is_same_v<T, float32_t>) {
            same = idet::bit_cast_memcpy<uint32_t>(rv) == idet::bit_cast_memcpy<uint32_t>(ev);
        } else if constexpr (std::is_same_v<T, float64_t>) {
            same = idet::bit_cast_memcpy<uint64_t>(rv) == idet::bit_cast_memcpy<uint64_t>(ev);
        } else {
            same = rv == ev;
        }
        if (!same) {
            ++failures;
            std::cerr << "FAIL: contig " << opname << " bit mismatch at [" << i << "] of " << n << "\n";
            return;
        }
    }
}

template<typename T>
static void run_contig_case(const char* const name, const DType dt, const std::vector<uint64_t>& pool, const size_t n) {
    auto a = Tensor({n}, dt, Device::CPU);
    auto b = Tensor({n}, dt, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        a.set<T>({i}, (i % 2 == 0) ? bits_to<T>(pool[i % pool.size()]) : bits_to<T>(simd_rand()));
        b.set<T>({i}, (i % 3 == 0) ? bits_to<T>(pool[(i + 5) % pool.size()]) : bits_to<T>(simd_rand()));
    }
    auto r_add = apply_op(a, b, ElementwiseOp::Add);
    check_contig_op_bits<T, ElementwiseOp::Add>((std::string(name) + " add").c_str(), n, r_add, a, b);
    auto r_sub = apply_op(a, b, ElementwiseOp::Sub);
    check_contig_op_bits<T, ElementwiseOp::Sub>((std::string(name) + " sub").c_str(), n, r_sub, a, b);
    auto r_mul = apply_op(a, b, ElementwiseOp::Mul);
    check_contig_op_bits<T, ElementwiseOp::Mul>((std::string(name) + " mul").c_str(), n, r_mul, a, b);
    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
        auto r_div = apply_op(a, b, ElementwiseOp::Div);
        check_contig_op_bits<T, ElementwiseOp::Div>((std::string(name) + " div").c_str(), n, r_div, a, b);
    }
}

static void test_simd_contig_dispatch_bits() {
    const auto sizes = std::vector<size_t>{1, 7, 31, 37, 4096};
    for (const size_t n : sizes) {
        run_contig_case<float32_t>("f32", DType::Float32, simd_f32_pool(), n);
        run_contig_case<float64_t>("f64", DType::Float64, simd_f64_pool(), n);
        // Unsigned twins: the dispatch reinterprets onto the signed-width kernels.
        run_contig_case<uint8_t>("u8", DType::UInt8, simd_int_pool(), n);
        run_contig_case<uint16_t>("u16", DType::UInt16, simd_int_pool(), n);
        run_contig_case<uint32_t>("u32", DType::UInt32, simd_int_pool(), n);
        run_contig_case<uint64_t>("u64", DType::UInt64, simd_int_pool(), n);
    }
}

template<typename P, typename H, ElementwiseOp Op>
static void check_packed_op_bits(const char* const opname, const size_t n, Tensor& r, Tensor& a, Tensor& b) {
    const auto shape = std::vector{n};
    for (size_t i = 0; i < n; ++i) {
        const auto abits = idet::bit_cast_memcpy<uint32_t>(get_flat<P>(a, shape, i));
        const auto bbits = idet::bit_cast_memcpy<uint32_t>(get_flat<P>(b, shape, i));
        const auto rbits = idet::bit_cast_memcpy<uint32_t>(get_flat<P>(r, shape, i));
        const auto exp_lo = static_cast<uint32_t>(simd_ref_lane_op<H, Op>(static_cast<uint16_t>(abits), static_cast<uint16_t>(bbits)));
        const auto exp_hi = static_cast<uint32_t>(simd_ref_lane_op<H, Op>(static_cast<uint16_t>(abits >> 16), static_cast<uint16_t>(bbits >> 16)));
        const auto expected = static_cast<uint32_t>((exp_lo & 0xFFFFu) | (exp_hi << 16));
        if (rbits != expected) {
            ++failures;
            std::cerr << "FAIL: packed " << opname << " bit mismatch at [" << i << "] of " << n << "\n";
            return;
        }
    }
}

template<typename P, typename H>
static void run_packed_contig_case(const char* const name, const DType dt, const size_t n) {
    const auto exp_mask = std::is_same_v<H, float16_t> ? uint16_t{0x7C00} : uint16_t{0x7F80};
    const auto mant_mask = std::is_same_v<H, float16_t> ? uint16_t{0x03FF} : uint16_t{0x007F};
    auto a = Tensor({n}, dt, Device::CPU);
    auto b = Tensor({n}, dt, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        const auto lo = no_nan_lane(static_cast<uint16_t>(simd_rand()), exp_mask, mant_mask);
        const auto hi = no_nan_lane(static_cast<uint16_t>(simd_rand()), exp_mask, mant_mask);
        a.set<P>({i}, idet::bit_cast_memcpy<P>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16)));
        const auto blo = no_nan_lane(static_cast<uint16_t>(simd_rand()), exp_mask, mant_mask);
        const auto bhi = no_nan_lane(static_cast<uint16_t>(simd_rand()), exp_mask, mant_mask);
        b.set<P>({i}, idet::bit_cast_memcpy<P>(static_cast<uint32_t>(blo) | (static_cast<uint32_t>(bhi) << 16)));
    }
    auto r_add = apply_op(a, b, ElementwiseOp::Add);
    check_packed_op_bits<P, H, ElementwiseOp::Add>((std::string(name) + " add").c_str(), n, r_add, a, b);
    auto r_sub = apply_op(a, b, ElementwiseOp::Sub);
    check_packed_op_bits<P, H, ElementwiseOp::Sub>((std::string(name) + " sub").c_str(), n, r_sub, a, b);
    auto r_mul = apply_op(a, b, ElementwiseOp::Mul);
    check_packed_op_bits<P, H, ElementwiseOp::Mul>((std::string(name) + " mul").c_str(), n, r_mul, a, b);
    auto r_div = apply_op(a, b, ElementwiseOp::Div);
    check_packed_op_bits<P, H, ElementwiseOp::Div>((std::string(name) + " div").c_str(), n, r_div, a, b);
}

static void test_simd_packed_contig_dispatch() {
    // 40000 packed elements (80000 lanes) is past the thread-pool inline
    // threshold, so chunked ranges and their lane doubling are exercised too.
    const auto sizes = std::vector<size_t>{1, 7, 37, 40000};
    for (const size_t n : sizes) {
        run_packed_contig_case<float16x2_t, float16_t>("f16x2", DType::Float16x2, n);
        run_packed_contig_case<bfloat16x2_t, bfloat16_t>("bf16x2", DType::BFloat16x2, n);
    }
}

// NaN payloads propagated through IEEE arithmetic are compiler-dependent (LLVM
// reorders commutative FP operands at -O3; GCC at -O0 does not — the HIP and
// CPU builds of the same scalar path select different payloads), so a NaN on
// either side only requires a NaN on the other. Everything else is bitwise.
static bool lane_is_nan(const uint16_t bits, const uint16_t exp_mask, const uint16_t mant_mask) {
    return (bits & exp_mask) == exp_mask && (bits & mant_mask) != 0;
}

static bool nan_relaxed_bits_equal(const DType dt, Tensor& x, Tensor& y,
                                   const std::vector<size_t>& shape, const size_t i) {
    switch (dt) {
        case DType::Float16: {
            const auto xb = lane_bits(get_flat<float16_t>(x, shape, i));
            const auto yb = lane_bits(get_flat<float16_t>(y, shape, i));
            if (lane_is_nan(xb, 0x7C00, 0x03FF) || lane_is_nan(yb, 0x7C00, 0x03FF)) {
                return lane_is_nan(xb, 0x7C00, 0x03FF) && lane_is_nan(yb, 0x7C00, 0x03FF);
            }
            return xb == yb;
        }
        case DType::BFloat16: {
            const auto xb = lane_bits(get_flat<bfloat16_t>(x, shape, i));
            const auto yb = lane_bits(get_flat<bfloat16_t>(y, shape, i));
            if (lane_is_nan(xb, 0x7F80, 0x007F) || lane_is_nan(yb, 0x7F80, 0x007F)) {
                return lane_is_nan(xb, 0x7F80, 0x007F) && lane_is_nan(yb, 0x7F80, 0x007F);
            }
            return xb == yb;
        }
        case DType::Float16x2:
        case DType::BFloat16x2: {
            const auto exp_mask = dt == DType::Float16x2 ? uint16_t{0x7C00} : uint16_t{0x7F80};
            const auto mant_mask = dt == DType::Float16x2 ? uint16_t{0x03FF} : uint16_t{0x007F};
            const auto xb = idet::bit_cast_memcpy<uint32_t>(get_flat<float16x2_t>(x, shape, i));
            const auto yb = idet::bit_cast_memcpy<uint32_t>(get_flat<float16x2_t>(y, shape, i));
            for (const auto lane : {uint16_t{0}, uint16_t{16}}) {
                const auto xn = lane_is_nan(static_cast<uint16_t>(xb >> lane), exp_mask, mant_mask);
                const auto yn = lane_is_nan(static_cast<uint16_t>(yb >> lane), exp_mask, mant_mask);
                if (xn || yn) {
                    if (xn != yn) {
                        return false;
                    }
                } else if (static_cast<uint16_t>(xb >> lane) != static_cast<uint16_t>(yb >> lane)) {
                    return false;
                }
            }
            return true;
        }
        case DType::Float32: {
            const auto xb = idet::bit_cast_memcpy<uint32_t>(get_flat<float32_t>(x, shape, i));
            const auto yb = idet::bit_cast_memcpy<uint32_t>(get_flat<float32_t>(y, shape, i));
            const auto xn = (xb & 0x7F800000u) == 0x7F800000u && (xb & 0x007FFFFFu) != 0;
            const auto yn = (yb & 0x7F800000u) == 0x7F800000u && (yb & 0x007FFFFFu) != 0;
            if (xn || yn) {
                return xn && yn;
            }
            return xb == yb;
        }
        case DType::Float64: {
            const auto xb = idet::bit_cast_memcpy<uint64_t>(get_flat<float64_t>(x, shape, i));
            const auto yb = idet::bit_cast_memcpy<uint64_t>(get_flat<float64_t>(y, shape, i));
            const auto xn = (xb & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (xb & 0x000FFFFFFFFFFFFFull) != 0;
            const auto yn = (yb & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (yb & 0x000FFFFFFFFFFFFFull) != 0;
            if (xn || yn) {
                return xn && yn;
            }
            return xb == yb;
        }
        [[unlikely]] default:
            return elem_bits_equal(dt, x, y, shape, i);
    }
}

// NaN payload parity, end to end: the same NaN-laden left operand through the
// SIMD contiguous path (exact shape) and through the library's real scalar
// path (a shape-{1} right operand forces plan.exact = false), compared
// bitwise except across NaN payloads (compiler-dependent, see
// nan_relaxed_bits_equal). The right operand cycles through 1.0 / +Inf / QNaN
// to cover NaN-vs-value, NaN-vs-Inf and NaN-vs-NaN selections.
template<typename T>
static void check_nan_cross(const DType dt, const std::vector<uint64_t>& pool,
                            const std::vector<uint64_t>& b_choices, const size_t n) {
    auto a = Tensor({n}, dt, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        a.set<T>({i}, bits_to<T>(pool[i % pool.size()]));
    }
    const auto shape = std::vector{n};
    for (const uint64_t b_bits : b_choices) {
        auto b = Tensor({n}, dt, Device::CPU);
        auto b_bcast = Tensor({1}, dt, Device::CPU);
        for (size_t i = 0; i < n; ++i) {
            b.set<T>({i}, bits_to<T>(b_bits));
        }
        b_bcast.set<T>({0}, bits_to<T>(b_bits));
        for (const auto op : parity_ops()) {
            auto r_simd = apply_op(a, b, op);
            auto r_scalar = apply_op(a, b_bcast, op);
            for (size_t i = 0; i < n; ++i) {
                if (!nan_relaxed_bits_equal(dt, r_simd, r_scalar, shape, i)) {
                    ++failures;
                    std::cerr << "FAIL: NaN cross " << op_name(op) << " dtype="
                              << BeanTensor::DataTypes::dtype_to_string(dt)
                              << " mismatch at [" << i << "] of " << n << "\n";
                    break;
                }
            }
        }
    }
}

template<typename P, typename H>
static void check_nan_cross_packed(const DType dt, const std::vector<uint64_t>& pool,
                                   const std::vector<uint32_t>& b_lanes, const size_t n) {
    auto a = Tensor({n}, dt, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        const auto lo = static_cast<uint32_t>(pool[i % pool.size()]);
        const auto hi = static_cast<uint32_t>(pool[(i + 9) % pool.size()]);
        a.set<P>({i}, idet::bit_cast_memcpy<P>(static_cast<uint32_t>((lo & 0xFFFFu) | (hi << 16))));
    }
    const auto shape = std::vector{n};
    for (const uint32_t vb : b_lanes) {
        const auto both = static_cast<uint32_t>((vb & 0xFFFFu) | (static_cast<uint32_t>(vb) << 16));
        auto b = Tensor({n}, dt, Device::CPU);
        auto b_bcast = Tensor({1}, dt, Device::CPU);
        for (size_t i = 0; i < n; ++i) {
            b.set<P>({i}, idet::bit_cast_memcpy<P>(both));
        }
        b_bcast.set<P>({0}, idet::bit_cast_memcpy<P>(both));
        for (const auto op : parity_ops()) {
            auto r_simd = apply_op(a, b, op);
            auto r_scalar = apply_op(a, b_bcast, op);
            for (size_t i = 0; i < n; ++i) {
                if (!nan_relaxed_bits_equal(dt, r_simd, r_scalar, shape, i)) {
                    ++failures;
                    std::cerr << "FAIL: NaN cross (packed) " << op_name(op) << " dtype="
                              << BeanTensor::DataTypes::dtype_to_string(dt)
                              << " mismatch at [" << i << "] of " << n << "\n";
                    break;
                }
            }
        }
    }
}

static void test_simd_nan_parity_e2e() {
    const auto sizes = std::vector<size_t>{7, 31, 37, 4096};
    for (const size_t n : sizes) {
        check_nan_cross<float32_t>(DType::Float32, simd_f32_pool(),
            std::vector<uint64_t>{0x3F800000ull, 0x7F800000ull, 0x7FC00000ull}, n);
        check_nan_cross<float64_t>(DType::Float64, simd_f64_pool(),
            std::vector<uint64_t>{0x3FF0000000000000ull, 0x7FF0000000000000ull, 0x7FF8000000000000ull}, n);
        check_nan_cross<float16_t>(DType::Float16, simd_f16_pool(),
            std::vector<uint64_t>{0x3C00, 0x7C00, 0x7E00}, n);
        check_nan_cross<bfloat16_t>(DType::BFloat16, simd_bf16_pool(),
            std::vector<uint64_t>{0x3F80, 0x7F80, 0x7FC0}, n);
        check_nan_cross_packed<float16x2_t, float16_t>(DType::Float16x2, simd_f16_pool(),
            std::vector<uint32_t>{0x3C00, 0x7C00, 0x7E00}, n);
        check_nan_cross_packed<bfloat16x2_t, bfloat16_t>(DType::BFloat16x2, simd_bf16_pool(),
            std::vector<uint32_t>{0x3F80, 0x7F80, 0x7FC0}, n);
    }
}

// ==================== Phase 2: splat / row-decomposition / int4x2 ====================
// Direct splat-kernel parity in the established probe style. Lhs selects the
// scalar-on-the-left kernels (broadcast rows with operand A on inner stride 0);
// the reference flips the operand order to match. NaN scalars stay out of the
// direct comparison (payload selection is compiler-dependent, same reasoning as
// the binary-kernel fills); +-Inf/0/clamp sentinels cover the stable edges.
// The e2e NaN cross test below exercises the SPLAT dispatch path (a shape-{1}
// right operand is a broadcast plan) with the nan_relaxed comparator.

// Scalar sentinels per dtype family: 0, -0/-1, +Inf, large-magnitude, and the
// fp16 clamp boundary (reinterpreted per dtype through bits_to<T>).
static const std::vector<uint64_t>& splat_scalar_f32() {
    static const auto s = std::vector<uint64_t>{
        0x00000000ull, 0x80000000ull, 0xBF800000ull, 0x7F800000ull, 0xC69C4000ull, 0x477FE000ull,
    };
    return s;
}

static const std::vector<uint64_t>& splat_scalar_f64() {
    static const auto s = std::vector<uint64_t>{
        0x0000000000000000ull, 0x8000000000000000ull, 0xBFF0000000000000ull,
        0x7FF0000000000000ull, 0x40F86A0000000000ull, 0xC0F86A0000000000ull,
    };
    return s;
}

static const std::vector<uint64_t>& splat_scalar_f16() {
    static const auto s = std::vector<uint64_t>{
        0x0000, 0x8000, 0xBC00, 0x7C00, 0x7BFF, 0xFBFF, 0x1400,
    };
    return s;
}

static const std::vector<uint64_t>& splat_scalar_bf16() {
    static const auto s = std::vector<uint64_t>{
        0x0000, 0x8000, 0xBF80, 0x7F80, 0x7F7F, 0x0001, 0x8080,
    };
    return s;
}

static const std::vector<uint64_t>& splat_scalar_int() {
    static const auto s = std::vector<uint64_t>{
        0x0000000000000000ull, 0x0000000000000001ull, 0xFFFFFFFFFFFFFFFFull,
        0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
    };
    return s;
}

// One (dtype, op) splat case against both ISAs' kernels, cycling the scalar
// through its sentinel set. A thrown CPUIllegalInstruction marks that ISA
// skipped (the CPU lacks the feature).
template<typename T, ElementwiseOp Op, bool Lhs>
static void simd_splat_native_case(const char* const label,
                                   void(*const k512)(T*, const T*, T, size_t, size_t),
                                   void(*const k2)(T*, const T*, T, size_t, size_t),
                                   const std::vector<uint64_t>& pool,
                                   const std::vector<uint64_t>& scalars,
                                   bool& skipped512,
                                   bool& skipped2) {
    for (const size_t n : simd_sizes()) {
        auto a = std::vector<T>(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = simd_no_nan((i % 2 == 0) ? bits_to<T>(pool[i % pool.size()]) : bits_to<T>(simd_rand()));
        }
        for (const uint64_t sbits : scalars) {
            const auto s = bits_to<T>(sbits);
            auto ref = std::vector<T>(n);
            for (size_t i = 0; i < n; ++i) {
                ref[i] = Lhs ? simd_ref_op<T, Op>(s, a[i]) : simd_ref_op<T, Op>(a[i], s);
            }
            auto got = std::vector<T>(n);
            try {
                k512(got.data(), a.data(), s, 0, n);
                check_bitmatch((std::string(label) + " [avx512]").c_str(), got, ref);
            } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
                skipped512 = true;
            }
            try {
                k2(got.data(), a.data(), s, 0, n);
                check_bitmatch((std::string(label) + " [avx2]").c_str(), got, ref);
            } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
                skipped2 = true;
            }
        }
    }
}

// One (half kind, op) splat case against both ISAs' flat-lane kernels.
template<typename H, ElementwiseOp Op, bool Lhs>
static void simd_splat_half_case(const char* const label,
                                 void(*const k512)(uint16_t*, const uint16_t*, uint16_t, size_t, size_t),
                                 void(*const k2)(uint16_t*, const uint16_t*, uint16_t, size_t, size_t),
                                 const std::vector<uint64_t>& pool,
                                 const std::vector<uint64_t>& scalars,
                                 bool& skipped512,
                                 bool& skipped2) {
    const auto exp_mask = std::is_same_v<H, float16_t> ? uint16_t{0x7C00} : uint16_t{0x7F80};
    const auto mant_mask = std::is_same_v<H, float16_t> ? uint16_t{0x03FF} : uint16_t{0x007F};
    for (const size_t n : simd_sizes()) {
        auto a = std::vector<uint16_t>(n);
        for (size_t i = 0; i < n; ++i) {
            const auto a_bits = (i % 2 == 0) ? static_cast<uint16_t>(pool[i % pool.size()]) : static_cast<uint16_t>(simd_rand());
            a[i] = no_nan_lane(a_bits, exp_mask, mant_mask);
        }
        for (const uint64_t sbits : scalars) {
            const auto s = static_cast<uint16_t>(sbits);
            auto ref = std::vector<uint16_t>(n);
            for (size_t i = 0; i < n; ++i) {
                ref[i] = Lhs ? simd_ref_lane_op<H, Op>(s, a[i]) : simd_ref_lane_op<H, Op>(a[i], s);
            }
            auto got = std::vector<uint16_t>(n);
            try {
                k512(got.data(), a.data(), s, 0, n);
                check_bitmatch((std::string(label) + " [avx512]").c_str(), got, ref);
            } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
                skipped512 = true;
            }
            try {
                k2(got.data(), a.data(), s, 0, n);
                check_bitmatch((std::string(label) + " [avx2]").c_str(), got, ref);
            } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
                skipped2 = true;
            }
        }
    }
}

// Scalar reference mirroring the scalar apply_int4x2: the signed per-nibble
// wrap is reproduced by unsigned-nibble arithmetic masked to 4 bits (the low
// bits of x op y depend only on the low bits of the operands).
template<ElementwiseOp Op>
static uint8_t i4x2_ref_op(const uint8_t a, const uint8_t b) {
    const auto alo = static_cast<uint32_t>(a & 0x0Fu);
    const auto ahi = static_cast<uint32_t>(a >> 4);
    const auto blo = static_cast<uint32_t>(b & 0x0Fu);
    const auto bhi = static_cast<uint32_t>(b >> 4);
    uint32_t rlo;
    uint32_t rhi;
    if constexpr (Op == ElementwiseOp::Add) {
        rlo = alo + blo;
        rhi = ahi + bhi;
    } else if constexpr (Op == ElementwiseOp::Sub) {
        rlo = alo - blo;
        rhi = ahi - bhi;
    } else {
        rlo = alo * blo;
        rhi = ahi * bhi;
    }
    return static_cast<uint8_t>((rlo & 0x0Fu) | ((rhi & 0x0Fu) << 4));
}

// One int4x2 (op) case against both ISAs' per-nibble kernels: sentinel bytes
// (0x00/0xFF/0x88/0x7F/0x01/alternating) mixed with random fills at odd sizes.
template<ElementwiseOp Op>
static void simd_i4x2_case(const char* const label,
                           void(*const k512)(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t),
                           void(*const k2)(uint8_t*, const uint8_t*, const uint8_t*, size_t, size_t),
                           bool& skipped512,
                           bool& skipped2) {
    static const auto byte_pool = std::vector<uint8_t>{0x00, 0xFF, 0x88, 0x7F, 0x01, 0xF0, 0x0F, 0x55, 0xAA};
    const auto sizes = std::vector<size_t>{1, 7, 31, 37, 4097};
    for (const size_t n : sizes) {
        auto a = std::vector<uint8_t>(n);
        auto b = std::vector<uint8_t>(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = (i % 2 == 0) ? byte_pool[i % byte_pool.size()] : static_cast<uint8_t>(simd_rand());
            b[i] = (i % 3 == 0) ? byte_pool[(i + 3) % byte_pool.size()] : static_cast<uint8_t>(simd_rand());
        }
        auto ref = std::vector<uint8_t>(n);
        for (size_t i = 0; i < n; ++i) {
            ref[i] = i4x2_ref_op<Op>(a[i], b[i]);
        }
        auto got = std::vector<uint8_t>(n);
        try {
            k512(got.data(), a.data(), b.data(), 0, n);
            check_bitmatch((std::string(label) + " [avx512]").c_str(), got, ref);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            skipped512 = true;
        }
        try {
            k2(got.data(), a.data(), b.data(), 0, n);
            check_bitmatch((std::string(label) + " [avx2]").c_str(), got, ref);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            skipped2 = true;
        }
    }
}

static void test_simd_splat_kernel_parity() {
    auto skipped512 = false;
    auto skipped2 = false;

    // Plain splats (scalar on the right — the tensor-scalar API order).
    simd_splat_native_case<float32_t, ElementwiseOp::Add, false>("f32 splat add", &ew::add_f32_splat_avx512, &ew::add_f32_splat_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float32_t, ElementwiseOp::Sub, false>("f32 splat sub", &ew::sub_f32_splat_avx512, &ew::sub_f32_splat_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float32_t, ElementwiseOp::Mul, false>("f32 splat mul", &ew::mul_f32_splat_avx512, &ew::mul_f32_splat_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float32_t, ElementwiseOp::Div, false>("f32 splat div", &ew::div_f32_splat_avx512, &ew::div_f32_splat_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Add, false>("f64 splat add", &ew::add_f64_splat_avx512, &ew::add_f64_splat_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Sub, false>("f64 splat sub", &ew::sub_f64_splat_avx512, &ew::sub_f64_splat_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Mul, false>("f64 splat mul", &ew::mul_f64_splat_avx512, &ew::mul_f64_splat_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Div, false>("f64 splat div", &ew::div_f64_splat_avx512, &ew::div_f64_splat_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<int8_t, ElementwiseOp::Add, false>("i8 splat add", &ew::add_i8_splat_avx512, &ew::add_i8_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int8_t, ElementwiseOp::Sub, false>("i8 splat sub", &ew::sub_i8_splat_avx512, &ew::sub_i8_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int8_t, ElementwiseOp::Mul, false>("i8 splat mul", &ew::mul_i8_splat_avx512, &ew::mul_i8_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int16_t, ElementwiseOp::Add, false>("i16 splat add", &ew::add_i16_splat_avx512, &ew::add_i16_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int16_t, ElementwiseOp::Sub, false>("i16 splat sub", &ew::sub_i16_splat_avx512, &ew::sub_i16_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int16_t, ElementwiseOp::Mul, false>("i16 splat mul", &ew::mul_i16_splat_avx512, &ew::mul_i16_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int32_t, ElementwiseOp::Add, false>("i32 splat add", &ew::add_i32_splat_avx512, &ew::add_i32_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int32_t, ElementwiseOp::Sub, false>("i32 splat sub", &ew::sub_i32_splat_avx512, &ew::sub_i32_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int32_t, ElementwiseOp::Mul, false>("i32 splat mul", &ew::mul_i32_splat_avx512, &ew::mul_i32_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int64_t, ElementwiseOp::Add, false>("i64 splat add", &ew::add_i64_splat_avx512, &ew::add_i64_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int64_t, ElementwiseOp::Sub, false>("i64 splat sub", &ew::sub_i64_splat_avx512, &ew::sub_i64_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int64_t, ElementwiseOp::Mul, false>("i64 splat mul", &ew::mul_i64_splat_avx512, &ew::mul_i64_splat_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Add, false>("f16 splat add", &ew::add_f16_splat_avx512, &ew::add_f16_splat_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Sub, false>("f16 splat sub", &ew::sub_f16_splat_avx512, &ew::sub_f16_splat_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Mul, false>("f16 splat mul", &ew::mul_f16_splat_avx512, &ew::mul_f16_splat_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Div, false>("f16 splat div", &ew::div_f16_splat_avx512, &ew::div_f16_splat_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Add, false>("bf16 splat add", &ew::add_bf16_splat_avx512, &ew::add_bf16_splat_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Sub, false>("bf16 splat sub", &ew::sub_bf16_splat_avx512, &ew::sub_bf16_splat_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Mul, false>("bf16 splat mul", &ew::mul_bf16_splat_avx512, &ew::mul_bf16_splat_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Div, false>("bf16 splat div", &ew::div_bf16_splat_avx512, &ew::div_bf16_splat_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);

    // Left splats (scalar on the left — broadcast rows with A on inner stride 0).
    simd_splat_native_case<float32_t, ElementwiseOp::Add, true>("f32 lhs splat add", &ew::add_f32_splat_lhs_avx512, &ew::add_f32_splat_lhs_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float32_t, ElementwiseOp::Sub, true>("f32 lhs splat sub", &ew::sub_f32_splat_lhs_avx512, &ew::sub_f32_splat_lhs_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float32_t, ElementwiseOp::Mul, true>("f32 lhs splat mul", &ew::mul_f32_splat_lhs_avx512, &ew::mul_f32_splat_lhs_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float32_t, ElementwiseOp::Div, true>("f32 lhs splat div", &ew::div_f32_splat_lhs_avx512, &ew::div_f32_splat_lhs_avx2, simd_f32_pool(), splat_scalar_f32(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Add, true>("f64 lhs splat add", &ew::add_f64_splat_lhs_avx512, &ew::add_f64_splat_lhs_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Sub, true>("f64 lhs splat sub", &ew::sub_f64_splat_lhs_avx512, &ew::sub_f64_splat_lhs_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Mul, true>("f64 lhs splat mul", &ew::mul_f64_splat_lhs_avx512, &ew::mul_f64_splat_lhs_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<float64_t, ElementwiseOp::Div, true>("f64 lhs splat div", &ew::div_f64_splat_lhs_avx512, &ew::div_f64_splat_lhs_avx2, simd_f64_pool(), splat_scalar_f64(), skipped512, skipped2);
    simd_splat_native_case<int8_t, ElementwiseOp::Sub, true>("i8 lhs splat sub", &ew::sub_i8_splat_lhs_avx512, &ew::sub_i8_splat_lhs_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int16_t, ElementwiseOp::Sub, true>("i16 lhs splat sub", &ew::sub_i16_splat_lhs_avx512, &ew::sub_i16_splat_lhs_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int32_t, ElementwiseOp::Sub, true>("i32 lhs splat sub", &ew::sub_i32_splat_lhs_avx512, &ew::sub_i32_splat_lhs_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_native_case<int64_t, ElementwiseOp::Sub, true>("i64 lhs splat sub", &ew::sub_i64_splat_lhs_avx512, &ew::sub_i64_splat_lhs_avx2, simd_int_pool(), splat_scalar_int(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Add, true>("f16 lhs splat add", &ew::add_f16_splat_lhs_avx512, &ew::add_f16_splat_lhs_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Sub, true>("f16 lhs splat sub", &ew::sub_f16_splat_lhs_avx512, &ew::sub_f16_splat_lhs_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Mul, true>("f16 lhs splat mul", &ew::mul_f16_splat_lhs_avx512, &ew::mul_f16_splat_lhs_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<float16_t, ElementwiseOp::Div, true>("f16 lhs splat div", &ew::div_f16_splat_lhs_avx512, &ew::div_f16_splat_lhs_avx2, simd_f16_pool(), splat_scalar_f16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Add, true>("bf16 lhs splat add", &ew::add_bf16_splat_lhs_avx512, &ew::add_bf16_splat_lhs_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Sub, true>("bf16 lhs splat sub", &ew::sub_bf16_splat_lhs_avx512, &ew::sub_bf16_splat_lhs_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Mul, true>("bf16 lhs splat mul", &ew::mul_bf16_splat_lhs_avx512, &ew::mul_bf16_splat_lhs_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);
    simd_splat_half_case<bfloat16_t, ElementwiseOp::Div, true>("bf16 lhs splat div", &ew::div_bf16_splat_lhs_avx512, &ew::div_bf16_splat_lhs_avx2, simd_bf16_pool(), splat_scalar_bf16(), skipped512, skipped2);

    if (ew::elementwise_avx512_supported()) {
        CHECK(!skipped512);
        std::cout << "AVX512 elementwise splat kernels executed and bit-matched the scalar path\n";
    } else {
        std::cout << "SKIP: AVX512 elementwise splat kernels (CPU lacks the feature)\n";
    }
    if (ew::elementwise_avx2_supported()) {
        CHECK(!skipped2);
        std::cout << "AVX2 elementwise splat kernels executed and bit-matched the scalar path\n";
    } else {
        std::cout << "SKIP: AVX2 elementwise splat kernels (CPU lacks the feature)\n";
    }
}

static void test_simd_int4x2_kernel_parity() {
    auto skipped512 = false;
    auto skipped2 = false;
    simd_i4x2_case<ElementwiseOp::Add>("i4x2 add", &ew::add_i4x2_range_avx512, &ew::add_i4x2_range_avx2, skipped512, skipped2);
    simd_i4x2_case<ElementwiseOp::Sub>("i4x2 sub", &ew::sub_i4x2_range_avx512, &ew::sub_i4x2_range_avx2, skipped512, skipped2);
    simd_i4x2_case<ElementwiseOp::Mul>("i4x2 mul", &ew::mul_i4x2_range_avx512, &ew::mul_i4x2_range_avx2, skipped512, skipped2);
    if (ew::elementwise_avx512_supported()) {
        CHECK(!skipped512);
        std::cout << "AVX512 int4x2 kernels executed and bit-matched the scalar path\n";
    } else {
        std::cout << "SKIP: AVX512 int4x2 kernels (CPU lacks the feature)\n";
    }
    if (ew::elementwise_avx2_supported()) {
        CHECK(!skipped2);
        std::cout << "AVX2 int4x2 kernels executed and bit-matched the scalar path\n";
    } else {
        std::cout << "SKIP: AVX2 int4x2 kernels (CPU lacks the feature)\n";
    }
}

// End-to-end tensor-scalar checks through the new splat dispatch: odd sizes
// past every vector width, manually computed expectations, narrowing/clamp
// semantics from test_scalar_narrowing now exercising the SIMD path.
static void test_simd_splat_dispatch_e2e() {
    constexpr size_t n = 4099; // odd, > all vector widths, below the pool inline threshold
    const auto positions = std::vector<size_t>{0, 1, 2048, 4097, 4098};
    // fp32 ramp: mul/add/sub/div against exact expectations.
    auto a = Tensor({n}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        a.set<float32_t>({i}, static_cast<float32_t>(static_cast<int32_t>(i % 13) - 6));
    }
    auto mul = a.mul(3.0);
    auto add = a.add(0.5);
    auto sub = a.sub(-2.25);
    auto div = a.div(0.5);
    for (const size_t i : positions) {
        const auto v = static_cast<float32_t>(static_cast<int32_t>(i % 13) - 6);
        CHECK(near(mul.at<float32_t>({i}), v * 3.0f, 1e-6));
        CHECK(near(add.at<float32_t>({i}), v + 0.5f, 1e-6));
        CHECK(near(sub.at<float32_t>({i}), v + 2.25f, 1e-6));
        CHECK(near(div.at<float32_t>({i}), v * 2.0f, 1e-6));
    }
    // fp64 scalar op in the storage width.
    auto a64 = Tensor({n}, DType::Float64, Device::CPU);
    a64.set<float64_t>({0}, 1.5);
    a64.set<float64_t>({n - 1}, -0.25);
    auto m64 = a64.mul(-2.0);
    CHECK(near64(m64.at<float64_t>({0}), -3.0, 1e-12));
    CHECK(near64(m64.at<float64_t>({n - 1}), 0.5, 1e-12));
    // Integer narrowing: truncation, clamping, NaN -> 0 — now via splat kernels.
    auto ai = Tensor({n}, DType::Int32, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        ai.set<int32_t>({i}, static_cast<int32_t>(i % 7) - 3);
    }
    auto ai_add = ai.add(2.9);
    auto ai_nan = ai.add(std::numeric_limits<float64_t>::quiet_NaN());
    auto au = Tensor({n}, DType::UInt8, Device::CPU);
    au.set<uint8_t>({0}, uint8_t{3});
    au.set<uint8_t>({n - 1}, uint8_t{200});
    auto au_clamp = au.add(1000.0);
    for (const size_t i : positions) {
        const auto v = static_cast<int32_t>(i % 7) - 3;
        CHECK(ai_add.at<int32_t>({i}) == v + 2);
        CHECK(ai_nan.at<int32_t>({i}) == v);
    }
    // The scalar clamps to 255 first, then the op wraps: 3+255 = 258 -> 2,
    // 200+255 = 455 -> 199.
    CHECK(au_clamp.at<uint8_t>({0}) == uint8_t{2});
    CHECK(au_clamp.at<uint8_t>({n - 1}) == uint8_t{199});
    // fp16 overflow clamps to 65504 through the splat mul; bf16 subnormal flush.
    auto h = Tensor({n}, DType::Float16, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        h.set<float16_t>({i}, narrow_lane<float16_t>(300.0f));
    }
    auto hm = h.mul(300.0);
    CHECK(near(widen_lane(hm.at<float16_t>({0})), 65504.0f, 0.0));
    CHECK(near(widen_lane(hm.at<float16_t>({n - 1})), 65504.0f, 0.0));
    auto bf = Tensor({n}, DType::BFloat16, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        bf.set<bfloat16_t>({i}, narrow_lane<bfloat16_t>(1.0e-20f));
    }
    auto bfm = bf.mul(1.0e-20);
    CHECK(near(widen_lane(bfm.at<bfloat16_t>({0})), 0.0f, 0.0));
    CHECK(near(widen_lane(bfm.at<bfloat16_t>({n - 1})), 0.0f, 0.0));
    // Packed dtypes: the scalar duplicates into both lanes.
    auto pk = Tensor({n}, DType::Float16x2, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        pk.set<float16x2_t>({i}, pack_lanes<float16x2_t, float16_t>(1.0f, 2.0f));
    }
    auto pk2 = pk.add(0.5);
    CHECK(near(unpack_lo(pk2.at<float16x2_t>({0})), 1.5f, 1e-3));
    CHECK(near(unpack_hi(pk2.at<float16x2_t>({n - 1})), 2.5f, 1e-3));
    // int4x2 tensor-scalar stays scalar (per-nibble splat not wired): the wrap
    // semantics still hold across a large tensor.
    auto q = Tensor({n}, DType::Int4x2, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        q.set<int4x2_t>({i}, i4x2(int8_t{7}, int8_t{-3}));
    }
    auto q2 = q.add(2.0);
    CHECK(i4x2_lo(q2.at<int4x2_t>({0})) == int8_t{-7}); // 7 + 2 = 9 -> -7 wrap
    CHECK(i4x2_hi(q2.at<int4x2_t>({n - 1})) == int8_t{-1}); // -3 + 2 = -1
}

// End-to-end broadcast checks through the row-decomposition dispatch: odd inner
// sizes hitting vector tails, both inner-stride-0 operand orders (plain and
// left splats), rank-3/4 plans, the degenerate (0,0) row, integer wrap, and
// the scalar Int4x2 broadcast fallback. All expectations manual.
static void test_broadcast_rows_dispatch() {
    constexpr size_t rows = 5;
    constexpr size_t cols = 37; // odd: every vector kernel takes a scalar/masked tail
    auto a = Tensor({rows, cols}, DType::Float32, Device::CPU);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            a.set<float32_t>({r, c}, static_cast<float32_t>(r * 100 + c));
        }
    }
    // Row-vector broadcast ({5,37} op {37}): rows hit the binary kernel, odd tail.
    auto bias = Tensor({cols}, DType::Float32, Device::CPU);
    for (size_t c = 0; c < cols; ++c) {
        bias.set<float32_t>({c}, static_cast<float32_t>(static_cast<int32_t>(c % 5) - 2));
    }
    auto rb = a.add(bias);
    CHECK(near(rb.at<float32_t>({0, 0}), -2.0f, 1e-6));
    CHECK(near(rb.at<float32_t>({0, 36}), 36.0f - 1.0f, 1e-6)); // 36 + (36%5-2=-1)
    CHECK(near(rb.at<float32_t>({4, 36}), 436.0f - 1.0f, 1e-6));
    // Column broadcast ({5,37} op {5,1} and swapped): splat rows both orders,
    // including sub/div where the operand order must be replayed exactly.
    auto col = Tensor({rows, 1}, DType::Float32, Device::CPU);
    for (size_t r = 0; r < rows; ++r) {
        col.set<float32_t>({r, 0}, static_cast<float32_t>(r + 1));
    }
    auto mulb = a.mul(col);
    CHECK(near(mulb.at<float32_t>({0, 36}), 36.0f * 1.0f, 1e-4));
    CHECK(near(mulb.at<float32_t>({4, 36}), 436.0f * 5.0f, 1e-4));
    auto inv = col.sub(a); // lhs splat sub: (r+1) - a[r,c]
    CHECK(near(inv.at<float32_t>({0, 0}), 1.0f - 0.0f, 1e-6));
    CHECK(near(inv.at<float32_t>({1, 0}), 2.0f - 100.0f, 1e-6));
    auto subb = a.sub(col); // plain splat sub: a[r,c] - (r+1)
    CHECK(near(subb.at<float32_t>({4, 36}), 436.0f - 5.0f, 1e-6));
    auto divb = col.div(a); // lhs splat div: (r+1) / a[r,c]
    CHECK(near(divb.at<float32_t>({1, 0}), 2.0f / 100.0f, 1e-4));
    auto divf = a.div(col); // plain splat div: a[r,c] / (r+1)
    CHECK(near(divf.at<float32_t>({4, 36}), 436.0f / 5.0f, 1e-4));
    // (0,0) degenerate rows: {1,1} against {5,37}.
    auto one = Tensor({1, 1}, DType::Float32, Device::CPU);
    one.set<float32_t>({0, 0}, 4.0f);
    auto fill = a.add(one);
    CHECK(near(fill.at<float32_t>({0, 0}), 4.0f, 1e-6));
    CHECK(near(fill.at<float32_t>({4, 36}), 440.0f, 1e-6));
    // Rank-3 plans: outer-dim broadcast, mid-dim broadcast, inner-dim splat.
    auto c3 = Tensor({2, 3, 5}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < 30; ++i) {
        c3.set<float32_t>({i / 15, i / 5 % 3, i % 5}, static_cast<float32_t>(i));
    }
    auto mid = Tensor({2, 1, 5}, DType::Float32, Device::CPU); // inner (1,1), mid broadcast
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            mid.set<float32_t>({r, 0, c}, static_cast<float32_t>(r * 10 + c));
        }
    }
    auto m3 = c3.add(mid);
    CHECK(near(m3.at<float32_t>({0, 0, 0}), 0.0f + 0.0f, 1e-6));
    CHECK(near(m3.at<float32_t>({0, 2, 4}), 14.0f + 4.0f, 1e-6)); // elem 14 + mid[0,0,4]
    CHECK(near(m3.at<float32_t>({1, 1, 2}), 22.0f + 12.0f, 1e-6)); // elem 22 + mid[1,0,2]
    auto inn = Tensor({2, 3, 1}, DType::Float32, Device::CPU); // inner splat rows
    for (size_t r = 0; r < 2; ++r) {
        for (size_t s = 0; s < 3; ++s) {
            inn.set<float32_t>({r, s, 0}, static_cast<float32_t>(r + s));
        }
    }
    auto i3 = c3.mul(inn);
    CHECK(near(i3.at<float32_t>({0, 1, 4}), 9.0f * 1.0f, 1e-4));  // elem 9 * inn[0,1,0]=1
    CHECK(near(i3.at<float32_t>({1, 2, 3}), 28.0f * 3.0f, 1e-4)); // elem 28 * inn[1,2,0]=3
    // Integer wrap through the broadcast splat rows.
    auto wi = Tensor({rows, cols}, DType::Int32, Device::CPU);
    for (size_t i = 0; i < rows * cols; ++i) {
        wi.set<int32_t>({i / cols, i % cols}, 2000000000);
    }
    auto ci = Tensor({rows, 1}, DType::Int32, Device::CPU);
    for (size_t r = 0; r < rows; ++r) {
        ci.set<int32_t>({r, 0}, 3);
    }
    auto wm = wi.mul(ci); // wrap: 2e9 * 3 mod 2^32 = 1705032704
    CHECK(wm.at<int32_t>({0, 0}) == 1705032704);
    CHECK(wm.at<int32_t>({4, 36}) == 1705032704);
    // int4x2 broadcast rows stay scalar; wrap semantics hold.
    auto qa = Tensor({2, 17}, DType::Int4x2, Device::CPU);
    for (size_t i = 0; i < 34; ++i) {
        qa.set<int4x2_t>({i / 17, i % 17}, i4x2(int8_t{7}, int8_t{-3}));
    }
    auto qb = Tensor({2, 1}, DType::Int4x2, Device::CPU);
    qb.set<int4x2_t>({0, 0}, i4x2(int8_t{1}, int8_t{-1}));
    qb.set<int4x2_t>({1, 0}, i4x2(int8_t{2}, int8_t{0}));
    auto qm = qa.add(qb);
    CHECK(i4x2_lo(qm.at<int4x2_t>({0, 16})) == int8_t{-8});  // 7 + 1 = 8 -> -8
    CHECK(i4x2_hi(qm.at<int4x2_t>({0, 16})) == int8_t{-4});  // -3 + -1 = -4
    CHECK(i4x2_lo(qm.at<int4x2_t>({1, 0})) == int8_t{-7});   // 7 + 2 = 9 -> -7
    CHECK(i4x2_hi(qm.at<int4x2_t>({1, 0})) == int8_t{-3});   // -3 + 0 = -3
    // Packed broadcast rows with DISTINCT lanes per packed element: the row
    // splat must not flatten the (lo, hi) pair into one repeated lane. The
    // column operand packs (1.0, -2.0) and every element of the matrix packs
    // (10.0, 20.0); adding gives (11.0, 18.0) everywhere.
    auto pa = Tensor({3, 29}, DType::Float16x2, Device::CPU);
    for (size_t i = 0; i < 3 * 29; ++i) {
        pa.set<float16x2_t>({i / 29, i % 29}, pack_lanes<float16x2_t, float16_t>(10.0f, 20.0f));
    }
    auto pc = Tensor({3, 1}, DType::Float16x2, Device::CPU);
    for (size_t r = 0; r < 3; ++r) {
        pc.set<float16x2_t>({r, 0}, pack_lanes<float16x2_t, float16_t>(1.0f, -2.0f));
    }
    auto pr = pa.add(pc);
    CHECK(near(unpack_lo(pr.at<float16x2_t>({0, 0})), 11.0f, 1e-3));
    CHECK(near(unpack_hi(pr.at<float16x2_t>({0, 0})), 18.0f, 1e-3));
    CHECK(near(unpack_lo(pr.at<float16x2_t>({2, 28})), 11.0f, 1e-3));
    CHECK(near(unpack_hi(pr.at<float16x2_t>({2, 28})), 18.0f, 1e-3));
    // Same through the tensor-scalar packed path (scalar duplicates into both lanes).
    auto ps = pa.add(-1.0);
    CHECK(near(unpack_lo(ps.at<float16x2_t>({1, 5})), 9.0f, 1e-3));
    CHECK(near(unpack_hi(ps.at<float16x2_t>({1, 5})), 19.0f, 1e-3));
}

int main() {
    test_arith_f32_f64();
    test_arith_ints();
    test_arith_int4x2();
    test_broadcast();
    test_broadcast_errors();
    test_scalar_narrowing();
    test_edge_semantics();
    test_half_and_packed();
    test_threaded_path();
    test_simd_kernel_parity();
    test_simd_contig_dispatch_bits();
    test_simd_packed_contig_dispatch();
    test_simd_nan_parity_e2e();
    test_simd_splat_kernel_parity();
    test_simd_int4x2_kernel_parity();
    test_simd_splat_dispatch_e2e();
    test_broadcast_rows_dispatch();
    if (gpu_available()) {
        test_gpu_arith_f32_f64();
        test_gpu_arith_ints();
        test_gpu_arith_int4x2();
        test_gpu_half_and_packed();
        test_gpu_broadcast();
        test_gpu_scalar_narrowing();
        test_gpu_edge_semantics();
        test_gpu_cpu_parity_exact();
        test_gpu_cpu_parity_broadcast();
        test_gpu_cpu_parity_scalar();
        test_gpu_cpu_parity_large();
    } else {
        std::cerr << "SKIP: no GPU device available\n";
    }
    if (failures == 0) {
        std::cout << "all elementwise tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
