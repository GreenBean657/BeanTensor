/**
 * @file elementwise_tests.cpp
 * @author GreenBean
 * @brief Elementwise arithmetic/broadcasting tests - framework-free assertion suite.
 * Covers add/sub/mul/div/pow (tensor-tensor with NumPy-style broadcasting and
 * tensor-scalar), exp/log, across the full dtype family on CPU: arithmetic
 * correctness, every broadcasting case, scalar narrowing into non-float dtypes,
 * division-by-zero and pow/exp/log edge semantics, and shape/dtype errors.
 * @version 0.1
 * @date 2026-10-22
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/Tensor.h"

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
    const auto av = std::vector<float32_t>{1.5f, -2.25f, 0.0f, 100.0f};
    const auto bv = std::vector<float32_t>{0.25f, 4.0f, -3.0f, 0.5f};
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
        c3.set<float32_t>({i / 6, (i / 3) % 2, i % 3}, static_cast<float32_t>(i + 1));
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
    if (failures == 0) {
        std::cout << "all elementwise tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
