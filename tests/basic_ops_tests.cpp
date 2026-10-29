/**
 * @file basic_ops_tests.cpp
 * @author GreenBean
 * @brief Reduction, optimization and shape ops tests - framework-free assertion suite.
 * Covers reshape, sgd_update/sgd_step, sum and argmax across the full dtype family (CPU threaded + SIMD tiers),
 * and the GPU (CUDA/HIP) paths for all four ops when a backend and device exist.
 * @version 0.3
 * @date 2026-10-26
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Math/Reductions/Sum/SumKernels.h"
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

// --- reshape ---

static void test_reshape_cpu() {
    auto t = Tensor({2, 3}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < 6; ++i) {
        t.set<float32_t>({i / 3, i % 3}, static_cast<float32_t>(i + 1));
    }
    auto view = t.reshape({3, 2});
    CHECK(view.is_same_shape(Tensor({3, 2})));
    // Flat order preserved: view {1,0} is flat index 2 == t {0,2}.
    CHECK(near(view.at<float32_t>({1, 0}), t.at<float32_t>({0, 2}), 1e-6));
    // View aliases the parent's buffer.
    view.set<float32_t>({0, 1}, 42.0f);
    CHECK(near(t.at<float32_t>({0, 1}), 42.0f, 1e-9));

    // numel change is rejected.
    auto threw = false;
    try {
        auto bad = t.reshape({4, 2});
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);

    // Rank beyond BT_TENSOR_MAX_DIMS is rejected.
    threw = false;
    try {
        auto bad = t.reshape({1, 1, 2, 3, 1});
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);

    // Parent death: the view must take its own copy and stay readable.
    auto orphan = Tensor({6}, DType::Float32, Device::CPU);
    {
        auto parent = Tensor({2, 3}, DType::Float32, Device::CPU);
        parent.set_one();
        orphan = parent.reshape({6});
    }
    orphan.sync();
    CHECK(near(orphan.at<float32_t>({5}), 1.0f, 1e-9));
}

// --- sum ---

static void test_sum_cpu() {
    {
        auto t = Tensor({4}, DType::Float32, Device::CPU);
        for (size_t i = 0; i < 4; ++i) {
            t.set<float32_t>({i}, static_cast<float32_t>(i + 1));
        }
        auto s = t.sum();
        CHECK(s.is_same_shape(Tensor({1}, DType::Float32, Device::CPU)));
        CHECK(near(s.at<float32_t>({0}), 10.0f, 1e-6));
    }
    {
        auto t = Tensor({2, 2}, DType::Float64, Device::CPU);
        for (size_t i = 0; i < 4; ++i) {
            t.set<float64_t>({i / 2, i % 2}, static_cast<float64_t>(i + 1) + 0.5);
        }
        auto s = t.sum();
        CHECK(near64(s.at<float64_t>({0}), 12.0, 1e-12));
    }
    // Threaded + SIMD tiers: 100k elements exceeds the inline threshold.
    {
        constexpr auto n = size_t{100000};
        auto t = Tensor({n}, DType::Float32, Device::CPU);
        t.set_one();
        auto s = t.sum();
        CHECK(near(s.at<float32_t>({0}), static_cast<float32_t>(n), 1e-3));
    }
    {
        constexpr auto n = size_t{100000};
        auto t = Tensor({n}, DType::Float64, Device::CPU);
        t.set_one();
        auto s = t.sum();
        CHECK(near64(s.at<float64_t>({0}), static_cast<float64_t>(n), 1e-9));
    }
    // fp16/bf16: fp32 accumulation, RNE narrow on store.
    {
        auto t = Tensor({3}, DType::Float16, Device::CPU);
        for (size_t i = 0; i < 3; ++i) {
            t.set<float16_t>({i}, narrow_lane<float16_t>(static_cast<float32_t>(i + 1)));
        }
        auto s = t.sum();
        CHECK(lane_bits(s.at<float16_t>({0})) == lane_bits(narrow_lane<float16_t>(6.0f)));
    }
    {
        auto t = Tensor({3}, DType::BFloat16, Device::CPU);
        for (size_t i = 0; i < 3; ++i) {
            t.set<bfloat16_t>({i}, narrow_lane<bfloat16_t>(static_cast<float32_t>(i + 1)));
        }
        auto s = t.sum();
        CHECK(lane_bits(s.at<bfloat16_t>({0})) == lane_bits(narrow_lane<bfloat16_t>(6.0f)));
    }
    // Packed: per-lane sums.
    {
        auto t = Tensor({2}, DType::Float16x2, Device::CPU);
        t.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 10.0f));
        t.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(2.0f, 20.0f));
        auto s = t.sum();
        CHECK(near(unpack_lo(s.at<float16x2_t>({0})), 3.0f, 1e-3));
        CHECK(near(unpack_hi(s.at<float16x2_t>({0})), 30.0f, 1e-2));
    }
    {
        auto t = Tensor({2}, DType::BFloat16x2, Device::CPU);
        t.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(1.0f, 10.0f));
        t.set<bfloat16x2_t>({1}, pack_lanes<bfloat16x2_t, bfloat16_t>(2.0f, 20.0f));
        auto s = t.sum();
        CHECK(near(unpack_lo(s.at<bfloat16x2_t>({0})), 3.0f, 1e-2));
        CHECK(near(unpack_hi(s.at<bfloat16x2_t>({0})), 30.0f, 0.5));
    }
    // Integers: uint64 modular accumulate, wrap on store.
    {
        auto t = Tensor({3}, DType::Int8, Device::CPU);
        t.set<int8_t>({0}, int8_t{1});
        t.set<int8_t>({1}, int8_t{2});
        t.set<int8_t>({2}, int8_t{-1});
        auto s = t.sum();
        CHECK(s.at<int8_t>({0}) == int8_t{2});
    }
    {
        // 200 + 100 = 300 wraps to 44 in uint8.
        auto t = Tensor({2}, DType::UInt8, Device::CPU);
        t.set<uint8_t>({0}, uint8_t{200});
        t.set<uint8_t>({1}, uint8_t{100});
        auto s = t.sum();
        CHECK(s.at<uint8_t>({0}) == uint8_t{44});
    }
    {
        auto t = Tensor({2}, DType::Int16, Device::CPU);
        t.set<int16_t>({0}, int16_t{-300});
        t.set<int16_t>({1}, int16_t{1000});
        auto s = t.sum();
        CHECK(s.at<int16_t>({0}) == int16_t{700});
    }
    {
        auto t = Tensor({2}, DType::UInt16, Device::CPU);
        t.set<uint16_t>({0}, uint16_t{60000});
        t.set<uint16_t>({1}, uint16_t{10000});
        auto s = t.sum();
        CHECK(s.at<uint16_t>({0}) == static_cast<uint16_t>(70000));
    }
    {
        auto t = Tensor({2}, DType::Int32, Device::CPU);
        t.set<int32_t>({0}, int32_t{-2000000});
        t.set<int32_t>({1}, int32_t{1000000});
        auto s = t.sum();
        CHECK(s.at<int32_t>({0}) == int32_t{-1000000});
    }
    {
        auto t = Tensor({2}, DType::UInt32, Device::CPU);
        t.set<uint32_t>({0}, uint32_t{4000000000u});
        t.set<uint32_t>({1}, uint32_t{500000000u});
        auto s = t.sum();
        CHECK(s.at<uint32_t>({0}) == static_cast<uint32_t>(4500000000ull));
    }
    {
        auto t = Tensor({2}, DType::Int64, Device::CPU);
        t.set<int64_t>({0}, std::numeric_limits<int64_t>::max());
        t.set<int64_t>({1}, int64_t{1});
        auto s = t.sum();
        CHECK(s.at<int64_t>({0}) == std::numeric_limits<int64_t>::min());
    }
    {
        auto t = Tensor({2}, DType::UInt64, Device::CPU);
        t.set<uint64_t>({0}, std::numeric_limits<uint64_t>::max());
        t.set<uint64_t>({1}, uint64_t{1});
        auto s = t.sum();
        CHECK(s.at<uint64_t>({0}) == uint64_t{0});
    }
    {
        // int4x2: per-lane modular sums. lo: 7+7+7 = 21 -> 5 (mod 16);
        // hi: (-8)*3 -> nibble 8+8+8 = 24 -> 8 (mod 16) == -8.
        auto t = Tensor({3}, DType::Int4x2, Device::CPU);
        t.set<int4x2_t>({0}, i4x2(int8_t{7}, int8_t{-8}));
        t.set<int4x2_t>({1}, i4x2(int8_t{7}, int8_t{-8}));
        t.set<int4x2_t>({2}, i4x2(int8_t{7}, int8_t{-8}));
        auto s = t.sum();
        CHECK(i4x2_lo(s.at<int4x2_t>({0})) == int8_t{5});
        CHECK(i4x2_hi(s.at<int4x2_t>({0})) == int8_t{-8});
    }
}

// --- argmax ---

static void test_argmax_cpu() {
    {
        auto t = Tensor({3}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, 5.0f);
        t.set<float32_t>({2}, 3.0f);
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // Ties: the first strict maximum wins.
        auto t = Tensor({2}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 5.0f);
        t.set<float32_t>({1}, 5.0f);
        CHECK(t.argmax() == uint64_t{0});
    }
    {
        // PyTorch parity: NaN beats every non-NaN; the first NaN wins.
        auto t = Tensor({3}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, std::numeric_limits<float32_t>::quiet_NaN());
        t.set<float32_t>({2}, 5.0f);
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        auto t = Tensor({2}, DType::Float64, Device::CPU);
        t.set<float64_t>({0}, std::numeric_limits<float64_t>::quiet_NaN());
        t.set<float64_t>({1}, std::numeric_limits<float64_t>::quiet_NaN());
        CHECK(t.argmax() == uint64_t{0});
    }
    {
        auto t = Tensor({2}, DType::Float16, Device::CPU);
        t.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
        t.set<float16_t>({1}, narrow_lane<float16_t>(2.0f));
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        auto t = Tensor({2}, DType::BFloat16, Device::CPU);
        t.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(2.0f));
        t.set<bfloat16_t>({1}, narrow_lane<bfloat16_t>(1.0f));
        CHECK(t.argmax() == uint64_t{0});
    }
    {
        // Packed: lexicographic (lo, hi). (1,9) beats (2,3)? No — lo 2 > 1.
        auto t = Tensor({2}, DType::Float16x2, Device::CPU);
        t.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 9.0f));
        t.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(2.0f, 3.0f));
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // Packed: equal lo lanes fall through to hi.
        auto t = Tensor({2}, DType::BFloat16x2, Device::CPU);
        t.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(1.0f, 2.0f));
        t.set<bfloat16x2_t>({1}, pack_lanes<bfloat16x2_t, bfloat16_t>(1.0f, 9.0f));
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // Packed: NaN in the lo lane wins outright.
        auto t = Tensor({2}, DType::Float16x2, Device::CPU);
        t.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(std::numeric_limits<float32_t>::quiet_NaN(), 0.0f));
        t.set<float16x2_t>({1}, pack_lanes<float16x2_t, float16_t>(9.0f, 9.0f));
        CHECK(t.argmax() == uint64_t{0});
    }
    {
        auto t = Tensor({2}, DType::Int8, Device::CPU);
        t.set<int8_t>({0}, int8_t{-5});
        t.set<int8_t>({1}, int8_t{3});
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        auto t = Tensor({2}, DType::UInt8, Device::CPU);
        t.set<uint8_t>({0}, uint8_t{0});
        t.set<uint8_t>({1}, uint8_t{255});
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        auto t = Tensor({2}, DType::Int64, Device::CPU);
        t.set<int64_t>({0}, std::numeric_limits<int64_t>::min());
        t.set<int64_t>({1}, std::numeric_limits<int64_t>::max());
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // Threaded tier with the max in the last chunk.
        constexpr auto n = size_t{100000};
        auto t = Tensor({n}, DType::Int32, Device::CPU);
        t.set_zero();
        t.set<int32_t>({n - 7}, int32_t{1});
        CHECK(t.argmax() == uint64_t{n - 7});
    }
    {
        constexpr auto n = size_t{100000};
        auto t = Tensor({n}, DType::Float32, Device::CPU);
        t.set_zero();
        t.set<float32_t>({12345}, 0.5f);
        CHECK(t.argmax() == uint64_t{12345});
    }
    {
        // int4x2: lexicographic signed (lo, hi); the lo lane dominates.
        auto t = Tensor({2}, DType::Int4x2, Device::CPU);
        t.set<int4x2_t>({0}, i4x2(int8_t{1}, int8_t{7}));
        t.set<int4x2_t>({1}, i4x2(int8_t{2}, int8_t{-8}));
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // int4x2: equal lo lanes fall through to hi.
        auto t = Tensor({2}, DType::Int4x2, Device::CPU);
        t.set<int4x2_t>({0}, i4x2(int8_t{-1}, int8_t{3}));
        t.set<int4x2_t>({1}, i4x2(int8_t{-1}, int8_t{4}));
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // Regression: argmax(axis=1) async path must not use-after-free the
        // axis locals captured by the worker lambdas. out_numel 40000 exceeds
        // threads*1024, so launch_axis queues futures that outlive this frame.
        constexpr auto rows = size_t{40000};
        auto t = Tensor({rows, 2}, DType::Float32, Device::CPU);
        t.set_zero();
        for (auto r = size_t{0}; r < rows; ++r) {
            t.set<float32_t>({r, 1}, 2.0f);
        }
        auto idx = t.argmax(1);
        for (auto r = size_t{0}; r < rows; ++r) {
            CHECK(idx.at<int64_t>({r}) == int64_t{1});
        }
    }
    {
        // Same async-path regression check for an integer dtype.
        constexpr auto rows = size_t{40000};
        auto t = Tensor({rows, 2}, DType::Int32, Device::CPU);
        t.set_zero();
        for (auto r = size_t{0}; r < rows; ++r) {
            t.set<int32_t>({r, 1}, int32_t{100});
        }
        auto idx = t.argmax(1);
        for (auto r = size_t{0}; r < rows; ++r) {
            CHECK(idx.at<int64_t>({r}) == int64_t{1});
        }
    }
}

// --- sgd_update / sgd_step ---

static void test_sgd_cpu() {
    {
        auto w = Tensor({2}, DType::Float32, Device::CPU);
        w.set<float32_t>({0}, 1.0f);
        w.set<float32_t>({1}, 2.0f);
        auto g = Tensor({2}, DType::Float32, Device::CPU);
        g.set<float32_t>({0}, 0.5f);
        g.set<float32_t>({1}, -1.0f);
        w.sgd_update(g, 0.1);
        w.sync();
        CHECK(near(w.at<float32_t>({0}), 0.95f, 1e-7));
        CHECK(near(w.at<float32_t>({1}), 2.1f, 1e-7));
    }
    {
        auto w = Tensor({2}, DType::Float64, Device::CPU);
        w.set<float64_t>({0}, 1.0);
        w.set<float64_t>({1}, 2.0);
        auto g = Tensor({2}, DType::Float64, Device::CPU);
        g.set<float64_t>({0}, 0.5);
        g.set<float64_t>({1}, -1.0);
        w.sgd_update(g, 0.1);
        w.sync();
        CHECK(near64(w.at<float64_t>({0}), 0.95, 1e-15));
        CHECK(near64(w.at<float64_t>({1}), 2.1, 1e-15));
    }
    {
        // Threaded + SIMD tier: every element must update.
        constexpr auto n = size_t{100000};
        auto w = Tensor({n}, DType::Float32, Device::CPU);
        w.set_one();
        auto g = Tensor({n}, DType::Float32, Device::CPU);
        g.set_one();
        w.sgd_update(g, 0.5);
        w.sync();
        CHECK(near(w.at<float32_t>({0}), 0.5f, 1e-7));
        CHECK(near(w.at<float32_t>({n - 1}), 0.5f, 1e-7));
    }
    {
        // fp16: widen, fmaf, narrow RNE — compare bit-exact against the same
        // host computation through the unaccel pathway.
        auto w = Tensor({1}, DType::Float16, Device::CPU);
        w.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
        auto g = Tensor({1}, DType::Float16, Device::CPU);
        g.set<float16_t>({0}, narrow_lane<float16_t>(0.5f));
        w.sgd_update(g, 0.1);
        w.sync();
        const auto expected = narrow_lane<float16_t>(std::fmaf(-0.1f, 0.5f, 1.0f));
        CHECK(lane_bits(w.at<float16_t>({0})) == lane_bits(expected));
    }
    {
        auto w = Tensor({1}, DType::BFloat16, Device::CPU);
        w.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(1.0f));
        auto g = Tensor({1}, DType::BFloat16, Device::CPU);
        g.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(0.5f));
        w.sgd_update(g, 0.1);
        w.sync();
        const auto expected = narrow_lane<bfloat16_t>(std::fmaf(-0.1f, 0.5f, 1.0f));
        CHECK(lane_bits(w.at<bfloat16_t>({0})) == lane_bits(expected));
    }
    {
        // Packed: per-lane update.
        auto w = Tensor({1}, DType::Float16x2, Device::CPU);
        w.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(1.0f, 2.0f));
        auto g = Tensor({1}, DType::Float16x2, Device::CPU);
        g.set<float16x2_t>({0}, pack_lanes<float16x2_t, float16_t>(0.5f, -1.0f));
        w.sgd_update(g, 0.1);
        w.sync();
        CHECK(near(unpack_lo(w.at<float16x2_t>({0})), 0.95f, 1e-3));
        CHECK(near(unpack_hi(w.at<float16x2_t>({0})), 2.1f, 1e-3));
    }
    {
        // int32: fp64 compute, truncation.
        auto w = Tensor({2}, DType::Int32, Device::CPU);
        w.set<int32_t>({0}, int32_t{10});
        w.set<int32_t>({1}, int32_t{10});
        auto g = Tensor({2}, DType::Int32, Device::CPU);
        g.set<int32_t>({0}, int32_t{3});
        g.set<int32_t>({1}, int32_t{3});
        w.sgd_update(g, 1.0);
        w.sync();
        CHECK(w.at<int32_t>({0}) == int32_t{7});
        // 7 - 0.5*3 = 5.5 truncates toward zero to 5.
        w.sgd_update(g, 0.5);
        w.sync();
        CHECK(w.at<int32_t>({1}) == int32_t{5});
    }
    {
        // Round-toward-zero asymmetry: -0.5 truncates to 0.
        auto w = Tensor({1}, DType::Int32, Device::CPU);
        w.set<int32_t>({0}, int32_t{0});
        auto g = Tensor({1}, DType::Int32, Device::CPU);
        g.set<int32_t>({0}, int32_t{1});
        w.sgd_update(g, 0.5);
        w.sync();
        CHECK(w.at<int32_t>({0}) == int32_t{0});
    }
    {
        // int8 clamps to the dtype range.
        auto w = Tensor({1}, DType::Int8, Device::CPU);
        w.set<int8_t>({0}, int8_t{-100});
        auto g = Tensor({1}, DType::Int8, Device::CPU);
        g.set<int8_t>({0}, int8_t{100});
        w.sgd_update(g, 2.0);
        w.sync();
        CHECK(w.at<int8_t>({0}) == std::numeric_limits<int8_t>::lowest());
    }
    {
        // uint8 clamps at zero.
        auto w = Tensor({1}, DType::UInt8, Device::CPU);
        w.set<uint8_t>({0}, uint8_t{5});
        auto g = Tensor({1}, DType::UInt8, Device::CPU);
        g.set<uint8_t>({0}, uint8_t{10});
        w.sgd_update(g, 1.0);
        w.sync();
        CHECK(w.at<uint8_t>({0}) == uint8_t{0});
    }
    {
        // Remaining int dtypes take the same templated path; smoke them.
        auto w16 = Tensor({1}, DType::Int16, Device::CPU);
        w16.set<int16_t>({0}, int16_t{100});
        auto g16 = Tensor({1}, DType::Int16, Device::CPU);
        g16.set<int16_t>({0}, int16_t{10});
        w16.sgd_update(g16, 1.0);
        w16.sync();
        CHECK(w16.at<int16_t>({0}) == int16_t{90});

        auto wu16 = Tensor({1}, DType::UInt16, Device::CPU);
        wu16.set<uint16_t>({0}, uint16_t{100});
        auto gu16 = Tensor({1}, DType::UInt16, Device::CPU);
        gu16.set<uint16_t>({0}, uint16_t{10});
        wu16.sgd_update(gu16, 1.0);
        wu16.sync();
        CHECK(wu16.at<uint16_t>({0}) == uint16_t{90});

        auto wu32 = Tensor({1}, DType::UInt32, Device::CPU);
        wu32.set<uint32_t>({0}, uint32_t{100});
        auto gu32 = Tensor({1}, DType::UInt32, Device::CPU);
        gu32.set<uint32_t>({0}, uint32_t{10});
        wu32.sgd_update(gu32, 1.0);
        wu32.sync();
        CHECK(wu32.at<uint32_t>({0}) == uint32_t{90});

        auto w64 = Tensor({1}, DType::Int64, Device::CPU);
        w64.set<int64_t>({0}, int64_t{100});
        auto g64 = Tensor({1}, DType::Int64, Device::CPU);
        g64.set<int64_t>({0}, int64_t{10});
        w64.sgd_update(g64, 1.0);
        w64.sync();
        CHECK(w64.at<int64_t>({0}) == int64_t{90});

        auto wu64 = Tensor({1}, DType::UInt64, Device::CPU);
        wu64.set<uint64_t>({0}, uint64_t{100});
        auto gu64 = Tensor({1}, DType::UInt64, Device::CPU);
        gu64.set<uint64_t>({0}, uint64_t{10});
        wu64.sgd_update(gu64, 1.0);
        wu64.sync();
        CHECK(wu64.at<uint64_t>({0}) == uint64_t{90});
    }
    {
        // NaN update leaves an integer element unchanged.
        auto w = Tensor({1}, DType::Int32, Device::CPU);
        w.set<int32_t>({0}, int32_t{42});
        auto g = Tensor({1}, DType::Int32, Device::CPU);
        g.set<int32_t>({0}, int32_t{1});
        w.sgd_update(g, std::numeric_limits<float64_t>::quiet_NaN());
        w.sync();
        CHECK(w.at<int32_t>({0}) == int32_t{42});
    }
    {
        // int4x2: per-lane fp64 compute, truncate toward zero.
        auto w = Tensor({1}, DType::Int4x2, Device::CPU);
        w.set<int4x2_t>({0}, i4x2(int8_t{5}, int8_t{-5}));
        auto g = Tensor({1}, DType::Int4x2, Device::CPU);
        g.set<int4x2_t>({0}, i4x2(int8_t{2}, int8_t{2}));
        w.sgd_update(g, 1.0);
        w.sync();
        CHECK(i4x2_lo(w.at<int4x2_t>({0})) == int8_t{3});
        CHECK(i4x2_hi(w.at<int4x2_t>({0})) == int8_t{-7});
    }
    {
        // int4x2: lanes clamp to [-8, 7].
        auto w = Tensor({1}, DType::Int4x2, Device::CPU);
        w.set<int4x2_t>({0}, i4x2(int8_t{6}, int8_t{-6}));
        auto g = Tensor({1}, DType::Int4x2, Device::CPU);
        g.set<int4x2_t>({0}, i4x2(int8_t{-7}, int8_t{7}));
        w.sgd_update(g, 2.0);
        w.sync();
        // lo: 6 - 2*(-7) = 20 clamps to 7; hi: -6 - 2*7 = -20 clamps to -8.
        CHECK(i4x2_lo(w.at<int4x2_t>({0})) == int8_t{7});
        CHECK(i4x2_hi(w.at<int4x2_t>({0})) == int8_t{-8});
    }
    {
        // Validation: shape / dtype / device mismatches throw.
        auto w = Tensor({2}, DType::Float32, Device::CPU);
        auto g_shape = Tensor({3}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            w.sgd_update(g_shape, 0.1);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);

        // A grad of a different dtype no longer throws: sgd_update converts a
        // private copy into the weight dtype (the single grad-conversion rule).
        auto g_dtype = Tensor({2}, DType::Float64, Device::CPU);
        g_dtype.set<float64_t>({0}, 0.5);
        g_dtype.set<float64_t>({1}, -0.25);
        w.set<float32_t>({0}, 1.0f);
        w.set<float32_t>({1}, 2.0f);
        w.sgd_update(g_dtype, 0.1);
        w.sync();
        CHECK(near(w.at<float32_t>({0}), 0.95f, 1e-6));
        CHECK(near(w.at<float32_t>({1}), 2.025f, 1e-6));
    }
    {
        // sgd_step with no gradient throws NoGradient.
        auto w = Tensor({2}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            w.sgd_step(0.1);
        } catch (const BeanTensor::ErrorHandling::NoGradient&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// --- GPU paths ---

static void test_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    {
        // reshape is metadata-only; verify it on-device too.
        auto t = Tensor({2, 3}, DType::Float32, Device::GPU);
        t.set_one();
        auto view = t.reshape({3, 2});
        view.set<float32_t>({1, 0}, 7.0f);
        CHECK(near(t.at<float32_t>({0, 2}), 7.0f, 1e-9));
    }
    {
        auto t = Tensor({4}, DType::Float32, Device::GPU);
        for (size_t i = 0; i < 4; ++i) {
            t.set<float32_t>({i}, static_cast<float32_t>(i + 1));
        }
        auto s = t.sum();
        s.sync();
        CHECK(near(s.at<float32_t>({0}), 10.0f, 1e-5));
    }
    {
        auto t = Tensor({3}, DType::Int32, Device::GPU);
        t.set<int32_t>({0}, int32_t{-5});
        t.set<int32_t>({1}, int32_t{10});
        t.set<int32_t>({2}, int32_t{3});
        auto s = t.sum();
        s.sync();
        CHECK(s.at<int32_t>({0}) == int32_t{8});
    }
    {
        auto t = Tensor({3}, DType::Float16, Device::GPU);
        for (size_t i = 0; i < 3; ++i) {
            t.set<float16_t>({i}, narrow_lane<float16_t>(static_cast<float32_t>(i + 1)));
        }
        auto s = t.sum();
        s.sync();
        CHECK(near(widen_lane(s.at<float16_t>({0})), 6.0f, 1e-3));
    }
    {
        auto t = Tensor({2}, DType::BFloat16x2, Device::GPU);
        t.set<bfloat16x2_t>({0}, pack_lanes<bfloat16x2_t, bfloat16_t>(1.0f, 10.0f));
        t.set<bfloat16x2_t>({1}, pack_lanes<bfloat16x2_t, bfloat16_t>(2.0f, 20.0f));
        auto s = t.sum();
        s.sync();
        CHECK(near(unpack_lo(s.at<bfloat16x2_t>({0})), 3.0f, 1e-2));
        CHECK(near(unpack_hi(s.at<bfloat16x2_t>({0})), 30.0f, 0.5));
    }
    {
        auto t = Tensor({3}, DType::Float32, Device::GPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, 9.0f);
        t.set<float32_t>({2}, 3.0f);
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // NaN wins on GPU too (PyTorch parity).
        auto t = Tensor({3}, DType::Float32, Device::GPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, std::numeric_limits<float32_t>::quiet_NaN());
        t.set<float32_t>({2}, 9.0f);
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        auto t = Tensor({3}, DType::Int8, Device::GPU);
        t.set<int8_t>({0}, int8_t{-5});
        t.set<int8_t>({1}, int8_t{3});
        t.set<int8_t>({2}, int8_t{0});
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        auto w = Tensor({2}, DType::Float32, Device::GPU);
        w.set<float32_t>({0}, 1.0f);
        w.set<float32_t>({1}, 2.0f);
        auto g = Tensor({2}, DType::Float32, Device::GPU);
        g.set<float32_t>({0}, 0.5f);
        g.set<float32_t>({1}, -1.0f);
        w.sgd_update(g, 0.1);
        w.sync();
        CHECK(near(w.at<float32_t>({0}), 0.95f, 1e-6));
        CHECK(near(w.at<float32_t>({1}), 2.1f, 1e-6));
    }
    {
        auto w = Tensor({1}, DType::Float16, Device::GPU);
        w.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
        auto g = Tensor({1}, DType::Float16, Device::GPU);
        g.set<float16_t>({0}, narrow_lane<float16_t>(0.5f));
        w.sgd_update(g, 0.1);
        w.sync();
        const auto expected = narrow_lane<float16_t>(std::fmaf(-0.1f, 0.5f, 1.0f));
        CHECK(lane_bits(w.at<float16_t>({0})) == lane_bits(expected));
    }
    {
        // int clamp on GPU.
        auto w = Tensor({1}, DType::Int8, Device::GPU);
        w.set<int8_t>({0}, int8_t{-100});
        auto g = Tensor({1}, DType::Int8, Device::GPU);
        g.set<int8_t>({0}, int8_t{100});
        w.sgd_update(g, 2.0);
        w.sync();
        CHECK(w.at<int8_t>({0}) == std::numeric_limits<int8_t>::lowest());
    }
    {
        // Device mismatch is rejected even with a GPU present.
        auto w = Tensor({2}, DType::Float32, Device::GPU);
        auto g = Tensor({2}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            w.sgd_update(g, 0.1);
        } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // int4x2 on GPU: per-lane modular sum.
        auto t = Tensor({3}, DType::Int4x2, Device::GPU);
        t.set<int4x2_t>({0}, i4x2(int8_t{7}, int8_t{-8}));
        t.set<int4x2_t>({1}, i4x2(int8_t{7}, int8_t{-8}));
        t.set<int4x2_t>({2}, i4x2(int8_t{7}, int8_t{-8}));
        auto s = t.sum();
        s.sync();
        CHECK(i4x2_lo(s.at<int4x2_t>({0})) == int8_t{5});
        CHECK(i4x2_hi(s.at<int4x2_t>({0})) == int8_t{-8});
    }
    {
        // int4x2 on GPU: lexicographic argmax.
        auto t = Tensor({2}, DType::Int4x2, Device::GPU);
        t.set<int4x2_t>({0}, i4x2(int8_t{1}, int8_t{7}));
        t.set<int4x2_t>({1}, i4x2(int8_t{2}, int8_t{-8}));
        CHECK(t.argmax() == uint64_t{1});
    }
    {
        // int4x2 on GPU: per-lane sgd with clamp to [-8, 7].
        auto w = Tensor({1}, DType::Int4x2, Device::GPU);
        w.set<int4x2_t>({0}, i4x2(int8_t{6}, int8_t{-6}));
        auto g = Tensor({1}, DType::Int4x2, Device::GPU);
        g.set<int4x2_t>({0}, i4x2(int8_t{-7}, int8_t{7}));
        w.sgd_update(g, 2.0);
        w.sync();
        CHECK(i4x2_lo(w.at<int4x2_t>({0})) == int8_t{7});
        CHECK(i4x2_hi(w.at<int4x2_t>({0})) == int8_t{-8});
    }
}

// Direct probe of the AVX2 sum kernels. SumCPU dispatches AVX512-first, so on
// an AVX512 machine these kernels never run through the normal tensor path —
// this test is their only exercise. Tree-reduction order differs from the
// scalar loop, so the exact checks use order-insensitive values (all ones);
// the random fills compare against an fp64 reference with tolerance.
static void test_sum_avx2_kernels() {
    namespace sr = BeanTensor::Math::Reductions;
    const auto hw_avx2 = sr::sum_avx2_available();

    {
        constexpr auto n = size_t{100003}; // odd, exercises the scalar tail
        auto ones = std::vector<float32_t>(n, 1.0f);
        try {
            // All ones: every partial sum is an exact small integer, so the
            // tree order is irrelevant and the result must be exact.
            CHECK(near(sr::sum_f32_range_avx2(ones.data(), 0, n), static_cast<float32_t>(n), 0.0f));
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            CHECK(!hw_avx2);
        }
        auto rnd = std::vector<float32_t>(n);
        auto rng = std::mt19937{42};
        for (auto i = size_t{0}; i < n; ++i) {
            rnd[i] = static_cast<float32_t>(static_cast<int32_t>(rng() % 2001) - 1000) / 250.0f;
        }
        auto expect = float64_t{0};
        for (const auto v : rnd) {
            expect += static_cast<float64_t>(v);
        }
        try {
            const auto got = static_cast<float64_t>(sr::sum_f32_range_avx2(rnd.data(), 0, n));
            CHECK(std::fabs(got - expect) <= 1e-3 * std::fabs(expect) + 1e-3);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            CHECK(!hw_avx2);
        }
    }
    {
        constexpr auto n = size_t{100003};
        auto ones = std::vector<float64_t>(n, 1.0);
        try {
            // All ones: exact for any summation order.
            CHECK(near64(sr::sum_f64_range_avx2(ones.data(), 0, n), static_cast<float64_t>(n), 0.0));
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            CHECK(!hw_avx2);
        }
        auto rnd = std::vector<float64_t>(n);
        auto rng = std::mt19937_64{42};
        for (auto i = size_t{0}; i < n; ++i) {
            rnd[i] = static_cast<float64_t>(static_cast<int64_t>(rng() % 2001) - 1000) / 250.0;
        }
        auto expect = float64_t{0};
        for (const auto v : rnd) {
            expect += v;
        }
        try {
            const auto got = sr::sum_f64_range_avx2(rnd.data(), 0, n);
            CHECK(std::fabs(got - expect) <= 1e-9 * std::fabs(expect) + 1e-9);
        } catch (const BeanTensor::ErrorHandling::CPUIllegalInstruction&) {
            CHECK(!hw_avx2);
        }
    }
}

int main() {
    test_reshape_cpu();
    test_sum_cpu();
    test_sum_avx2_kernels();
    test_argmax_cpu();
    test_sgd_cpu();
    test_gpu();
    if (failures == 0) {
        std::cout << "all reduction/optimization/shape tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
