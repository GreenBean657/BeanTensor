/**
 * @file metrics_tests.cpp
 * @author GreenBean
 * @brief Classification-metrics and KernelTune invariance tests - framework-free
 * assertion suite. Covers row-wise argmax(axis) (per-slice int64 indices,
 * tie-breaking, NaN-wins), elementwise equal masks across dtypes (IEEE ==
 * semantics: NaN never equal, -0.0 equals 0.0), mask-to-float casts, the
 * end-to-end accuracy pipeline argmax -> equal -> cast -> mean, full-tensor
 * min/max value reductions (planted extremes, ties, NaN propagation, per-lane
 * packed rules, bit-identity across backends), and KernelTune reduction block-size
 * invariance (min/max bit-identical, int sum bit-identical, float sum
 * order-dependent so only near-equal) on CPU and on the GPU (CUDA/HIP) paths
 * when a backend and device exist.
 * @version 0.3
 * @date 2026-10-25
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Helpers/KernelTune.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::bfloat16_t;
using BeanTensor::bfloat16x2_t;
using BeanTensor::float16_t;
using BeanTensor::float16x2_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using BeanTensor::int4x2_t;
using DType = BeanTensor::DataTypes::DType;
namespace idet = BeanTensor::Intrinsics::detail;
using BeanTensor::KernelTune::GpuFamily;
using BeanTensor::KernelTune::SizeClass;
using BeanTensor::KernelTune::clear_override;
using BeanTensor::KernelTune::set_override;

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

// --- Half helpers (host-side, via the unaccel conversion pathway) ---

template<typename H>
static H narrow_lane(const float32_t v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return idet::unaccel_fp32_to_fp16_untracked(v);
    } else {
        return idet::unaccel_fp32_to_bf16_untracked(v);
    }
}

// --- full-tensor min/max helpers (host-side lane widening + raw bits) ---

template<typename H>
static float32_t widen_h(const H v) {
    if constexpr (std::is_same_v<H, float16_t>) {
        return idet::unaccel_fp16_to_fp32_untracked(v);
    } else {
        return idet::unaccel_bf16_to_fp32(v);
    }
}

template<typename H>
static uint16_t h_bits(const H v) {
    return idet::bit_cast_memcpy<uint16_t>(v);
}

template<typename P>
static P pack2_bits(const uint16_t lo_bits, const uint16_t hi_bits) {
    return idet::bit_cast_memcpy<P>(static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16));
}

template<typename P, typename H>
static P pack2(const float32_t lo, const float32_t hi) {
    return pack2_bits<P>(h_bits(narrow_lane<H>(lo)), h_bits(narrow_lane<H>(hi)));
}

template<typename P>
static uint16_t lane_lo_bits(const P v) {
    return static_cast<uint16_t>(idet::bit_cast_memcpy<uint32_t>(v) & 0xFFFFu);
}

template<typename P>
static uint16_t lane_hi_bits(const P v) {
    return static_cast<uint16_t>(idet::bit_cast_memcpy<uint32_t>(v) >> 16);
}

static int8_t nib_lo(const int4x2_t v) {
    const auto n = static_cast<int8_t>(idet::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
    return n > 7 ? static_cast<int8_t>(n - 16) : n;
}

static int8_t nib_hi(const int4x2_t v) {
    const auto n = static_cast<int8_t>(idet::bit_cast_memcpy<uint8_t>(v) >> 4);
    return n > 7 ? static_cast<int8_t>(n - 16) : n;
}

static int4x2_t nib_pack(const int8_t lo, const int8_t hi) {
    return idet::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>((static_cast<uint8_t>(lo) & 0x0Fu) | ((static_cast<uint8_t>(hi) & 0x0Fu) << 4)));
}

// --- row-wise argmax ---

static void test_argmax_axis_cpu() {
    {
        // [batch, classes] logits: one distinct max per row.
        auto logits = Tensor({4, 10}, DType::Float32, Device::CPU);
        for (size_t r = 0; r < 4; ++r) {
            for (size_t c = 0; c < 10; ++c) {
                logits.set<float32_t>({r, c}, 0.5f);
            }
        }
        logits.set<float32_t>({0, 3}, 9.0f);
        logits.set<float32_t>({1, 0}, 2.0f);
        logits.set<float32_t>({2, 9}, 7.25f);
        logits.set<float32_t>({3, 5}, 100.0f);
        auto pred = logits.argmax(1);
        CHECK(pred.is_same_shape(Tensor({4})));
        CHECK(pred.get_dtype() == DType::Int64);
        CHECK(pred.at<int64_t>({0}) == 3);
        CHECK(pred.at<int64_t>({1}) == 0);
        CHECK(pred.at<int64_t>({2}) == 9);
        CHECK(pred.at<int64_t>({3}) == 5);
    }
    {
        // Reducing axis 0 leaves the trailing dims.
        auto t = Tensor({3, 4}, DType::Float32, Device::CPU);
        for (size_t i = 0; i < 12; ++i) {
            t.set<float32_t>({i / 4, i % 4}, static_cast<float32_t>(i));
        }
        // Column c peaks at row 2 for every column.
        for (size_t c = 0; c < 4; ++c) {
            t.set<float32_t>({2, c}, 100.0f);
        }
        auto pred = t.argmax(0);
        CHECK(pred.is_same_shape(Tensor({4})));
        for (size_t c = 0; c < 4; ++c) {
            CHECK(pred.at<int64_t>({c}) == 2);
        }
    }
    {
        // Rank-1 input reduces to shape {1}.
        auto t = Tensor({5}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, 2.0f);
        t.set<float32_t>({2}, 8.0f);
        t.set<float32_t>({3}, 4.0f);
        t.set<float32_t>({4}, 0.5f);
        auto pred = t.argmax(0);
        CHECK(pred.is_same_shape(Tensor({1})));
        CHECK(pred.at<int64_t>({0}) == 2);
    }
    {
        // Tie-breaking: the first strict maximum wins, so ties keep the
        // lowest index.
        auto t = Tensor({6}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, 3.0f);
        t.set<float32_t>({2}, 3.0f);
        t.set<float32_t>({3}, 2.0f);
        t.set<float32_t>({4}, 3.0f);
        t.set<float32_t>({5}, -1.0f);
        CHECK(t.argmax(0).at<int64_t>({0}) == 1);
    }
    {
        // NaN beats every non-NaN value; the first NaN wins against later NaNs.
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto t = Tensor({3}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, nan);
        t.set<float32_t>({2}, 2.0f);
        CHECK(t.argmax(0).at<int64_t>({0}) == 1);
        auto u = Tensor({3}, DType::Float32, Device::CPU);
        u.set<float32_t>({0}, nan);
        u.set<float32_t>({1}, nan);
        u.set<float32_t>({2}, 1.0f);
        CHECK(u.argmax(0).at<int64_t>({0}) == 0);
    }
    {
        // Integer rows: strict greater wins, ties keep the low index.
        auto t = Tensor({2, 4}, DType::Int32, Device::CPU);
        t.set<int32_t>({0, 0}, int32_t{-5});
        t.set<int32_t>({0, 1}, int32_t{7});
        t.set<int32_t>({0, 2}, int32_t{7});
        t.set<int32_t>({0, 3}, int32_t{2});
        t.set<int32_t>({1, 0}, int32_t{0});
        t.set<int32_t>({1, 1}, int32_t{-1});
        t.set<int32_t>({1, 2}, int32_t{-1});
        t.set<int32_t>({1, 3}, int32_t{0});
        auto pred = t.argmax(1);
        CHECK(pred.get_dtype() == DType::Int64);
        CHECK(pred.at<int64_t>({0}) == 1);
        CHECK(pred.at<int64_t>({1}) == 0);
    }
    {
        // fp16 rows rank on widened fp64 keys (PyTorch-compatible NaN rule).
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto t = Tensor({4}, DType::Float16, Device::CPU);
        t.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
        t.set<float16_t>({1}, narrow_lane<float16_t>(5.0f));
        t.set<float16_t>({2}, narrow_lane<float16_t>(5.0f));
        t.set<float16_t>({3}, narrow_lane<float16_t>(nan));
        CHECK(t.argmax(0).at<int64_t>({0}) == 3);
    }
    {
        // Axis out of range throws.
        auto t = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = t.argmax(2);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Non-contiguous operand throws.
        auto t = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
        auto inner = t.slice(ranges);
        auto threw = false;
        try {
            auto bad = inner.argmax(1);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// --- equality masks ---

static void test_equal_cpu() {
    {
        // IEEE == semantics on widened fp64 keys.
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto a = Tensor({5}, DType::Float32, Device::CPU);
        a.set<float32_t>({0}, 1.0f);
        a.set<float32_t>({1}, nan);
        a.set<float32_t>({2}, -0.0f);
        a.set<float32_t>({3}, 4.5f);
        a.set<float32_t>({4}, 4.5f);
        auto b = Tensor({5}, DType::Float32, Device::CPU);
        b.set<float32_t>({0}, 2.0f);
        b.set<float32_t>({1}, nan);
        b.set<float32_t>({2}, 0.0f);
        b.set<float32_t>({3}, 4.5f);
        b.set<float32_t>({4}, -4.5f);
        auto mask = a.equal(b);
        CHECK(mask.is_same_shape(Tensor({5})));
        CHECK(mask.get_dtype() == DType::UInt8);
        CHECK(mask.at<uint8_t>({0}) == 0); // 1.0 != 2.0
        CHECK(mask.at<uint8_t>({1}) == 0); // NaN never equals NaN
        CHECK(mask.at<uint8_t>({2}) == 1); // -0.0 == 0.0
        CHECK(mask.at<uint8_t>({3}) == 1); // exact match
        CHECK(mask.at<uint8_t>({4}) == 0); // sign differs
    }
    {
        // Masks across integer dtypes.
        auto a = Tensor({4}, DType::Int32, Device::CPU);
        a.set<int32_t>({0}, int32_t{-7});
        a.set<int32_t>({1}, int32_t{42});
        a.set<int32_t>({2}, int32_t{0});
        a.set<int32_t>({3}, int32_t{1});
        auto b = Tensor({4}, DType::Int32, Device::CPU);
        b.set<int32_t>({0}, int32_t{-7});
        b.set<int32_t>({1}, int32_t{43});
        b.set<int32_t>({2}, int32_t{0});
        b.set<int32_t>({3}, int32_t{1});
        auto mask = a.equal(b);
        CHECK(mask.at<uint8_t>({0}) == 1);
        CHECK(mask.at<uint8_t>({1}) == 0);
        CHECK(mask.at<uint8_t>({2}) == 1);
        CHECK(mask.at<uint8_t>({3}) == 1);
        // Self-equality is all-true.
        auto self = a.equal(a);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(self.at<uint8_t>({i}) == 1);
        }
    }
    {
        // fp16 equality compares widened lanes; NaN lanes never equal.
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto a = Tensor({3}, DType::Float16, Device::CPU);
        a.set<float16_t>({0}, narrow_lane<float16_t>(1.5f));
        a.set<float16_t>({1}, narrow_lane<float16_t>(nan));
        a.set<float16_t>({2}, narrow_lane<float16_t>(-0.0f));
        auto b = Tensor({3}, DType::Float16, Device::CPU);
        b.set<float16_t>({0}, narrow_lane<float16_t>(1.5f));
        b.set<float16_t>({1}, narrow_lane<float16_t>(nan));
        b.set<float16_t>({2}, narrow_lane<float16_t>(0.0f));
        auto mask = a.equal(b);
        CHECK(mask.at<uint8_t>({0}) == 1);
        CHECK(mask.at<uint8_t>({1}) == 0);
        CHECK(mask.at<uint8_t>({2}) == 1);
    }
    {
        // Shape mismatch throws.
        auto a = Tensor({3}, DType::Float32, Device::CPU);
        auto b = Tensor({4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.equal(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Dtype mismatch throws.
        auto a = Tensor({3}, DType::Float32, Device::CPU);
        auto b = Tensor({3}, DType::Int32, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.equal(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Non-contiguous operand throws.
        auto a = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto b = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto ranges_a = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
        auto ranges_b = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
        auto inner = a.slice(ranges_a);
        auto inner_b = b.slice(ranges_b);
        auto threw = false;
        try {
            auto bad = inner.equal(inner_b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// --- mask cast + mean: the accuracy pipeline ---

static void test_accuracy_cpu() {
    {
        // Mask semantics of the UInt8 -> float cast: false -> 0.0, true -> 1.0.
        auto mask = Tensor({4}, DType::UInt8, Device::CPU);
        mask.set<uint8_t>({0}, 1);
        mask.set<uint8_t>({1}, 0);
        mask.set<uint8_t>({2}, 1);
        mask.set<uint8_t>({3}, 0);
        auto as_float = mask.clone();
        as_float.convert_dtype(DType::Float32);
        as_float.sync(); // the dtype flip materializes at sync()
        CHECK(as_float.get_dtype() == DType::Float32);
        CHECK(near(as_float.at<float32_t>({0}), 1.0f, 1e-9));
        CHECK(near(as_float.at<float32_t>({1}), 0.0f, 1e-9));
        CHECK(near(as_float.at<float32_t>({2}), 1.0f, 1e-9));
        CHECK(near(as_float.at<float32_t>({3}), 0.0f, 1e-9));
        auto as_double = mask.clone();
        as_double.convert_dtype(DType::Float64);
        CHECK(near64(as_double.at<float64_t>({0}), 1.0, 1e-15));
        CHECK(near64(as_double.at<float64_t>({1}), 0.0, 1e-15));
    }
#if !defined(USE_CUDA)
    {
        // Mean consumes the mask dtype directly, too.
        auto mask = Tensor({4}, DType::UInt8, Device::CPU);
        mask.set<uint8_t>({0}, 1);
        mask.set<uint8_t>({1}, 0);
        mask.set<uint8_t>({2}, 0);
        mask.set<uint8_t>({3}, 1);
        auto m = mask.mean();
        CHECK(m.is_same_shape(Tensor({1})));
        CHECK(m.get_dtype() == DType::UInt8);
        CHECK(m.at<uint8_t>({0}) == 0); // 0.5 truncates toward zero
    }
#endif
    {
        // End-to-end accuracy: argmax -> equal -> cast -> mean.
        auto logits = Tensor({5, 10}, DType::Float32, Device::CPU);
        for (size_t r = 0; r < 5; ++r) {
            for (size_t c = 0; c < 10; ++c) {
                logits.set<float32_t>({r, c}, 0.1f);
            }
        }
        // Argmax picks class 9 on every row but row 2 (class 4).
        for (size_t r = 0; r < 5; ++r) {
            if (r == 2) {
                logits.set<float32_t>({r, 4}, 5.0f);
            } else {
                logits.set<float32_t>({r, 9}, 5.0f);
            }
        }
        auto labels = Tensor({5}, DType::Int64, Device::CPU);
        labels.set<int64_t>({0}, 9);
        labels.set<int64_t>({1}, 9);
        labels.set<int64_t>({2}, 4);
        labels.set<int64_t>({3}, 9);
        labels.set<int64_t>({4}, 7); // wrong on purpose
        auto pred = logits.argmax(1);
        CHECK(pred.get_dtype() == DType::Int64);
        auto mask = pred.equal(labels);
        CHECK(mask.get_dtype() == DType::UInt8);
        auto as_float = mask.clone();
        as_float.convert_dtype(DType::Float32);
#if !defined(USE_CUDA)
        // No MeanCUDA.cu exists yet, so mean() is only linkable on CPU-only
        // and HIP builds; see Math/Reductions/Mean.
        auto acc = as_float.mean();
        CHECK(near(acc.at<float32_t>({0}), 0.8f, 1e-6));
#endif
        // Cross-check the mean against the raw mask: 4 of 5 correct.
        CHECK(mask.at<uint8_t>({0}) == 1);
        CHECK(mask.at<uint8_t>({1}) == 1);
        CHECK(mask.at<uint8_t>({2}) == 1);
        CHECK(mask.at<uint8_t>({3}) == 1);
        CHECK(mask.at<uint8_t>({4}) == 0);
    }
}

// --- full-tensor min/max value reductions ---

static void test_minmax_cpu() {
    {
        // Float32: planted extremes, duplicated tie, exact bit identity,
        // shape-{1} output of the input dtype.
        auto t = Tensor({6}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 3.0f);
        t.set<float32_t>({1}, -1.5f);
        t.set<float32_t>({2}, 2.0f);
        t.set<float32_t>({3}, -1.5f); // tie
        t.set<float32_t>({4}, 10.25f);
        t.set<float32_t>({5}, 0.0f);
        auto mn = t.min();
        auto mx = t.max();
        CHECK(mn.is_same_shape(Tensor({1})));
        CHECK(mn.get_dtype() == DType::Float32);
        CHECK(mx.get_dtype() == DType::Float32);
        CHECK(idet::bit_cast_memcpy<uint32_t>(mn.at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(-1.5f));
        CHECK(idet::bit_cast_memcpy<uint32_t>(mx.at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(10.25f));
    }
    {
        // PyTorch-compatible NaN rule: any NaN -> NaN result, both ways.
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto t = Tensor({4}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 1.0f);
        t.set<float32_t>({1}, nan);
        t.set<float32_t>({2}, -7.0f);
        t.set<float32_t>({3}, 42.0f);
        CHECK(std::isnan(t.min().at<float32_t>({0})));
        CHECK(std::isnan(t.max().at<float32_t>({0})));
    }
    {
        // Zero-sign tie: min canonicalizes to -0.0, max to +0.0, so every
        // backend and chunking agrees bit-for-bit.
        auto t = Tensor({2}, DType::Float32, Device::CPU);
        t.set<float32_t>({0}, 0.0f);
        t.set<float32_t>({1}, -0.0f);
        CHECK(idet::bit_cast_memcpy<uint32_t>(t.min().at<float32_t>({0})) == UINT32_C(0x80000000));
        CHECK(idet::bit_cast_memcpy<uint32_t>(t.max().at<float32_t>({0})) == UINT32_C(0x00000000));
    }
    {
        // Large input: exercises launch_reduce's threaded path (far past
        // threads*1024 elements), min planted at the last element.
        constexpr auto n = size_t{500001};
        auto t = Tensor({n}, DType::Float32, Device::CPU);
        for (size_t i = 0; i < n; ++i) {
            t.set<float32_t>({i}, static_cast<float32_t>(static_cast<int32_t>(i % 1000) - 500));
        }
        t.set<float32_t>({n - 1}, -12345.0f);
        t.set<float32_t>({0}, 99999.0f);
        CHECK(near(t.min().at<float32_t>({0}), -12345.0f, 0.0));
        CHECK(near(t.max().at<float32_t>({0}), 99999.0f, 0.0));
    }
    {
        // Float64: planted extremes and NaN propagation.
        auto t = Tensor({5}, DType::Float64, Device::CPU);
        t.set<float64_t>({0}, -2.5);
        t.set<float64_t>({1}, 8.75);
        t.set<float64_t>({2}, 0.0);
        t.set<float64_t>({3}, -2.5); // tie
        t.set<float64_t>({4}, 1.0);
        CHECK(idet::bit_cast_memcpy<uint64_t>(t.min().at<float64_t>({0})) == idet::bit_cast_memcpy<uint64_t>(-2.5));
        CHECK(idet::bit_cast_memcpy<uint64_t>(t.max().at<float64_t>({0})) == idet::bit_cast_memcpy<uint64_t>(8.75));
        constexpr auto nan = std::numeric_limits<float64_t>::quiet_NaN();
        auto u = Tensor({3}, DType::Float64, Device::CPU);
        u.set<float64_t>({0}, 1.0);
        u.set<float64_t>({1}, nan);
        u.set<float64_t>({2}, -1.0);
        CHECK(std::isnan(u.min().at<float64_t>({0})));
        CHECK(std::isnan(u.max().at<float64_t>({0})));
    }
    {
        // Float16: compare on widened lanes; NaN propagates.
        auto t = Tensor({4}, DType::Float16, Device::CPU);
        t.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
        t.set<float16_t>({1}, narrow_lane<float16_t>(-4.0f));
        t.set<float16_t>({2}, narrow_lane<float16_t>(-4.0f)); // tie
        t.set<float16_t>({3}, narrow_lane<float16_t>(3.0f));
        CHECK(h_bits(t.min().at<float16_t>({0})) == h_bits(narrow_lane<float16_t>(-4.0f)));
        CHECK(h_bits(t.max().at<float16_t>({0})) == h_bits(narrow_lane<float16_t>(3.0f)));
        auto u = Tensor({3}, DType::Float16, Device::CPU);
        u.set<float16_t>({0}, narrow_lane<float16_t>(1.0f));
        u.set<float16_t>({1}, idet::bit_cast_memcpy<float16_t>(uint16_t{0x7E00})); // canonical NaN
        u.set<float16_t>({2}, narrow_lane<float16_t>(2.0f));
        CHECK(std::isnan(widen_h(u.min().at<float16_t>({0}))));
        CHECK(std::isnan(widen_h(u.max().at<float16_t>({0}))));
    }
    {
        // BFloat16: same contract on the bf16 lane type.
        auto t = Tensor({4}, DType::BFloat16, Device::CPU);
        t.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(1.0f));
        t.set<bfloat16_t>({1}, narrow_lane<bfloat16_t>(-4.0f));
        t.set<bfloat16_t>({2}, narrow_lane<bfloat16_t>(3.0f));
        t.set<bfloat16_t>({3}, narrow_lane<bfloat16_t>(3.0f)); // tie
        CHECK(h_bits(t.min().at<bfloat16_t>({0})) == h_bits(narrow_lane<bfloat16_t>(-4.0f)));
        CHECK(h_bits(t.max().at<bfloat16_t>({0})) == h_bits(narrow_lane<bfloat16_t>(3.0f)));
        auto u = Tensor({3}, DType::BFloat16, Device::CPU);
        u.set<bfloat16_t>({0}, narrow_lane<bfloat16_t>(1.0f));
        u.set<bfloat16_t>({1}, idet::bit_cast_memcpy<bfloat16_t>(uint16_t{0x7FC0})); // canonical NaN
        u.set<bfloat16_t>({2}, narrow_lane<bfloat16_t>(2.0f));
        CHECK(std::isnan(widen_h(u.min().at<bfloat16_t>({0}))));
        CHECK(std::isnan(widen_h(u.max().at<bfloat16_t>({0}))));
    }
    {
        // Float16x2 per-lane: the lo-lane min and the hi-lane min come from
        // different elements; lanes are checked by exact lane bits.
        auto t = Tensor({3}, DType::Float16x2, Device::CPU);
        t.set<float16x2_t>({0}, pack2<float16x2_t, float16_t>(5.0f, -3.0f));
        t.set<float16x2_t>({1}, pack2<float16x2_t, float16_t>(1.0f, 7.0f));
        t.set<float16x2_t>({2}, pack2<float16x2_t, float16_t>(2.0f, -8.0f));
        auto mn = t.min();
        auto mx = t.max();
        CHECK(lane_lo_bits(mn.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(1.0f)));
        CHECK(lane_hi_bits(mn.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(-8.0f)));
        CHECK(lane_lo_bits(mx.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(5.0f)));
        CHECK(lane_hi_bits(mx.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(7.0f)));
    }
    {
        // Packed per-lane NaN: one poisoned lane leaves the other lane's
        // winner intact.
        auto t = Tensor({2}, DType::Float16x2, Device::CPU);
        t.set<float16x2_t>({0}, pack2_bits<float16x2_t>(uint16_t{0x7E00}, h_bits(narrow_lane<float16_t>(4.0f))));
        t.set<float16x2_t>({1}, pack2<float16x2_t, float16_t>(2.0f, 9.0f));
        auto mn = t.min();
        CHECK(lane_lo_bits(mn.at<float16x2_t>({0})) == uint16_t{0x7E00});
        CHECK(lane_hi_bits(mn.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(4.0f)));
        auto mx = t.max();
        CHECK(lane_lo_bits(mx.at<float16x2_t>({0})) == uint16_t{0x7E00});
        CHECK(lane_hi_bits(mx.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(9.0f)));
    }
    {
        // BFloat16x2 per-lane, same convention.
        auto t = Tensor({3}, DType::BFloat16x2, Device::CPU);
        t.set<bfloat16x2_t>({0}, pack2<bfloat16x2_t, bfloat16_t>(5.0f, -3.0f));
        t.set<bfloat16x2_t>({1}, pack2<bfloat16x2_t, bfloat16_t>(1.0f, 7.0f));
        t.set<bfloat16x2_t>({2}, pack2<bfloat16x2_t, bfloat16_t>(2.0f, -8.0f));
        auto mn = t.min();
        auto mx = t.max();
        CHECK(lane_lo_bits(mn.at<bfloat16x2_t>({0})) == h_bits(narrow_lane<bfloat16_t>(1.0f)));
        CHECK(lane_hi_bits(mn.at<bfloat16x2_t>({0})) == h_bits(narrow_lane<bfloat16_t>(-8.0f)));
        CHECK(lane_lo_bits(mx.at<bfloat16x2_t>({0})) == h_bits(narrow_lane<bfloat16_t>(5.0f)));
        CHECK(lane_hi_bits(mx.at<bfloat16x2_t>({0})) == h_bits(narrow_lane<bfloat16_t>(7.0f)));
    }
    {
        // Int4x2 per-lane signed: lo-lane min from one element, hi-lane min
        // from another; max likewise.
        auto t = Tensor({3}, DType::Int4x2, Device::CPU);
        t.set<int4x2_t>({0}, nib_pack(3, -5));
        t.set<int4x2_t>({1}, nib_pack(-8, 2));
        t.set<int4x2_t>({2}, nib_pack(1, 7));
        CHECK(nib_lo(t.min().at<int4x2_t>({0})) == int8_t{-8});
        CHECK(nib_hi(t.min().at<int4x2_t>({0})) == int8_t{-5});
        CHECK(nib_lo(t.max().at<int4x2_t>({0})) == int8_t{3});
        CHECK(nib_hi(t.max().at<int4x2_t>({0})) == int8_t{7});
    }
    {
        // Empty tensor throws IndexOutOfRange (same contract as mean/argmax).
        auto empty = Tensor({0}, DType::Float32, Device::CPU);
        auto threw_min = false;
        auto threw_max = false;
        try {
            auto bad = empty.min();
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
            threw_min = true;
        }
        try {
            auto bad = empty.max();
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::IndexOutOfRange&) {
            threw_max = true;
        }
        CHECK(threw_min);
        CHECK(threw_max);
    }
    {
        // Non-contiguous operand throws NotImplemented (same as mean).
        auto t = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
        auto inner = t.slice(ranges);
        auto threw = false;
        try {
            auto bad = inner.min();
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
    }
}

template<typename T>
static void check_int_minmax(const DType dt) {
    // Planted extremes at the dtype limits, duplicated tie in the middle.
    auto t = Tensor({6}, dt, Device::CPU);
    const auto lo = std::numeric_limits<T>::lowest();
    const auto hi = std::numeric_limits<T>::max();
    t.set<T>({0}, T{3});
    t.set<T>({1}, T{0});
    t.set<T>({2}, lo);
    t.set<T>({3}, hi);
    t.set<T>({4}, lo); // tie
    t.set<T>({5}, T{1});
    auto mn = t.min();
    auto mx = t.max();
    CHECK(mn.is_same_shape(Tensor({1})));
    CHECK(mn.get_dtype() == dt);
    CHECK(mn.at<T>({0}) == lo);
    CHECK(mx.at<T>({0}) == hi);
}

static void test_minmax_int_cpu() {
    check_int_minmax<int8_t>(DType::Int8);
    check_int_minmax<uint8_t>(DType::UInt8);
    check_int_minmax<int16_t>(DType::Int16);
    check_int_minmax<uint16_t>(DType::UInt16);
    check_int_minmax<int32_t>(DType::Int32);
    check_int_minmax<uint32_t>(DType::UInt32);
    check_int_minmax<int64_t>(DType::Int64);
    check_int_minmax<uint64_t>(DType::UInt64);
}

// --- GPU paths ---

static void test_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    {
        // Row-wise argmax on device: same per-row indices as CPU.
        auto logits = Tensor({3, 10}, DType::Float32, Device::GPU);
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 10; ++c) {
                logits.set<float32_t>({r, c}, 0.25f);
            }
        }
        logits.set<float32_t>({0, 2}, 9.0f);
        logits.set<float32_t>({1, 7}, 9.0f);
        logits.set<float32_t>({2, 0}, 9.0f);
        auto pred = logits.argmax(1);
        pred.sync();
        CHECK(pred.at<int64_t>({0}) == 2);
        CHECK(pred.at<int64_t>({1}) == 7);
        CHECK(pred.at<int64_t>({2}) == 0);
        // Flat argmax agrees with the row-wise result.
        CHECK(logits.argmax() == uint64_t{2}); // first tied maximum wins (flat 2; contract: basic_ops_tests tie test)
    }
    {
        // Equal on device: NaN never equals, -0.0 equals 0.0.
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto a = Tensor({4}, DType::Float32, Device::GPU);
        a.set<float32_t>({0}, nan);
        a.set<float32_t>({1}, -0.0f);
        a.set<float32_t>({2}, 3.0f);
        a.set<float32_t>({3}, 1.0f);
        auto b = Tensor({4}, DType::Float32, Device::GPU);
        b.set<float32_t>({0}, nan);
        b.set<float32_t>({1}, 0.0f);
        b.set<float32_t>({2}, 3.0f);
        b.set<float32_t>({3}, 9.0f);
        auto mask = a.equal(b);
        mask.sync();
        CHECK(mask.at<uint8_t>({0}) == 0);
        CHECK(mask.at<uint8_t>({1}) == 1);
        CHECK(mask.at<uint8_t>({2}) == 1);
        CHECK(mask.at<uint8_t>({3}) == 0);
    }
#if defined(USE_HIP)
    {
        // Mean on device (HIP backend): mask -> float -> mean is expressible
        // end-to-end on the GPU.
        auto mask = Tensor({4}, DType::UInt8, Device::GPU);
        mask.set<uint8_t>({0}, 1);
        mask.set<uint8_t>({1}, 1);
        mask.set<uint8_t>({2}, 0);
        mask.set<uint8_t>({3}, 0);
        mask.convert_dtype(DType::Float32);
        auto m = mask.mean();
        m.sync();
        CHECK(near(m.at<float32_t>({0}), 0.5f, 1e-6));
    }
#endif
    {
        // Full-tensor min/max on device: same planted extremes as a CPU twin,
        // and the device result is bit-identical to the CPU result (the
        // comparison combine is order-independent).
        auto gpu = Tensor({7}, DType::Float32, Device::GPU);
        auto cpu = Tensor({7}, DType::Float32, Device::CPU);
        const float32_t vals[7] = {3.0f, -1.5f, 2.0f, -1.5f, 10.25f, 0.0f, -0.0f};
        for (size_t i = 0; i < 7; ++i) {
            gpu.set<float32_t>({i}, vals[i]);
            cpu.set<float32_t>({i}, vals[i]);
        }
        auto gpu_min = gpu.min();
        auto gpu_max = gpu.max();
        gpu_min.sync();
        gpu_max.sync();
        auto cpu_min = cpu.min();
        auto cpu_max = cpu.max();
        CHECK(idet::bit_cast_memcpy<uint32_t>(gpu_min.at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(cpu_min.at<float32_t>({0})));
        CHECK(idet::bit_cast_memcpy<uint32_t>(gpu_max.at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(cpu_max.at<float32_t>({0})));
        CHECK(idet::bit_cast_memcpy<uint32_t>(gpu_min.at<float32_t>({0})) == idet::bit_cast_memcpy<uint32_t>(-1.5f));
    }
    {
        // NaN on device propagates as the canonical quiet NaN (0x7FC00000),
        // matching the CPU result bit-for-bit.
        constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
        auto gpu = Tensor({4}, DType::Float32, Device::GPU);
        gpu.set<float32_t>({0}, 1.0f);
        gpu.set<float32_t>({1}, nan);
        gpu.set<float32_t>({2}, -7.0f);
        gpu.set<float32_t>({3}, 42.0f);
        auto gpu_min = gpu.min();
        gpu_min.sync();
        CHECK(idet::bit_cast_memcpy<uint32_t>(gpu_min.at<float32_t>({0})) == UINT32_C(0x7FC00000));
    }
    {
        // Integer min/max on device, bit-identical to the CPU twin.
        auto gpu = Tensor({6}, DType::Int32, Device::GPU);
        auto cpu = Tensor({6}, DType::Int32, Device::CPU);
        const int32_t vals[6] = {3, -5, 7, -5, 0, 1};
        for (size_t i = 0; i < 6; ++i) {
            gpu.set<int32_t>({i}, vals[i]);
            cpu.set<int32_t>({i}, vals[i]);
        }
        auto gpu_min = gpu.min();
        auto gpu_max = gpu.max();
        gpu_min.sync();
        gpu_max.sync();
        CHECK(gpu_min.at<int32_t>({0}) == cpu.min().at<int32_t>({0}));
        CHECK(gpu_max.at<int32_t>({0}) == cpu.max().at<int32_t>({0}));
        CHECK(gpu_min.at<int32_t>({0}) == int32_t{-5});
        CHECK(gpu_max.at<int32_t>({0}) == int32_t{7});
    }
    {
        // Packed per-lane min on device, bit-identical to the CPU twin.
        auto gpu = Tensor({3}, DType::Float16x2, Device::GPU);
        auto cpu = Tensor({3}, DType::Float16x2, Device::CPU);
        const uint32_t bits[3] = {
            idet::bit_cast_memcpy<uint32_t>(pack2_bits<float16x2_t>(uint16_t{0x7E00}, h_bits(narrow_lane<float16_t>(4.0f)))),
            idet::bit_cast_memcpy<uint32_t>(pack2<float16x2_t, float16_t>(2.0f, 9.0f)),
            idet::bit_cast_memcpy<uint32_t>(pack2<float16x2_t, float16_t>(-3.0f, 1.0f)),
        };
        for (size_t i = 0; i < 3; ++i) {
            gpu.set<float16x2_t>({i}, idet::bit_cast_memcpy<float16x2_t>(bits[i]));
            cpu.set<float16x2_t>({i}, idet::bit_cast_memcpy<float16x2_t>(bits[i]));
        }
        auto gpu_min = gpu.min();
        gpu_min.sync();
        auto cpu_min = cpu.min();
        CHECK(idet::bit_cast_memcpy<uint32_t>(gpu_min.at<float16x2_t>({0})) == idet::bit_cast_memcpy<uint32_t>(cpu_min.at<float16x2_t>({0})));
        CHECK(lane_lo_bits(gpu_min.at<float16x2_t>({0})) == uint16_t{0x7E00});
        CHECK(lane_hi_bits(gpu_min.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(1.0f)));
    }
    {
        // Int4x2 per-lane signed min on device, bit-identical to the CPU twin.
        auto gpu = Tensor({3}, DType::Int4x2, Device::GPU);
        auto cpu = Tensor({3}, DType::Int4x2, Device::CPU);
        const int4x2_t vals[3] = {nib_pack(3, -5), nib_pack(-8, 2), nib_pack(1, 7)};
        for (size_t i = 0; i < 3; ++i) {
            gpu.set<int4x2_t>({i}, vals[i]);
            cpu.set<int4x2_t>({i}, vals[i]);
        }
        auto gpu_min = gpu.min();
        auto gpu_max = gpu.max();
        gpu_min.sync();
        gpu_max.sync();
        CHECK(idet::bit_cast_memcpy<uint8_t>(gpu_min.at<int4x2_t>({0})) == idet::bit_cast_memcpy<uint8_t>(cpu.min().at<int4x2_t>({0})));
        CHECK(idet::bit_cast_memcpy<uint8_t>(gpu_max.at<int4x2_t>({0})) == idet::bit_cast_memcpy<uint8_t>(cpu.max().at<int4x2_t>({0})));
        CHECK(nib_lo(gpu_min.at<int4x2_t>({0})) == int8_t{-8});
        CHECK(nib_hi(gpu_max.at<int4x2_t>({0})) == int8_t{7});
    }
}

// --- KernelTune reduction block-size invariance ---

static void force_reduction_block_size(const size_t bt) noexcept {
    set_override(GpuFamily::Reduction, SizeClass::Tiny, bt);
    set_override(GpuFamily::Reduction, SizeClass::Small, bt);
    set_override(GpuFamily::Reduction, SizeClass::Medium, bt);
    set_override(GpuFamily::Reduction, SizeClass::Large, bt);
}

struct ReductionBlockOverrideGuard {
    explicit ReductionBlockOverrideGuard(const size_t bt) {
        force_reduction_block_size(bt);
    }
    ~ReductionBlockOverrideGuard() {
        clear_override();
    }
};

static void test_kernel_tune_gpu_invariance() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available for KernelTune invariance\n";
        return;
    }

    // Float32 min/max with planted extremes and a NaN. The comparison combine
    // is associative/commutative and canonicalizes ties, so the GPU result is
    // bit-identical across any valid block size and against the CPU twin.
    constexpr auto nan = std::numeric_limits<float32_t>::quiet_NaN();
    constexpr auto k_flen = size_t{1000};
    const auto make_float_case = [&]() {
        auto gpu = Tensor({k_flen}, DType::Float32, Device::GPU);
        auto cpu = Tensor({k_flen}, DType::Float32, Device::CPU);
        for (size_t i = 0; i < k_flen; ++i) {
            const auto v = static_cast<float32_t>(static_cast<int32_t>(i % 100) - 50);
            gpu.set<float32_t>({i}, v);
            cpu.set<float32_t>({i}, v);
        }
        gpu.set<float32_t>({42}, nan);
        cpu.set<float32_t>({42}, nan);
        gpu.set<float32_t>({123}, 12345.0f);
        cpu.set<float32_t>({123}, 12345.0f);
        gpu.set<float32_t>({456}, -12345.0f);
        cpu.set<float32_t>({456}, -12345.0f);
        return std::pair{std::move(gpu), std::move(cpu)};
    };

    const auto default_float = make_float_case();
    auto default_min = default_float.first.min();
    auto default_max = default_float.first.max();
    default_min.sync();
    default_max.sync();
    auto cpu_min = default_float.second.min();
    auto cpu_max = default_float.second.max();

    const auto default_min_bits = idet::bit_cast_memcpy<uint32_t>(default_min.at<float32_t>({0}));
    const auto default_max_bits = idet::bit_cast_memcpy<uint32_t>(default_max.at<float32_t>({0}));
    const auto cpu_min_bits = idet::bit_cast_memcpy<uint32_t>(cpu_min.at<float32_t>({0}));
    const auto cpu_max_bits = idet::bit_cast_memcpy<uint32_t>(cpu_max.at<float32_t>({0}));

    for (const auto bt : {size_t{128}, size_t{1024}}) {
        const ReductionBlockOverrideGuard guard(bt);
        const auto c = make_float_case();
        auto bt_min = c.first.min();
        auto bt_max = c.first.max();
        bt_min.sync();
        bt_max.sync();
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_min.at<float32_t>({0})) == default_min_bits);
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_max.at<float32_t>({0})) == default_max_bits);
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_min.at<float32_t>({0})) == cpu_min_bits);
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_max.at<float32_t>({0})) == cpu_max_bits);
    }

    // Packed fp16x2 min with a NaN lane: per-lane combine is independent, so
    // the lane winners are bit-identical across block sizes and against CPU.
    const auto make_fp16x2_case = [&]() {
        auto gpu = Tensor({k_flen}, DType::Float16x2, Device::GPU);
        auto cpu = Tensor({k_flen}, DType::Float16x2, Device::CPU);
        for (size_t i = 0; i < k_flen; ++i) {
            const auto lo = static_cast<float32_t>((static_cast<int32_t>(i) % 50) - 25);
            const auto hi = static_cast<float32_t>((static_cast<int32_t>(i) % 37) - 18);
            gpu.set<float16x2_t>({i}, pack2<float16x2_t, float16_t>(lo, hi));
            cpu.set<float16x2_t>({i}, pack2<float16x2_t, float16_t>(lo, hi));
        }
        gpu.set<float16x2_t>({42}, pack2_bits<float16x2_t>(uint16_t{0x7E00}, h_bits(narrow_lane<float16_t>(7.0f))));
        cpu.set<float16x2_t>({42}, pack2_bits<float16x2_t>(uint16_t{0x7E00}, h_bits(narrow_lane<float16_t>(7.0f))));
        return std::pair{std::move(gpu), std::move(cpu)};
    };

    const auto default_fp16x2 = make_fp16x2_case();
    auto default_p_min = default_fp16x2.first.min();
    default_p_min.sync();
    auto cpu_p_min = default_fp16x2.second.min();
    const auto default_p_min_bits = idet::bit_cast_memcpy<uint32_t>(default_p_min.at<float16x2_t>({0}));
    const auto cpu_p_min_bits = idet::bit_cast_memcpy<uint32_t>(cpu_p_min.at<float16x2_t>({0}));

    for (const auto bt : {size_t{128}, size_t{1024}}) {
        const ReductionBlockOverrideGuard guard(bt);
        const auto c = make_fp16x2_case();
        auto bt_min = c.first.min();
        bt_min.sync();
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_min.at<float16x2_t>({0})) == default_p_min_bits);
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_min.at<float16x2_t>({0})) == cpu_p_min_bits);
        CHECK(lane_lo_bits(bt_min.at<float16x2_t>({0})) == uint16_t{0x7E00});
        CHECK(lane_hi_bits(bt_min.at<float16x2_t>({0})) == h_bits(narrow_lane<float16_t>(-18.0f)));
    }

    // Int32 sum: the accumulator is modular (uint64 internally, stored as int32),
    // so the result is order-independent and bit-identical across block sizes
    // and against the CPU twin.
    constexpr auto k_ilen = size_t{500};
    const auto make_int_case = [&]() {
        auto gpu = Tensor({k_ilen}, DType::Int32, Device::GPU);
        auto cpu = Tensor({k_ilen}, DType::Int32, Device::CPU);
        for (size_t i = 0; i < k_ilen; ++i) {
            const auto v = static_cast<int32_t>((static_cast<int32_t>(i) % 20) - 10);
            gpu.set<int32_t>({i}, v);
            cpu.set<int32_t>({i}, v);
        }
        return std::pair{std::move(gpu), std::move(cpu)};
    };

    const auto default_int = make_int_case();
    auto default_i_sum = default_int.first.sum();
    default_i_sum.sync();
    auto cpu_i_sum = default_int.second.sum();
    const auto default_i_bits = idet::bit_cast_memcpy<uint32_t>(default_i_sum.at<int32_t>({0}));
    const auto cpu_i_bits = idet::bit_cast_memcpy<uint32_t>(cpu_i_sum.at<int32_t>({0}));

    for (const auto bt : {size_t{128}, size_t{1024}}) {
        const ReductionBlockOverrideGuard guard(bt);
        const auto c = make_int_case();
        auto bt_sum = c.first.sum();
        bt_sum.sync();
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_sum.at<int32_t>({0})) == default_i_bits);
        CHECK(idet::bit_cast_memcpy<uint32_t>(bt_sum.at<int32_t>({0})) == cpu_i_bits);
    }

    // Float32 sum: the order of accumulation depends on the block size, so the
    // result is only approximately equal across forced sizes and against CPU.
    const auto make_float_sum_case = [&]() {
        auto gpu = Tensor({k_flen}, DType::Float32, Device::GPU);
        auto cpu = Tensor({k_flen}, DType::Float32, Device::CPU);
        for (size_t i = 0; i < k_flen; ++i) {
            const auto v = 1.0f + static_cast<float32_t>(static_cast<int32_t>(i % 10)) * 0.1f;
            gpu.set<float32_t>({i}, v);
            cpu.set<float32_t>({i}, v);
        }
        return std::pair{std::move(gpu), std::move(cpu)};
    };

    const auto default_float_sum = make_float_sum_case();
    auto default_s_sum = default_float_sum.first.sum();
    default_s_sum.sync();
    auto cpu_s_sum = default_float_sum.second.sum();
    const auto default_s = default_s_sum.at<float32_t>({0});
    const auto cpu_s = cpu_s_sum.at<float32_t>({0});

    for (const auto bt : {size_t{128}, size_t{1024}}) {
        const ReductionBlockOverrideGuard guard(bt);
        const auto c = make_float_sum_case();
        auto bt_sum = c.first.sum();
        bt_sum.sync();
        CHECK(near(bt_sum.at<float32_t>({0}), default_s, 1e-3f));
        CHECK(near(bt_sum.at<float32_t>({0}), cpu_s, 1e-3f));
    }
}

int main() {
    test_argmax_axis_cpu();
    test_equal_cpu();
    test_accuracy_cpu();
    test_minmax_cpu();
    test_minmax_int_cpu();
    test_gpu();
    test_kernel_tune_gpu_invariance();
    if (failures == 0) {
        std::cout << "ALL METRICS TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
