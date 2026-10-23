/**
 * @file metrics_tests.cpp
 * @author GreenBean
 * @brief Classification-metrics tests - framework-free assertion suite.
 * Covers row-wise argmax(axis) (per-slice int64 indices, tie-breaking,
 * NaN-wins), elementwise equal masks across dtypes (IEEE == semantics:
 * NaN never equal, -0.0 equals 0.0), mask-to-float casts, and the
 * end-to-end accuracy pipeline argmax -> equal -> cast -> mean, on CPU
 * and on the GPU (CUDA/HIP) paths when a backend and device exist.
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

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float16_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
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

// --- Half helpers (host-side, via the unaccel conversion pathway) ---

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
        const auto nan = std::numeric_limits<float32_t>::quiet_NaN();
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
        const auto nan = std::numeric_limits<float32_t>::quiet_NaN();
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
        const auto nan = std::numeric_limits<float32_t>::quiet_NaN();
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
        const auto nan = std::numeric_limits<float32_t>::quiet_NaN();
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
        const auto nan = std::numeric_limits<float32_t>::quiet_NaN();
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
}

int main() {
    test_argmax_axis_cpu();
    test_equal_cpu();
    test_accuracy_cpu();
    test_gpu();
    if (failures == 0) {
        std::cout << "ALL METRICS TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
