/**
 * @file bce_tests.cpp
 * @author GreenBean
 * @brief BCE loss tests — framework-free assertion suite.
 * Covers both probability form (bce) and stable logit form (bce_with_logits):
 * hand forward/backward values including soft targets, clamp boundaries,
 * invalid targets, shape/dtype/device/contiguity/empty error paths, huge and
 * infinite logits with formula-exact NaN checks, F64 finite-difference
 * gradients away from the clamp kink, non-unit upstream, autograd chains,
 * repeated backward accumulation, per-call grad_dtype probes, SGD/Adam
 * loss-decrease smoke, and CPU/GPU parity when a backend is present.
 * @version 0.1
 * @date 2026-11-10
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
#include "Math/Autograd/GradDType.h"
#include "Math/Loss/Loss.h"
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
using DType = BeanTensor::DataTypes::DType;
namespace idet = BeanTensor::Intrinsics::detail;
namespace tdetail = BeanTensor::Tensors::detail;

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

template<typename A, typename B>
static bool near64(const A a, const B b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - static_cast<float64_t>(b)) <= eps;
}

static bool is_nan64(const float64_t v) {
    return std::isnan(v);
}

// Reference probability-form BCE for one lane, using the same clamp.
template<typename Acc>
static Acc ref_bce_lane(const Acc p, const Acc y) {
    const auto eps = [] {
        if constexpr (std::is_same_v<Acc, float64_t>) {
            return float64_t{1e-15};
        } else {
            return Acc{1e-7f};
        }
    }();
    auto pc = p;
    if (pc < eps) {
        pc = eps;
    }
    if (pc > Acc{1} - eps) {
        pc = Acc{1} - eps;
    }
    return -(y * std::log(pc)) - ((Acc{1} - y) * std::log(Acc{1} - pc));
}

// Reference stable logit-form BCE for one lane.
template<typename Acc>
static Acc ref_bce_logits_lane(const Acc x, const Acc y) {
    const auto abs_x = std::fabs(x);
    const auto max_term = std::max(x, Acc{0});
    return max_term - x * y + std::log1p(std::exp(-abs_x));
}

// Stable sigmoid reference.
template<typename Acc>
static Acc ref_stable_sigmoid(const Acc x) {
    if (x >= Acc{0}) {
        const auto z = std::exp(-x);
        return Acc{1} / (Acc{1} + z);
    }
    const auto z = std::exp(x);
    return z / (Acc{1} + z);
}

// =====================================================================
// forward: probability form
// =====================================================================

static void test_bce_forward_hand() {
    {
        // p = [0.9, 0.1, 0.75], y = [1, 0, 1]
        // lane losses: -log(0.9), -log(0.9), -log(0.75)
        auto pred = Tensor({3}, DType::Float32, Device::CPU);
        auto target = Tensor({3}, DType::Float32, Device::CPU);
        pred.set<float32_t>({0}, 0.9f);
        pred.set<float32_t>({1}, 0.1f);
        pred.set<float32_t>({2}, 0.75f);
        target.set<float32_t>({0}, 1.0f);
        target.set<float32_t>({1}, 0.0f);
        target.set<float32_t>({2}, 1.0f);
        auto loss = pred.bce(target);
        const auto want = (std::log(1.0f / 0.9f) + std::log(1.0f / 0.9f) + std::log(1.0f / 0.75f)) / 3.0f;
        CHECK(loss.is_same_shape(Tensor({1})));
        CHECK(loss.get_dtype() == DType::Float32);
        CHECK(near64(loss.at<float32_t>({0}), want, 1e-6));
    }
    {
        // Soft targets: p = [0.2, 0.8], y = [0.3, 0.6]
        auto pred = Tensor({2}, DType::Float64, Device::CPU);
        auto target = Tensor({2}, DType::Float64, Device::CPU);
        pred.set<float64_t>({0}, 0.2);
        pred.set<float64_t>({1}, 0.8);
        target.set<float64_t>({0}, 0.3);
        target.set<float64_t>({1}, 0.6);
        auto loss = pred.bce(target);
        const auto want = (ref_bce_lane(0.2, 0.3) + ref_bce_lane(0.8, 0.6)) / 2.0;
        CHECK(loss.is_same_shape(Tensor({1})));
        CHECK(loss.get_dtype() == DType::Float64);
        CHECK(near64(loss.at<float64_t>({0}), want, 1e-12));
    }
}

static void test_bce_clamp_boundaries() {
    {
        // p=0 and p=1 hit the clamp.
        auto pred = Tensor({2}, DType::Float32, Device::CPU);
        auto target = Tensor({2}, DType::Float32, Device::CPU);
        pred.set<float32_t>({0}, 0.0f);
        pred.set<float32_t>({1}, 1.0f);
        target.set_zero();
        target.set<float32_t>({1}, 1.0f);
        auto loss = pred.bce(target);
        const auto want = (ref_bce_lane(0.0f, 0.0f) + ref_bce_lane(1.0f, 1.0f)) / 2.0f;
        CHECK(near64(loss.at<float32_t>({0}), want, 1e-6));
    }
    {
        // Just inside/outside the clamp region.
        auto pred = Tensor({4}, DType::Float32, Device::CPU);
        auto target = Tensor({4}, DType::Float32, Device::CPU);
        pred.set<float32_t>({0}, 1e-8f);
        pred.set<float32_t>({1}, 1.0f - 1e-8f);
        pred.set<float32_t>({2}, -1e-8f);
        pred.set<float32_t>({3}, 1.0f + 1e-8f);
        target.set_zero();
        target.set<float32_t>({1}, 1.0f);
        target.set<float32_t>({2}, 0.0f);
        target.set<float32_t>({3}, 1.0f);
        auto loss = pred.bce(target);
        const auto want = (ref_bce_lane(1e-8f, 0.0f) + ref_bce_lane(1.0f - 1e-8f, 1.0f) +
                           ref_bce_lane(-1e-8f, 0.0f) + ref_bce_lane(1.0f + 1e-8f, 1.0f)) / 4.0f;
        CHECK(near64(loss.at<float32_t>({0}), want, 1e-6));
    }
    {
        // fp64 clamp eps = 1e-15.
        auto pred = Tensor({2}, DType::Float64, Device::CPU);
        auto target = Tensor({2}, DType::Float64, Device::CPU);
        pred.set<float64_t>({0}, 1e-16);
        pred.set<float64_t>({1}, 1.0 - 1e-16);
        target.set_zero();
        target.set<float64_t>({1}, 1.0);
        auto loss = pred.bce(target);
        const auto want = (ref_bce_lane(1e-16, 0.0) + ref_bce_lane(1.0 - 1e-16, 1.0)) / 2.0;
        CHECK(near64(loss.at<float64_t>({0}), want, 1e-12));
    }
}

static void test_bce_logits_forward_hand() {
    {
        // x = [0, 2, -1], y = [1, 1, 0]
        auto logits = Tensor({3}, DType::Float32, Device::CPU);
        auto target = Tensor({3}, DType::Float32, Device::CPU);
        logits.set<float32_t>({0}, 0.0f);
        logits.set<float32_t>({1}, 2.0f);
        logits.set<float32_t>({2}, -1.0f);
        target.set<float32_t>({0}, 1.0f);
        target.set<float32_t>({1}, 1.0f);
        target.set<float32_t>({2}, 0.0f);
        auto loss = logits.bce_with_logits(target);
        const auto want = (ref_bce_logits_lane(0.0f, 1.0f) + ref_bce_logits_lane(2.0f, 1.0f) +
                           ref_bce_logits_lane(-1.0f, 0.0f)) / 3.0f;
        CHECK(loss.is_same_shape(Tensor({1})));
        CHECK(near64(loss.at<float32_t>({0}), want, 1e-6));
    }
    {
        // Huge logits should not overflow.
        auto logits = Tensor({2}, DType::Float32, Device::CPU);
        auto target = Tensor({2}, DType::Float32, Device::CPU);
        logits.set<float32_t>({0}, 1e4f);
        logits.set<float32_t>({1}, -1e4f);
        target.set<float32_t>({0}, 1.0f);
        target.set<float32_t>({1}, 0.0f);
        auto loss = logits.bce_with_logits(target);
        loss.sync();
        CHECK(std::isfinite(loss.at<float32_t>({0})));
    }
    {
        // Soft targets.
        auto logits = Tensor({2}, DType::Float64, Device::CPU);
        auto target = Tensor({2}, DType::Float64, Device::CPU);
        logits.set<float64_t>({0}, 0.5);
        logits.set<float64_t>({1}, -0.5);
        target.set<float64_t>({0}, 0.25);
        target.set<float64_t>({1}, 0.75);
        auto loss = logits.bce_with_logits(target);
        const auto want = (ref_bce_logits_lane(0.5, 0.25) + ref_bce_logits_lane(-0.5, 0.75)) / 2.0;
        CHECK(near64(loss.at<float64_t>({0}), want, 1e-12));
    }
}

static void test_bce_logits_infinities() {
    {
        // x = +Inf, y = 1 -> max(Inf,0)=Inf, -Inf*1=-Inf, log1p(exp(-Inf))=0
        // => Inf - Inf = NaN. Formula-exact, matches PyTorch.
        auto logits = Tensor({2}, DType::Float64, Device::CPU);
        auto target = Tensor({2}, DType::Float64, Device::CPU);
        logits.set<float64_t>({0}, std::numeric_limits<float64_t>::infinity());
        logits.set<float64_t>({1}, -std::numeric_limits<float64_t>::infinity());
        target.set<float64_t>({0}, 1.0);
        target.set<float64_t>({1}, 0.0);
        auto loss = logits.bce_with_logits(target);
        loss.sync();
        CHECK(is_nan64(loss.at<float64_t>({0})));
    }
    {
        // x = +Inf, y = 0 -> max(Inf,0)=Inf, Inf*0=NaN, log1p(exp(-Inf))=0
        // => NaN. Formula-exact, matches PyTorch.
        auto logits = Tensor({1}, DType::Float64, Device::CPU);
        auto target = Tensor({1}, DType::Float64, Device::CPU);
        logits.set<float64_t>({0}, std::numeric_limits<float64_t>::infinity());
        target.set_zero();
        auto loss = logits.bce_with_logits(target);
        loss.sync();
        CHECK(is_nan64(loss.at<float64_t>({0})));
    }
    {
        // x = -Inf, y = 1 -> max(-Inf,0)=0, -(-Inf)*1=+Inf, log1p(exp(-Inf))=0
        // => Inf. Finite, not NaN.
        auto logits = Tensor({1}, DType::Float64, Device::CPU);
        auto target = Tensor({1}, DType::Float64, Device::CPU);
        logits.set<float64_t>({0}, -std::numeric_limits<float64_t>::infinity());
        target.set<float64_t>({0}, 1.0);
        auto loss = logits.bce_with_logits(target);
        loss.sync();
        CHECK(std::isinf(loss.at<float64_t>({0})));
    }
}

// =====================================================================
// backward: probability form
// =====================================================================

static void test_bce_g_values(const Device dev) {
    auto pred = Tensor({3}, DType::Float64, dev, true);
    auto target = Tensor({3}, DType::Float64, dev, false);
    pred.set<float64_t>({0}, 0.9);
    pred.set<float64_t>({1}, 0.1);
    pred.set<float64_t>({2}, 0.75);
    target.set<float64_t>({0}, 1.0);
    target.set<float64_t>({1}, 0.0);
    target.set<float64_t>({2}, 1.0);
    auto loss = pred.bce(target);
    loss.backward(DType::Float64);
    CHECK(pred.has_grad());
    CHECK(!target.has_grad());
    for (auto i = size_t{0}; i < 3; ++i) {
        const auto p = pred.at<float64_t>({i});
        const auto y = target.at<float64_t>({i});
        const auto pc = [] (const float64_t v) {
            constexpr auto eps = float64_t{1e-15};
            if (v < eps) {
                return eps;
            }
            if (v > 1.0 - eps) {
                return 1.0 - eps;
            }
            return v;
        }(p);
        const auto want = (((1.0 - y) / (1.0 - pc)) - (y / pc)) / 3.0;
        CHECK(near64(pred.gradient().at<float64_t>({i}), want, 1e-12));
    }
}

static void test_bce_g_clamp(const Device dev) {
    auto pred = Tensor({2}, DType::Float32, dev, true);
    auto target = Tensor({2}, DType::Float32, dev, false);
    pred.set<float32_t>({0}, 0.0f);
    pred.set<float32_t>({1}, 1.0f);
    target.set_zero();
    target.set<float32_t>({1}, 1.0f);
    auto loss = pred.bce(target);
    loss.backward(DType::Float32);
    // p=0, y=0: clamped to eps=1e-7; grad = (1-0)/(1-1e-7) - 0/1e-7 = ~1 / 2.
    // p=1, y=1: clamped to 1-eps; grad = 0/(eps) - 1/(1-eps) = ~-1 / 2.
    CHECK(near64(pred.gradient().at<float32_t>({0}), 1.0f / (1.0f - 1e-7f) / 2.0f, 1e-6f));
    CHECK(near64(pred.gradient().at<float32_t>({1}), -1.0f / (1.0f - 1e-7f) / 2.0f, 1e-6f));
}

// =====================================================================
// backward: logit form
// =====================================================================

static void test_bce_logits_g_values(const Device dev) {
    auto logits = Tensor({3}, DType::Float64, dev, true);
    auto target = Tensor({3}, DType::Float64, dev, false);
    logits.set<float64_t>({0}, 0.0);
    logits.set<float64_t>({1}, 2.0);
    logits.set<float64_t>({2}, -1.0);
    target.set<float64_t>({0}, 1.0);
    target.set<float64_t>({1}, 1.0);
    target.set<float64_t>({2}, 0.0);
    auto loss = logits.bce_with_logits(target);
    loss.backward(DType::Float64);
    CHECK(logits.has_grad());
    CHECK(!target.has_grad());
    for (auto i = size_t{0}; i < 3; ++i) {
        const auto x = logits.at<float64_t>({i});
        const auto y = target.at<float64_t>({i});
        const auto want = (ref_stable_sigmoid(x) - y) / 3.0;
        CHECK(near64(logits.gradient().at<float64_t>({i}), want, 1e-12));
    }
}

// =====================================================================
// finite-difference gradient check (fp64, away from clamp kink)
// =====================================================================

static void test_bce_fd_probability() {
    const auto n = size_t{4};
    auto pred = Tensor({n}, DType::Float64, Device::CPU, true);
    auto target = Tensor({n}, DType::Float64, Device::CPU, false);
    const auto pv = std::vector<float64_t>{0.25, 0.55, 0.40, 0.80};
    const auto tv = std::vector<float64_t>{0.0, 1.0, 0.5, 0.2};
    for (auto i = size_t{0}; i < n; ++i) {
        pred.set<float64_t>({i}, pv[i]);
        target.set<float64_t>({i}, tv[i]);
    }
    auto loss = pred.bce(target);
    loss.backward(DType::Float64);
    const auto h = 1e-6;
    for (auto i = size_t{0}; i < n; ++i) {
        auto pred_plus = Tensor({n}, DType::Float64, Device::CPU, false);
        auto pred_minus = Tensor({n}, DType::Float64, Device::CPU, false);
        for (auto j = size_t{0}; j < n; ++j) {
            pred_plus.set<float64_t>({j}, pv[j] + (i == j ? h : 0.0));
            pred_minus.set<float64_t>({j}, pv[j] - (i == j ? h : 0.0));
        }
        auto f_plus = pred_plus.bce(target);
        auto f_minus = pred_minus.bce(target);
        const auto fd = (f_plus.at<float64_t>({0}) - f_minus.at<float64_t>({0})) / (2.0 * h);
        CHECK(near64(pred.gradient().at<float64_t>({i}), fd, 1e-8));
    }
}

static void test_bce_fd_logits() {
    const auto n = size_t{4};
    auto logits = Tensor({n}, DType::Float64, Device::CPU, true);
    auto target = Tensor({n}, DType::Float64, Device::CPU, false);
    const auto xv = std::vector<float64_t>{-0.5, 1.2, 0.0, -2.0};
    const auto tv = std::vector<float64_t>{0.0, 1.0, 0.5, 0.3};
    for (auto i = size_t{0}; i < n; ++i) {
        logits.set<float64_t>({i}, xv[i]);
        target.set<float64_t>({i}, tv[i]);
    }
    auto loss = logits.bce_with_logits(target);
    loss.backward(DType::Float64);
    const auto h = 1e-6;
    for (auto i = size_t{0}; i < n; ++i) {
        auto logits_plus = Tensor({n}, DType::Float64, Device::CPU, false);
        auto logits_minus = Tensor({n}, DType::Float64, Device::CPU, false);
        for (auto j = size_t{0}; j < n; ++j) {
            logits_plus.set<float64_t>({j}, xv[j] + (i == j ? h : 0.0));
            logits_minus.set<float64_t>({j}, xv[j] - (i == j ? h : 0.0));
        }
        auto f_plus = logits_plus.bce_with_logits(target);
        auto f_minus = logits_minus.bce_with_logits(target);
        const auto fd = (f_plus.at<float64_t>({0}) - f_minus.at<float64_t>({0})) / (2.0 * h);
        CHECK(near64(logits.gradient().at<float64_t>({i}), fd, 1e-8));
    }
}

// =====================================================================
// error paths
// =====================================================================

static void test_bce_errors() {
    {
        // mismatched shapes
        auto a = Tensor({3}, DType::Float32, Device::CPU);
        auto b = Tensor({4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // mismatched dtypes
        auto a = Tensor({3}, DType::Float32, Device::CPU);
        auto b = Tensor({3}, DType::Float64, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // integer storage
        auto a = Tensor({3}, DType::Int32, Device::CPU);
        auto b = Tensor({3}, DType::Int32, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // invalid target below 0
        auto a = Tensor({2}, DType::Float32, Device::CPU);
        auto b = Tensor({2}, DType::Float32, Device::CPU);
        a.set<float32_t>({0}, 0.5f);
        a.set<float32_t>({1}, 0.5f);
        b.set<float32_t>({0}, -0.1f);
        b.set<float32_t>({1}, 0.5f);
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::InvalidProbability&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // invalid target above 1
        auto a = Tensor({1}, DType::Float32, Device::CPU);
        auto b = Tensor({1}, DType::Float32, Device::CPU);
        a.set<float32_t>({0}, 0.5f);
        b.set<float32_t>({0}, 1.1f);
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::InvalidProbability&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // invalid target NaN
        auto a = Tensor({1}, DType::Float32, Device::CPU);
        auto b = Tensor({1}, DType::Float32, Device::CPU);
        a.set<float32_t>({0}, 0.5f);
        b.set<float32_t>({0}, std::numeric_limits<float32_t>::quiet_NaN());
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::InvalidProbability&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // empty
        auto a = Tensor(std::vector<size_t>{0}, DType::Float32, Device::CPU);
        auto b = Tensor(std::vector<size_t>{0}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // non-contiguous
        auto a = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
        auto inner = a.slice(ranges);
        auto b = Tensor({2, 2}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = inner.bce(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // no gradient path
        auto a = Tensor({2}, DType::Float32, Device::CPU);
        auto b = Tensor({2}, DType::Float32, Device::CPU);
        a.set_zero();
        b.set_zero();
        auto threw = false;
        try {
            auto bad = a.bce(b);
            static_cast<void>(bad);
            bad.backward(DType::Float32);
        } catch (const BeanTensor::ErrorHandling::NoGradient&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// =====================================================================
// autograd plumbing
// =====================================================================

static void test_bce_target_no_grad(const Device dev) {
    auto pred = Tensor({3}, DType::Float32, dev, true);
    auto target = Tensor({3}, DType::Float32, dev, true);
    pred.set<float32_t>({0}, 0.9f);
    pred.set<float32_t>({1}, 0.1f);
    pred.set<float32_t>({2}, 0.5f);
    target.set<float32_t>({0}, 1.0f);
    target.set<float32_t>({1}, 0.0f);
    target.set<float32_t>({2}, 1.0f);
    auto loss = pred.bce(target);
    loss.backward(DType::Float32);
    CHECK(pred.has_grad());
    CHECK(!target.has_grad());
}

static void test_bce_logits_chain(const Device dev) {
    // logits -> sigmoid -> bce should match bce_with_logits gradients on the logits.
    auto logits = Tensor({4}, DType::Float64, dev, true);
    auto target = Tensor({4}, DType::Float64, dev, false);
    const auto xv = std::vector<float64_t>{-0.5, 1.0, 0.25, -1.5};
    const auto tv = std::vector<float64_t>{0.0, 1.0, 0.5, 0.2};
    for (auto i = size_t{0}; i < 4; ++i) {
        logits.set<float64_t>({i}, xv[i]);
        target.set<float64_t>({i}, tv[i]);
    }
    auto prob = logits.sigmoid();
    auto loss = prob.bce(target);
    tdetail::grad_probe_reset();
    loss.backward(DType::Float64);
    CHECK(tdetail::grad_probe_count(DType::Float64) == 1); // bce_g delivery into the sigmoid output
    logits.zero_grad();
    auto direct_loss = logits.bce_with_logits(target);
    tdetail::grad_probe_reset();
    direct_loss.backward(DType::Float64);
    CHECK(tdetail::grad_probe_count(DType::Float64) == 0); // logits is a leaf, no intermediate delta
    for (auto i = size_t{0}; i < 4; ++i) {
        CHECK(near64(logits.gradient().at<float64_t>({i}), (ref_stable_sigmoid(xv[i]) - tv[i]) / 4.0, 1e-12));
    }
}

static void test_bce_grad_dtype_probe(const Device dev) {
    auto pred = Tensor({4}, DType::Float32, dev, true);
    auto target = Tensor({4}, DType::Float32, dev, false);
    pred.set_one();
    target.set_zero();
    for (const auto gd : std::vector{DType::Float64, DType::Float16, DType::Float32}) {
        auto loss = pred.bce(target);
        tdetail::grad_probe_reset();
        loss.backward(gd);
        CHECK(tdetail::grad_probe_count(gd) == 0); // pred is a leaf, no intermediate delta
        CHECK(pred.gradient().get_dtype() == gd);
        pred.zero_grad();
    }
}

static void test_bce_logits_grad_dtype_probe(const Device dev) {
    auto logits = Tensor({4}, DType::Float32, dev, true);
    auto target = Tensor({4}, DType::Float32, dev, false);
    logits.set_one();
    target.set_zero();
    for (const auto gd : std::vector{DType::Float64, DType::Float16, DType::Float32}) {
        auto loss = logits.bce_with_logits(target);
        tdetail::grad_probe_reset();
        loss.backward(gd);
        CHECK(tdetail::grad_probe_count(gd) == 0); // logits is a leaf, no intermediate delta
        CHECK(logits.gradient().get_dtype() == gd);
        logits.zero_grad();
    }
}

static void test_bce_repeated_accumulation(const Device dev) {
    auto pred = Tensor({2}, DType::Float32, dev, true);
    auto target = Tensor({2}, DType::Float32, dev, false);
    pred.set<float32_t>({0}, 0.8f);
    pred.set<float32_t>({1}, 0.2f);
    target.set<float32_t>({0}, 1.0f);
    target.set<float32_t>({1}, 0.0f);
    auto loss1 = pred.bce(target);
    loss1.backward(DType::Float32);
    auto g0 = pred.gradient().clone(true);
    auto loss2 = pred.bce(target);
    loss2.backward(DType::Float32);
    // Gradients should be doubled after two identical backward calls.
    for (auto i = size_t{0}; i < 2; ++i) {
        CHECK(near64(pred.gradient().at<float32_t>({i}), 2.0f * g0.at<float32_t>({i}), 1e-6f));
    }
}

static void test_bce_non_unit_upstream(const Device dev) {
    auto pred = Tensor({3}, DType::Float64, dev, true);
    auto target = Tensor({3}, DType::Float64, dev, false);
    pred.set<float64_t>({0}, 0.9);
    pred.set<float64_t>({1}, 0.1);
    pred.set<float64_t>({2}, 0.75);
    target.set<float64_t>({0}, 1.0);
    target.set<float64_t>({1}, 0.0);
    target.set<float64_t>({2}, 1.0);
    auto loss = pred.bce(target);
    auto scaled = loss.mul(2.5);
    scaled.backward(DType::Float64);
    for (auto i = size_t{0}; i < 3; ++i) {
        const auto p = pred.at<float64_t>({i});
        const auto y = target.at<float64_t>({i});
        const auto pc = [] (const float64_t v) {
            constexpr auto eps = float64_t{1e-15};
            if (v < eps) {
                return eps;
            }
            if (v > 1.0 - eps) {
                return 1.0 - eps;
            }
            return v;
        }(p);
        const auto want = 2.5 * (((1.0 - y) / (1.0 - pc)) - (y / pc)) / 3.0;
        CHECK(near64(pred.gradient().at<float64_t>({i}), want, 1e-12));
    }
}

// =====================================================================
// optimizer smoke
// =====================================================================

static void test_bce_sgd_smoke(const Device dev) {
    const auto n = size_t{8};
    auto w = Tensor({n}, DType::Float32, dev, true);
    auto target = Tensor({n}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        w.set<float32_t>({i}, 0.1f * static_cast<float32_t>(i) - 0.3f);
        target.set<float32_t>({i}, i % 2 == 0 ? 1.0f : 0.0f);
    }
    auto first = float64_t{0.0};
    for (auto step = size_t{0}; step < 200; ++step) {
        auto prob = w.sigmoid();
        auto loss = prob.bce(target);
        if (step == 0) {
            first = static_cast<float64_t>(loss.at<float32_t>({0}));
        }
        loss.backward(DType::Float32);
        w.sgd_step(0.5f);
        w.zero_grad();
    }
    auto prob = w.sigmoid();
    auto loss = prob.bce(target);
    const auto fin = static_cast<float64_t>(loss.at<float32_t>({0}));
    CHECK(fin < first);
}

static void test_bce_logits_adam_smoke(const Device dev) {
    const auto n = size_t{8};
    auto w = Tensor({n}, DType::Float32, dev, true);
    auto target = Tensor({n}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        w.set<float32_t>({i}, 0.1f * static_cast<float32_t>(i) - 0.3f);
        target.set<float32_t>({i}, i % 2 == 0 ? 1.0f : 0.0f);
    }
    auto first = float64_t{0.0};
    for (auto step = size_t{0}; step < 400; ++step) {
        auto loss = w.bce_with_logits(target);
        if (step == 0) {
            first = static_cast<float64_t>(loss.at<float32_t>({0}));
        }
        loss.backward(DType::Float32);
        w.adam_step(0.05, 0.9, 0.999, 1e-8);
        w.zero_grad();
    }
    auto loss = w.bce_with_logits(target);
    const auto fin = static_cast<float64_t>(loss.at<float32_t>({0}));
    CHECK(fin < first);
}

// =====================================================================
// free-function form
// =====================================================================

static void test_bce_free_functions() {
    auto pred = Tensor({2}, DType::Float32, Device::CPU);
    auto target = Tensor({2}, DType::Float32, Device::CPU);
    pred.set<float32_t>({0}, 0.9f);
    pred.set<float32_t>({1}, 0.1f);
    target.set<float32_t>({0}, 1.0f);
    target.set<float32_t>({1}, 0.0f);
    auto loss1 = BeanTensor::Loss::bce(pred, target);
    auto loss2 = pred.bce(target);
    CHECK(near64(loss1.at<float32_t>({0}), loss2.at<float32_t>({0}), 1e-6f));

    auto logits = Tensor({2}, DType::Float32, Device::CPU);
    logits.set<float32_t>({0}, 0.0f);
    logits.set<float32_t>({1}, 2.0f);
    auto loss3 = BeanTensor::Loss::bce_with_logits(logits, target);
    auto loss4 = logits.bce_with_logits(target);
    CHECK(near64(loss3.at<float32_t>({0}), loss4.at<float32_t>({0}), 1e-6f));
}

// =====================================================================
// GPU parity
// =====================================================================

static void test_bce_gpu_parity() {
    if (!gpu_available()) {
        return;
    }
    auto pred_cpu = Tensor({8}, DType::Float32, Device::CPU);
    auto target_cpu = Tensor({8}, DType::Float32, Device::CPU);
    for (auto i = size_t{0}; i < 8; ++i) {
        pred_cpu.set<float32_t>({i}, 0.1f + 0.1f * static_cast<float32_t>(i));
        target_cpu.set<float32_t>({i}, i % 2 == 0 ? 1.0f : 0.0f);
    }
    auto pred_gpu = pred_cpu.clone(true);
    pred_gpu.to(Device::GPU);
    auto target_gpu = target_cpu.clone(true);
    target_gpu.to(Device::GPU);
    auto loss_cpu = pred_cpu.bce(target_cpu);
    auto loss_gpu = pred_gpu.bce(target_gpu);
    loss_gpu.sync();
    CHECK(near64(loss_cpu.at<float32_t>({0}), loss_gpu.at<float32_t>({0}), 1e-4f));

    pred_cpu = Tensor({8}, DType::Float32, Device::CPU, true);
    target_cpu = Tensor({8}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 8; ++i) {
        pred_cpu.set<float32_t>({i}, 0.1f + 0.1f * static_cast<float32_t>(i));
        target_cpu.set<float32_t>({i}, i % 2 == 0 ? 1.0f : 0.0f);
    }
    pred_gpu = pred_cpu.clone(true);
    pred_gpu.to(Device::GPU);
    target_gpu = target_cpu.clone(true);
    target_gpu.to(Device::GPU);
    auto loss_g = pred_gpu.bce(target_gpu);
    loss_g.backward(DType::Float32);
    auto grad_gpu = pred_gpu.gradient().clone(true);
    grad_gpu.to(Device::CPU);
    grad_gpu.sync();
    auto loss_c = pred_cpu.bce(target_cpu);
    loss_c.backward(DType::Float32);
    for (auto i = size_t{0}; i < 8; ++i) {
        CHECK(near64(grad_gpu.at<float32_t>({i}), pred_cpu.gradient().at<float32_t>({i}), 1e-4f));
    }
}

static void test_bce_logits_gpu_parity() {
    if (!gpu_available()) {
        return;
    }
    auto logits_cpu = Tensor({8}, DType::Float32, Device::CPU);
    auto target_cpu = Tensor({8}, DType::Float32, Device::CPU);
    for (auto i = size_t{0}; i < 8; ++i) {
        logits_cpu.set<float32_t>({i}, -2.0f + 0.5f * static_cast<float32_t>(i));
        target_cpu.set<float32_t>({i}, i % 2 == 0 ? 1.0f : 0.0f);
    }
    auto logits_gpu = logits_cpu.clone(true);
    logits_gpu.to(Device::GPU);
    auto target_gpu = target_cpu.clone(true);
    target_gpu.to(Device::GPU);
    auto loss_cpu = logits_cpu.bce_with_logits(target_cpu);
    auto loss_gpu = logits_gpu.bce_with_logits(target_gpu);
    loss_gpu.sync();
    CHECK(near64(loss_cpu.at<float32_t>({0}), loss_gpu.at<float32_t>({0}), 1e-4f));

    logits_cpu = Tensor({8}, DType::Float32, Device::CPU, true);
    target_cpu = Tensor({8}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 8; ++i) {
        logits_cpu.set<float32_t>({i}, -2.0f + 0.5f * static_cast<float32_t>(i));
        target_cpu.set<float32_t>({i}, i % 2 == 0 ? 1.0f : 0.0f);
    }
    logits_gpu = logits_cpu.clone(true);
    logits_gpu.to(Device::GPU);
    target_gpu = target_cpu.clone(true);
    target_gpu.to(Device::GPU);
    auto loss_g = logits_gpu.bce_with_logits(target_gpu);
    loss_g.backward(DType::Float32);
    auto grad_gpu = logits_gpu.gradient().clone(true);
    grad_gpu.to(Device::CPU);
    grad_gpu.sync();
    auto loss_c = logits_cpu.bce_with_logits(target_cpu);
    loss_c.backward(DType::Float32);
    for (auto i = size_t{0}; i < 8; ++i) {
        CHECK(near64(grad_gpu.at<float32_t>({i}), logits_cpu.gradient().at<float32_t>({i}), 1e-4f));
    }
}

static void test_bce_device_mismatch() {
    if (!gpu_available()) {
        return;
    }
    auto pred_cpu = Tensor({4}, DType::Float32, Device::CPU);
    auto target_cpu = Tensor({4}, DType::Float32, Device::CPU);
    pred_cpu.set_one();
    target_cpu.set_zero();
    auto target_gpu = target_cpu.clone(true);
    target_gpu.to(Device::GPU);
    target_gpu.sync();

    auto threw_bce = false;
    try {
        auto bad = pred_cpu.bce(target_gpu);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
        threw_bce = true;
    }
    CHECK(threw_bce);

    auto threw_logits = false;
    try {
        auto bad = pred_cpu.bce_with_logits(target_gpu);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
        threw_logits = true;
    }
    CHECK(threw_logits);
}

// =====================================================================
// main
// =====================================================================

int main() {
    test_bce_forward_hand();
    test_bce_clamp_boundaries();
    test_bce_logits_forward_hand();
    test_bce_logits_infinities();
    test_bce_g_values(Device::CPU);
    test_bce_g_clamp(Device::CPU);
    test_bce_logits_g_values(Device::CPU);
    test_bce_fd_probability();
    test_bce_fd_logits();
    test_bce_errors();
    test_bce_target_no_grad(Device::CPU);
    test_bce_logits_chain(Device::CPU);
    test_bce_grad_dtype_probe(Device::CPU);
    test_bce_logits_grad_dtype_probe(Device::CPU);
    test_bce_repeated_accumulation(Device::CPU);
    test_bce_non_unit_upstream(Device::CPU);
    test_bce_sgd_smoke(Device::CPU);
    test_bce_logits_adam_smoke(Device::CPU);
    test_bce_free_functions();
    test_bce_gpu_parity();
    test_bce_logits_gpu_parity();
    test_bce_device_mismatch();

    if (failures == 0) {
        std::cout << "ALL BCE TESTS PASSED\n";
        return 0;
    }
    std::cerr << "FAILURES: " << failures << "\n";
    return 1;
}
