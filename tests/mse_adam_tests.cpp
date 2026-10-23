// BeanTensorV5 mse + adam tests — framework-free assertion suite.
// Covers: mse forward values (rank 1/2/3, fp32/fp64/fp16/int32/packed,
// free-function form, error paths), mse backward vs hand-computed
// d/dpred = 2*(pred-target)/numel (leaf and chained-through-matmul, the
// intermediate-delta probe, per-call grad dtype override, targets never
// receiving a gradient), Adam (bias-correction hand check against an fp64
// reference mirror, constant-gradient trajectory, convergence on a quadratic
// via the mse -> backward -> adam_step loop, error paths, mixed grad dtype),
// and — when a backend + device exist — the same checks on the GPU plus
// CPU/GPU parity.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

// Mixed-precision comparisons happen at float64 through explicit casts, so no
// call site relies on an implicit float->double promotion.
template<typename A, typename B>
static bool near64(const A a, const B b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - static_cast<float64_t>(b)) <= eps;
}

// --- half helpers (host-side, via the unaccel conversion pathway) ---

static float16_t narrow_half(const float32_t v) {
    return idet::unaccel_fp32_to_fp16_untracked(v);
}

static float32_t widen_half(const float16_t v) {
    return idet::unaccel_fp16_to_fp32_untracked(v);
}

// =====================================================================
// mse forward
// =====================================================================

static void test_mse_forward_cpu() {
    {
        // fp32 rank-1: mean of (1,4,1,0)/4 = 1.5.
        auto pred = Tensor({4}, DType::Float32, Device::CPU);
        auto target = Tensor({4}, DType::Float32, Device::CPU);
        pred.set<float32_t>({0}, 1.0f);
        pred.set<float32_t>({1}, 2.0f);
        pred.set<float32_t>({2}, 3.0f);
        pred.set<float32_t>({3}, 4.0f);
        target.set<float32_t>({0}, 0.0f);
        target.set<float32_t>({1}, 0.0f);
        target.set<float32_t>({2}, 2.0f);
        target.set<float32_t>({3}, 4.0f);
        auto loss = pred.mse(target);
        CHECK(loss.is_same_shape(Tensor({1})));
        CHECK(loss.get_dtype() == DType::Float32);
        CHECK(near64(loss.at<float32_t>({0}), 1.5, 1e-6));
    }
    {
        // fp64 rank-2, hand-computed: diffs (2, -1, 0.5, -3, 1, 2),
        // squares (4, 1, 0.25, 9, 1, 4) sum 19.25, mean 19.25/6.
        auto pred = Tensor({2, 3}, DType::Float64, Device::CPU);
        auto target = Tensor({2, 3}, DType::Float64, Device::CPU);
        const auto pv = std::vector<float64_t>{3.0, 1.0, 2.5, 0.0, 5.0, -1.0};
        const auto tv = std::vector<float64_t>{1.0, 2.0, 2.0, 3.0, 4.0, -3.0};
        for (auto i = size_t{0}; i < 6; ++i) {
            pred.set<float64_t>({i / 3, i % 3}, pv[i]);
            target.set<float64_t>({i / 3, i % 3}, tv[i]);
        }
        auto loss = pred.mse(target);
        CHECK(near64(loss.at<float64_t>({0}), 19.25 / 6.0, 1e-12));
    }
    {
        // rank-3: rank-agnostic, mean over all 24 elements.
        auto pred = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
        auto target = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
        pred.set_one();
        target.set_zero();
        auto loss = pred.mse(target);
        CHECK(near64(loss.at<float32_t>({0}), 1.0, 1e-6));
    }
    {
        // fp16: widened fp32 compute, RNE store.
        auto pred = Tensor({3}, DType::Float16, Device::CPU);
        auto target = Tensor({3}, DType::Float16, Device::CPU);
        pred.set<float16_t>({0}, narrow_half(1.0f));
        pred.set<float16_t>({1}, narrow_half(0.5f));
        pred.set<float16_t>({2}, narrow_half(-2.0f));
        target.set<float16_t>({0}, narrow_half(0.0f));
        target.set<float16_t>({1}, narrow_half(0.5f));
        target.set<float16_t>({2}, narrow_half(2.0f));
        auto loss = pred.mse(target);
        // 17/3 = 5.6667; fp16 spacing at that magnitude is 2^-8, so the RNE
        // store lands within 4e-3.
        CHECK(near64(widen_half(loss.at<float16_t>({0})), (1.0 + 0.0 + 16.0) / 3.0, 1e-2));
    }
    {
        // int32: fp64 accumulate, clamp + truncate toward zero on store.
        auto pred = Tensor({4}, DType::Int32, Device::CPU);
        auto target = Tensor({4}, DType::Int32, Device::CPU);
        pred.set<int32_t>({0}, 3);
        pred.set<int32_t>({1}, 0);
        pred.set<int32_t>({2}, 0);
        pred.set<int32_t>({3}, 0);
        target.set_zero();
        auto loss = pred.mse(target);
        // 9/4 = 2.25 -> truncates to 2.
        CHECK(loss.at<int32_t>({0}) == 2);
    }
    {
        // Float16x2: both lanes are logical elements; the scalar mean is
        // duplicated into both lanes. Element i holds lane value 1+i (lo=hi).
        auto pred = Tensor({2}, DType::Float16x2, Device::CPU);
        auto target = Tensor({2}, DType::Float16x2, Device::CPU);
        for (auto i = size_t{0}; i < 2; ++i) {
            const auto lane = narrow_half(1.0f + static_cast<float32_t>(i));
            const auto bits = idet::bit_cast_memcpy<uint16_t>(lane);
            pred.set<float16x2_t>({i}, idet::bit_cast_memcpy<float16x2_t>(static_cast<uint32_t>(bits) | (static_cast<uint32_t>(bits) << 16)));
            target.set<float16x2_t>({i}, idet::bit_cast_memcpy<float16x2_t>(uint32_t{0}));
        }
        auto loss = pred.mse(target);
        // lane squares: 1,1,4,4 -> mean 10/4 = 2.5 (exact in fp16).
        const auto loss_bits = idet::bit_cast_memcpy<uint32_t>(loss.at<float16x2_t>({0}));
        CHECK(near64(widen_half(idet::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(loss_bits & 0xFFFFu))), 2.5, 1e-3));
        CHECK(near64(widen_half(idet::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(loss_bits >> 16))), 2.5, 1e-3));
    }
    {
        // Free-function form: identical contract to the Tensor method.
        auto pred = Tensor({2}, DType::Float32, Device::CPU);
        auto target = Tensor({2}, DType::Float32, Device::CPU);
        pred.set<float32_t>({0}, 3.0f);
        pred.set<float32_t>({1}, -1.0f);
        target.set_zero();
        auto loss = BeanTensor::Loss::mse(pred, target);
        CHECK(near64(loss.at<float32_t>({0}), 5.0, 1e-6));
    }
    {
        // Error paths.
        auto a = Tensor({3}, DType::Float32, Device::CPU);
        auto b = Tensor({4}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = a.mse(b);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        auto c = Tensor({3}, DType::Int32, Device::CPU);
        try {
            auto bad = a.mse(c);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        auto d = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {0, 2}};
        auto inner = d.slice(ranges);
        auto e = Tensor({2, 2}, DType::Float32, Device::CPU);
        try {
            auto bad = inner.mse(e);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            auto bad = a.mse(a); // no operand requires grad: no graph
            static_cast<void>(bad);
            bad.backward(DType::Float32);
        } catch (const BeanTensor::ErrorHandling::NoGradient&) {
            threw = true;
        }
        CHECK(threw);
    }
}

// =====================================================================
// mse backward
// =====================================================================

static void test_mse_g_values(const Device dev) {
    // Leaf pred: d/dpred = 2*(pred - target)/numel (ones seed).
    auto pred = Tensor({2, 3}, DType::Float32, dev, true);
    auto target = Tensor({2, 3}, DType::Float32, dev, false);
    const auto pv = std::vector<float32_t>{1.0f, -2.0f, 3.5f, 0.0f, 2.0f, -1.0f};
    const auto tv = std::vector<float32_t>{0.5f, -2.0f, 1.5f, 1.0f, 0.0f, -3.0f};
    for (auto i = size_t{0}; i < 6; ++i) {
        pred.set<float32_t>({i / 3, i % 3}, pv[i]);
        target.set<float32_t>({i / 3, i % 3}, tv[i]);
    }
    auto loss = pred.mse(target);
    tdetail::grad_probe_reset();
    loss.backward(DType::Float32);
    // pred is a leaf: its delta lands in a leaf grad, not an intermediate.
    CHECK(tdetail::grad_probe_count(DType::Float32) == 0);
    CHECK(pred.has_grad());
    CHECK(!target.has_grad());
    CHECK(loss.gradient().get_dtype() == DType::Float32);
    for (auto i = size_t{0}; i < 6; ++i) {
        const auto want = 2.0 * (static_cast<float64_t>(pv[i]) - static_cast<float64_t>(tv[i])) / 6.0;
        CHECK(near64(pred.gradient().at<float32_t>({i / 3, i % 3}), want, 1e-5));
    }
}

static void test_mse_g_chain_probe(const Device dev) {
    // pred = x·w is an intermediate: mse_g delivers its delta in the assigned
    // grad dtype (probe hit), then gemm_g propagates to the leaves.
    const auto xv = std::vector<float32_t>{1.0f, -1.0f, 2.0f, 0.5f, 3.0f, -2.0f, 1.5f, 1.0f, 0.25f, 2.5f, -0.5f, 0.75f};
    const auto wv = std::vector<float32_t>{0.5f, -1.5f, 2.0f};
    const auto tv = std::vector<float32_t>{1.0f, -2.0f, 0.5f, 2.0f};
    for (const auto gd : std::vector{DType::Float64, DType::Float16, DType::Float32}) {
        auto x = Tensor({4, 3}, DType::Float32, dev, true);
        auto w = Tensor({3, 1}, DType::Float32, dev, true);
        auto target = Tensor({4, 1}, DType::Float32, dev, false);
        for (auto i = size_t{0}; i < 12; ++i) {
            x.set<float32_t>({i / 3, i % 3}, xv[i]);
        }
        for (auto i = size_t{0}; i < 3; ++i) {
            w.set<float32_t>({i, 0}, wv[i]);
        }
        for (auto i = size_t{0}; i < 4; ++i) {
            target.set<float32_t>({i, 0}, tv[i]);
        }
        auto pred = x.matmul(w);
        auto loss = pred.mse(target);
        tdetail::grad_probe_reset();
        loss.backward(gd);
        // Exactly one intermediate delta carries the assigned dtype: the
        // mse_g delivery into pred (gemm_g feeds only leaves).
        CHECK(tdetail::grad_probe_count(gd) == 1);
        CHECK(pred.gradient().get_dtype() == gd);
        // d/dpred = 2*(pred - target)/4 in fp32 (the operand dtype).
        std::vector<float32_t> pv(4);
        for (auto i = size_t{0}; i < 4; ++i) {
            pv[i] = 0.0f;
            for (auto k = size_t{0}; k < 3; ++k) {
                pv[i] += xv[i * 3 + k] * wv[k];
            }
            const auto want = 2.0f * (pv[i] - tv[i]) / 4.0f;
            if (gd == DType::Float64) {
                CHECK(near64(pred.gradient().at<float64_t>({i, 0}), want, 1e-6));
            } else if (gd == DType::Float16) {
                CHECK(near64(widen_half(pred.gradient().at<float16_t>({i, 0})), want, 1e-2));
            } else {
                CHECK(near64(pred.gradient().at<float32_t>({i, 0}), want, 1e-5));
            }
        }
        // Leaf grads pinned to the first call's dtype.
        CHECK(x.gradient().get_dtype() == gd);
        CHECK(w.gradient().get_dtype() == gd);
    }
}

static void test_mse_target_requires_grad(const Device dev) {
    // Targets never receive a gradient, even when flagged requires_grad.
    auto pred = Tensor({3}, DType::Float32, dev, true);
    auto target = Tensor({3}, DType::Float32, dev, true);
    pred.set<float32_t>({0}, 1.0f);
    pred.set<float32_t>({1}, 2.0f);
    pred.set<float32_t>({2}, 3.0f);
    target.set_zero();
    auto loss = pred.mse(target);
    loss.backward(DType::Float32);
    CHECK(pred.has_grad());
    CHECK(!target.has_grad());
    CHECK(near64(pred.gradient().at<float32_t>({0}), 2.0 * 1.0 / 3.0, 1e-5));
    CHECK(near64(pred.gradient().at<float32_t>({1}), 2.0 * 2.0 / 3.0, 1e-5));
    CHECK(near64(pred.gradient().at<float32_t>({2}), 2.0 * 3.0 / 3.0, 1e-5));
}

// =====================================================================
// Adam
// =====================================================================

// fp64 reference mirror of the Adam kernels (storage-width compute; the
// fp64 kernel path is bit-comparable at 1e-12).
static float64_t ref_adam_step(const float64_t w, float64_t& m, float64_t& v, const float64_t g,
                               const float64_t lr, const float64_t b1, const float64_t b2,
                               const uint64_t t, const float64_t eps) {
    m = std::fma(b1, m, (1.0 - b1) * g);
    v = std::fma(b2, v, (1.0 - b2) * g * g);
    const auto bc1 = 1.0 - std::pow(b1, static_cast<float64_t>(t));
    const auto bc2 = 1.0 - std::pow(b2, static_cast<float64_t>(t));
    return std::fma(-lr, (m / bc1) / (std::sqrt(v / bc2) + eps), w);
}

static void test_adam_hand_check() {
    // Step 1 with w=0, g=1: m=(1-b1), v=(1-b2), m_hat=1, v_hat=1, so
    // w = -lr/(1+eps) — the bias correction must exactly cancel the
    // (1-beta) initialization factors.
    {
        auto w = Tensor({2}, DType::Float64, Device::CPU);
        auto g = Tensor({2}, DType::Float64, Device::CPU);
        w.set_zero();
        g.set<float64_t>({0}, 1.0);
        g.set<float64_t>({1}, -1.0);
        w.adam_update(g, 0.1, 0.9, 0.999, 1e-8);
        w.sync();
        CHECK(near64(w.at<float64_t>({0}), -0.1 / (1.0 + 1e-8), 1e-15));
        CHECK(near64(w.at<float64_t>({1}), 0.1 / (1.0 + 1e-8), 1e-15));
    }
    // Multi-step trajectory against the fp64 reference mirror, mixed signs.
    {
        auto w = Tensor({4}, DType::Float64, Device::CPU);
        auto g = Tensor({4}, DType::Float64, Device::CPU);
        const auto gv = std::vector<float64_t>{2.0, -1.0, 0.5, 3.0};
        for (auto i = size_t{0}; i < 4; ++i) {
            w.set<float64_t>({i}, 0.25 * static_cast<float64_t>(i) - 0.5);
            g.set<float64_t>({i}, gv[i]);
        }
        auto ref_w = std::vector<float64_t>{-0.5, -0.25, 0.0, 0.25};
        auto ref_m = std::vector<float64_t>{0.0, 0.0, 0.0, 0.0};
        auto ref_v = std::vector<float64_t>{0.0, 0.0, 0.0, 0.0};
        for (auto t = uint64_t{1}; t <= 5; ++t) {
            w.adam_update(g, 0.05, 0.9, 0.999, 1e-8);
            w.sync();
            for (auto i = size_t{0}; i < 4; ++i) {
                ref_w[i] = ref_adam_step(ref_w[i], ref_m[i], ref_v[i], gv[i], 0.05, 0.9, 0.999, t, 1e-8);
                CHECK(near64(w.at<float64_t>({i}), ref_w[i], 1e-12));
            }
        }
    }
}

static void test_adam_convergence_quadratic(const Device dev) {
    // Minimize f(w) = mean((w - c)^2) with Adam via the mse loss:
    // the loop exercises mse -> backward -> adam_step end to end.
    const auto n = size_t{8};
    auto w = Tensor({n}, DType::Float32, dev, true);
    auto c = Tensor({n}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        w.set<float32_t>({i}, 0.1f * static_cast<float32_t>(i) - 0.3f);
        c.set<float32_t>({i}, std::sin(0.7f * static_cast<float32_t>(i)));
    }
    auto first = float64_t{0.0};
    for (auto step = size_t{0}; step < 400; ++step) {
        auto loss = w.mse(c);
        if (step == 0) {
            first = static_cast<float64_t>(loss.at<float32_t>({0}));
        }
        loss.backward(DType::Float32);
        w.adam_step(0.05, 0.9, 0.999, 1e-8);
        w.zero_grad();
    }
    auto loss = w.mse(c);
    const auto fin = static_cast<float64_t>(loss.at<float32_t>({0}));
    CHECK(fin < first);
    CHECK(fin < 1e-5);
    for (auto i = size_t{0}; i < n; ++i) {
        CHECK(near64(w.at<float32_t>({i}), std::sin(0.7f * static_cast<float32_t>(i)), 5e-2));
    }
}

static void test_adam_fp64_convergence(const Device dev) {
    // fp64 quadratic converges tighter; also exercises the fp64 kernel width.
    const auto n = size_t{4};
    auto w = Tensor({n}, DType::Float64, dev, true);
    auto c = Tensor({n}, DType::Float64, dev, false);
    for (auto i = size_t{0}; i < n; ++i) {
        w.set<float64_t>({i}, 4.0 - static_cast<float64_t>(i));
        c.set<float64_t>({i}, 0.5 * static_cast<float64_t>(i) - 1.0);
    }
    for (auto step = size_t{0}; step < 400; ++step) {
        auto loss = w.mse(c);
        loss.backward(DType::Float64);
        w.adam_step(0.1, 0.9, 0.999, 1e-12);
        w.zero_grad();
    }
    auto loss = w.mse(c);
    CHECK(near64(loss.at<float64_t>({0}), 0.0, 1e-10));
}

static void test_adam_errors(const Device dev) {
    {
        // NoGradient: no accumulated grad.
        auto w = Tensor({2}, DType::Float32, dev, true);
        auto threw = false;
        try {
            w.adam_step(0.1, 0.9, 0.999, 1e-8);
        } catch (const BeanTensor::ErrorHandling::NoGradient&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Integer weights: Adam is float-only.
        auto w = Tensor({2}, DType::Int32, dev, false);
        auto g = Tensor({2}, DType::Int32, dev, false);
        auto threw = false;
        try {
            w.adam_update(g, 0.1, 0.9, 0.999, 1e-8);
        } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Shape mismatch.
        auto w = Tensor({2}, DType::Float32, dev, false);
        auto g = Tensor({3}, DType::Float32, dev, false);
        auto threw = false;
        try {
            w.adam_update(g, 0.1, 0.9, 0.999, 1e-8);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Mixed grad dtype converts into the weight dtype (GradDType rule).
        auto w = Tensor({2}, DType::Float32, dev, true);
        w.set<float32_t>({0}, 1.5f);
        w.set<float32_t>({1}, 2.5f);
        auto y = w.relu();
        y.backward(DType::Float64);
        CHECK(w.gradient().get_dtype() == DType::Float64);
        w.adam_step(0.1, 0.9, 0.999, 1e-8);
        w.sync();
        // grad is ones where relu passed through (both positive here).
        // Step 1: w -= lr * 1/(1+eps) for both elements.
        CHECK(near64(w.at<float32_t>({0}), 1.4, 1e-5));
        CHECK(near64(w.at<float32_t>({1}), 2.4, 1e-5));
    }
}

// =====================================================================
// GPU parity
// =====================================================================

static void test_mse_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    for (const auto dt : std::vector{DType::Float32, DType::Float64}) {
        auto pred = Tensor({4, 5}, dt, Device::GPU, true);
        auto target = Tensor({4, 5}, dt, Device::GPU, false);
        for (auto i = size_t{0}; i < 20; ++i) {
            const auto pv = std::sin(0.37 * static_cast<float64_t>(i));
            const auto tv = std::cos(0.91 * static_cast<float64_t>(i));
            if (dt == DType::Float32) {
                pred.set<float32_t>({i / 5, i % 5}, static_cast<float32_t>(pv));
                target.set<float32_t>({i / 5, i % 5}, static_cast<float32_t>(tv));
            } else {
                pred.set<float64_t>({i / 5, i % 5}, pv);
                target.set<float64_t>({i / 5, i % 5}, tv);
            }
        }
        auto get64 = [&](Tensor& t, const size_t idx) -> float64_t {
            if (dt == DType::Float32) {
                return static_cast<float64_t>(t.at<float32_t>({idx / 5, idx % 5}));
            }
            return t.at<float64_t>({idx / 5, idx % 5});
        };
        auto loss64 = [&](Tensor& t) -> float64_t {
            if (dt == DType::Float32) {
                return static_cast<float64_t>(t.at<float32_t>({0}));
            }
            return t.at<float64_t>({0});
        };
        auto loss = pred.mse(target);
        loss.sync();
        auto want = 0.0;
        for (auto i = size_t{0}; i < 20; ++i) {
            const auto d = std::sin(0.37 * static_cast<float64_t>(i)) - std::cos(0.91 * static_cast<float64_t>(i));
            want += d * d;
        }
        want /= 20.0;
        CHECK(near64(loss64(loss), want, dt == DType::Float32 ? 1e-5 : 1e-12));
        // Backward on GPU: d/dpred vs hand-computed, and CPU/GPU parity.
        loss.backward(dt);
        auto g_gpu = pred.gradient().clone(true);
        g_gpu.to(Device::CPU);
        g_gpu.sync();
        auto pred_cpu = pred.clone(true);
        auto target_cpu = target.clone(true);
        pred_cpu.to(Device::CPU);
        target_cpu.to(Device::CPU);
        pred_cpu.sync();
        target_cpu.sync();
        auto loss_cpu = pred_cpu.mse(target_cpu);
        loss_cpu.backward(dt);
        auto bad = size_t{0};
        const auto tol = dt == DType::Float32 ? 1e-5 : 1e-12;
        for (auto i = size_t{0}; i < 20; ++i) {
            const auto p = get64(pred_cpu, i);
            const auto t = get64(target_cpu, i);
            const auto want_grad = 2.0 * (p - t) / 20.0;
            if (!near64(get64(g_gpu, i), want_grad, tol) || !near64(get64(g_gpu, i), get64(pred_cpu.gradient(), i), tol)) {
                ++bad;
            }
        }
        CHECK(bad == 0);
    }
}

static void test_adam_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    // GPU Adam: trajectory parity with the CPU result over a few steps.
    {
        auto w = Tensor({4}, DType::Float32, Device::GPU);
        auto g = Tensor({4}, DType::Float32, Device::GPU);
        for (auto i = size_t{0}; i < 4; ++i) {
            w.set<float32_t>({i}, 0.25f * static_cast<float32_t>(i) - 0.5f);
            g.set<float32_t>({i}, 0.5f - static_cast<float32_t>(i));
        }
        for (auto t = uint64_t{1}; t <= 3; ++t) {
            w.adam_update(g, 0.05, 0.9, 0.999, 1e-8);
            w.sync();
        }
        auto w_cpu = w.clone(true);
        w_cpu.to(Device::CPU);
        w_cpu.sync();
        auto w_ref = Tensor({4}, DType::Float32, Device::CPU);
        auto g_cpu = Tensor({4}, DType::Float32, Device::CPU);
        for (auto i = size_t{0}; i < 4; ++i) {
            w_ref.set<float32_t>({i}, 0.25f * static_cast<float32_t>(i) - 0.5f);
            g_cpu.set<float32_t>({i}, 0.5f - static_cast<float32_t>(i));
        }
        for (auto t = uint64_t{1}; t <= 3; ++t) {
            w_ref.adam_update(g_cpu, 0.05, 0.9, 0.999, 1e-8);
            w_ref.sync();
        }
        for (auto i = size_t{0}; i < 4; ++i) {
            CHECK(near64(w_cpu.at<float32_t>({i}), w_ref.at<float32_t>({i}), 1e-5));
        }
    }
    // GPU quadratic convergence through mse -> backward -> adam_step.
    {
        const auto n = size_t{6};
        auto w = Tensor({n}, DType::Float32, Device::GPU, true);
        auto c = Tensor({n}, DType::Float32, Device::GPU, false);
        for (auto i = size_t{0}; i < n; ++i) {
            w.set<float32_t>({i}, 1.0f - 0.2f * static_cast<float32_t>(i));
            c.set<float32_t>({i}, std::cos(0.5f * static_cast<float32_t>(i)));
        }
        for (auto step = size_t{0}; step < 300; ++step) {
            auto loss = w.mse(c);
            loss.backward(DType::Float32);
            w.adam_step(0.05, 0.9, 0.999, 1e-8);
            w.zero_grad();
        }
        auto loss = w.mse(c);
        loss.sync();
        CHECK(near64(loss.at<float32_t>({0}), 0.0, 1e-4));
    }
}

int main() {
    test_mse_forward_cpu();
    test_mse_g_values(Device::CPU);
    test_mse_g_chain_probe(Device::CPU);
    test_mse_target_requires_grad(Device::CPU);
    test_adam_hand_check();
    test_adam_convergence_quadratic(Device::CPU);
    test_adam_fp64_convergence(Device::CPU);
    test_adam_errors(Device::CPU);
#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        test_mse_gpu();
        test_mse_g_values(Device::GPU);
        test_mse_g_chain_probe(Device::GPU);
        test_mse_target_requires_grad(Device::GPU);
        test_adam_convergence_quadratic(Device::GPU);
        test_adam_fp64_convergence(Device::GPU);
        test_adam_errors(Device::GPU);
        test_adam_gpu();
    } else {
        std::cout << "GPU not available — skipping GPU mse/adam tests\n";
    }
#endif

    if (failures == 0) {
        std::cout << "ALL MSE/ADAM TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
