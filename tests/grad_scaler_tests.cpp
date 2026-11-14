/**
 * @file grad_scaler_tests.cpp
 * @author GreenBean
 * @brief GradScaler tests — framework-free assertion suite.
 * Covers: scale(enabled/disabled/static), unscale_and_check finite and
 * non-finite (NaN/Inf), dynamic growth/backoff, static mode, Adam step-counter
 * preservation on skipped steps, SGD param preservation on skipped steps,
 * fp16-storage underflow retention, mixed grad dtypes, exception safety, and
 * a real training step on CPU and CUDA.
 * @version 0.1
 * @date 2026-11-14
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Math/Optimization/GradScaler/GradScaler.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Module;
using BeanTensor::Optimization::GradScaler;
using BeanTensor::float16_t;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

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

static bool exact_eq64(const float64_t a, const float64_t b) {
    return !(a < b) && !(a > b);
}

// =====================================================================
// Test helper: a module that owns a single parameter named "p".
// =====================================================================

class SingleParam final : public Module {
public:
    explicit SingleParam(Tensor t) {
        add_parameter("p", std::move(t));
    }

    [[nodiscard]] std::string name() const override {
        return "SingleParam";
    }

    [[nodiscard]] Tensor forward(const Tensor& x) override {
        return x.mul(float64_t{1.0});
    }

    [[nodiscard]] Tensor& param() {
        return parameter("p");
    }
};

// =====================================================================
// scale
// =====================================================================

static void test_scale_enabled_disabled() {
    {
        auto scaler = GradScaler{float64_t{64.0}};
        auto loss = Tensor({1}, DType::Float32, Device::CPU);
        loss.set<float32_t>({0}, 2.0f);
        auto scaled = scaler.scale(loss);
        scaled.sync();
        CHECK(near64(scaled.at<float32_t>({0}), 128.0f, 1e-6));
        CHECK(exact_eq64(scaler.get_scale(), float64_t{64.0}));
    }
    {
        auto scaler = GradScaler{float64_t{64.0}, float64_t{2.0}, float64_t{0.5}, 2000, false, true};
        auto loss = Tensor({1}, DType::Float32, Device::CPU);
        loss.set<float32_t>({0}, 2.0f);
        auto scaled = scaler.scale(loss);
        scaled.sync();
        CHECK(near64(scaled.at<float32_t>({0}), 2.0f, 1e-6));
    }
}

// =====================================================================
// unscale_and_check: finite
// =====================================================================

static void test_unscale_finite_cpu() {
    auto scaler = GradScaler{float64_t{4.0}};
    auto module = SingleParam(Tensor({3}, DType::Float32, Device::CPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);
    w.set<float32_t>({2}, 3.0f);

    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float32);
    CHECK(near64(w.gradient().at<float32_t>({0}), 2.0f, 1e-6));

    const auto finite = scaler.unscale_and_check(module);
    w.gradient().sync();
    CHECK(finite);
    CHECK(near64(w.gradient().at<float32_t>({0}), 0.5f, 1e-6));
    CHECK(near64(w.gradient().at<float32_t>({1}), 0.5f, 1e-6));
    CHECK(near64(w.gradient().at<float32_t>({2}), 0.5f, 1e-6));
}

// =====================================================================
// unscale_and_check: NaN / Inf
// =====================================================================

static void test_unscale_nonfinite_cpu() {
    {
        auto scaler = GradScaler{float64_t{4.0}};
        auto module = SingleParam(Tensor({2}, DType::Float32, Device::CPU, true));
        auto& w = module.param();
        w.set<float32_t>({0}, 1.0f);
        w.set<float32_t>({1}, 2.0f);
        auto y = w.mul(float64_t{2.0});
        y.backward(DType::Float32);
        w.gradient().set<float32_t>({1}, std::numeric_limits<float32_t>::quiet_NaN());

        const auto finite = scaler.unscale_and_check(module);
        CHECK(!finite);
        w.gradient().sync();
        CHECK(near64(w.gradient().at<float32_t>({0}), 0.5f, 1e-6));
    }
    {
        auto scaler = GradScaler{float64_t{4.0}};
        auto module = SingleParam(Tensor({2}, DType::Float32, Device::CPU, true));
        auto& w = module.param();
        w.set<float32_t>({0}, 1.0f);
        w.set<float32_t>({1}, 2.0f);
        auto y = w.mul(float64_t{2.0});
        y.backward(DType::Float32);
        w.gradient().set<float32_t>({0}, std::numeric_limits<float32_t>::infinity());

        const auto finite = scaler.unscale_and_check(module);
        CHECK(!finite);
    }
}

// =====================================================================
// dynamic growth / backoff / static mode
// =====================================================================

static void test_dynamic_growth_and_backoff() {
    auto scaler = GradScaler{float64_t{1.0}, float64_t{2.0}, float64_t{0.5}, 3, true, true};
    CHECK(exact_eq64(scaler.get_scale(), float64_t{1.0}));
    CHECK(scaler.get_growth_tracker() == 0);

    scaler.update(true);
    CHECK(exact_eq64(scaler.get_scale(), float64_t{1.0}));
    CHECK(scaler.get_growth_tracker() == 1);

    scaler.update(true);
    CHECK(exact_eq64(scaler.get_scale(), float64_t{1.0}));
    CHECK(scaler.get_growth_tracker() == 2);

    scaler.update(true);
    CHECK(exact_eq64(scaler.get_scale(), float64_t{2.0}));
    CHECK(scaler.get_growth_tracker() == 0);

    scaler.update(false);
    CHECK(exact_eq64(scaler.get_scale(), float64_t{1.0}));
    CHECK(scaler.get_growth_tracker() == 0);
}

static void test_static_mode() {
    auto scaler = GradScaler{float64_t{16.0}, float64_t{2.0}, float64_t{0.5}, 2, true, false};
    scaler.update(true);
    scaler.update(true);
    CHECK(exact_eq64(scaler.get_scale(), float64_t{16.0}));
    CHECK(scaler.get_growth_tracker() == 0);
    scaler.update(false);
    CHECK(exact_eq64(scaler.get_scale(), float64_t{16.0}));
}

// =====================================================================
// Adam step counter preservation on skip
// =====================================================================

static void test_adam_skip_preserves_step_count_and_moments() {
    auto module = SingleParam(Tensor({2}, DType::Float32, Device::CPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);

    auto g1 = Tensor({2}, DType::Float32, Device::CPU);
    g1.set<float32_t>({0}, 1.0f);
    g1.set<float32_t>({1}, 1.0f);
    w.adam_update(g1, 0.1, 0.9, 0.999, 1e-8);
    w.sync();

    const auto t1 = w.adam_step_count();
    CHECK(t1.has_value() && t1.value() == uint64_t{1});
    auto moments1 = w.adam_moments();
    CHECK(moments1.has_value());
    moments1.value().first->sync();
    moments1.value().second->sync();
    const auto m1_0 = const_cast<Tensor*>(moments1.value().first)->at<float32_t>({0});
    const auto v1_0 = const_cast<Tensor*>(moments1.value().second)->at<float32_t>({0});

    auto scaler = GradScaler{float64_t{4.0}};
    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float32);
    w.gradient().set<float32_t>({0}, std::numeric_limits<float32_t>::quiet_NaN());
    const auto finite = scaler.step(module, 0.1, 0.9, 0.999, 1e-8);
    CHECK(!finite);

    const auto t2 = w.adam_step_count();
    CHECK(t2.has_value() && t2.value() == uint64_t{1});
    auto moments2 = w.adam_moments();
    CHECK(moments2.has_value());
    moments2.value().first->sync();
    moments2.value().second->sync();
    CHECK(near64(const_cast<Tensor*>(moments2.value().first)->at<float32_t>({0}), m1_0, 1e-12));
    CHECK(near64(const_cast<Tensor*>(moments2.value().second)->at<float32_t>({0}), v1_0, 1e-12));
}

// =====================================================================
// SGD skip leaves parameters untouched
// =====================================================================

static void test_sgd_skip_leaves_params() {
    auto module = SingleParam(Tensor({2}, DType::Float32, Device::CPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);

    auto scaler = GradScaler{float64_t{4.0}};
    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float32);
    w.gradient().set<float32_t>({0}, std::numeric_limits<float32_t>::quiet_NaN());

    w.sync();
    const auto p0 = w.at<float32_t>({0});
    const auto p1 = w.at<float32_t>({1});

    const auto finite = scaler.step(module, 0.1);
    CHECK(!finite);
    w.sync();
    CHECK(near64(w.at<float32_t>({0}), p0, 1e-12));
    CHECK(near64(w.at<float32_t>({1}), p1, 1e-12));
}

// =====================================================================
// FP16-storage underflow retention
// =====================================================================

static float32_t widen_half(const float16_t v) {
    return BeanTensor::Intrinsics::detail::unaccel_fp16_to_fp32_untracked(v);
}

static float16_t narrow_half(const float32_t v) {
    return BeanTensor::Intrinsics::detail::unaccel_fp32_to_fp16_untracked(v);
}

static void test_fp16_underflow_retention() {
    auto scaler = GradScaler{float64_t{65536.0}};
    auto module = SingleParam(Tensor({1}, DType::Float16, Device::CPU, true));
    auto& w = module.param();
    w.set<float16_t>({0}, narrow_half(1.0f));

    auto y = w.mul(float64_t{1.0});
    y.backward(DType::Float16);

    const auto finite = scaler.unscale_and_check(module);
    CHECK(finite);
    w.gradient().sync();

    const auto unscaled = widen_half(w.gradient().at<float16_t>({0}));
    CHECK(unscaled > 0.0f);
    CHECK(near64(unscaled, 1.0f / 65536.0f, 1e-7));
}

// =====================================================================
// Mixed grad dtype
// =====================================================================

static void test_mixed_grad_dtype() {
    auto module = SingleParam(Tensor({2}, DType::Float32, Device::CPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);

    auto scaler = GradScaler{float64_t{4.0}};
    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float64);
    CHECK(w.gradient().get_dtype() == DType::Float64);

    const auto finite = scaler.unscale_and_check(module);
    CHECK(finite);
    w.gradient().sync();
    CHECK(near64(w.gradient().at<float64_t>({0}), 0.5, 1e-12));
    CHECK(near64(w.gradient().at<float64_t>({1}), 0.5, 1e-12));
}

// =====================================================================
// Exception safety: integer gradient
// =====================================================================

static void test_integer_grad_rejected() {
    auto module = SingleParam(Tensor({1}, DType::Int32, Device::CPU, true));
    auto& w = module.param();
    w.set<int32_t>({0}, 4);
    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Int32);

    auto scaler = GradScaler{float64_t{4.0}};
    auto threw = false;
    try {
        scaler.unscale_and_check(module);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

// =====================================================================
// Real training step
// =====================================================================

static void test_real_training_step(const Device dev) {
    const auto n_in = size_t{8};
    const auto n_out = size_t{1};
    auto layer = Linear(n_in, n_out, DType::Float32, dev);
    layer.train();

    auto x = Tensor({4, n_in}, DType::Float32, dev, false);
    auto target = Tensor({4, n_out}, DType::Float32, dev, false);
    for (auto i = size_t{0}; i < 4 * n_in; ++i) {
        x.set<float32_t>({i / n_in, i % n_in}, 0.1f * static_cast<float32_t>(i) - 0.3f);
    }
    for (auto i = size_t{0}; i < 4; ++i) {
        target.set<float32_t>({i, 0}, static_cast<float32_t>(i % 2));
    }

    auto scaler = GradScaler{float64_t{256.0}};
    auto first_loss = float64_t{0.0};
    for (auto step = size_t{0}; step < 50; ++step) {
        layer.zero_grad();
        auto logits = layer.forward(x);
        auto raw_loss = logits.mse(target);
        auto loss = scaler.scale(raw_loss);
        loss.backward(DType::Float32);
        const auto finite = scaler.step(layer, 0.05, 0.9, 0.999, 1e-8);
        CHECK(finite);
        if (step == 0) {
            first_loss = static_cast<float64_t>(loss.at<float32_t>({0})) / scaler.get_scale();
        }
    }
    auto logits = layer.forward(x);
    auto final_loss = logits.mse(target);
    final_loss.sync();
    const auto final_val = static_cast<float64_t>(final_loss.at<float32_t>({0}));
    CHECK(final_val < first_loss);
}

// =====================================================================
// GPU parity
// =====================================================================

static void test_unscale_finite_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto scaler = GradScaler{float64_t{4.0}};
    auto module = SingleParam(Tensor({3}, DType::Float32, Device::GPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);
    w.set<float32_t>({2}, 3.0f);

    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float32);
    const auto finite = scaler.unscale_and_check(module);
    w.gradient().sync();
    CHECK(finite);
    CHECK(near64(w.gradient().at<float32_t>({0}), 0.5f, 1e-5));
}

static void test_nonfinite_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto scaler = GradScaler{float64_t{4.0}};
    auto module = SingleParam(Tensor({2}, DType::Float32, Device::GPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);
    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float32);
    w.gradient().set<float32_t>({1}, std::numeric_limits<float32_t>::quiet_NaN());

    const auto finite = scaler.unscale_and_check(module);
    CHECK(!finite);
}

static void test_adam_skip_preserves_step_count_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto module = SingleParam(Tensor({2}, DType::Float32, Device::GPU, true));
    auto& w = module.param();
    w.set<float32_t>({0}, 1.0f);
    w.set<float32_t>({1}, 2.0f);

    auto g1 = Tensor({2}, DType::Float32, Device::GPU);
    g1.set<float32_t>({0}, 1.0f);
    g1.set<float32_t>({1}, 1.0f);
    w.adam_update(g1, 0.1, 0.9, 0.999, 1e-8);
    w.sync();
    CHECK(w.adam_step_count().value() == uint64_t{1});

    auto scaler = GradScaler{float64_t{4.0}};
    auto y = w.mul(float64_t{2.0});
    y.backward(DType::Float32);
    w.gradient().set<float32_t>({0}, std::numeric_limits<float32_t>::quiet_NaN());
    const auto finite = scaler.step(module, 0.1, 0.9, 0.999, 1e-8);
    CHECK(!finite);
    CHECK(w.adam_step_count().value() == uint64_t{1});
}

int main() {
    test_scale_enabled_disabled();
    test_unscale_finite_cpu();
    test_unscale_nonfinite_cpu();
    test_dynamic_growth_and_backoff();
    test_static_mode();
    test_adam_skip_preserves_step_count_and_moments();
    test_sgd_skip_leaves_params();
    test_fp16_underflow_retention();
    test_mixed_grad_dtype();
    test_integer_grad_rejected();
    test_real_training_step(Device::CPU);

#if defined(USE_HIP) || defined(USE_CUDA)
    if (gpu_available()) {
        test_unscale_finite_gpu();
        test_nonfinite_gpu();
        test_adam_skip_preserves_step_count_gpu();
        test_real_training_step(Device::GPU);
    } else {
        std::cout << "GPU not available — skipping GPU grad scaler tests\n";
    }
#endif

    if (failures == 0) {
        std::cout << "ALL GRADSCALER TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
