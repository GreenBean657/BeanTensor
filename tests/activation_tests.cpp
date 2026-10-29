/**
 * @file activation_tests.cpp
 * @author GreenBean
 * @brief Framework-free tests for sigmoid, tanh, and softmax as activation ops,
 * their recompute-from-parent backward nodes, and the matching parameter-free
 * modules. Covers CPU values, extreme-input stability, analytic backward
 * gradients, and module forward/name wiring.
 * @version 0.3
 * @date 2026-10-28
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "Module/Sigmoid.h"
#include "Module/Softmax.h"
#include "Module/Tanh.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float64_t a, const float64_t b, const float64_t eps) {
    return std::fabs(a - b) <= eps;
}

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

static void test_sigmoid_values() {
    auto x = Tensor({5}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, -2.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 2.0f);
    x.set<float32_t>({3}, 100.0f);
    x.set<float32_t>({4}, -100.0f);

    auto s = x.sigmoid();
    const auto inputs = std::vector<float32_t>{-2.0f, 0.0f, 2.0f, 100.0f, -100.0f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto expected = static_cast<float64_t>(1.0) / (static_cast<float64_t>(1.0) + std::exp(-static_cast<float64_t>(inputs[i])));
        CHECK(near64(static_cast<float64_t>(s.at<float32_t>({i})), expected, 1e-6));
    }
    CHECK(near64(static_cast<float64_t>(s.at<float32_t>({3})), 1.0, 1e-6));
    CHECK(near64(static_cast<float64_t>(s.at<float32_t>({4})), 0.0, 1e-6));
}

static void test_tanh_values() {
    auto x = Tensor({5}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, -2.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 2.0f);
    x.set<float32_t>({3}, 100.0f);
    x.set<float32_t>({4}, -100.0f);

    auto t = x.tanh();
    const auto inputs = std::vector<float32_t>{-2.0f, 0.0f, 2.0f, 100.0f, -100.0f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto e2x = std::exp(static_cast<float64_t>(2.0) * static_cast<float64_t>(inputs[i]));
        const auto expected = (e2x - static_cast<float64_t>(1.0)) / (e2x + static_cast<float64_t>(1.0));
        CHECK(near64(static_cast<float64_t>(t.at<float32_t>({i})), expected, 1e-6));
    }
    CHECK(near64(static_cast<float64_t>(t.at<float32_t>({3})), 1.0, 1e-6));
    CHECK(near64(static_cast<float64_t>(t.at<float32_t>({4})), -1.0, 1e-6));
}

static void test_sigmoid_backward() {
    auto x = Tensor({5}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -2.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 2.0f);
    x.set<float32_t>({3}, 100.0f);
    x.set<float32_t>({4}, -100.0f);

    auto y = x.sigmoid();
    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    for (auto i = size_t{0}; i < 5; ++i) {
        const auto xi = static_cast<float64_t>(x.at<float32_t>({i}));
        const auto s = static_cast<float64_t>(1.0) / (static_cast<float64_t>(1.0) + std::exp(-xi));
        const auto expected = s * (static_cast<float64_t>(1.0) - s);
        CHECK(near64(static_cast<float64_t>(x.gradient().at<float32_t>({i})), expected, 1e-5));
    }
}

static void test_tanh_backward() {
    auto x = Tensor({5}, DType::Float32, Device::CPU, true);
    x.set<float32_t>({0}, -2.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 2.0f);
    x.set<float32_t>({3}, 100.0f);
    x.set<float32_t>({4}, -100.0f);

    auto y = x.tanh();
    auto loss = y.sum();
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    for (auto i = size_t{0}; i < 5; ++i) {
        const auto xi = static_cast<float64_t>(x.at<float32_t>({i}));
        const auto e2x = std::exp(static_cast<float64_t>(2.0) * xi);
        const auto t = (e2x - static_cast<float64_t>(1.0)) / (e2x + static_cast<float64_t>(1.0));
        const auto expected = static_cast<float64_t>(1.0) - t * t;
        CHECK(near64(static_cast<float64_t>(x.gradient().at<float32_t>({i})), expected, 1e-5));
    }
}

static void test_modules() {
    BeanTensor::Modules::Sigmoid sigmoid_mod;
    BeanTensor::Modules::Tanh tanh_mod;
    BeanTensor::Modules::Softmax softmax_mod;

    CHECK(sigmoid_mod.name() == "Sigmoid");
    CHECK(tanh_mod.name() == "Tanh");
    CHECK(softmax_mod.name() == "Softmax");

    auto x = Tensor({3}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, -1.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 1.0f);

    auto s = sigmoid_mod.forward(x);
    for (auto i = size_t{0}; i < 3; ++i) {
        const auto xi = static_cast<float64_t>(x.at<float32_t>({i}));
        const auto expected = static_cast<float64_t>(1.0) / (static_cast<float64_t>(1.0) + std::exp(-xi));
        CHECK(near64(static_cast<float64_t>(s.at<float32_t>({i})), expected, 1e-6));
    }

    auto t = tanh_mod.forward(x);
    for (auto i = size_t{0}; i < 3; ++i) {
        const auto xi = static_cast<float64_t>(x.at<float32_t>({i}));
        const auto e2x = std::exp(static_cast<float64_t>(2.0) * xi);
        const auto expected = (e2x - static_cast<float64_t>(1.0)) / (e2x + static_cast<float64_t>(1.0));
        CHECK(near64(static_cast<float64_t>(t.at<float32_t>({i})), expected, 1e-6));
    }

    auto sm = softmax_mod.forward(x.reshape({1, 3}));
    auto sum = float64_t{0.0};
    for (auto i = size_t{0}; i < 3; ++i) {
        sum += static_cast<float64_t>(sm.at<float32_t>({0, i}));
    }
    CHECK(near64(sum, 1.0, 1e-5));
}

static void test_softmax_values() {
    auto x = Tensor({4, 5}, DType::Float32, Device::CPU);
    const auto xv = std::vector<float32_t>{
        0.1f, -0.5f, 2.0f, 0.0f, 1.5f,
        -1.0f, 3.0f, 0.5f, -0.25f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({i / 5, i % 5}, xv[i]);
    }
    auto y = x.softmax();
    for (auto r = size_t{0}; r < 4; ++r) {
        auto row_sum = float64_t{0.0};
        for (auto c = size_t{0}; c < 5; ++c) {
            row_sum += static_cast<float64_t>(y.at<float32_t>({r, c}));
        }
        CHECK(near64(row_sum, 1.0, 1e-5));
    }
}

static void test_softmax_stability() {
    auto x = Tensor({1, 3}, DType::Float32, Device::CPU);
    x.set<float32_t>({0, 0}, 10000.0f);
    x.set<float32_t>({0, 1}, 10001.0f);
    x.set<float32_t>({0, 2}, 9999.0f);
    auto y = x.softmax();

    const auto xs = std::vector<float64_t>{10000.0, 10001.0, 9999.0};
    auto m = xs[0];
    for (const auto v : xs) {
        if (v > m) {
            m = v;
        }
    }
    auto denom = float64_t{0.0};
    for (const auto v : xs) {
        denom += std::exp(v - m);
    }
    for (auto c = size_t{0}; c < 3; ++c) {
        const auto expected = std::exp(xs[c] - m) / denom;
        const auto got = static_cast<float64_t>(y.at<float32_t>({0, c}));
        CHECK(std::isfinite(got));
        CHECK(near64(got, expected, 1e-5));
    }
}

static void test_softmax_backward() {
    const auto rows = size_t{3};
    const auto cols = size_t{4};
    auto x = Tensor({rows, cols}, DType::Float32, Device::CPU, true);
    const auto xv = std::vector<float32_t>{
        0.5f, -1.0f, 2.0f, 0.25f,
        -0.5f, 1.5f, 0.0f, 1.0f,
        2.0f, -2.0f, 0.5f, -0.25f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({i / cols, i % cols}, xv[i]);
    }
    auto target = Tensor({rows, cols}, DType::Float32, Device::CPU, false);
    const auto tv = std::vector<float32_t>{
        0.1f, 0.2f, 0.3f, 0.4f,
        -0.1f, 0.5f, 0.2f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    for (auto i = size_t{0}; i < tv.size(); ++i) {
        target.set<float32_t>({i / cols, i % cols}, tv[i]);
    }
    auto y = x.softmax();
    auto loss = y.mse(target);
    loss.backward(DType::Float32);
    CHECK(x.has_grad());

    // Analytic dx = y * (dy - rowsum(dy*y)); dy here is d(mse)/dy = 2*(y-target)/numel.
    for (auto r = size_t{0}; r < rows; ++r) {
        auto dy_y = std::vector<float64_t>(cols);
        auto row_sum = float64_t{0.0};
        for (auto c = size_t{0}; c < cols; ++c) {
            const auto yi = static_cast<float64_t>(y.at<float32_t>({r, c}));
            const auto ti = static_cast<float64_t>(target.at<float32_t>({r, c}));
            const auto dy = 2.0 * (yi - ti) / static_cast<float64_t>(rows * cols);
            dy_y[c] = dy * yi;
            row_sum += dy_y[c];
        }
        for (auto c = size_t{0}; c < cols; ++c) {
            const auto yi = static_cast<float64_t>(y.at<float32_t>({r, c}));
            const auto ti = static_cast<float64_t>(target.at<float32_t>({r, c}));
            const auto dy = 2.0 * (yi - ti) / static_cast<float64_t>(rows * cols);
            const auto expected = yi * (dy - row_sum);
            CHECK(near64(static_cast<float64_t>(x.gradient().at<float32_t>({r, c})), expected, 1e-4));
        }
    }
}

static void test_softmax_gradient_convention() {
    const auto rows = size_t{3};
    const auto cols = size_t{4};
    auto x = Tensor({rows, cols}, DType::Float32, Device::CPU, true);
    const auto xv = std::vector<float32_t>{
        0.5f, -1.0f, 2.0f, 0.25f,
        -0.5f, 1.5f, 0.0f, 1.0f,
        2.0f, -2.0f, 0.5f, -0.25f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({i / cols, i % cols}, xv[i]);
    }

    auto labels = Tensor({rows}, DType::Int64, Device::CPU, false);
    const auto label_of = std::vector<size_t>{2, 0, 1};
    for (auto r = size_t{0}; r < rows; ++r) {
        labels.set<int64_t>({r}, static_cast<int64_t>(label_of[r]));
    }

    // softmax_ce gives dlogits = (softmax(x) - onehot) / rows.
    auto ce_loss = x.softmax_ce(labels);
    ce_loss.backward(DType::Float32);
    auto dlogits = Tensor({rows, cols}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < rows * cols; ++i) {
        dlogits.set<float32_t>({i / cols, i % cols}, x.gradient().at<float32_t>({i / cols, i % cols}));
    }
    x.zero_grad();

    // Standalone softmax with upstream dy = onehot / rows: dx should be proportional to dlogits.
    auto dy = Tensor({rows, cols}, DType::Float32, Device::CPU, false);
    for (auto r = size_t{0}; r < rows; ++r) {
        for (auto c = size_t{0}; c < cols; ++c) {
            dy.set<float32_t>({r, c}, c == label_of[r] ? 1.0f / static_cast<float32_t>(rows) : 0.0f);
        }
    }
    auto y = x.softmax();
    auto y_dy = y.mul(dy);
    auto loss = y_dy.sum();
    loss.backward(DType::Float32);

    for (auto r = size_t{0}; r < rows; ++r) {
        const auto scale = -static_cast<float64_t>(y.at<float32_t>({r, label_of[r]}));
        for (auto c = size_t{0}; c < cols; ++c) {
            const auto expected = scale * static_cast<float64_t>(dlogits.at<float32_t>({r, c}));
            CHECK(near64(static_cast<float64_t>(x.gradient().at<float32_t>({r, c})), expected, 1e-4));
        }
    }
}

static void test_softmax_rank3_throws() {
    auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
    auto threw = false;
    try {
        auto bad = x.softmax();
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_sigmoid_values_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({5}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, -2.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 2.0f);
    x.set<float32_t>({3}, 100.0f);
    x.set<float32_t>({4}, -100.0f);
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto s = x_g.sigmoid();
    s.sync();
    s.to(Device::CPU);
    s.sync();
    const auto inputs = std::vector<float32_t>{-2.0f, 0.0f, 2.0f, 100.0f, -100.0f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto expected = static_cast<float64_t>(1.0) / (static_cast<float64_t>(1.0) + std::exp(-static_cast<float64_t>(inputs[i])));
        CHECK(near64(static_cast<float64_t>(s.at<float32_t>({i})), expected, 1e-6));
    }
    CHECK(near64(static_cast<float64_t>(s.at<float32_t>({3})), 1.0, 1e-6));
    CHECK(near64(static_cast<float64_t>(s.at<float32_t>({4})), 0.0, 1e-6));
}

static void test_tanh_values_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({5}, DType::Float32, Device::CPU);
    x.set<float32_t>({0}, -2.0f);
    x.set<float32_t>({1}, 0.0f);
    x.set<float32_t>({2}, 2.0f);
    x.set<float32_t>({3}, 100.0f);
    x.set<float32_t>({4}, -100.0f);
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto t = x_g.tanh();
    t.sync();
    t.to(Device::CPU);
    t.sync();
    const auto inputs = std::vector<float32_t>{-2.0f, 0.0f, 2.0f, 100.0f, -100.0f};
    for (auto i = size_t{0}; i < inputs.size(); ++i) {
        const auto e2x = std::exp(static_cast<float64_t>(2.0) * static_cast<float64_t>(inputs[i]));
        const auto expected = (e2x - static_cast<float64_t>(1.0)) / (e2x + static_cast<float64_t>(1.0));
        CHECK(near64(static_cast<float64_t>(t.at<float32_t>({i})), expected, 1e-6));
    }
    CHECK(near64(static_cast<float64_t>(t.at<float32_t>({3})), 1.0, 1e-6));
    CHECK(near64(static_cast<float64_t>(t.at<float32_t>({4})), -1.0, 1e-6));
}

static void test_softmax_values_gpu() {
    if (!gpu_available()) {
        return;
    }
    auto x = Tensor({4, 5}, DType::Float32, Device::CPU);
    const auto xv = std::vector<float32_t>{
        0.1f, -0.5f, 2.0f, 0.0f, 1.5f,
        -1.0f, 3.0f, 0.5f, -0.25f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (auto i = size_t{0}; i < xv.size(); ++i) {
        x.set<float32_t>({i / 5, i % 5}, xv[i]);
    }
    auto x_g = x.clone(true);
    x_g.to(Device::GPU);
    auto y = x_g.softmax();
    y.sync();
    y.to(Device::CPU);
    y.sync();
    for (auto r = size_t{0}; r < 4; ++r) {
        auto row_sum = float64_t{0.0};
        for (auto c = size_t{0}; c < 5; ++c) {
            row_sum += static_cast<float64_t>(y.at<float32_t>({r, c}));
        }
        CHECK(near64(row_sum, 1.0, 1e-5));
    }
}

int main() {
    test_sigmoid_values();
    test_tanh_values();
    test_sigmoid_backward();
    test_tanh_backward();
    test_softmax_values();
    test_softmax_stability();
    test_softmax_backward();
    test_softmax_gradient_convention();
    test_softmax_rank3_throws();
    test_modules();
    test_sigmoid_values_gpu();
    test_tanh_values_gpu();
    test_softmax_values_gpu();

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed\n";
    return 1;
}
