// BeanTensorV5 flatten tests — framework-free assertion suite.
// Covers: Tensor::flatten values and shape, gradient round-trip through
// FlattenG, non-contiguous input rejection, and the Flatten module inside
// a Sequential.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Module/Flatten.h"
#include "Module/Sequential.h"
#include "Tensor/Tensor.h"

using BeanTensor::Modules::Flatten;
using BeanTensor::Modules::Sequential;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Tensors::Device;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static bool near64(const float32_t a, const float64_t b, const float64_t eps) {
    return std::fabs(static_cast<float64_t>(a) - b) <= eps;
}

static void test_flatten_values() {
    auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 24; ++i) {
        x.set<float32_t>({i / 12, (i / 4) % 3, i % 4}, static_cast<float32_t>(i));
    }
    auto y = x.flatten();
    CHECK(y.is_same_shape(Tensor({2, 12})));
    CHECK(y.get_dtype() == DType::Float32);
    for (auto i = size_t{0}; i < 24; ++i) {
        CHECK(near64(y.at<float32_t>({i / 12, i % 12}), static_cast<float64_t>(i), 1e-6));
    }
}

static void test_flatten_grad() {
    auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU, true);
    for (auto i = size_t{0}; i < 24; ++i) {
        x.set<float32_t>({i / 12, (i / 4) % 3, i % 4}, static_cast<float32_t>(i));
    }
    auto target = Tensor({2, 12}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 24; ++i) {
        target.set<float32_t>({i / 12, i % 12}, static_cast<float32_t>(i % 5));
    }
    auto y = x.flatten();
    auto loss = y.mse(target);
    loss.backward(DType::Float32);
    CHECK(x.has_grad());
    CHECK(x.gradient().is_same_shape(Tensor({2, 3, 4})));
    // Expected dy = 2*(y - target)/24, then reshaped to {2,3,4}.
    for (auto i = size_t{0}; i < 24; ++i) {
        const auto yi = static_cast<float64_t>(i);
        const auto ti = static_cast<float64_t>(i % 5);
        const auto dy = 2.0 * (yi - ti) / 24.0;
        CHECK(near64(x.gradient().at<float32_t>({i / 12, (i / 4) % 3, i % 4}), dy, 1e-5));
    }
}

static void test_flatten_noncontiguous_throws() {
    auto base = Tensor({2, 4}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 8; ++i) {
        base.set<float32_t>({i / 4, i % 4}, static_cast<float32_t>(i));
    }
    auto ranges = std::vector<std::pair<size_t, size_t>>{{0, 2}, {1, 3}};
    auto view = base.slice(ranges); // inner box: non-contiguous view
    auto threw = false;
    try {
        auto bad = view.flatten();
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::NotImplemented&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_flatten_module() {
    auto net = Sequential();
    net.append("flat", std::make_unique<Flatten>());
    auto x = Tensor({2, 3, 4}, DType::Float32, Device::CPU, false);
    for (auto i = size_t{0}; i < 24; ++i) {
        x.set<float32_t>({i / 12, (i / 4) % 3, i % 4}, static_cast<float32_t>(i));
    }
    auto y = net.forward(x);
    CHECK(y.is_same_shape(Tensor({2, 12})));
    CHECK(net[0].name() == "Flatten");
}

int main() {
    test_flatten_values();
    test_flatten_grad();
    test_flatten_noncontiguous_throws();
    test_flatten_module();
    if (failures == 0) {
        std::cout << "ALL FLATTEN TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
