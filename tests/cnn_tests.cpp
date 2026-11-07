/**
 * @file cnn_tests.cpp
 * @author GreenBean
 * @brief CNN training-convergence tests - framework-free assertion suite.
 * Trains a small Conv2d/BatchNorm/MaxPool2d/Flatten/Linear network on
 * synthetic random data and asserts the loss decreases over the steps:
 * the spatial stack is exercised under a real optimizer, not just
 * forward/backward correctness. Data is generated, never read from disk,
 * so the test runs identically on every machine.
 * @version 0.1
 * @date 2026-11-06
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "Module/BatchNorm.h"
#include "Module/Conv2d.h"
#include "Module/Flatten.h"
#include "Module/Linear.h"
#include "Module/MaxPool2d.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Random/Generator.h"
#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Modules::BatchNorm;
using BeanTensor::Modules::Conv2d;
using BeanTensor::Modules::Flatten;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::MaxPool2d;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Random::set_seed;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

static void test_cnn_convergence_cpu() {
    set_seed(4242);

    // Synthetic dataset: 128 random 8x8 single-channel images, fixed random
    // labels. 8 -> conv3 -> 6 -> pool2 -> 3 -> conv3 -> 1; flatten 4.
    constexpr size_t n = 128;
    auto x = Tensor(std::vector<size_t>{n, 1, 8, 8}, DType::Float32, Device::CPU, false);
    x.random(-1.0, 1.0, 1);
    auto labels = Tensor(std::vector<size_t>{n}, DType::Int32, Device::CPU, false);
    for (auto i = int32_t{0}; i < static_cast<int32_t>(n); ++i) {
        const auto cls = (i * 7) % 4;
        labels.write_element(static_cast<size_t>(i), &cls, sizeof(int32_t));
    }

    auto net = Sequential();
    net.append("conv1", std::make_unique<Conv2d>(1, 4, 3, 1, 0, DType::Float32, Device::CPU, true, std::optional<int64_t>{}));
    net.append("bn1", std::make_unique<BatchNorm>(4, DType::Float32, Device::CPU, 0.1, 1e-5));
    net.append("relu1", std::make_unique<Relu>());
    net.append("pool1", std::make_unique<MaxPool2d>(2));
    net.append("conv2", std::make_unique<Conv2d>(4, 4, 3, 1, 0, DType::Float32, Device::CPU, true, std::optional<int64_t>{}));
    net.append("relu2", std::make_unique<Relu>());
    net.append("flat", std::make_unique<Flatten>());
    net.append("fc", std::make_unique<Linear>(4, 4, DType::Float32, Device::CPU));

    constexpr int32_t kSteps = 40;
    auto first_loss = float64_t{0};
    auto last_loss = float64_t{0};
    for (auto step = int32_t{0}; step < kSteps; ++step) {
        net.zero_grad();
        auto logits = net.forward(x);
        auto loss = logits.softmax_ce(labels);
        loss.backward(DType::Float32);
        net.adam_step(1e-3, 0.9, 0.999, 1e-8);
        loss.sync();
        const auto v = static_cast<float64_t>(loss.at<float32_t>({0}));
        if (step == 0) {
            first_loss = v;
        }
        last_loss = v;
        CHECK(std::isfinite(v));
    }
    CHECK(last_loss < first_loss);
    std::cout << "cnn convergence: first loss " << first_loss << " -> last loss " << last_loss << "\n";

    // Eval mode with the trained running statistics must produce finite
    // accuracy-style output through the same forward.
    net.eval();
    auto eval_logits = net.forward(x);
    eval_logits.sync();
    CHECK(std::isfinite(eval_logits.at<float32_t>({0, 0})));
    net.train();
    std::cout << "cnn eval-mode forward ok\n";
}

int main() {
    test_cnn_convergence_cpu();
    if (failures == 0) {
        std::cout << "ALL CNN TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
