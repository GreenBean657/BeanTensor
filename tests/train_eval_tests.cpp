// BeanTensorV5 Module train()/eval() mode-flag tests.
// Covers: default training mode on fresh modules, recursive eval/train on
// child modules via Sequential, and the train(false) override.
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "Module/Module.h"
#include "Module/Sequential.h"
#include "Tensor/Tensor.h"

using BeanTensor::Modules::Module;
using BeanTensor::Modules::Sequential;
using BeanTensor::Tensors::Tensor;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

// Probe module: records the training flag seen during forward().
struct FlagProbe : Module {
    bool seen{};

    [[nodiscard]] Tensor forward(const Tensor& x) override {
        this->seen = this->is_training();
        static_cast<void>(x);
        return Tensor({1}, BeanTensor::DataTypes::DType::Float32, BeanTensor::Tensors::Device::CPU);
    }

    [[nodiscard]] std::string name() const override {
        return "FlagProbe";
    }
};

static void test_fresh_modules_are_training() {
    auto probe = FlagProbe();
    CHECK(probe.is_training());

    auto container = Sequential();
    CHECK(container.is_training());
}

static void test_eval_recurses_into_children() {
    auto probe = std::make_unique<FlagProbe>();
    auto* raw_probe = probe.get();
    auto container = Sequential();
    container.append("probe", std::move(probe));

    auto x = Tensor({2, 3}, BeanTensor::DataTypes::DType::Float32, BeanTensor::Tensors::Device::CPU);
    x.set_zero();

    container.eval();
    static_cast<void>(container.forward(x));
    CHECK(!raw_probe->is_training());
    CHECK(!raw_probe->seen);

    container.train();
    static_cast<void>(container.forward(x));
    CHECK(raw_probe->is_training());
    CHECK(raw_probe->seen);
}

static void test_train_false_leaves_eval() {
    auto probe = std::make_unique<FlagProbe>();
    auto* raw_probe = probe.get();
    auto container = Sequential();
    container.append("probe", std::move(probe));

    container.eval();
    container.train(false);
    CHECK(!container.is_training());
    CHECK(!raw_probe->is_training());
}

int main() {
    test_fresh_modules_are_training();
    test_eval_recurses_into_children();
    test_train_false_leaves_eval();

    if (failures == 0) {
        std::cout << "ALL TRAIN/EVAL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
