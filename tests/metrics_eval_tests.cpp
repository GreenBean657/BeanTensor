/**
 * @file metrics_eval_tests.cpp
 * @author GreenBean
 * @brief Metrics family tests: accuracy composition and evaluate() with
 * requires_grad restoration. Framework-free assertion suite.
 * @version 0.1
 * @date 2026-10-30
 */
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Data/DataLoader.h"
#include "Helpers/Errors.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Metrics/Metrics.h"
#include "Tensor/Tensor.h"

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

using BeanTensor::Data::DataLoader;
using BeanTensor::DataTypes::DType;
using BeanTensor::Metrics::accuracy;
using BeanTensor::Metrics::evaluate;
using BeanTensor::Metrics::EvalResult;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Modules::softmax_cross_entropy;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;

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

// Snapshot every named parameter and its gradient into a map.
static std::map<std::string, Tensor> snapshot_params_and_grads(Sequential& model) {
    std::map<std::string, Tensor> snap;
    for (auto& [name, param] : model.named_parameters()) {
        CHECK(param->has_grad());
        auto cloned = param->clone(true);
        cloned.sync();
        snap.emplace(name, std::move(cloned));
        auto grad_clone = param->gradient().clone(true);
        grad_clone.sync();
        snap.emplace(name + ".grad", std::move(grad_clone));
    }
    return snap;
}

static bool same_bit_for_bit(Tensor& a, Tensor& b) {
    a.sync();
    b.sync();
    if (a.get_dtype() != b.get_dtype()) {
        return false;
    }
    if (!a.is_same_shape(b)) {
        return false;
    }
    auto flat_a = a.reshape({a.element_count()});
    auto flat_b = b.reshape({b.element_count()});
    const auto n = flat_a.element_count();
    if (flat_a.get_dtype() == DType::Float32) {
        for (size_t i = 0; i < n; ++i) {
            if (std::bit_cast<uint32_t>(flat_a.at<float32_t>({i})) != std::bit_cast<uint32_t>(flat_b.at<float32_t>({i}))) {
                return false;
            }
        }
        return true;
    }
    if (flat_a.get_dtype() == DType::Float64) {
        for (size_t i = 0; i < n; ++i) {
            if (std::bit_cast<uint64_t>(flat_a.at<float64_t>({i})) != std::bit_cast<uint64_t>(flat_b.at<float64_t>({i}))) {
                return false;
            }
        }
        return true;
    }
    if (flat_a.get_dtype() == DType::Int64) {
        for (size_t i = 0; i < n; ++i) {
            if (flat_a.at<int64_t>({i}) != flat_b.at<int64_t>({i})) {
                return false;
            }
        }
        return true;
    }
    return false;
}

static bool snapshots_equal(std::map<std::string, Tensor>& a,
                            std::map<std::string, Tensor>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (auto& [name, t] : a) {
        const auto it = b.find(name);
        if (it == b.end()) {
            return false;
        }
        if (!same_bit_for_bit(t, it->second)) {
            return false;
        }
    }
    return true;
}

static void do_train_step(Sequential& model, const Tensor& x, const Tensor& y) {
    model.zero_grad();
    auto logits = model.forward(x);
    auto loss = softmax_cross_entropy(logits, y);
    loss.backward(DType::Float32);
    constexpr auto lr = float64_t{0.01};
    constexpr auto beta1 = float64_t{0.9};
    constexpr auto beta2 = float64_t{0.999};
    constexpr auto epsilon = float64_t{1e-8};
    model.adam_step(lr, beta1, beta2, epsilon);
}

static void test_accuracy_known_answer() {
    auto logits = Tensor({4, 5}, DType::Float32, Device::CPU);
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            logits.set<float32_t>({r, c}, float32_t{0.1f});
        }
    }
    logits.set<float32_t>({0, 2}, float32_t{5.0f});
    logits.set<float32_t>({1, 0}, float32_t{5.0f});
    logits.set<float32_t>({2, 4}, float32_t{5.0f});
    logits.set<float32_t>({3, 1}, float32_t{5.0f});

    auto labels = Tensor({4}, DType::Int64, Device::CPU);
    labels.set<int64_t>({0}, int64_t{2});
    labels.set<int64_t>({1}, int64_t{0});
    labels.set<int64_t>({2}, int64_t{3}); // wrong on purpose
    labels.set<int64_t>({3}, int64_t{1});

    auto acc = accuracy(logits, labels);
    CHECK(acc.is_same_shape(Tensor({1})));
    CHECK(acc.get_dtype() == DType::Float32);
    CHECK(near64(static_cast<float64_t>(acc.at<float32_t>({0})), float64_t{0.75}, float64_t{1e-6}));
}

static void test_accuracy_tie_inherits_argmax() {
    // Two equal maxima in a row: argmax contract picks the first/lowest index.
    auto logits = Tensor({2, 4}, DType::Float32, Device::CPU);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            logits.set<float32_t>({r, c}, float32_t{0.0f});
        }
    }
    logits.set<float32_t>({0, 1}, float32_t{3.0f});
    logits.set<float32_t>({0, 2}, float32_t{3.0f});
    logits.set<float32_t>({1, 0}, float32_t{2.0f});
    logits.set<float32_t>({1, 3}, float32_t{2.0f});

    auto labels = Tensor({2}, DType::Int64, Device::CPU);
    labels.set<int64_t>({0}, int64_t{1}); // first of the tied maxima
    labels.set<int64_t>({1}, int64_t{0}); // first of the tied maxima

    auto acc = accuracy(logits, labels);
    CHECK(near64(static_cast<float64_t>(acc.at<float32_t>({0})), float64_t{1.0}, float64_t{1e-6}));
}

static void test_accuracy_all_right_all_wrong_single_row() {
    {
        auto logits = Tensor({1, 3}, DType::Float32, Device::CPU);
        logits.set<float32_t>({0, 0}, float32_t{1.0f});
        logits.set<float32_t>({0, 1}, float32_t{5.0f});
        logits.set<float32_t>({0, 2}, float32_t{2.0f});
        auto labels = Tensor({1}, DType::Int64, Device::CPU);
        labels.set<int64_t>({0}, int64_t{1});
        auto acc = accuracy(logits, labels);
        CHECK(near64(static_cast<float64_t>(acc.at<float32_t>({0})), float64_t{1.0}, float64_t{1e-6}));
    }
    {
        auto logits = Tensor({1, 3}, DType::Float32, Device::CPU);
        logits.set<float32_t>({0, 0}, float32_t{1.0f});
        logits.set<float32_t>({0, 1}, float32_t{5.0f});
        logits.set<float32_t>({0, 2}, float32_t{2.0f});
        auto labels = Tensor({1}, DType::Int64, Device::CPU);
        labels.set<int64_t>({0}, int64_t{2});
        auto acc = accuracy(logits, labels);
        CHECK(near64(static_cast<float64_t>(acc.at<float32_t>({0})), float64_t{0.0}, float64_t{1e-6}));
    }
    {
        auto logits = Tensor({3, 2}, DType::Float32, Device::CPU);
        logits.set<float32_t>({0, 0}, float32_t{1.0f});
        logits.set<float32_t>({0, 1}, float32_t{0.0f});
        logits.set<float32_t>({1, 0}, float32_t{0.0f});
        logits.set<float32_t>({1, 1}, float32_t{1.0f});
        logits.set<float32_t>({2, 0}, float32_t{1.0f});
        logits.set<float32_t>({2, 1}, float32_t{0.0f});
        auto labels = Tensor({3}, DType::Int64, Device::CPU);
        labels.set<int64_t>({0}, int64_t{0});
        labels.set<int64_t>({1}, int64_t{1});
        labels.set<int64_t>({2}, int64_t{0});
        auto acc = accuracy(logits, labels);
        CHECK(near64(static_cast<float64_t>(acc.at<float32_t>({0})), float64_t{1.0}, float64_t{1e-6}));
    }
}

static void test_accuracy_rank_errors() {
    {
        auto logits = Tensor({2, 3, 4}, DType::Float32, Device::CPU);
        auto labels = Tensor({2, 3}, DType::Int64, Device::CPU);
        auto threw = false;
        try {
            auto bad = accuracy(logits, labels);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto logits = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto labels = Tensor({2, 3}, DType::Int64, Device::CPU);
        auto threw = false;
        try {
            auto bad = accuracy(logits, labels);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto logits = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto labels = Tensor({4}, DType::Int64, Device::CPU);
        auto threw = false;
        try {
            auto bad = accuracy(logits, labels);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedShapes&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        auto logits = Tensor({2, 3}, DType::Float32, Device::CPU);
        auto labels = Tensor({2}, DType::Float32, Device::CPU);
        auto threw = false;
        try {
            auto bad = accuracy(logits, labels);
            static_cast<void>(bad);
        } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
            threw = true;
        }
        CHECK(threw);
    }
}

static void set_linear_weights(Linear& lin,
                               const std::vector<float32_t>& w,
                               const std::vector<float32_t>& b) {
    auto& weight = lin.parameter("weight");
    auto& bias = lin.parameter("bias");
    const auto out = lin.out_features();
    const auto in = lin.in_features();
    for (size_t r = 0; r < in; ++r) {
        for (size_t c = 0; c < out; ++c) {
            weight.set<float32_t>({r, c}, w[r * out + c]);
        }
    }
    for (size_t c = 0; c < out; ++c) {
        bias.set<float32_t>({c}, b[c]);
    }
}

static void test_evaluate_weighted_partial_batch() {
    auto model = Sequential();
    model.append("lin", std::make_unique<Linear>(2, 2, DType::Float32, Device::CPU, true, int64_t{42}));
    auto& lin = dynamic_cast<Linear&>(model.at("lin"));
    // Weight maps [a,b] to [10*a, 10*b]; bias is zero.
    set_linear_weights(lin,
                       {float32_t{10.0f}, float32_t{0.0f},
                        float32_t{0.0f}, float32_t{10.0f}},
                       {float32_t{0.0f}, float32_t{0.0f}});

    auto features = Tensor({5, 2}, DType::Float32, Device::CPU);
    auto labels = Tensor({5}, DType::Int64, Device::CPU);
    // First four rows predict class 0 but label is class 1 (wrong).
    for (size_t r = 0; r < 4; ++r) {
        features.set<float32_t>({r, 0}, float32_t{1.0f});
        features.set<float32_t>({r, 1}, float32_t{0.0f});
        labels.set<int64_t>({r}, int64_t{1});
    }
    // Last row predicts class 1 and is labeled class 1 (right).
    features.set<float32_t>({4, 0}, float32_t{0.0f});
    features.set<float32_t>({4, 1}, float32_t{1.0f});
    labels.set<int64_t>({4}, int64_t{1});

    auto loader = DataLoader(features, labels, 2, false, std::nullopt, false);

    // Compute expected weighted loss by iterating the same loader (without
    // evaluate()s requires_grad toggle) so the data path matches.
    auto expected_loader = DataLoader(features, labels, 2, false, std::nullopt, false);
    float64_t expected_loss = float64_t{0.0};
    size_t expected_rows = 0;
    while (true) {
        auto batch_opt = expected_loader.next_batch();
        if (!batch_opt.has_value()) {
            break;
        }
        auto& [batch_x, batch_y] = batch_opt.value();
        auto logits = model.forward(batch_x);
        auto loss = softmax_cross_entropy(logits, batch_y);
        loss.sync();
        expected_loss += static_cast<float64_t>(loss.at<float32_t>({0})) * static_cast<float64_t>(batch_x.dim(0));
        expected_rows += batch_x.dim(0);
    }
    expected_loss /= static_cast<float64_t>(expected_rows);

    auto result = evaluate(model, loader);
    CHECK(near64(result.accuracy, float64_t{1.0} / float64_t{5.0}, float64_t{1e-6}));
    CHECK(near64(result.mean_loss, expected_loss, float64_t{1e-5}));
}

static void test_evaluate_bit_identity() {
    auto model = Sequential();
    model.append("fc1", std::make_unique<Linear>(2, 3, DType::Float32, Device::CPU, true, int64_t{1}));
    model.append("relu", std::make_unique<Relu>());
    model.append("fc2", std::make_unique<Linear>(3, 2, DType::Float32, Device::CPU, true, int64_t{2}));

    auto features = Tensor({4, 2}, DType::Float32, Device::CPU);
    auto labels = Tensor({4}, DType::Int64, Device::CPU);
    for (size_t i = 0; i < 4; ++i) {
        features.set<float32_t>({i, 0}, float32_t{0.1f});
        features.set<float32_t>({i, 1}, float32_t{-0.1f});
        labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
    }

    do_train_step(model, features, labels);
    auto after_train = snapshot_params_and_grads(model);

    auto loader = DataLoader(features, labels, 2, false, std::nullopt, false);
    static_cast<void>(evaluate(model, loader));

    auto after_eval = snapshot_params_and_grads(model);
    CHECK(snapshots_equal(after_train, after_eval));
}

static void test_evaluate_no_graph_state() {
    auto make_model = []() {
        auto model = Sequential();
        model.append("fc1", std::make_unique<Linear>(2, 3, DType::Float32, Device::CPU, true, int64_t{3}));
        model.append("relu", std::make_unique<Relu>());
        model.append("fc2", std::make_unique<Linear>(3, 2, DType::Float32, Device::CPU, true, int64_t{4}));
        return model;
    };

    auto features = Tensor({4, 2}, DType::Float32, Device::CPU);
    auto labels = Tensor({4}, DType::Int64, Device::CPU);
    for (size_t i = 0; i < 4; ++i) {
        features.set<float32_t>({i, 0}, float32_t{0.2f});
        features.set<float32_t>({i, 1}, float32_t{-0.2f});
        labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
    }

    auto ref_model = make_model();
    do_train_step(ref_model, features, labels);
    do_train_step(ref_model, features, labels);
    auto ref_grads = snapshot_params_and_grads(ref_model);

    auto test_model = make_model();
    do_train_step(test_model, features, labels);
    auto loader = DataLoader(features, labels, 2, false, std::nullopt, false);
    auto eval1 = evaluate(test_model, loader);
    auto eval2 = evaluate(test_model, loader);
    do_train_step(test_model, features, labels);
    auto test_grads = snapshot_params_and_grads(test_model);

    CHECK(snapshots_equal(ref_grads, test_grads));
    CHECK(near64(eval1.accuracy, eval2.accuracy, float64_t{1e-9}));
    CHECK(near64(eval1.mean_loss, eval2.mean_loss, float64_t{1e-9}));
}

static void test_evaluate_restores_requires_grad_on_throw() {
    auto model = Sequential();
    model.append("lin", std::make_unique<Linear>(2, 2, DType::Float32, Device::CPU, true, int64_t{5}));

    auto good_features = Tensor({3, 2}, DType::Float32, Device::CPU);
    auto good_labels = Tensor({3}, DType::Int64, Device::CPU);
    for (size_t i = 0; i < 3; ++i) {
        good_features.set<float32_t>({i, 0}, float32_t{0.3f});
        good_features.set<float32_t>({i, 1}, float32_t{-0.3f});
        good_labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
    }

    auto ref_model = Sequential();
    ref_model.append("lin", std::make_unique<Linear>(2, 2, DType::Float32, Device::CPU, true, int64_t{5}));
    do_train_step(ref_model, good_features, good_labels);
    auto ref_grads = snapshot_params_and_grads(ref_model);

    // Corrupt labels to a float dtype so the first batch throws from softmax_ce.
    auto bad_labels = Tensor({3}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < 3; ++i) {
        bad_labels.set<float32_t>({i}, float32_t{0.0f});
    }
    auto bad_loader = DataLoader(good_features, bad_labels, 2, false, std::nullopt, false);

    auto threw = false;
    try {
        static_cast<void>(evaluate(model, bad_loader));
    } catch (const BeanTensor::ErrorHandling::MismatchedDTypes&) {
        threw = true;
    }
    CHECK(threw);

    // A subsequent training step must behave as if requires_grad were restored.
    do_train_step(model, good_features, good_labels);
    auto test_grads = snapshot_params_and_grads(model);
    CHECK(snapshots_equal(ref_grads, test_grads));
}

#if defined(USE_CUDA) || defined(USE_HIP)
static void test_accuracy_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto logits = Tensor({4, 5}, DType::Float32, Device::GPU);
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            logits.set<float32_t>({r, c}, float32_t{0.1f});
        }
    }
    logits.set<float32_t>({0, 2}, float32_t{5.0f});
    logits.set<float32_t>({1, 0}, float32_t{5.0f});
    logits.set<float32_t>({2, 4}, float32_t{5.0f});
    logits.set<float32_t>({3, 1}, float32_t{5.0f});

    auto labels = Tensor({4}, DType::Int64, Device::GPU);
    labels.set<int64_t>({0}, int64_t{2});
    labels.set<int64_t>({1}, int64_t{0});
    labels.set<int64_t>({2}, int64_t{3});
    labels.set<int64_t>({3}, int64_t{1});

    auto acc = accuracy(logits, labels);
    acc.sync();
    CHECK(near64(static_cast<float64_t>(acc.at<float32_t>({0})), float64_t{0.75}, float64_t{1e-6}));
}

static void test_evaluate_bit_identity_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto model = Sequential();
    model.append("fc1", std::make_unique<Linear>(2, 3, DType::Float32, Device::GPU, true, int64_t{6}));
    model.append("relu", std::make_unique<Relu>());
    model.append("fc2", std::make_unique<Linear>(3, 2, DType::Float32, Device::GPU, true, int64_t{7}));

    auto features = Tensor({4, 2}, DType::Float32, Device::GPU);
    auto labels = Tensor({4}, DType::Int64, Device::GPU);
    for (size_t i = 0; i < 4; ++i) {
        features.set<float32_t>({i, 0}, float32_t{0.1f});
        features.set<float32_t>({i, 1}, float32_t{-0.1f});
        labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
    }

    do_train_step(model, features, labels);
    auto after_train = snapshot_params_and_grads(model);

    auto loader = DataLoader(features, labels, 2, false, std::nullopt, false);
    static_cast<void>(evaluate(model, loader));

    auto after_eval = snapshot_params_and_grads(model);
    CHECK(snapshots_equal(after_train, after_eval));
}

static void test_evaluate_no_graph_state_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto make_model = []() {
        auto model = Sequential();
        model.append("fc1", std::make_unique<Linear>(2, 3, DType::Float32, Device::GPU, true, int64_t{8}));
        model.append("relu", std::make_unique<Relu>());
        model.append("fc2", std::make_unique<Linear>(3, 2, DType::Float32, Device::GPU, true, int64_t{9}));
        return model;
    };

    auto features = Tensor({4, 2}, DType::Float32, Device::GPU);
    auto labels = Tensor({4}, DType::Int64, Device::GPU);
    for (size_t i = 0; i < 4; ++i) {
        features.set<float32_t>({i, 0}, float32_t{0.2f});
        features.set<float32_t>({i, 1}, float32_t{-0.2f});
        labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
    }

    auto ref_model = make_model();
    do_train_step(ref_model, features, labels);
    do_train_step(ref_model, features, labels);
    auto ref_grads = snapshot_params_and_grads(ref_model);

    auto test_model = make_model();
    do_train_step(test_model, features, labels);
    auto loader = DataLoader(features, labels, 2, false, std::nullopt, false);
    auto eval1 = evaluate(test_model, loader);
    auto eval2 = evaluate(test_model, loader);
    do_train_step(test_model, features, labels);
    auto test_grads = snapshot_params_and_grads(test_model);

    CHECK(snapshots_equal(ref_grads, test_grads));
    CHECK(near64(eval1.accuracy, eval2.accuracy, float64_t{1e-9}));
    CHECK(near64(eval1.mean_loss, eval2.mean_loss, float64_t{1e-9}));
}

static void test_accuracy_cross_device_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto gpu_logits = Tensor({2, 3}, DType::Float32, Device::GPU);
    auto cpu_labels = Tensor({2}, DType::Int64, Device::CPU);
    auto threw = false;
    try {
        static_cast<void>(accuracy(gpu_logits, cpu_labels));
    } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_evaluate_cross_device_throws_and_restores_gpu() {
    if (!gpu_available()) {
        std::cerr << "SKIP: no GPU device available\n";
        return;
    }
    auto model = Sequential();
    model.append("lin", std::make_unique<Linear>(2, 2, DType::Float32, Device::GPU, true, int64_t{10}));

    auto gpu_features = Tensor({3, 2}, DType::Float32, Device::GPU);
    auto gpu_labels = Tensor({3}, DType::Int64, Device::GPU);
    auto cpu_features = Tensor({3, 2}, DType::Float32, Device::CPU);
    auto cpu_labels = Tensor({3}, DType::Int64, Device::CPU);
    for (size_t i = 0; i < 3; ++i) {
        gpu_features.set<float32_t>({i, 0}, float32_t{0.3f});
        gpu_features.set<float32_t>({i, 1}, float32_t{-0.3f});
        gpu_labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
        cpu_features.set<float32_t>({i, 0}, float32_t{0.3f});
        cpu_features.set<float32_t>({i, 1}, float32_t{-0.3f});
        cpu_labels.set<int64_t>({i}, static_cast<int64_t>(i % 2));
    }

    // Model on GPU, loader on CPU: evaluate must surface ErrorHandling::*.
    auto cpu_loader = DataLoader(cpu_features, cpu_labels, 2, false, std::nullopt, false);
    auto threw = false;
    try {
        static_cast<void>(evaluate(model, cpu_loader));
    } catch (const BeanTensor::ErrorHandling::MismatchedDevices&) {
        threw = true;
    }
    CHECK(threw);

    // requires_grad flags must have been restored despite the throw.
    for (auto& [name, param] : model.named_parameters()) {
        static_cast<void>(name);
        CHECK(param->requires_gradient());
    }

    // A subsequent training step on the matching device must succeed.
    auto gpu_loader = DataLoader(gpu_features, gpu_labels, 2, false, std::nullopt, false);
    static_cast<void>(evaluate(model, gpu_loader));

    do_train_step(model, gpu_features, gpu_labels);
    CHECK(model.named_parameters().size() > 0);
    for (auto& [name, param] : model.named_parameters()) {
        static_cast<void>(name);
        CHECK(param->has_grad());
    }
}
#endif

int main() {
    test_accuracy_known_answer();
    test_accuracy_tie_inherits_argmax();
    test_accuracy_all_right_all_wrong_single_row();
    test_accuracy_rank_errors();
    test_evaluate_weighted_partial_batch();
    test_evaluate_bit_identity();
    test_evaluate_no_graph_state();
    test_evaluate_restores_requires_grad_on_throw();
#if defined(USE_CUDA) || defined(USE_HIP)
    test_accuracy_gpu();
    test_accuracy_cross_device_gpu();
    test_evaluate_bit_identity_gpu();
    test_evaluate_no_graph_state_gpu();
    test_evaluate_cross_device_throws_and_restores_gpu();
#endif
    if (failures == 0) {
        std::cout << "ALL METRICS EVAL TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
