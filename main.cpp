/**
 * @file main.cpp
 * @author GreenBean
 * @brief MNIST demo over the Module layer stack: reads the uncompressed IDX
 * files from datasets/mnist/, prints live thread-pool telemetry (snapshot + format) on the first
 * batch of each epoch, trains a Sequential MLP (Linear -> Relu -> Linear) with
 * a seeded DataLoader, Adam, and softmax cross-entropy, and
 * reports test accuracy after every epoch. Every tensor access is gated on
 * sync() or an explicit dtype check. Data loading lives in demos/Mnist.h,
 * shared with the CNN demo (cnn.cpp).
 */
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/Relu.h"
#include "Module/ModuleStream.h"
#include "Module/Sequential.h"
#include "Data/DataLoader.h"
#include "Metrics/Metrics.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"
#include "demos/Mnist.h"

using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Modules::softmax_cross_entropy;
using BeanTensor::Data::DataLoader;
using BeanTensor::Metrics::EvalResult;
using BeanTensor::Metrics::evaluate;
using BeanTensor::Random::set_seed;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

namespace {

    constexpr size_t kInputs = demos::mnist::kPixels;
    constexpr size_t kHidden = 128;
    constexpr size_t kClasses = 10;
    constexpr size_t kBatch = 100;
    constexpr size_t kEpochs = 3;
    constexpr float64_t kLearningRate = 0.001;
    constexpr float64_t kBeta1 = 0.9;
    constexpr float64_t kBeta2 = 0.999;
    constexpr float64_t kEpsilon = 1e-8;
    // Float16's smallest positive value is ~6e-8, so 1e-8 rounds to zero and
    // Adam divides by zero on tiny gradients. Half-precision runs use this instead.
    constexpr float64_t kEpsilonHalf = 1e-4;
    constexpr uint32_t kShuffleSeed = 1234;

    // The model (weights, inputs, loss) runs in kDType; gradients in gDType.
    constexpr auto kDType = DType::Float32;
    constexpr auto gDType = DType::Float32;

    // Host staging dtype for dataset upload.
    constexpr auto kHostDType = DType::Float32;

    // Train and evaluate the same MLP on the requested device. Both splits are
    // decoded and moved to the device before the timer starts; batches are drawn
    // from DataLoader on the device. The model and its optimizer state live on
    // the device.
    [[nodiscard]] bool run_training(const demos::mnist::Split& train, const demos::mnist::Split& test, const Device device) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Device: " << BeanTensor::Tensors::stringify_device_source(device) << "\n";
        std::cout << std::string(60, '=') << "\n";

        auto net = Sequential();
        net.append("fc1", std::make_unique<Linear>(kInputs, kHidden, kDType, device));
        net.append("relu", std::make_unique<Relu>());
        net.append("fc2", std::make_unique<Linear>(kHidden, kClasses, kDType, device));
        std::cout << net;

        auto train_data = demos::mnist::upload_flat(train, device, kDType, kHostDType);
        auto test_data = demos::mnist::upload_flat(test, device, kDType, kHostDType);
        if (
            const auto uploaded = train_data.has_value() && test_data.has_value();
            !uploaded
        ) {
            return false;
        }

        auto train_loader = DataLoader(
            train_data->images,
            train_data->labels,
            kBatch,
            true,
            kShuffleSeed,
            true
        );
        auto test_loader = DataLoader(
            test_data->images,
            test_data->labels,
            kBatch,
            false,
            std::nullopt,
            true
        );

        const auto run_start = std::chrono::steady_clock::now();
        for (auto epoch = size_t{0}; epoch < kEpochs; ++epoch) {
            const auto epoch_start = std::chrono::steady_clock::now();
            train_loader.reset();
            auto loss_acc = Tensor(std::vector<size_t>{1}, kDType, device);
            loss_acc.set_zero();
            auto steps = size_t{0};
            while (true) {
                auto batch_opt = train_loader.next_batch();
                if (!batch_opt.has_value()) {
                    break;
                }
                auto& [batch, labels] = batch_opt.value();
                if (
                    const auto batch_ok = demos::mnist::expect_dtype(batch, kDType, "train: input batch");
                    !batch_ok
                ) {
                    return false;
                }

                net.zero_grad();
                auto out = net.forward(batch);
                auto loss = softmax_cross_entropy(out, labels);
                loss.backward(gDType);
                constexpr auto epsilon = (kDType == DType::Float16 || kDType == DType::Float16x2) ? kEpsilonHalf : kEpsilon;
                net.adam_step(kLearningRate, kBeta1, kBeta2, epsilon);

                if (
                    const auto loss_ok = demos::mnist::expect_dtype(loss, kDType, "train loss");
                    !loss_ok
                ) {
                    return false;
                }
                loss_acc = loss_acc.add(loss);
                ++steps;
            }
            // Drain any pending optimizer/gradient work before running the
            // synchronous test-only forward pass.
            for (auto& [name, param] : net.named_parameters()) {
                static_cast<void>(name);
                param->sync();
            }
            net.eval();
            const auto result = evaluate(net, test_loader);
            net.train();
            const auto epoch_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - epoch_start).count();
            std::cout << "epoch " << (epoch + 1) << "/" << kEpochs
                      << "  train mean loss " << (demos::mnist::read_scalar(loss_acc, kHostDType) / static_cast<float64_t>(steps))
                      << "  test accuracy " << (result.accuracy * float64_t{100}) << "%"
                      << "  test mean loss " << result.mean_loss
                      << "  time " << epoch_seconds << " s\n";
        }
        const auto total_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - run_start).count();
        std::cout << "total time " << total_seconds << " s\n";
        std::cout << "\nafter training:\n" << net;
        return true;
    }
}

int main() {
    set_seed(kShuffleSeed);

    const auto dir = std::filesystem::path(demos::mnist::kDir);
    auto train = demos::mnist::Split{};
    auto test = demos::mnist::Split{};
    if (
        const auto loaded = demos::mnist::load_split(dir, demos::mnist::kTrainImages, demos::mnist::kTrainLabels, train)
            && demos::mnist::load_split(dir, demos::mnist::kTestImages, demos::mnist::kTestLabels, test);
        !loaded
    ) {
        std::cerr << "Place the uncompressed MNIST IDX files under " << dir.string() << "/ and rerun.\n";
        return 1;
    }
    std::cout << "Loaded " << train.count << " training and " << test.count << " test images\n";

    // CPU pass.
    if (!run_training(train, test, Device::CPU)) {
        return 1;
    }

    // GPU pass when the build includes a GPU backend.
#if defined(USE_CUDA) || defined(USE_HIP)
    if (!run_training(train, test, Device::GPU)) {
        return 1;
    }
#else
    std::cout << "\nNo GPU backend compiled in; skipping GPU run.\n";
#endif

    return 0;
}
