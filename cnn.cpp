/**
 * @file cnn.cpp
 * @author GreenBean
 * @brief MNIST CNN demo over the spatial layer stack: Conv2d -> BatchNorm ->
 * Relu -> MaxPool2d twice, then Flatten -> Dropout -> Linear -> Relu ->
 * Linear, trained with a seeded DataLoader, Adam, and softmax
 * cross-entropy, evaluated per epoch through Metrics::evaluate. Reads the
 * same IDX files as the MLP demo (demos/Mnist.h) as NCHW images. Runs CPU
 * always and GPU when the build includes a backend. Complemented by
 * comparison/cnn_pytorch.py and recorded in docs/cnn-results.md.
 */
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Module/BatchNorm.h"
#include "Module/Conv2d.h"
#include "Module/Dropout.h"
#include "Module/Flatten.h"
#include "Module/Linear.h"
#include "Module/MaxPool2d.h"
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
using BeanTensor::Modules::BatchNorm;
using BeanTensor::Modules::Conv2d;
using BeanTensor::Modules::Dropout;
using BeanTensor::Modules::Flatten;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::MaxPool2d;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Modules::softmax_cross_entropy;
using BeanTensor::Data::DataLoader;
using BeanTensor::Metrics::evaluate;
using BeanTensor::Random::set_seed;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

namespace {

    // 28 -> conv3 -> 26 -> pool2 -> 13 -> conv3 -> 11 -> pool2 -> 5, so the
    // flatten width below is 16 * 5 * 5.
    constexpr size_t kChannels1 = 8;
    constexpr size_t kChannels2 = 16;
    constexpr size_t kFinalSide = 5;
    constexpr size_t kFlattened = kChannels2 * kFinalSide * kFinalSide;
    constexpr size_t kHidden = 64;
    constexpr size_t kClasses = 10;
    constexpr size_t kBatch = 128;
    constexpr size_t kEpochs = 2;
    constexpr float64_t kDropout = 0.25;
    constexpr float64_t kLearningRate = 0.001;
    constexpr float64_t kBeta1 = 0.9;
    constexpr float64_t kBeta2 = 0.999;
    constexpr float64_t kEpsilon = 1e-8;
    constexpr uint32_t kShuffleSeed = 1234;

    constexpr auto kDType = DType::Float32;
    constexpr auto gDType = DType::Float32;
    constexpr auto kHostDType = DType::Float32;

    // A CPU epoch over the full 60k-image set is minutes long; the ctest
    // entry uses a small subsample so the suite (and CI) stay fast, and the
    // recorded results run picks a size that finishes in sensible wall time.
    // The PyTorch counterpart takes the same flags for an apples-to-apples
    // comparison.
    struct RunConfig {
        size_t train_rows = 60000;
        size_t test_rows = 10000;
        size_t epochs = kEpochs;
    };

    [[nodiscard]] RunConfig parse_args(const int argc, char** argv) {
        auto cfg = RunConfig{};
        for (auto i = int32_t{1}; i < argc; ++i) {
            const auto arg = std::string(argv[i]);
            const auto next = [&](const char* flag) -> std::optional<size_t> {
                if (arg == flag && i + 1 < argc) {
                    ++i;
                    return static_cast<size_t>(std::stoull(argv[i]));
                }
                return std::nullopt;
            };
            if (
                const auto v = next("--train-rows");
                v.has_value()
            ) {
                cfg.train_rows = *v;
            } else if (
                const auto v = next("--test-rows");
                v.has_value()
            ) {
                cfg.test_rows = *v;
            } else if (
                const auto v = next("--epochs");
                v.has_value()
            ) {
                cfg.epochs = *v;
            }
        }
        cfg.train_rows = std::min(cfg.train_rows, size_t{60000});
        cfg.test_rows = std::min(cfg.test_rows, size_t{10000});
        return cfg;
    }

    demos::mnist::Split subsample(const demos::mnist::Split& in, const size_t rows) {
        auto out = demos::mnist::Split{};
        out.count = std::min(rows, in.count);
        out.images.resize(out.count * demos::mnist::kPixels);
        out.labels.resize(out.count);
        std::copy_n(in.images.begin(), static_cast<std::ptrdiff_t>(out.count * demos::mnist::kPixels), out.images.begin());
        std::copy_n(in.labels.begin(), static_cast<std::ptrdiff_t>(out.count), out.labels.begin());
        return out;
    }

    [[nodiscard]] bool run_training(const demos::mnist::Split& train, const demos::mnist::Split& test,
                                    const Device device, const size_t epochs) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Device: " << BeanTensor::Tensors::stringify_device_source(device) << "\n";
        std::cout << std::string(60, '=') << "\n";

        auto net = Sequential();
        net.append("conv1", std::make_unique<Conv2d>(1, kChannels1, 3, 1, 0, kDType, device, true, std::optional<int64_t>{}));
        net.append("bn1", std::make_unique<BatchNorm>(kChannels1, kDType, device, 0.1, 1e-5));
        net.append("relu1", std::make_unique<Relu>());
        net.append("pool1", std::make_unique<MaxPool2d>(2));
        net.append("conv2", std::make_unique<Conv2d>(kChannels1, kChannels2, 3, 1, 0, kDType, device, true, std::optional<int64_t>{}));
        net.append("bn2", std::make_unique<BatchNorm>(kChannels2, kDType, device, 0.1, 1e-5));
        net.append("relu2", std::make_unique<Relu>());
        net.append("pool2", std::make_unique<MaxPool2d>(2));
        net.append("flat", std::make_unique<Flatten>());
        net.append("drop", std::make_unique<Dropout>(kDropout));
        net.append("fc1", std::make_unique<Linear>(kFlattened, kHidden, kDType, device));
        net.append("relu3", std::make_unique<Relu>());
        net.append("fc2", std::make_unique<Linear>(kHidden, kClasses, kDType, device));
        std::cout << net;

        auto train_data = demos::mnist::upload_nchw(train, device, kDType, kHostDType);
        auto test_data = demos::mnist::upload_nchw(test, device, kDType, kHostDType);
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
        for (auto epoch = size_t{0}; epoch < epochs; ++epoch) {
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
                net.adam_step(kLearningRate, kBeta1, kBeta2, kEpsilon);

                if (
                    const auto loss_ok = demos::mnist::expect_dtype(loss, kDType, "train loss");
                    !loss_ok
                ) {
                    return false;
                }
                loss_acc = loss_acc.add(loss);
                ++steps;
            }
            // Drain pending optimizer/gradient work before the synchronous
            // eval pass; BatchNorm must also be in eval mode so the running
            // statistics (not batch statistics) drive the accuracy number.
            for (auto& [name, param] : net.named_parameters()) {
                static_cast<void>(name);
                param->sync();
            }
            net.eval();
            const auto result = evaluate(net, test_loader);
            net.train();
            const auto epoch_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - epoch_start).count();
            std::cout << "epoch " << (epoch + 1) << "/" << epochs
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

int main(const int argc, char** argv) {
    set_seed(kShuffleSeed);
    const auto cfg = [&] {
        try {
            return parse_args(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "arguments: [--train-rows N] [--test-rows N] [--epochs N]\n";
            throw;
        }
    }();

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

    const auto train_sub = subsample(train, cfg.train_rows);
    const auto test_sub = subsample(test, cfg.test_rows);
    std::cout << "Using " << train_sub.count << " training and " << test_sub.count << " test images\n";

    if (!run_training(train_sub, test_sub, Device::CPU, cfg.epochs)) {
        return 1;
    }

#if defined(USE_CUDA) || defined(USE_HIP)
    if (!run_training(train_sub, test_sub, Device::GPU, cfg.epochs)) {
        return 1;
    }
#else
    std::cout << "\nNo GPU backend compiled in; skipping GPU run.\n";
#endif

    return 0;
}
