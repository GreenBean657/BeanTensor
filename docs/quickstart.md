# Quickstart

This page is a complete, copy-paste MNIST training example. It mirrors the
repository's `main.cpp` but is stripped to a single CPU-only run so you can
build and run it immediately.

## Build

```bash
cd <your-beantensor-clone>
cmake -B build-cpu -S . -DUSE_CUDA=OFF -DUSE_HIP=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target BeanTensorV5Core -j 6
```

Ninja is recommended if it is installed (`-G Ninja` on the configure line). The
example below links against the resulting `build-cpu/libBeanTensorV5Core.a`.

## MNIST training example

Save as `mnist_quickstart.cpp` in the repository root and fetch the MNIST data:

```bash
./fetch_mnist.sh
```

This places the uncompressed IDX files under `datasets/mnist/`. Then create the
source file:

```cpp
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Data/DataLoader.h"
#include "Metrics/Metrics.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/ModuleStream.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"

using BeanTensor::Data::DataLoader;
using BeanTensor::Metrics::evaluate;
using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Modules::softmax_cross_entropy;
using BeanTensor::Random::set_seed;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

namespace {

    constexpr auto kMnistDir = "datasets/mnist";
    constexpr auto kTrainImages = "train-images-idx3-ubyte";
    constexpr auto kTrainLabels = "train-labels-idx1-ubyte";
    constexpr auto kTestImages = "t10k-images-idx3-ubyte";
    constexpr auto kTestLabels = "t10k-labels-idx1-ubyte";

    constexpr uint32_t kImageMagic = 0x00000803;
    constexpr uint32_t kLabelMagic = 0x00000801;
    constexpr uint32_t kImageSide = 28;
    constexpr size_t kImageHeaderBytes = 16;
    constexpr size_t kLabelHeaderBytes = 8;

    constexpr size_t kInputs = 784;
    constexpr size_t kHidden = 128;
    constexpr size_t kClasses = 10;
    constexpr size_t kBatch = 100;
    constexpr size_t kEpochs = 3;
    constexpr float64_t kLearningRate = 0.001;
    constexpr float64_t kBeta1 = 0.9;
    constexpr float64_t kBeta2 = 0.999;
    constexpr float64_t kEpsilon = 1e-8;
    constexpr uint32_t kShuffleSeed = 1234;
    constexpr auto kDType = DType::Float32;
    constexpr auto gDType = DType::Float32;

    struct MnistSplit {
        size_t count = 0;
        std::vector<uint8_t> images;
        std::vector<uint8_t> labels;
    };

    struct FileCloser {
        void operator()(std::FILE* file) const {
            std::fclose(file);
        }
    };
    using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

    [[nodiscard]] uint32_t read_be32(const std::vector<uint8_t>& bytes, const size_t offset) {
        return (static_cast<uint32_t>(bytes[offset]) << 24)
             | (static_cast<uint32_t>(bytes[offset + 1]) << 16)
             | (static_cast<uint32_t>(bytes[offset + 2]) << 8)
             | static_cast<uint32_t>(bytes[offset + 3]);
    }

    [[nodiscard]] bool read_file(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
        const auto file = FilePtr(std::fopen(path.string().c_str(), "rb"));
        if (!file) {
            std::cerr << "Cannot open " << path.string() << "\n";
            return false;
        }
        bytes.resize(static_cast<size_t>(std::filesystem::file_size(path)));
        if (
            const auto read_count = std::fread(bytes.data(), 1, bytes.size(), file.get());
            read_count != bytes.size()
        ) {
            std::cerr << "Short read on " << path.string() << "\n";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool load_split(const std::filesystem::path& dir, const std::string_view image_file,
                                  const std::string_view label_file, MnistSplit& split) {
        auto image_bytes = std::vector<uint8_t>{};
        auto label_bytes = std::vector<uint8_t>{};
        if (
            const auto read_ok = read_file(dir / image_file, image_bytes) && read_file(dir / label_file, label_bytes);
            !read_ok
        ) {
            return false;
        }
        if (
            const auto header_ok = image_bytes.size() >= kImageHeaderBytes && label_bytes.size() >= kLabelHeaderBytes;
            !header_ok
        ) {
            std::cerr << "Truncated IDX header in " << dir.string() << "\n";
            return false;
        }
        if (
            const auto image_format_ok = read_be32(image_bytes, 0) == kImageMagic
                && read_be32(image_bytes, 8) == kImageSide
                && read_be32(image_bytes, 12) == kImageSide;
            !image_format_ok
        ) {
            std::cerr << "Not a 28x28 IDX image file: " << image_file << "\n";
            return false;
        }
        if (
            const auto label_format_ok = read_be32(label_bytes, 0) == kLabelMagic;
            !label_format_ok
        ) {
            std::cerr << "Not an IDX label file: " << label_file << "\n";
            return false;
        }
        const auto count = static_cast<size_t>(read_be32(image_bytes, 4));
        if (
            const auto counts_ok = read_be32(label_bytes, 4) == count
                && image_bytes.size() == kImageHeaderBytes + count * kInputs
                && label_bytes.size() == kLabelHeaderBytes + count;
            !counts_ok
        ) {
            std::cerr << "Image/label counts disagree in " << dir.string() << "\n";
            return false;
        }
        split.count = count;
        split.images.assign(image_bytes.begin() + kImageHeaderBytes, image_bytes.end());
        split.labels.assign(label_bytes.begin() + kLabelHeaderBytes, label_bytes.end());
        return true;
    }

    [[nodiscard]] Tensor upload_images(const MnistSplit& split) {
        auto images = Tensor(std::vector<size_t>{split.count, kInputs}, kDType);
        for (auto row = size_t{0}; row < split.count; ++row) {
            for (auto col = size_t{0}; col < kInputs; ++col) {
                const auto pixel = static_cast<float32_t>(split.images[row * kInputs + col]);
                images.set<float32_t>({row, col}, pixel / float32_t{255});
            }
        }
        return images;
    }

    [[nodiscard]] Tensor upload_labels(const MnistSplit& split) {
        auto labels = Tensor(std::vector<size_t>{split.count}, DType::Int64);
        for (auto row = size_t{0}; row < split.count; ++row) {
            labels.set<int64_t>({row}, static_cast<int64_t>(split.labels[row]));
        }
        return labels;
    }

    [[nodiscard]] float64_t read_scalar(Tensor& t) {
        auto host = t.clone();
        host.convert_dtype(DType::Float32);
        host.sync();
        return static_cast<float64_t>(host.at<float32_t>({0}));
    }

    [[nodiscard]] bool run_training(const MnistSplit& train, const MnistSplit& test) {
        const auto device = Device::CPU;
        auto net = Sequential();
        net.append("fc1", std::make_unique<Linear>(kInputs, kHidden, kDType, device));
        net.append("relu", std::make_unique<Relu>());
        net.append("fc2", std::make_unique<Linear>(kHidden, kClasses, kDType, device));
        std::cout << net;

        auto train_images = upload_images(train);
        auto train_labels = upload_labels(train);
        auto test_images = upload_images(test);
        auto test_labels = upload_labels(test);

        auto train_loader = DataLoader(train_images, train_labels, kBatch, true, kShuffleSeed, true);
        auto test_loader = DataLoader(test_images, test_labels, kBatch, false, std::nullopt, true);

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

                net.zero_grad();
                auto out = net.forward(batch);
                auto loss = softmax_cross_entropy(out, labels);
                loss.backward(gDType);
                net.adam_step(kLearningRate, kBeta1, kBeta2, kEpsilon);

                loss_acc = loss_acc.add(loss);
                ++steps;
            }
            for (auto& [name, param] : net.named_parameters()) {
                static_cast<void>(name);
                param->sync();
            }
            net.eval();
            const auto result = evaluate(net, test_loader);
            net.train();
            const auto epoch_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - epoch_start).count();
            std::cout << "epoch " << (epoch + 1) << "/" << kEpochs
                      << "  train mean loss " << (read_scalar(loss_acc) / static_cast<float64_t>(steps))
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

    const auto dir = std::filesystem::path(kMnistDir);
    auto train = MnistSplit{};
    auto test = MnistSplit{};
    if (
        const auto loaded = load_split(dir, kTrainImages, kTrainLabels, train)
            && load_split(dir, kTestImages, kTestLabels, test);
        !loaded
    ) {
        std::cerr << "Place the uncompressed MNIST IDX files under " << dir.string() << "/ and rerun.\n";
        return 1;
    }
    std::cout << "Loaded " << train.count << " training and " << test.count << " test images\n";

    if (!run_training(train, test)) {
        return 1;
    }
    return 0;
}
```

Compile and run:

```bash
./fetch_mnist.sh
g++ -std=c++23 -O2 -I. \
    mnist_quickstart.cpp \
    -Lbuild-cpu -lBeanTensorV5Core \
    -lpthread -o mnist_quickstart
./mnist_quickstart
```

You should see three epochs of training output ending with test accuracy around
97% on MNIST. See [`docs/mnist-results.md`](mnist-results.md) for the full
acceptance numbers.
