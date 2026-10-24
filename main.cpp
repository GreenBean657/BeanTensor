/**
 * @file main.cpp
 * @author GreenBean
 * @brief MNIST demo over the Module layer stack: reads the uncompressed IDX
 * files from data/mnist/, trains a Sequential MLP (Linear -> Relu -> Linear)
 * with softmax_cross_entropy against class labels and Module::adam_step, and
 * reports test accuracy after every epoch. Every tensor access is gated on
 * its dtype once per batch; a mismatch prints to std::cerr and exits 1.
 * Batches are staged in Float32 on the host and converted to the model dtype
 * with Tensor::convert_dtype, so packed dtypes (Float16x2) never need a
 * host-side scalar type.
 * @version 0.3.8
 * @date 2026-10-23
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Tensor/Tensor.h"

using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Module;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Modules::softmax_cross_entropy;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

namespace {

    constexpr auto kMnistDir = "data/mnist";
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
    // Float16's smallest positive value is ~6e-8, so 1e-8 rounds to zero and
    // Adam divides by zero on tiny gradients. Half-precision runs use this instead.
    constexpr float64_t kEpsilonHalf = 1e-4;
    constexpr uint32_t kShuffleSeed = 1234;

    // The model (weights, inputs, loss) runs in kDType; gradients in gDType.
    constexpr auto kDType = DType::Float32;
    constexpr auto gDType = DType::Float32;
    constexpr auto kPacked = kDType == DType::Float16x2 || kDType == DType::BFloat16x2;

    // Host side: every value the demo writes or reads is this scalar type.
    // Packed dtypes hold two values per element, so batches are staged here
    // and converted to kDType rather than written into it directly.
    constexpr auto kHostDType = DType::Float32;
    using Host = float32_t;

    // Raw MNIST split: pixels stay as bytes until a batch is filled, so the
    // full 60k-image training set costs ~47 MB instead of ~188 MB of floats.
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

    // Per-batch dtype gate: Tensor::set/at stride by sizeof(T) and do not check
    // T, so a tensor of the wrong dtype would be corrupted silently. Prints the
    // mismatch and returns false; callers stop instead of writing.
    [[nodiscard]] bool expect_dtype(const Tensor& t, const DType expected, const std::string_view name) {
        if (
            const auto dtype_ok = t.get_dtype() == expected;
            !dtype_ok
        ) {
            std::cerr << name << " has dtype " << BeanTensor::DataTypes::dtype_to_string(t.get_dtype())
                      << ", expected " << BeanTensor::DataTypes::dtype_to_string(expected) << "\n";
            return false;
        }
        return true;
    }

    // Convert a host-staged tensor to the model dtype. Packed targets duplicate
    // each value into their lanes; the conversion runs on the tensor itself.
    [[nodiscard]] Tensor to_model_dtype(Tensor staged) {
        if (staged.get_dtype() != kDType) {
            staged.convert_dtype(kDType);
        }
        staged.sync();
        return staged;
    }

    // Stage kBatch samples (looked up through order) as Host values scaled to
    // [0, 1], then convert to kDType. Returns nullopt if the result is not kDType.
    [[nodiscard]] std::optional<Tensor> fill_inputs(const MnistSplit& split, const std::vector<size_t>& order, const size_t start) {
        auto staged = Tensor(std::vector<size_t>{kBatch, kInputs}, kHostDType);
        for (auto row = size_t{0}; row < kBatch; ++row) {
            const auto sample = order[start + row];
            for (auto col = size_t{0}; col < kInputs; ++col) {
                const auto pixel = static_cast<Host>(split.images[sample * kInputs + col]);
                staged.set<Host>({row, col}, pixel / Host{255});
            }
        }
        auto batch = to_model_dtype(std::move(staged));
        if (
            const auto batch_ok = expect_dtype(batch, kDType, "fill_inputs: input batch");
            !batch_ok
        ) {
            return std::nullopt;
        }
        return batch;
    }

    // Class index per sample for the same batch as fill_inputs. Returns false
    // if labels is not Int64.
    [[nodiscard]] bool fill_labels(const MnistSplit& split, const std::vector<size_t>& order, const size_t start, Tensor& labels) {
        if (
            const auto labels_ok = expect_dtype(labels, DType::Int64, "fill_labels: label tensor");
            !labels_ok
        ) {
            return false;
        }
        for (auto row = size_t{0}; row < kBatch; ++row) {
            labels.set<int64_t>({row}, static_cast<int64_t>(split.labels[order[start + row]]));
        }
        return true;
    }

    // Read a one-element tensor as a float64 value. Works on a copy, converted
    // to the host dtype first, so packed storage is never read directly.
    [[nodiscard]] float64_t read_scalar(Tensor& t) {
        auto host = t.clone();
        host.convert_dtype(kHostDType);
        host.sync();
        return static_cast<float64_t>(host.at<Host>({0}));
    }

    [[nodiscard]] std::vector<size_t> identity_order(const size_t count) {
        auto order = std::vector<size_t>(count);
        for (auto i = size_t{0}; i < count; ++i) {
            order[i] = i;
        }
        return order;
    }

    // Fraction of samples whose argmax prediction matches the label. Forward
    // only: no backward() follows, so nothing is trained on this pass.
    // Returns nullopt if a tensor has the wrong dtype.
    [[nodiscard]] std::optional<float64_t> evaluate(Sequential& net, const MnistSplit& split) {
        const auto order = identity_order(split.count);
        auto correct = size_t{0};
        auto seen = size_t{0};
        for (auto start = size_t{0}; start + kBatch <= split.count; start += kBatch) {
            auto batch = fill_inputs(split, order, start);
            if (
                const auto filled = batch.has_value();
                !filled
            ) {
                return std::nullopt;
            }
            auto logits = net.forward(*batch);
            // Packed logits need the lane-aware argmax: each element is two classes.
            auto predictions = kPacked ? logits.argmax_lanes(1) : logits.argmax(1);
            if (
                const auto predictions_ok = expect_dtype(predictions, DType::Int64, "evaluate: predictions");
                !predictions_ok
            ) {
                return std::nullopt;
            }
            for (auto row = size_t{0}; row < kBatch; ++row) {
                if (
                    const auto predicted = static_cast<size_t>(predictions.at<int64_t>({row}));
                    predicted == static_cast<size_t>(split.labels[start + row])
                ) {
                    ++correct;
                }
            }
            seen += kBatch;
        }
        return static_cast<float64_t>(correct) / static_cast<float64_t>(seen);
    }
}

int main() {
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

    auto net = Sequential();
    net.append("fc1", std::make_unique<Linear>(kInputs, kHidden, kDType));
    net.append("relu", std::make_unique<Relu>());
    net.append("fc2", std::make_unique<Linear>(kHidden, kClasses, kDType));
    std::cout << net;

    // Labels live outside the training loop and are refilled each step. The
    // input batch is built inside the step, so it outlives backward() on that step.
    auto labels = Tensor(std::vector<size_t>{kBatch}, DType::Int64);

    auto rng = std::mt19937{kShuffleSeed};
    auto order = identity_order(train.count);
    const auto run_start = std::chrono::steady_clock::now();
    for (auto epoch = size_t{0}; epoch < kEpochs; ++epoch) {
        const auto epoch_start = std::chrono::steady_clock::now();
        std::ranges::shuffle(order, rng);
        auto loss_sum = float64_t{0};
        auto steps = size_t{0};
        for (auto start = size_t{0}; start + kBatch <= train.count; start += kBatch) {
            auto batch = fill_inputs(train, order, start);
            if (
                const auto filled = batch.has_value() && fill_labels(train, order, start, labels);
                !filled
            ) {
                return 1;
            }

            auto out = net.forward(*batch);
            auto loss = softmax_cross_entropy(out, labels);
            loss.backward(gDType);
            constexpr auto epsilon = (kDType == DType::Float16 || kDType == DType::Float16x2) ? kEpsilonHalf : kEpsilon;
            net.adam_step(kLearningRate, kBeta1, kBeta2, epsilon);
            net.zero_grad();

            if (
                const auto loss_ok = expect_dtype(loss, kDType, "train loss");
                !loss_ok
            ) {
                return 1;
            }
            loss_sum += read_scalar(loss);
            ++steps;
        }
        const auto accuracy = evaluate(net, test);
        if (
            const auto evaluated = accuracy.has_value();
            !evaluated
        ) {
            return 1;
        }
        const auto epoch_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - epoch_start).count();
        std::cout << "epoch " << (epoch + 1) << "/" << kEpochs
                  << "  mean loss " << (loss_sum / static_cast<float64_t>(steps))
                  << "  test accuracy " << (*accuracy * float64_t{100}) << "%"
                  << "  time " << epoch_seconds << " s\n";
    }
    const auto total_seconds = std::chrono::duration<float64_t>(std::chrono::steady_clock::now() - run_start).count();
    std::cout << "total time " << total_seconds << " s\n";
    std::cout << "\nafter training:\n" << net;
    return 0;
}
