/**
 * @file main.cpp
 * @author GreenBean
 * @brief MNIST demo over the Module layer stack: reads the uncompressed IDX
 * files from data/mnist/, prints live thread-pool telemetry (snapshot + format) on the first
 * batch of each epoch, trains a Sequential MLP (Linear -> Relu -> Linear)
 * with softmax_cross_entropy against class labels and Module::adam_step, and
 * reports test accuracy after every epoch. Every tensor access is gated on
 * its dtype once per batch; a mismatch prints to std::cerr and exits 1.
 * Each split is staged once in Float32 on the host, converted to the model
 * dtype with Tensor::convert_dtype (so packed dtypes like Float16x2 never
 * need a host-side scalar type), and moved to the run's device before the
 * timer starts. Per step, batches are gathered on that device with
 * index_select through a shuffled index tensor uploaded once per epoch, so
 * no per-batch host staging or host<->device transfer is timed.
 *
 * The demo runs the same model twice: once on the CPU, then once on the GPU
 * (when the build includes a GPU backend). Both passes use identical
 * hyperparameters and report the same metrics.
 * @version 0.5.0
 * @date 2026-10-26
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

#include "Helpers/CPUThreading.h"
#include "Helpers/PoolMonitor.h"
#include "Module/Linear.h"
#include "Module/Module.h"
#include "Module/ModuleStream.h"
#include "Module/Relu.h"
#include "Module/Sequential.h"
#include "Tensor/Tensor.h"

using BeanTensor::Modules::Linear;
using BeanTensor::Modules::Module;
using BeanTensor::Modules::Relu;
using BeanTensor::Modules::Sequential;
using BeanTensor::Modules::softmax_cross_entropy;
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

    // Raw MNIST split as read from disk: pixels stay as bytes until upload_split
    // decodes the whole split into a device tensor once per run.
    struct MnistSplit {
        size_t count = 0;
        std::vector<uint8_t> images;
        std::vector<uint8_t> labels;
    };

    // One split decoded and resident on the run's device: images in kDType
    // ({count, kInputs} before packing), labels as Int64 class indices, and
    // identity row indices for the in-order evaluation gather.
    struct DeviceSplit {
        size_t count = 0;
        Tensor images;
        Tensor labels;
        Tensor identity;
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

    // Row indices as a rank-1 UInt64 tensor on device, the index operand
    // index_select expects. Called once per epoch for the shuffled order.
    [[nodiscard]] Tensor upload_order(const std::vector<size_t>& order, const Device device) {
        auto indices = Tensor(std::vector<size_t>{order.size()}, DType::UInt64);
        for (auto i = size_t{0}; i < order.size(); ++i) {
            indices.set<uint64_t>({i}, static_cast<uint64_t>(order[i]));
        }
        if (device == Device::GPU) {
            indices.to(Device::GPU);
        }
        return indices;
    }

    [[nodiscard]] std::vector<size_t> identity_order(const size_t count) {
        auto order = std::vector<size_t>(count);
        for (auto i = size_t{0}; i < count; ++i) {
            order[i] = i;
        }
        return order;
    }

    // Decode a whole split once, outside the timed region: pixels staged as
    // Host values scaled to [0, 1] and converted to kDType, labels as Int64,
    // everything moved to device. Returns nullopt if a dtype gate fails.
    [[nodiscard]] std::optional<DeviceSplit> upload_split(const MnistSplit& split, const Device device) {
        auto staged = Tensor(std::vector<size_t>{split.count, kInputs}, kHostDType);
        for (auto row = size_t{0}; row < split.count; ++row) {
            for (auto col = size_t{0}; col < kInputs; ++col) {
                const auto pixel = static_cast<Host>(split.images[row * kInputs + col]);
                staged.set<Host>({row, col}, pixel / Host{255});
            }
        }
        auto images = to_model_dtype(std::move(staged));
        if (
            const auto images_ok = expect_dtype(images, kDType, "upload_split: images");
            !images_ok
        ) {
            return std::nullopt;
        }
        auto labels = Tensor(std::vector<size_t>{split.count}, DType::Int64);
        for (auto row = size_t{0}; row < split.count; ++row) {
            labels.set<int64_t>({row}, static_cast<int64_t>(split.labels[row]));
        }
        if (device == Device::GPU) {
            images.to(Device::GPU);
            labels.to(Device::GPU);
        }
        return DeviceSplit{split.count, std::move(images), std::move(labels), upload_order(identity_order(split.count), device)};
    }

    // Read a one-element tensor as a float64 value. Works on a copy, converted
    // to the host dtype first, so packed storage is never read directly.
    [[nodiscard]] float64_t read_scalar(Tensor& t) {
        auto host = t.clone();
        host.convert_dtype(kHostDType);
        host.sync();
        return static_cast<float64_t>(host.at<Host>({0}));
    }

    // Fraction of samples whose argmax prediction matches the label. Forward
    // only: no backward() follows, so nothing is trained on this pass. Hits
    // are counted on the device and read back once.
    // Returns nullopt if a tensor has the wrong dtype.
    [[nodiscard]] std::optional<float64_t> evaluate(Sequential& net, DeviceSplit& split, const Device device) {
        auto correct = Tensor(std::vector<size_t>{1}, DType::Int64, device);
        correct.set_zero();
        auto seen = size_t{0};
        for (auto start = size_t{0}; start + kBatch <= split.count; start += kBatch) {
            auto batch_indices = split.identity.slice({{start, start + kBatch}});
            auto batch = split.images.index_select(0, batch_indices);
            auto labels = split.labels.slice({{start, start + kBatch}});
            auto logits = net.forward(batch);
            logits.sync();
            // Packed logits need the lane-aware argmax: each element is two classes.
            auto predictions = kPacked ? logits.argmax_lanes(1) : logits.argmax(1);
            predictions.sync();
            if (
                const auto predictions_ok = expect_dtype(predictions, DType::Int64, "evaluate: predictions");
                !predictions_ok
            ) {
                return std::nullopt;
            }
            auto hits = predictions.equal(labels);
            hits.convert_dtype(DType::Int64);
            const auto batch_hits = hits.sum();
            auto updated = correct.add(batch_hits);
            updated.sync();
            correct = std::move(updated);
            seen += kBatch;
        }
        if (device == Device::GPU) {
            correct.to(Device::CPU);
        }
        correct.sync();
        return static_cast<float64_t>(correct.at<int64_t>({0})) / static_cast<float64_t>(seen);
    }

    // Train and evaluate the same MLP on the requested device. Both splits are
    // decoded and moved to the device before the timer starts; each step
    // gathers its batch and labels on the device through the epoch's shuffled
    // index tensor. The model and its optimizer state live on the device.
    [[nodiscard]] bool run_training(const MnistSplit& train, const MnistSplit& test, const Device device) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Device: " << BeanTensor::Tensors::stringify_device_source(device) << "\n";
        std::cout << std::string(60, '=') << "\n";

        auto net = Sequential();
        net.append("fc1", std::make_unique<Linear>(kInputs, kHidden, kDType, device));
        net.append("relu", std::make_unique<Relu>());
        net.append("fc2", std::make_unique<Linear>(kHidden, kClasses, kDType, device));
        std::cout << net;

        auto train_data = upload_split(train, device);
        auto test_data = upload_split(test, device);
        if (
            const auto uploaded = train_data.has_value() && test_data.has_value();
            !uploaded
        ) {
            return false;
        }

        auto rng = std::mt19937{kShuffleSeed};
        auto order = identity_order(train.count);
        const auto run_start = std::chrono::steady_clock::now();
        for (auto epoch = size_t{0}; epoch < kEpochs; ++epoch) {
            const auto epoch_start = std::chrono::steady_clock::now();
            std::ranges::shuffle(order, rng);
            auto indices = upload_order(order, device);
            auto loss_sum = float64_t{0};
            auto steps = size_t{0};
            for (auto start = size_t{0}; start + kBatch <= train.count; start += kBatch) {
                auto batch_indices = indices.slice({{start, start + kBatch}});
                auto batch = train_data->images.index_select(0, batch_indices);
                if (
                    const auto batch_ok = expect_dtype(batch, kDType, "train: input batch");
                    !batch_ok
                ) {
                    return false;
                }
                auto labels = train_data->labels.index_select(0, batch_indices);

                auto out = net.forward(batch);
                auto loss = softmax_cross_entropy(out, labels);
                loss.backward(gDType);
                constexpr auto epsilon = (kDType == DType::Float16 || kDType == DType::Float16x2) ? kEpsilonHalf : kEpsilon;
                net.adam_step(kLearningRate, kBeta1, kBeta2, epsilon);
                net.zero_grad();

                if (
                    const auto loss_ok = expect_dtype(loss, kDType, "train loss");
                    !loss_ok
                ) {
                    return false;
                }
                loss_sum += read_scalar(loss);
                ++steps;
            }
            // Drain any pending optimizer/gradient work before running the
            // synchronous test-only forward pass.
            for (auto& [name, param] : net.named_parameters()) {
                (void)name;
                param->sync();
            }
            const auto accuracy = evaluate(net, *test_data, device);
            if (
                const auto evaluated = accuracy.has_value();
                !evaluated
            ) {
                return false;
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
        return true;
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
