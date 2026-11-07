/**
 * @file Mnist.h
 * @author GreenBean
 * @brief Shared MNIST IDX loading for the demo programs: raw split parsing
 * (magic/count/header validation) and one-shot device upload of a split as
 * either flat rank-2 {N, 784} rows (the MLP demo) or NCHW rank-4 {N, 1, 28,
 * 28} images (the CNN demo), plus the host scalar readback and dtype-gate
 * helpers both demos use. Pixels are scaled to [0, 1]; labels are Int64
 * class indices. Lives under demos/ because only demos consume it — the
 * library itself has no dataset reader (CSV loading is the planned general
 * path).
 * @version 0.1
 * @date 2026-11-06
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"

namespace demos::mnist {

    using BeanTensor::Tensors::Device;
    using BeanTensor::Tensors::Tensor;
    using DType = BeanTensor::DataTypes::DType;
    using BeanTensor::float32_t;
    using BeanTensor::float64_t;
    // int64_t/uint32_t/uint8_t below are the standard <cstdint> spellings,
    // not BeanTensor aliases.

    constexpr auto kDir = "datasets/mnist";
    constexpr auto kTrainImages = "train-images-idx3-ubyte";
    constexpr auto kTrainLabels = "train-labels-idx1-ubyte";
    constexpr auto kTestImages = "t10k-images-idx3-ubyte";
    constexpr auto kTestLabels = "t10k-labels-idx1-ubyte";

    constexpr uint32_t kImageMagic = 0x00000803;
    constexpr uint32_t kLabelMagic = 0x00000801;
    constexpr uint32_t kSide = 28;
    constexpr size_t kPixels = static_cast<size_t>(kSide) * kSide;
    constexpr size_t kImageHeaderBytes = 16;
    constexpr size_t kLabelHeaderBytes = 8;

    // Raw MNIST split as read from disk: pixels stay as bytes until an
    // upload_* call decodes the whole split into device tensors once.
    struct Split {
        size_t count = 0;
        std::vector<uint8_t> images;
        std::vector<uint8_t> labels;
    };

    // One split decoded and resident on the run's device.
    struct DeviceSplit {
        size_t count = 0;
        Tensor images;
        Tensor labels;
    };

    struct FileCloser {
        void operator()(std::FILE* file) const {
            std::fclose(file);
        }
    };
    using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

    [[nodiscard]] inline uint32_t read_be32(const std::vector<uint8_t>& bytes, const size_t offset) {
        return (static_cast<uint32_t>(bytes[offset]) << 24)
             | (static_cast<uint32_t>(bytes[offset + 1]) << 16)
             | (static_cast<uint32_t>(bytes[offset + 2]) << 8)
             | static_cast<uint32_t>(bytes[offset + 3]);
    }

    [[nodiscard]] inline bool read_file(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
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

    /**
     * Parse and validate one IDX image+label pair from @p dir.
     * @brief Read one MNIST split from disk.
     * @param dir Directory holding the uncompressed IDX files.
     * @param image_file Image filename within @p dir.
     * @param label_file Label filename within @p dir.
     * @param split Out: the raw split.
     * @return True when both files parsed and their counts agree.
     */
    [[nodiscard]] inline bool load_split(const std::filesystem::path& dir, const std::string_view image_file,
                                         const std::string_view label_file, Split& split) {
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
                && read_be32(image_bytes, 8) == kSide
                && read_be32(image_bytes, 12) == kSide;
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
                && image_bytes.size() == kImageHeaderBytes + count * kPixels
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

    // Per-write dtype gate: Tensor::set/at stride by sizeof(T) and do not
    // check T, so a tensor of the wrong dtype would be corrupted silently.
    [[nodiscard]] inline bool expect_dtype(const Tensor& t, const DType expected, const std::string_view name) {
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

    // Convert a host-staged tensor to the model dtype.
    [[nodiscard]] inline Tensor to_model_dtype(Tensor staged, const DType model_dtype) {
        if (staged.get_dtype() != model_dtype) {
            staged.convert_dtype(model_dtype);
        }
        staged.sync();
        return staged;
    }

    /**
     * Decode a whole split once, outside any timed region: pixels scaled to
     * [0, 1] as rank-2 {count, 784} rows in @p model_dtype, labels as Int64,
     * moved to @p device.
     * @brief Upload a split as flat rows for the MLP demo.
     */
    [[nodiscard]] inline std::optional<DeviceSplit> upload_flat(const Split& split, const Device device,
                                                                const DType model_dtype, const DType host_dtype) {
        // Stage the whole split in one host buffer and write it with a
        // single bulk write_element: per-pixel set() would sync per element
        // (47M syncs for MNIST) and dominate the run before training starts.
        auto staged = Tensor(std::vector<size_t>{split.count, kPixels}, host_dtype);
        switch (host_dtype) {
            case DType::Float32: {
                auto buf = std::vector<float32_t>(split.count * kPixels);
                for (auto i = size_t{0}; i < split.count * kPixels; ++i) {
                    buf[i] = static_cast<float32_t>(split.images[i]) / 255.0f;
                }
                staged.write_element(0, buf.data(), buf.size() * sizeof(float32_t));
                break;
            }
            case DType::Float64: {
                auto buf = std::vector<float64_t>(split.count * kPixels);
                for (auto i = size_t{0}; i < split.count * kPixels; ++i) {
                    buf[i] = static_cast<float64_t>(split.images[i]) / 255.0;
                }
                staged.write_element(0, buf.data(), buf.size() * sizeof(float64_t));
                break;
            }
            default: {
                std::cerr << "upload_flat: unsupported host dtype\n";
                return std::nullopt;
            }
        }
        auto images = to_model_dtype(std::move(staged), model_dtype);
        if (
            const auto images_ok = expect_dtype(images, model_dtype, "upload_flat: images");
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
        return DeviceSplit{split.count, std::move(images), std::move(labels)};
    }

    /**
     * Decode a whole split once as NCHW rank-4 {count, 1, 28, 28} images in
     * @p model_dtype (pixel (r, c) of image i at [i, 0, r, c]); labels Int64.
     * @brief Upload a split as NCHW images for the CNN demo.
     */
    [[nodiscard]] inline std::optional<DeviceSplit> upload_nchw(const Split& split, const Device device,
                                                                const DType model_dtype, const DType host_dtype) {
        // NCHW with C=1 is pixel-order contiguous, so the flat IDX payload
        // maps 1:1 — one bulk write, no per-pixel set()/sync.
        auto staged = Tensor(std::vector<size_t>{split.count, 1, kSide, kSide}, host_dtype);
        switch (host_dtype) {
            case DType::Float32: {
                auto buf = std::vector<float32_t>(split.count * kPixels);
                for (auto i = size_t{0}; i < split.count * kPixels; ++i) {
                    buf[i] = static_cast<float32_t>(split.images[i]) / 255.0f;
                }
                staged.write_element(0, buf.data(), buf.size() * sizeof(float32_t));
                break;
            }
            case DType::Float64: {
                auto buf = std::vector<float64_t>(split.count * kPixels);
                for (auto i = size_t{0}; i < split.count * kPixels; ++i) {
                    buf[i] = static_cast<float64_t>(split.images[i]) / 255.0;
                }
                staged.write_element(0, buf.data(), buf.size() * sizeof(float64_t));
                break;
            }
            default: {
                std::cerr << "upload_nchw: unsupported host dtype\n";
                return std::nullopt;
            }
        }
        auto images = to_model_dtype(std::move(staged), model_dtype);
        if (
            const auto images_ok = expect_dtype(images, model_dtype, "upload_nchw: images");
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
        return DeviceSplit{split.count, std::move(images), std::move(labels)};
    }

    // Read a one-element tensor as a float64 value. Works on a copy,
    // converted to the host dtype first, so packed storage is never read.
    [[nodiscard]] inline float64_t read_scalar(Tensor& t, const DType host_dtype) {
        auto host = t.clone();
        host.convert_dtype(host_dtype);
        host.sync();
        switch (host_dtype) {
            case DType::Float32:
                return static_cast<float64_t>(host.at<float32_t>({0}));
            case DType::Float64:
                return static_cast<float64_t>(host.at<float64_t>({0}));
            default:
                std::cerr << "read_scalar: unsupported host dtype\n";
                return float64_t{0};
        }
    }
}
