/**
 * @file Data/DataLoader.cpp
 * @author GreenBean
 * @brief DataLoader implementation.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Data/DataLoader.h"

#include <algorithm>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Data {

DataLoader::DataLoader(
    const Tensors::Tensor& features,
    const Tensors::Tensor& labels,
    const size_t batch_size,
    const bool shuffle,
    const std::optional<uint64_t> seed,
    const bool drop_last
)
    // const_cast relies on the documented precondition that the caller's
    // tensors are non-const objects; clone(false) needs to register the view.
    : features_(const_cast<Tensors::Tensor&>(features).clone(false))
    , labels_(const_cast<Tensors::Tensor&>(labels).clone(false))
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , drop_last_(drop_last)
    , gen_(seed.value_or(Random::default_generator().draw_seed()))
    , device_(features.get_device())
    , order_tensor_(std::vector<size_t>{size_t{1}}, BeanTensor::DataTypes::DType::UInt64, Tensors::Device::CPU) {
    if (
        features.rank() == 0 || labels.rank() != 1
    ) {
        throw ErrorHandling::MismatchedShapes("features must have rank >= 1 and labels must be rank-1");
    }

    const auto feature_rows = features.dim(0);
    const auto label_rows = labels.dim(0);
    if (feature_rows != label_rows) {
        throw ErrorHandling::MismatchedRowCounts(
            "features dim(0) = " + std::to_string(feature_rows) +
            " vs labels dim(0) = " + std::to_string(label_rows)
        );
    }
    if (feature_rows == 0) {
        throw ErrorHandling::EmptyDataset("features and labels contain zero rows");
    }
    if (batch_size_ == 0) {
        throw ErrorHandling::InvalidBatchSize("batch size must be > 0");
    }

    row_count_ = feature_rows;
    order_.resize(row_count_);
    for (auto i = size_t{0}; i < row_count_; ++i) {
        order_[i] = i;
    }

    reset();
}

void DataLoader::upload_order() {
    // UInt64 order tensor to match main.cpp's index_select idiom.
    order_tensor_ = Tensors::Tensor(std::vector<size_t>{row_count_}, BeanTensor::DataTypes::DType::UInt64, Tensors::Device::CPU);
    for (auto i = size_t{0}; i < row_count_; ++i) {
        order_tensor_.set<uint64_t>({i}, static_cast<uint64_t>(order_[i]));
    }
    if (device_ == Tensors::Device::GPU) {
        order_tensor_.to(device_);
    }
}

void DataLoader::reset() {
    for (auto i = size_t{0}; i < row_count_; ++i) {
        order_[i] = i;
    }
    if (shuffle_) {
        auto eng = std::mt19937_64{gen_.draw_seed()};
        std::ranges::shuffle(order_, eng);
    }
    upload_order();
    position_ = 0;
}

std::optional<std::pair<Tensors::Tensor, Tensors::Tensor>> DataLoader::next_batch() {
    if (position_ >= row_count_) {
        return std::nullopt;
    }

    const auto remaining = row_count_ - position_;
    if (
        remaining < batch_size_ && drop_last_
    ) {
        position_ = row_count_;
        return std::nullopt;
    }

    const auto chunk_size = std::min(batch_size_, remaining);
    auto chunk_indices = order_tensor_.slice({{position_, position_ + chunk_size}});
    auto batch_features = features_.index_select(0, chunk_indices);
    auto batch_labels = labels_.index_select(0, chunk_indices);

    position_ += chunk_size;
    return std::pair{std::move(batch_features), std::move(batch_labels)};
}

DataLoader::Iterator::Iterator(DataLoader& loader)
    : loader_(&loader)
    , current_(loader.next_batch()) {
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return current_.has_value() != other.current_.has_value();
}

std::pair<Tensors::Tensor, Tensors::Tensor>& DataLoader::Iterator::operator*() {
    return *current_;
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    if (loader_ != nullptr) {
        current_ = loader_->next_batch();
    }
    return *this;
}

DataLoader::Iterator DataLoader::begin() {
    return Iterator(*this);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator();
}

} // namespace BeanTensor::Data
