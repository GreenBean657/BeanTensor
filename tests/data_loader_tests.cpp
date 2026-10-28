// BeanTensorV5 DataLoader tests — framework-free assertion suite.
// Covers: every-sample-once across two epochs (shuffle on/off), same-seed
// determinism, omitted-seed/set_seed equivalence, partial final batch,
// drop_last, zero batches when count < batch + drop_last, ctor error paths,
// and CPU/GPU batch parity when a device is available.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "Data/DataLoader.h"
#include "Helpers/Errors.h"
#include "Random/Generator.h"
#include "Tensor/Tensor.h"

using BeanTensor::Data::DataLoader;
using BeanTensor::Tensors::Device;
using BeanTensor::Tensors::Tensor;
using BeanTensor::float32_t;
using BeanTensor::float64_t;
using DType = BeanTensor::DataTypes::DType;

static int32_t failures = 0;
#define CHECK(...) do { \
    if (!(__VA_ARGS__)) { ++failures; std::cerr << "FAIL: " << #__VA_ARGS__ << " @ line " << __LINE__ << "\n"; } \
} while (0)

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

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

// Build a tiny dataset where each row's single feature and integer label equal
// the row index, so batches can be identified without host↔device copies of
// the feature tensor.
static std::pair<Tensor, Tensor> make_index_dataset(const size_t rows, const Device dev) {
    auto features = Tensor({rows, size_t{1}}, DType::Float32, dev, false);
    auto labels = Tensor({rows}, DType::Int64, dev, false);
    for (auto i = size_t{0}; i < rows; ++i) {
        features.set<float32_t>({i, 0}, static_cast<float32_t>(static_cast<float64_t>(i)));
        labels.set<int64_t>({i}, static_cast<int64_t>(i));
    }
    return {std::move(features), std::move(labels)};
}

// Read a rank-1 Int64 tensor back to host as row indices.
static std::vector<size_t> read_label_indices(Tensor& labels) {
    labels.sync();
    auto out = std::vector<size_t>(labels.element_count());
    for (auto i = size_t{0}; i < out.size(); ++i) {
        out[i] = static_cast<size_t>(labels.at<int64_t>({i}));
    }
    return out;
}

// Collect every batch from one epoch into a flat list of selected row indices.
static std::vector<size_t> collect_epoch(DataLoader& loader) {
    loader.reset();
    auto selected = std::vector<size_t>{};
    while (const auto batch = loader.next_batch()) {
        const auto& labels = batch->second;
        const auto chunk = read_label_indices(const_cast<Tensor&>(labels));
        selected.insert(selected.end(), chunk.begin(), chunk.end());
    }
    return selected;
}

// =====================================================================
// Epoch coverage and ordering
// =====================================================================

static void test_shuffle_false_two_epochs() {
    const auto rows = size_t{11};
    const auto batch_size = size_t{3};
    auto [features, labels] = make_index_dataset(rows, Device::CPU);
    auto loader = DataLoader(features, labels, batch_size, false, uint64_t{1}, false);

    for (auto epoch = size_t{0}; epoch < 2; ++epoch) {
        const auto selected = collect_epoch(loader);
        CHECK(selected.size() == rows);
        for (auto i = size_t{0}; i < rows; ++i) {
            CHECK(selected[i] == i);
        }
    }
}

static void test_shuffle_true_every_sample_once() {
    const auto rows = size_t{13};
    const auto batch_size = size_t{4};
    auto [features, labels] = make_index_dataset(rows, Device::CPU);
    auto loader = DataLoader(features, labels, batch_size, true, uint64_t{123}, false);

    for (auto epoch = size_t{0}; epoch < 2; ++epoch) {
        const auto selected = collect_epoch(loader);
        CHECK(selected.size() == rows);
        auto seen = std::set<size_t>(selected.begin(), selected.end());
        CHECK(seen.size() == rows);
        for (auto i = size_t{0}; i < rows; ++i) {
            CHECK(seen.count(i) == 1);
        }
    }
}

// =====================================================================
// Determinism and seed contracts
// =====================================================================

static void test_same_seed_identical_sequences() {
    const auto rows = size_t{20};
    const auto batch_size = size_t{5};
    auto [fa, la] = make_index_dataset(rows, Device::CPU);
    auto [fb, lb] = make_index_dataset(rows, Device::CPU);
    auto loader_a = DataLoader(fa, la, batch_size, true, uint64_t{999}, false);
    auto loader_b = DataLoader(fb, lb, batch_size, true, uint64_t{999}, false);

    const auto epoch_a = collect_epoch(loader_a);
    const auto epoch_b = collect_epoch(loader_b);
    CHECK(epoch_a == epoch_b);
}

static void test_omitted_seed_set_seed_equivalence() {
    const auto rows = size_t{15};
    const auto batch_size = size_t{4};
    auto [fa, la] = make_index_dataset(rows, Device::CPU);
    auto [fb, lb] = make_index_dataset(rows, Device::CPU);

    BeanTensor::Random::set_seed(20261028);
    auto loader_a = DataLoader(fa, la, batch_size, true, std::nullopt, false);
    const auto epoch_a = collect_epoch(loader_a);

    BeanTensor::Random::set_seed(20261028);
    auto loader_b = DataLoader(fb, lb, batch_size, true, std::nullopt, false);
    const auto epoch_b = collect_epoch(loader_b);

    CHECK(epoch_a == epoch_b);
}

// =====================================================================
// Batch sizing and drop_last
// =====================================================================

static void test_partial_batch() {
    const auto rows = size_t{5};
    const auto batch_size = size_t{2};
    auto [features, labels] = make_index_dataset(rows, Device::CPU);
    auto loader = DataLoader(features, labels, batch_size, false, uint64_t{1}, false);

    loader.reset();
    auto b1 = loader.next_batch();
    auto b2 = loader.next_batch();
    auto b3 = loader.next_batch();
    auto b4 = loader.next_batch();
    CHECK(b1.has_value());
    CHECK(b2.has_value());
    CHECK(b3.has_value());
    CHECK(!b4.has_value());
    CHECK(b1->second.element_count() == batch_size);
    CHECK(b2->second.element_count() == batch_size);
    CHECK(b3->second.element_count() == size_t{1});

    const auto tail = read_label_indices(b3->second);
    CHECK(tail.size() == 1);
    CHECK(tail[0] == size_t{4});
}

static void test_drop_last() {
    const auto rows = size_t{5};
    const auto batch_size = size_t{2};
    auto [features, labels] = make_index_dataset(rows, Device::CPU);
    auto loader = DataLoader(features, labels, batch_size, false, uint64_t{1}, true);

    loader.reset();
    const auto b1 = loader.next_batch();
    const auto b2 = loader.next_batch();
    const auto b3 = loader.next_batch();
    CHECK(b1.has_value());
    CHECK(b2.has_value());
    CHECK(!b3.has_value());
    CHECK(b1->second.element_count() == batch_size);
    CHECK(b2->second.element_count() == batch_size);
}

static void test_drop_last_zero_batches() {
    const auto rows = size_t{2};
    const auto batch_size = size_t{5};
    auto [features, labels] = make_index_dataset(rows, Device::CPU);
    auto loader = DataLoader(features, labels, batch_size, false, std::nullopt, true);

    const auto selected = collect_epoch(loader);
    CHECK(selected.empty());
    CHECK(!loader.next_batch().has_value());
    CHECK(!loader.next_batch().has_value()); // repeated calls stay nullopt
}

// =====================================================================
// Ctor error paths
// =====================================================================

static void test_ctor_errors() {
    auto good_features = Tensor({size_t{4}, size_t{1}}, DType::Float32, Device::CPU, false);
    auto good_labels = Tensor({size_t{4}}, DType::Int64, Device::CPU, false);
    good_features.set_zero();
    good_labels.set_zero();

    auto empty_features = Tensor({size_t{0}, size_t{1}}, DType::Float32, Device::CPU, false);
    auto empty_labels = Tensor({size_t{0}}, DType::Int64, Device::CPU, false);
    empty_features.set_zero();
    empty_labels.set_zero();

    auto mismatched_labels = Tensor({size_t{3}}, DType::Int64, Device::CPU, false);
    mismatched_labels.set_zero();

    auto threw = false;

    threw = false;
    try {
        auto bad = DataLoader(empty_features, empty_labels, size_t{2}, false, uint64_t{1}, false);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::EmptyDataset&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        auto bad = DataLoader(good_features, mismatched_labels, size_t{2}, false, uint64_t{1}, false);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::MismatchedRowCounts&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        auto bad = DataLoader(good_features, good_labels, size_t{0}, false, uint64_t{1}, false);
        static_cast<void>(bad);
    } catch (const BeanTensor::ErrorHandling::InvalidBatchSize&) {
        threw = true;
    }
    CHECK(threw);
}

// =====================================================================
// Range-for convenience
// =====================================================================

static void test_range_for() {
    const auto rows = size_t{6};
    const auto batch_size = size_t{2};
    auto [features, labels] = make_index_dataset(rows, Device::CPU);
    auto loader = DataLoader(features, labels, batch_size, false, uint64_t{1}, false);

    loader.reset();
    auto count = size_t{0};
    for (const auto& [batch_features, batch_labels] : loader) {
        static_cast<void>(batch_features);
        CHECK(batch_labels.element_count() == batch_size);
        ++count;
    }
    CHECK(count == rows / batch_size);
}

// =====================================================================
// GPU parity
// =====================================================================

#if defined(USE_CUDA) || defined(USE_HIP)
static void test_gpu_parity() {
    if (!gpu_available()) {
        std::cout << "SKIP: no GPU device available\n";
        return;
    }

    const auto rows = size_t{17};
    const auto batch_size = size_t{5};
    auto [cpu_features, cpu_labels] = make_index_dataset(rows, Device::CPU);
    auto cpu_loader = DataLoader(cpu_features, cpu_labels, batch_size, true, uint64_t{555}, false);
    const auto cpu_epoch = collect_epoch(cpu_loader);

    auto [gpu_features, gpu_labels] = make_index_dataset(rows, Device::CPU);
    gpu_features.to(Device::GPU);
    gpu_labels.to(Device::GPU);
    auto gpu_loader = DataLoader(gpu_features, gpu_labels, batch_size, true, uint64_t{555}, false);
    const auto gpu_epoch = collect_epoch(gpu_loader);

    CHECK(cpu_epoch == gpu_epoch);
}
#endif

int main() {
    test_shuffle_false_two_epochs();
    test_shuffle_true_every_sample_once();
    test_same_seed_identical_sequences();
    test_omitted_seed_set_seed_equivalence();
    test_partial_batch();
    test_drop_last();
    test_drop_last_zero_batches();
    test_ctor_errors();
    test_range_for();
#if defined(USE_CUDA) || defined(USE_HIP)
    test_gpu_parity();
#endif

    if (failures == 0) {
        std::cout << "ALL DATALOADER TESTS PASSED\n";
        return 0;
    }
    std::cerr << failures << " FAILURE(S)\n";
    return 1;
}
