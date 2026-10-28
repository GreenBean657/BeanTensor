/**
 * @file Data/DataLoader.h
 * @author GreenBean
 * @brief Shuffling batch DataLoader over a feature/label tensor pair.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "Random/Generator.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Data {

/**
 * @class DataLoader
 * @brief Iterable batch producer over a contiguous dataset stored as tensors.
 *
 * The loader holds view handles (created with @c clone(false)) that share the
 * buffers of the supplied @p features and @p labels tensors. The caller's
 * tensors must outlive the DataLoader and any batches yielded from it;
 * destroying them early leaves the loader holding dangling views. Batches are
 * produced by index_select(0, chunk) on the dataset's device, so no host-side
 * data copies occur during iteration.
 *
 * Construction validates ranks, row counts, and batch size. @p reset() begins
 * an epoch: identity order, optionally shuffled with a seed drawn from the
 * loader's Generator. @p next_batch() returns the next (features, labels) pair
 * or nullopt at epoch end; repeated calls after the end continue to return
 * nullopt. When @p drop_last is true, a ragged final chunk is discarded.
 */
class DataLoader {
public:
    /**
     * @brief Construct a DataLoader.
     * @param features Rank >= 1 tensor; dim(0) is the dataset size.
     * @param labels Rank-1 tensor with one label per row.
     * @param batch_size Number of rows per batch; must be > 0.
     * @param shuffle If true, shuffle the row order each reset().
     * @param seed Explicit seed for the internal Generator, or nullopt to draw
     *             one seed from the process-wide default generator.
     * @param drop_last If true, discard the final ragged batch.
     * @pre The referenced @p features and @p labels tensors must be non-const
     *      objects. The loader creates view handles via @c Tensor::clone(false),
     *      which registers bookkeeping in the source tensor; passing a genuinely
     *      const object (e.g. through @c std::as_const) is unsupported and
     *      results in undefined behavior.
     * @throws ErrorHandling::EmptyDataset if the row count is zero.
     * @throws ErrorHandling::MismatchedShapes if features.rank() < 1 or labels.rank() != 1.
     * @throws ErrorHandling::MismatchedRowCounts if labels.dim(0) != features.dim(0).
     * @throws ErrorHandling::InvalidBatchSize if batch_size == 0.
     */
    DataLoader(
        const Tensors::Tensor& features,
        const Tensors::Tensor& labels,
        size_t batch_size,
        bool shuffle = true,
        std::optional<uint64_t> seed = std::nullopt,
        bool drop_last = false
    );

    /**
     * @brief Begin a new epoch.
     *
     * Resets the row counter to zero and rebuilds the order vector. If shuffle
     * is enabled, the order is shuffled once with std::ranges::shuffle using a
     * seed drawn from the internal Generator. The order is then uploaded to the
     * dataset's device as a rank-1 UInt64 tensor.
     */
    void reset();

    /**
     * @brief Return the next batch, or nullopt at epoch end.
     * @return A pair of (feature batch, label batch) tensors on the dataset's
     *         device, or std::nullopt when the epoch is exhausted.
     */
    [[nodiscard]] std::optional<std::pair<Tensors::Tensor, Tensors::Tensor>> next_batch();

    /**
     * @brief Input iterator for range-for over epochs.
     *
     * The iterator calls next_batch() on each increment and compares against an
     * end sentinel. The caller must call reset() before iterating if a fresh
     * epoch is desired.
     */
    class Iterator {
    public:
        Iterator() = default;

        bool operator!=(const Iterator& other) const;
        std::pair<Tensors::Tensor, Tensors::Tensor>& operator*();
        Iterator& operator++();

    private:
        explicit Iterator(DataLoader& loader);

        DataLoader* loader_ = nullptr;
        std::optional<std::pair<Tensors::Tensor, Tensors::Tensor>> current_ = std::nullopt;

        friend class DataLoader;
    };

    Iterator begin();
    Iterator end();

private:
    Tensors::Tensor features_;
    Tensors::Tensor labels_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    Random::Generator gen_;
    Tensors::Device device_;
    size_t row_count_ = 0;
    size_t position_ = 0;
    std::vector<size_t> order_;
    Tensors::Tensor order_tensor_;

    void upload_order();
};

} // namespace BeanTensor::Data
