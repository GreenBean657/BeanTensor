/**
 * @file Metrics.h
 * @author GreenBean
 * @brief Classification metrics and dataset evaluation: per-batch accuracy and
 * a weighted evaluate() entry point that toggles requires_grad off for the
 * duration and restores it on exit.
 * @version 0.1
 * @date 2026-10-30
 */
#pragma once

#include <cstdint>

#include "Data/DataLoader.h"
#include "Module/Module.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Metrics {

/**
 * Aggregate result returned by evaluate(). Both fields are host-side float64
 * values in the range implied by the dataset.
 * @brief Accuracy and mean loss from a dataset evaluation pass.
 */
struct EvalResult {
    float64_t accuracy;
    float64_t mean_loss;
};

/**
 * Per-batch classification accuracy. Computes
 * logits.argmax(1).equal(labels).convert_dtype(Float32).mean(), so the result
 * is a shape-{1} Float32 tensor whose sole element is the fraction of rows
 * whose predicted class matches the label. Ties inherit argmaxs first-index
 * rule.
 * @brief Row-wise top-1 accuracy.
 * @throws ErrorHandling::MismatchedShapes logits is not rank-2, labels is not rank-1,
 *         or their row counts differ.
 * @throws ErrorHandling::MismatchedDevices logits and labels live on different devices.
 * @throws ErrorHandling::MismatchedDTypes labels is not an integer kind.
 */
[[nodiscard]] Tensors::Tensor accuracy(const Tensors::Tensor&, const Tensors::Tensor&);

/**
 * Evaluate a model over a full DataLoader epoch. The function saves the
 * requires_grad flags of every named parameter, disables gradient tracking
 * during the forward pass, and restores the flags on exit (including when an
 * exception propagates). It does NOT change the module's train/eval mode;
 * stochastic layers such as Dropout remain stochastic if the caller leaves the
 * module in training mode. Accuracy is the fraction of correctly classified
 * rows, and mean_loss is the row-weighted mean of softmax_cross_entropy over
 * every batch. A single host readback happens at the end.
 * @brief Dataset-level accuracy and weighted mean loss.
 * @throws ErrorHandling::* The underlying forward/loss ops may raise any
 *         BeanTensor error (e.g. mismatched devices or dtypes), in which case
 *         the requires_grad guard still restores the original flags.
 */
[[nodiscard]] EvalResult evaluate(Modules::Module&, Data::DataLoader&);

} // namespace BeanTensor::Metrics
