/**
 * @file Metrics.cpp
 * @author GreenBean
 * @brief Implementation of Metrics::accuracy and Metrics::evaluate.
 * @version 0.1
 * @date 2026-10-30
 */
#include "Metrics/Metrics.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "Helpers/Errors.h"
#include "Module/Module.h"

namespace BeanTensor::Metrics {

namespace {

bool is_integer_kind(DataTypes::DType dt) {
    return dt == DataTypes::DType::Int8
        || dt == DataTypes::DType::Int16
        || dt == DataTypes::DType::Int32
        || dt == DataTypes::DType::Int64
        || dt == DataTypes::DType::UInt8
        || dt == DataTypes::DType::UInt16
        || dt == DataTypes::DType::UInt32
        || dt == DataTypes::DType::UInt64
        || dt == DataTypes::DType::Int4x2;
}

struct RequiresGradGuard {
    std::vector<std::pair<Tensors::Tensor*, bool>> saved;

    explicit RequiresGradGuard(Modules::Module& model) {
        for (auto& [name, param] : model.named_parameters()) {
            static_cast<void>(name);
            saved.emplace_back(param, param->requires_gradient());
        }
        for (auto& [param, flag] : saved) {
            param->set_requires_grad(false);
        }
    }

    ~RequiresGradGuard() {
        for (auto& [param, flag] : saved) {
            param->set_requires_grad(flag);
        }
    }
};

} // namespace

/**
 * Per-batch classification accuracy. Computes
 * logits.argmax(1).equal(labels).convert_dtype(Float32).mean(), so the result
 * is a shape-{1} Float32 tensor whose sole element is the fraction of rows
 * whose predicted class matches the label. Ties inherit argmaxs first-index
 * rule.
 * @brief Row-wise top-1 accuracy.
 * @param logits Rank-2 tensor of shape {rows, classes}.
 * @param labels Rank-1 integer-kind tensor of shape {rows}.
 * @return Shape-{1} Float32 tensor holding the accuracy in [0, 1].
 * @throws ErrorHandling::MismatchedShapes logits is not rank-2, labels is not rank-1,
 *         or their row counts differ.
 * @throws ErrorHandling::MismatchedDevices logits and labels live on different devices.
 * @throws ErrorHandling::MismatchedDTypes labels is not an integer kind.
 */
Tensors::Tensor accuracy(const Tensors::Tensor& logits, const Tensors::Tensor& labels) {
    if (logits.rank() != 2) {
        throw ErrorHandling::MismatchedShapes("accuracy expects rank-2 logits");
    }
    if (labels.rank() != 1) {
        throw ErrorHandling::MismatchedShapes("accuracy expects rank-1 labels");
    }
    if (logits.dim(0) != labels.dim(0)) {
        throw ErrorHandling::MismatchedShapes("accuracy: logits and labels row counts differ");
    }
    if (logits.get_device() != labels.get_device()) {
        throw ErrorHandling::MismatchedDevices();
    }
    if (!is_integer_kind(labels.get_dtype())) {
        throw ErrorHandling::MismatchedDTypes();
    }
    auto mask = logits.argmax(1).equal(labels);
    mask.convert_dtype(DataTypes::DType::Float32);
    return mask.mean();
}

/**
 * Evaluate a model over a full DataLoader epoch. The function saves the
 * requires_grad flags of every named parameter, disables gradient tracking
 * during the forward pass, and restores the flags on exit (including when an
 * exception propagates). Accuracy is the fraction of correctly classified rows,
 * and mean_loss is the row-weighted mean of softmax_cross_entropy over every
 * batch. A single host readback happens at the end.
 * @brief Dataset-level accuracy and weighted mean loss.
 * @param model Module to evaluate. Its parameters are toggled and restored.
 * @param loader DataLoader yielding feature/label batches. It is reset at the start.
 * @return EvalResult with accuracy and mean_loss as float64 values.
 * @note This function intentionally does not switch the module into evaluation
 *       mode. Callers must call `model.eval()` beforehand if they want
 *       deterministic behavior from Dropout or BatchNorm.
 * @note The guard only affects parameter `requires_grad` flags. Input batches
 *       that have `requires_grad=true` still participate in the autograd graph
 *       during the forward pass; no backward() is called, so no gradients are
 *       accumulated, but graph nodes are still wired.
 */
EvalResult evaluate(Modules::Module& model, Data::DataLoader& loader) {
    RequiresGradGuard guard(model);
    loader.reset();

    std::optional<Tensors::Tensor> correct_acc;
    std::optional<Tensors::Tensor> loss_acc;
    size_t total_rows = 0;

    while (true) {
        auto batch_opt = loader.next_batch();
        if (!batch_opt.has_value()) {
            break;
        }
        auto& [x, y] = batch_opt.value();

        if (!correct_acc.has_value()) {
            const auto device = x.get_device();
            correct_acc.emplace(std::vector<size_t>{1}, DataTypes::DType::Float32, device);
            correct_acc->set_zero();
            loss_acc.emplace(std::vector<size_t>{1}, DataTypes::DType::Float32, device);
            loss_acc->set_zero();
        }

        auto logits = model.forward(x);
        auto hits_mask = logits.argmax(1).equal(y);
        hits_mask.convert_dtype(DataTypes::DType::Float32);
        auto hits = hits_mask.sum();
        correct_acc = correct_acc->add(hits);

        auto batch_loss = Modules::softmax_cross_entropy(logits, y);
        auto weighted_loss = batch_loss.mul(static_cast<float64_t>(x.dim(0)));
        loss_acc = loss_acc->add(weighted_loss);

        total_rows += x.dim(0);
    }

    if (!correct_acc.has_value()) {
        return EvalResult{float64_t{0.0}, float64_t{0.0}};
    }

    std::vector<Tensors::Tensor> accs;
    accs.reserve(2);
    accs.emplace_back(std::move(*correct_acc));
    accs.emplace_back(std::move(*loss_acc));
    auto combined = Tensors::Tensor::cat(accs, 0);
    combined.convert_dtype(DataTypes::DType::Float64);
    combined.to(Tensors::Device::CPU);
    const auto correct = static_cast<float64_t>(combined.at<float64_t>({0}));
    const auto total_loss = static_cast<float64_t>(combined.at<float64_t>({1}));
    return EvalResult{
        correct / static_cast<float64_t>(total_rows),
        total_loss / static_cast<float64_t>(total_rows)
    };
}

} // namespace BeanTensor::Metrics
