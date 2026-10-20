/**
 * @file Reshape.cpp
 * @author GreenBean
 * @brief Metadata-only reshape: Tensor::reshape builds a same-buffer view.
 * @version 0.1
 * @date 2026-10-20
 */
#include <string>
#include <vector>

#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {

    /**
     * Metadata-only: the result is a view over the same buffer, requires contiguous storage.
     * @brief Reshape into a view with a new shape of equal element count.
     * @param new_shape Target shape, 1..BT_TENSOR_MAX_DIMS dimensions.
     * @return View tensor sharing this tensor's buffer.
     */
    Tensor Tensor::reshape(const std::vector<size_t>& new_shape) {
        sync();
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (new_shape.empty() || new_shape.size() > BT_TENSOR_MAX_DIMS) {
            throw ErrorHandling::MismatchedShapes("reshape rank must be 1.." + std::to_string(BT_TENSOR_MAX_DIMS));
        }
        auto new_numel = size_t{1};
        for (const auto d : new_shape) {
            new_numel *= d;
        }
        if (new_numel != this->numel) {
            throw ErrorHandling::MismatchedShapes("reshape changes numel (" + std::to_string(this->numel) + " vs " + std::to_string(new_numel) + ")");
        }
        // Build a same-buffer view. The constructor allocates a fresh buffer;
        // release it while owns_data is still true, then re-point at ours.
        auto view = Tensor(new_shape, this->dtype, this->device, this->requires_grad);
        view.kill_cpu_data();
        view.kill_gpu_data();
        view.owns_data = false;
        view.data = this->data;
        this->children.emplace_back(&view);
        view.view_parents.emplace_back(this);
        return view;
    }
}
