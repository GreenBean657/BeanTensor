/**
 * @file Flatten.cpp
 * @author GreenBean
 * @brief Metadata-only flatten: Tensor::flatten builds a rank-2 view and wires
 * a FlattenG autograd node.
 * @version 0.1
 * @date 2026-10-28
 */
#include <vector>

#include "Math/Autograd/FlattenG/FlattenG.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {

    /**
     * Metadata-only rank-2 view: {shape[0], numel / shape[0]}. Contiguous-only,
     * sync-free, and wires FlattenG so backward reshapes dy back to x.shape().
     * @brief Flatten to a rank-2 view.
     * @return Rank-2 tensor viewing this tensor's data.
     * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
     */
    Tensor Tensor::flatten() const {
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (this->ndim == 0) {
            throw ErrorHandling::NotImplemented();
        }
        auto new_shape = std::vector<size_t>{this->shape[0], this->numel / this->shape[0]};
        auto view = const_cast<Tensor*>(this)->reshape(new_shape);
        view.wire_node(detail::flatten_g_bridge, *this);
        return view;
    }
}
