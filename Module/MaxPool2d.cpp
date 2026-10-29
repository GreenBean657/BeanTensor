/**
 * @file MaxPool2d.cpp
 * @author GreenBean
 * @brief MaxPool2d layer implementation: forward delegates to Tensor::max_pool2d.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Module/MaxPool2d.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    MaxPool2d::MaxPool2d(const uint32_t kernel, const std::optional<uint32_t> stride)
        : kernel(kernel), stride(stride.value_or(kernel)) {
    }

    Tensors::Tensor MaxPool2d::forward(const Tensors::Tensor& x) {
        return x.max_pool2d(this->kernel, this->stride);
    }

    std::string MaxPool2d::name() const {
        return "MaxPool2d";
    }
}
