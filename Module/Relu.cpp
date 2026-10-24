/**
 * @file Relu.cpp
 * @author GreenBean
 * @brief ReLU layer implementation: forward delegates to Tensor::relu.
 * @version 0.1
 * @date 2026-10-23
 */
#include "Module/Relu.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Relu::forward(const Tensors::Tensor& x) {
        return x.relu();
    }

    std::string Relu::name() const {
        return "Relu";
    }
}
