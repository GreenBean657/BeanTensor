/**
 * @file Tanh.cpp
 * @author GreenBean
 * @brief Tanh layer implementation: forward delegates to Tensor::tanh.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Module/Tanh.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Tanh::forward(const Tensors::Tensor& x) {
        return x.tanh();
    }

    std::string Tanh::name() const {
        return "Tanh";
    }
}
