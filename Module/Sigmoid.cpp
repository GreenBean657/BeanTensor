/**
 * @file Sigmoid.cpp
 * @author GreenBean
 * @brief Sigmoid layer implementation: forward delegates to Tensor::sigmoid.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Module/Sigmoid.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Sigmoid::forward(const Tensors::Tensor& x) {
        return x.sigmoid();
    }

    std::string Sigmoid::name() const {
        return "Sigmoid";
    }
}
