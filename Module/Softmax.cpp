/**
 * @file Softmax.cpp
 * @author GreenBean
 * @brief Softmax layer implementation: forward delegates to Tensor::softmax.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Module/Softmax.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Softmax::forward(const Tensors::Tensor& x) {
        return x.softmax();
    }

    std::string Softmax::name() const {
        return "Softmax";
    }
}
