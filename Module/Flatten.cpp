/**
 * @file Flatten.cpp
 * @author GreenBean
 * @brief Flatten layer implementation: forward delegates to Tensor::flatten.
 * @version 0.1
 * @date 2026-10-28
 */
#include "Module/Flatten.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Modules {

    Tensors::Tensor Flatten::forward(const Tensors::Tensor& x) {
        return x.flatten();
    }

    std::string Flatten::name() const {
        return "Flatten";
    }
}
