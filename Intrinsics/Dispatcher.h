#pragma once
#include <future>
#include <vector>
#include <Tensor/Tensor.h>
namespace BeanTensor::Intrinsics::detail {
    [[nodiscard]] inline std::pmr::vector<std::vector<void>> launch_conv_cpu(void* data, BeanTensor::Casting::DType dtype) {

        return std::pmr::vector<std::vector<void>>();
    }
}