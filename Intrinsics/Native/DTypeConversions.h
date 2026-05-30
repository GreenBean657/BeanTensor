#pragma once
//
// Created by greenbean on 2026-05-30.
//
#include "Tensor/Tensor.h"

namespace BeanTensor::Intrinsics::detail {
    std::vector<std::shared_future<void>> unaccelerated_bf16_to_fp32(const Casting::bfloat16_t* src, float* dst, size_t end);
}