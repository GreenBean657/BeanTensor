#pragma once
#include "Tensor/Tensor.h"

namespace BeanTensor::Intrinsics::detail {
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const Casting::float32_t* src, Casting::bfloat16_t* dst, const size_t& end);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_bf16_to_fp32(const Casting::bfloat16_t* src, Casting::float32_t* dst, const size_t& end);
}