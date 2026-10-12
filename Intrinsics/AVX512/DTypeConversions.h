#pragma once

#include <atomic>
#include <cstddef>
#include <future>
#include <vector>

#include "Intrinsics/ConversionMethods.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Intrinsics::detail {
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const float32_t* src, bfloat16_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_bf16_to_fp32(const bfloat16_t* src, float32_t* dst, const size_t& end);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_fp16(const float32_t* src, float16_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp16_to_fp32(const float16_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
}
