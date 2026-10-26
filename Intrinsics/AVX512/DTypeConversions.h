/**
 * @file DTypeConversions.h
 * @author GreenBean
 * @brief AVX512 dtype conversion kernel declarations. Implementations gate NT stores on
 * pointer alignment + NtStoreMinBytes() and submit chunks CCD-locally when KeepWorkLocal() holds.
 * @version 0.2
 * @date 2026-10-24
 */
#pragma once

#include <atomic>
#include <future>
#include <vector>

#include "Intrinsics/ConversionMethods.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Intrinsics::detail {
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_bf16(const float32_t* src, bfloat16_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_bf16_to_fp32(const bfloat16_t* src, float32_t* dst, const size_t& end);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_fp16(const float32_t* src, float16_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp16_to_fp32(const float16_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_fp64(const float32_t* src, float64_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp64_to_fp32(const float64_t* src, float32_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_int32_to_fp32(const int32_t* src, float32_t* dst, const size_t& end);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_int32(const float32_t* src, int32_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_uint32_to_fp32(const uint32_t* src, float32_t* dst, const size_t& end);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp32_to_uint32(const float32_t* src, uint32_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_int64_to_fp64(const int64_t* src, float64_t* dst, const size_t& end);
    [[nodiscard]] std::vector<std::shared_future<void>> avx512_fp64_to_int64(const float64_t* src, int64_t* dst, const size_t& end, std::atomic<bool>& flag, ConversionClampMethod method);
}
