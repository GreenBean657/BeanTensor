/**
 * @file Dispatcher.cpp
 * @author GreenBean
 * @brief CPU dtype conversion dispatcher definitions.
 * @version 0.1
 * @date 2026-10-18
 */
#include "Dispatcher.h"

#include <memory>

#include "Tensor/Tensor.h"

namespace BeanTensor::Intrinsics::detail {
    /**
     * Dispatch a CPU tensor conversion to a new dtype. Explores branches to locate the correct conversion & acceleration type.
     * @brief Launch a CPU-based conversion operation for the given tensor.
     * @param t The tensor to convert.
     * @param new_dtype The target data type.
     * @param method The conversion clamp method to use.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource If the tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented If the conversion is not implemented for the given data types.
     */
    void launch_conv_cpu(Tensors::Tensor& t, const DataTypes::DType& new_dtype, const ConversionClampMethod method) {
        using DType = DataTypes::DType;
        if (t.dtype == new_dtype) return;
        if (t.device != Tensors::Device::CPU) throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", "CPU");

        const size_t new_nbytes = t.numel * DataTypes::dtype_bytes(new_dtype);
        if (t.convert_buffer == nullptr || t.convert_buffer_size < new_nbytes) {
            std::free(t.convert_buffer);
            // aligned_alloc requires the size to be a multiple of the alignment.
            const size_t alloc_bytes = (new_nbytes + 63) / 64 * 64;
            t.convert_buffer = std::aligned_alloc(64, alloc_bytes);
            if (t.convert_buffer == nullptr) throw ErrorHandling::OutOfMemory();
            t.convert_buffer_size = alloc_bytes;
        }

        std::vector<std::shared_future<void>> futures;
        auto flag = std::make_shared<std::atomic<bool>>(false);

        // Runtime-gated AVX512 fast paths.
        // has_feature() combines the compile-time macro with __builtin_cpu_supports.
        const bool avx512_bf16 = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512BF16);
        const bool avx512_f = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
        const auto avx512_dq = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512DQ);
        const auto avx2 = IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX2);
        const auto avx2_f16c = avx2 && IncludedCPUFeatures().has_feature(CPUFeatureSetType_F16C);
        if (avx512_bf16 && t.dtype == DType::Float32 && new_dtype == DType::BFloat16) {
            auto f = avx512_fp32_to_bf16(static_cast<const float32_t*>(t.data),
                                         static_cast<bfloat16_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_bf16 && t.dtype == DType::BFloat16 && new_dtype == DType::Float32) {
            auto f = avx512_bf16_to_fp32(static_cast<const bfloat16_t*>(t.data),
                                         static_cast<float32_t*>(t.convert_buffer), t.numel);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Float32 && new_dtype == DType::Float16) {
            auto f = avx512_fp32_to_fp16(static_cast<const float32_t*>(t.data),
                                         static_cast<float16_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Float16 && new_dtype == DType::Float32) {
            auto f = avx512_fp16_to_fp32(static_cast<const float16_t*>(t.data),
                                         static_cast<float32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Float32 && new_dtype == DType::Float64) {
            auto f = avx512_fp32_to_fp64(static_cast<const float32_t*>(t.data),
                                         static_cast<float64_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Float64 && new_dtype == DType::Float32) {
            auto f = avx512_fp64_to_fp32(static_cast<const float64_t*>(t.data),
                                         static_cast<float32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Int32 && new_dtype == DType::Float32) {
            auto f = avx512_int32_to_fp32(static_cast<const int32_t*>(t.data),
                                          static_cast<float32_t*>(t.convert_buffer), t.numel);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Float32 && new_dtype == DType::Int32) {
            auto f = avx512_fp32_to_int32(static_cast<const float32_t*>(t.data),
                                          static_cast<int32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::UInt32 && new_dtype == DType::Float32) {
            auto f = avx512_uint32_to_fp32(static_cast<const uint32_t*>(t.data),
                                           static_cast<float32_t*>(t.convert_buffer), t.numel);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_f && t.dtype == DType::Float32 && new_dtype == DType::UInt32) {
            auto f = avx512_fp32_to_uint32(static_cast<const float32_t*>(t.data),
                                           static_cast<uint32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_dq && t.dtype == DType::Int64 && new_dtype == DType::Float64) {
            auto f = avx512_int64_to_fp64(static_cast<const int64_t*>(t.data),
                                          static_cast<float64_t*>(t.convert_buffer), t.numel);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx512_dq && t.dtype == DType::Float64 && new_dtype == DType::Int64) {
            auto f = avx512_fp64_to_int64(static_cast<const float64_t*>(t.data),
                                          static_cast<int64_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        // AVX2/F16C fast paths — same conversions, for CPUs without AVX512.
        } else if (avx2 && t.dtype == DType::Float32 && new_dtype == DType::BFloat16) {
            auto f = avx2_fp32_to_bf16(static_cast<const float32_t*>(t.data),
                                       static_cast<bfloat16_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2 && t.dtype == DType::BFloat16 && new_dtype == DType::Float32) {
            auto f = avx2_bf16_to_fp32(static_cast<const bfloat16_t*>(t.data),
                                       static_cast<float32_t*>(t.convert_buffer), t.numel);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2_f16c && t.dtype == DType::Float32 && new_dtype == DType::Float16) {
            auto f = avx2_fp32_to_fp16(static_cast<const float32_t*>(t.data),
                                       static_cast<float16_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2_f16c && t.dtype == DType::Float16 && new_dtype == DType::Float32) {
            auto f = avx2_fp16_to_fp32(static_cast<const float16_t*>(t.data),
                                       static_cast<float32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2 && t.dtype == DType::Float32 && new_dtype == DType::Float64) {
            auto f = avx2_fp32_to_fp64(static_cast<const float32_t*>(t.data),
                                       static_cast<float64_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2 && t.dtype == DType::Float64 && new_dtype == DType::Float32) {
            auto f = avx2_fp64_to_fp32(static_cast<const float64_t*>(t.data),
                                       static_cast<float32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2 && t.dtype == DType::Int32 && new_dtype == DType::Float32) {
            auto f = avx2_int32_to_fp32(static_cast<const int32_t*>(t.data),
                                        static_cast<float32_t*>(t.convert_buffer), t.numel);
            futures.insert(futures.end(), f.begin(), f.end());
        } else if (avx2 && t.dtype == DType::Float32 && new_dtype == DType::Int32) {
            auto f = avx2_fp32_to_int32(static_cast<const float32_t*>(t.data),
                                        static_cast<int32_t*>(t.convert_buffer), t.numel, *flag, method);
            futures.insert(futures.end(), f.begin(), f.end());
        } else {
            switch (t.dtype) {
                case DType::BFloat16:   convert_to_dst<bfloat16_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::BFloat16x2: convert_to_dst<bfloat16x2_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Float16:    convert_to_dst<float16_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Float16x2:  convert_to_dst<float16x2_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Float32:    convert_to_dst<float32_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Float64:    convert_to_dst<float64_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Int64:      convert_to_dst<int64_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Int32:      convert_to_dst<int32_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Int16:      convert_to_dst<int16_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Int8:       convert_to_dst<int8_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::UInt64:     convert_to_dst<uint64_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::UInt32:     convert_to_dst<uint32_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::UInt16:     convert_to_dst<uint16_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::UInt8:      convert_to_dst<uint8_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                case DType::Int4x2:     convert_to_dst<int4x2_t>(new_dtype, t.data, t.convert_buffer, t.numel, *flag, method, futures); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }


        // Finalize on pool after chunks assigned. Move buffers and flip dtype metadata.
        // A clamp violation under a non-permission method is rethrown from the future at tensor.sync().
        const auto finalize = Threading::GetCPUThreadPool().submit(
            [&t, new_dtype, new_nbytes, flag, futures = std::move(futures)] {
                for (const auto& f : futures) f.get();
                if (flag->load(std::memory_order_relaxed)) {
                    throw std::range_error("convert_dtype: values clamped/invalid under the active ConversionClampMethod");
                }
                std::swap(t.data, t.convert_buffer);
                t.convert_buffer_size = t.nbytes;
                t.dtype = new_dtype;
                t.nbytes = new_nbytes;
            },
            Threading::TaskPriority::Normal
        ).share();
        t.give_future(finalize);
    }
}
