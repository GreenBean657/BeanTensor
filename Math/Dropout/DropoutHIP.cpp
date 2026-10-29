/**
 * @file DropoutHIP.cpp
 * @brief HIP/ROCm backend for inverted dropout mask construction
 * (train_mask_gpu; routed to by the train_mask dispatcher in DropoutCPU.cpp).
 * @version 0.2
 * @date 2026-10-29
 */
#if defined(USE_HIP)
#include <cstdint>
#include <type_traits>
#include <vector>

#include "Math/Dropout/Dropout.h"
#include "Math/Dropout/DropoutHipShared.h"
#include "Math/Dropout/DropoutShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"

namespace BeanTensor::Math::Dropout {

    namespace {
        using namespace BeanTensor::Tensors::detail::dropout_gpu;

        /**
         * @brief HIP threshold-scale mask kernel for float kinds.
         */
        template<typename T>
        __global__ void dropout_mask_kernel(const T* __restrict__ rand, T* __restrict__ out,
                                            const float64_t p, const float64_t scale, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = tid; i < numel; i += stride) {
                if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                    out[i] = dev_mask_native<T>(rand[i], p, scale);
                } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    out[i] = dev_mask_half<T>(rand[i], p, scale);
                } else {
                    out[i] = dev_mask_packed<T>(rand[i], p, scale);
                }
            }
        }

        template<typename T>
        __host__ void run_dropout_mask(const T* rand, T* out, const float64_t p, const float64_t scale, const size_t numel) {
            if (numel > 0) {
                dropout_mask_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(rand, out, p, scale, numel);
            }
        }
    }

    Tensors::Tensor train_mask_gpu(const Tensors::Tensor& x, const float64_t p, const int64_t seed) {
        using DType = DataTypes::DType;
        using namespace BeanTensor::Tensors::detail::dropout_shared;
        if (!is_dropout_float_kind(x.dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        if (x.device != Tensors::Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        if (p < float64_t{0.0} || p >= float64_t{1.0}) {
            throw ErrorHandling::InvalidProbability("p must satisfy 0 <= p < 1");
        }
        auto rand = Tensors::Tensor(std::vector(x.shape, x.shape + x.ndim), x.dtype, x.device, false);
        rand.random(0.0, 1.0, seed);
        rand.sync();
        auto out = Tensors::Tensor(std::vector(x.shape, x.shape + x.ndim), x.dtype, x.device, false);
        const auto scale = float64_t{1.0} / (float64_t{1.0} - p);
        const auto* const rand_data = rand.data;
        auto* const out_data = out.data;
        const auto rand_offset = rand.offset;
        const auto out_offset = out.offset;
        const auto dtype = x.dtype;
        const auto numel = x.numel;
        const auto worker = [=] {
            switch (dtype) {
                case DType::Float32: run_dropout_mask(static_cast<const float32_t*>(rand_data) + rand_offset, static_cast<float32_t*>(out_data) + out_offset, p, scale, numel); break;
                case DType::Float64: run_dropout_mask(static_cast<const float64_t*>(rand_data) + rand_offset, static_cast<float64_t*>(out_data) + out_offset, p, scale, numel); break;
                case DType::Float16: run_dropout_mask(static_cast<const float16_t*>(rand_data) + rand_offset, static_cast<float16_t*>(out_data) + out_offset, p, scale, numel); break;
                case DType::BFloat16: run_dropout_mask(static_cast<const bfloat16_t*>(rand_data) + rand_offset, static_cast<bfloat16_t*>(out_data) + out_offset, p, scale, numel); break;
                case DType::Float16x2: run_dropout_mask(static_cast<const float16x2_t*>(rand_data) + rand_offset, static_cast<float16x2_t*>(out_data) + out_offset, p, scale, numel); break;
                case DType::BFloat16x2: run_dropout_mask(static_cast<const bfloat16x2_t*>(rand_data) + rand_offset, static_cast<bfloat16x2_t*>(out_data) + out_offset, p, scale, numel); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        rand.give_future(future);
        out.give_future(future);
        return out;
    }

}
#endif
