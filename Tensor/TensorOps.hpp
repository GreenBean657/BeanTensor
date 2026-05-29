#pragma once
#include <cstdint>
#include "Errors/logic.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        inline void set_random_gpu(Tensor& t, double min, double max, std::int64_t seed) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor& t, int32_t amount) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_to_gpu(Tensor& t, uint32_t device) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_to_cpu(Tensor& t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_gpu_to_gpu(Tensor& t, uint32_t new_device) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void convert_gpu(Tensor& t, const BeanTensor::Casting::DType& new_dtype, bool use_inf_conversions) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}

#endif
