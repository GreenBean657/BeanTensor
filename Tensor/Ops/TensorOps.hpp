#pragma once
#include <cstdint>
#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        inline void set_random_gpu(Tensor& t, float64_t min, float64_t max, std::int64_t seed) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor& t, int64_t amount) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor& t, uint64_t amount) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor& t, float64_t amount) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_to_gpu(Tensor& t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_to_cpu(Tensor& t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void convert_gpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}

#endif
