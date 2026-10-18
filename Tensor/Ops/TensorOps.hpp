/**
 * @file TensorOps.hpp
 * @author GreenBean
 * @brief CPU only build fall back for GPU tensor ops - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-17
 */
#pragma once
#include <cstdint>
#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        inline void set_random_gpu(Tensor&, float64_t, float64_t, std::int64_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor&, int64_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor&, uint64_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void set_amount_gpu(Tensor&, float64_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_to_gpu(Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void move_to_cpu(Tensor&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        inline void convert_gpu(Tensor&, const DataTypes::DType&, ConversionClampMethod) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}

#endif
