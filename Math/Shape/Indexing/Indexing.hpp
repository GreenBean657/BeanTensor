/**
 * @file Indexing.hpp
 * @author GreenBean
 * @brief CPU only build fall back for the indexing ops - throws and does nothing else.
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#include <cstddef>
#include <vector>

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {
        // CPU-only build: no GPU backend exists to link against.
        /**
         * @brief CPU-only build stub for GPU index_select. Always throws.
         */
        inline void index_select_gpu(const Tensor&, const Tensor&, Tensor&, size_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU gather. Always throws.
         */
        inline void gather_gpu(const Tensor&, const Tensor&, Tensor&, size_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU cat. Always throws.
         */
        inline void cat_gpu(const std::vector<Tensor>&, Tensor&, size_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
