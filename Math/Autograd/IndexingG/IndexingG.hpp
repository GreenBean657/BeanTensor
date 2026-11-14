/**
 * @file IndexingG.hpp
 * @author GreenBean
 * @brief CPU-only build fallback for the indexing/view backward ops - throws
 * and does nothing else.
 * @version 0.1
 * @date 2026-11-14
 */
#pragma once

#include <cstddef>

#include "Helpers/Errors.h"
#include "Tensor/DTypes.h"

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Tensors {
    class Tensor;
    struct NodeMeta;
    namespace detail {
        // CPU-only build: no GPU backend exists to link against.
        /**
         * @brief CPU-only build stub for GPU index_select backward. Always throws.
         */
        inline void index_select_g_gpu(const Tensor&, const Tensor&, const Tensor&, size_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU gather backward. Always throws.
         */
        inline void gather_g_gpu(const Tensor&, const Tensor&, const Tensor&, size_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU cat backward. Always throws.
         */
        inline void cat_g_gpu(const Tensor&, const Tensor&, size_t, size_t) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
        /**
         * @brief CPU-only build stub for GPU slice backward. Always throws.
         */
        inline void slice_g_gpu(const Tensor&, const Tensor&, const NodeMeta&) {
            throw ErrorHandling::GPUTaskOnCPUBuild();
        }
    }
}
#endif
