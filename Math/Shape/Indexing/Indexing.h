/**
 * @file Indexing.h
 * @author GreenBean
 * @brief Entry point for the indexing ops (index_select, gather, cat): cpu vs gpu.
 * @version 0.2
 * @date 2026-10-22
 */
#pragma once

#include <cstddef>
#include <vector>

namespace BeanTensor::Tensors {
    class Tensor;
    namespace detail {

        /**
         * @brief CPU index_select: out[..., j, ...] = t[..., indices[j], ...] along axis.
         */
        void index_select_cpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief GPU index_select: out[..., j, ...] = t[..., indices[j], ...] along axis.
         */
        void index_select_gpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief CPU gather: out[m] = t[m with m[axis] replaced by indices[m]].
         */
        void gather_cpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief GPU gather: out[m] = t[m with m[axis] replaced by indices[m]].
         */
        void gather_gpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief CPU concatenation of a tensor list along axis into out.
         */
        void cat_cpu(const std::vector<Tensor>&, const Tensor&, size_t);
        /**
         * @brief GPU concatenation of a tensor list along axis into out.
         */
        void cat_gpu(const std::vector<Tensor>&, const Tensor&, size_t);
    }
}

#include "Indexing.hpp"
