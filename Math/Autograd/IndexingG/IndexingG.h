/**
 * @file IndexingG.h
 * @author GreenBean
 * @brief Entry point for the indexing/view backward ops (index_select, gather,
 * cat, slice; reshape reuses FlattenG): cpu vs gpu backends and the node
 * trampolines matching Node::_backward_fn's signature.
 * @version 0.1
 * @date 2026-11-14
 */
#pragma once

#include <cstddef>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    struct NodeMeta;
    namespace detail {

        /**
         * @brief CPU index_select backward: dx is pre-zeroed; dy slabs scatter-add into it.
         */
        void index_select_g_cpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief GPU index_select backward: dx is pre-zeroed; dy slabs scatter-add into it.
         */
        void index_select_g_gpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief CPU gather backward: dx is pre-zeroed; dy elements scatter-add into it.
         */
        void gather_g_cpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief GPU gather backward: dx is pre-zeroed; dy elements scatter-add into it.
         */
        void gather_g_gpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        /**
         * @brief CPU cat backward: fill dx (an input's shape) from dy's slab at the axis offset.
         */
        void cat_g_cpu(const Tensor&, const Tensor&, size_t, size_t);
        /**
         * @brief GPU cat backward: fill dx (an input's shape) from dy's slab at the axis offset.
         */
        void cat_g_gpu(const Tensor&, const Tensor&, size_t, size_t);
        /**
         * @brief CPU slice backward: embed dy's box into the pre-zeroed dx per the recorded ranges.
         */
        void slice_g_cpu(const Tensor&, const Tensor&, const NodeMeta&);
        /**
         * @brief GPU slice backward: embed dy's box into the pre-zeroed dx per the recorded ranges.
         */
        void slice_g_gpu(const Tensor&, const Tensor&, const NodeMeta&);

        /**
         * @brief Backward trampoline for index_select; matches Node::_backward_fn's
         * signature. The leading Node* is the owning node; the trailing DType is the
         * calling backward()'s assigned gradient dtype.
         */
        void index_select_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node and index_select_g_node.
         */
        void index_select_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for gather; same contract as index_select_g_node.
         */
        void gather_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node and gather_g_node.
         */
        void gather_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for cat; walks _inputs plus NodeMeta::extra_parents.
         */
        void cat_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node_meta and cat_g_node.
         */
        void cat_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for slice (and split's per-chunk nodes).
         */
        void slice_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node and slice_g_node.
         */
        void slice_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "IndexingG.hpp"
