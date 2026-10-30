/**
 * @file ElementwiseG.h
 * @author GreenBean
 * @brief Entry point for elementwise autograd backward (cpu vs gpu): the
 * add/sub/mul/div/pow (tensor-tensor and tensor-scalar) and exp/log backward
 * trampolines, plus the broadcast gradient reduction (bcast_sum_to) that sums
 * an out-shaped delta back over the broadcast dims of each operand.
 * The trampolines match Node::_backward_fn's signature; the trailing DType is
 * the calling backward()'s assigned grad dtype.
 * @version 0.2
 * @date 2026-10-23
 */
#pragma once

#include <cstddef>

#include "Math/Elementwise/Elementwise.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * @brief Backward trampoline for tensor-tensor add.
         */
        void add_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-tensor sub.
         */
        void sub_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-tensor mul.
         */
        void mul_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-tensor div.
         */
        void div_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-tensor pow.
         */
        void pow_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-scalar add.
         */
        void add_scalar_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-scalar sub.
         */
        void sub_scalar_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-scalar mul.
         */
        void mul_scalar_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-scalar div.
         */
        void div_scalar_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for tensor-scalar pow.
         */
        void pow_scalar_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for unary exp.
         */
        void exp_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Backward trampoline for unary log.
         */
        void log_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);

        /**
         * @brief Shared trampoline body for the tensor-tensor elementwise backward nodes (impl detail — friended by Tensor).
         */
        void ew_binary_g_node_impl(Node*, Tensor*, Tensor**, size_t, DataTypes::DType, ElementwiseOp);
        /**
         * @brief Shared trampoline body for the tensor-scalar elementwise backward nodes (impl detail — friended by Tensor).
         */
        void ew_scalar_g_node_impl(Node*, Tensor*, Tensor**, size_t, DataTypes::DType, ElementwiseOp);
        /**
         * @brief Shared trampoline body for the unary elementwise backward nodes (impl detail — friended by Tensor).
         */
        void ew_unary_g_node_impl(Node*, Tensor*, Tensor**, size_t, DataTypes::DType, ElementwiseUnaryOp);

        /**
         * Deep copy of op in dst (boundary-cast under the single
         * grad-conversion rule when the dtypes differ) with the
         * requires_grad flag cleared. The backward trampolines compose their
         * deltas from the public forward ops; any operand that still
         * requires grad would wire throwaway nodes onto those products, with
         * parent pointers into node-fn locals that die at scope end. The
         * copy keeps the composed products off the graph entirely (the
         * GemmG precedent is clean because its transposes are fresh
         * requires_grad=false tensors and its output_grad is a grad tensor).
         * @brief Graph-clean operand copy in the compute dtype.
         */
        [[nodiscard]] Tensor ew_operand_copy(const Tensor&, const DataTypes::DType&);

        /**
         * Shared tail of every elementwise trampoline: delivers a full
         * (output-shaped) delta into `operand`. When the operand was
         * broadcast (its shape differs from the delta's) the delta is first
         * sum-reduced over the broadcast dims by bcast_sum_to on the
         * operand's device; the result is then boundary-cast into
         * grad_dtype when the compute ran in a different working dtype
         * (pow/exp under an integer assigned grad dtype), hits the
         * intermediate-delta probe when the operand owns a node, and lands
         * in operand.accumulate_grad.
         * @brief Reduce (if broadcast), boundary-cast, probe and accumulate a full delta.
         * @throws ErrorHandling::UnReachableCode if the operand is on neither CPU nor GPU.
         */
        void ew_deliver_delta(Tensor&&, Tensor&, const DataTypes::DType&, const DataTypes::DType&);

        /**
         * Sum-reduces delta (the out-shaped gradient of a broadcast elementwise
         * op) into dx (the operand's shape): dx[j] = sum of delta[i] over every
         * output index i that maps to operand element j under the broadcast
         * plan. dx is written (each element by exactly one worker), not
         * accumulated, so no zero-fill is required. Floats sum in the storage
         * width per element in a fixed enumeration order (bit-identical across
         * chunkings and against the GPU twin); halfs accumulate in fp32 and
         * narrow RNE once; packed dtypes per lane; integers wrap through the
         * unsigned twin; int4x2 wraps per nibble. Blocks until the workers
         * finish is NOT performed here: futures land on delta and dx.
         * @brief CPU broadcast gradient reduction.
         * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
         * @throws ErrorHandling::NotImplemented for an unsupported dtype.
         */
        void bcast_sum_to_cpu(const Tensor&, const Tensor&, Tensor&);
        /**
         * GPU twin of bcast_sum_to_cpu; behaviorally identical (same compute
         * widths, same RNE narrowing, same wrap integer adds).
         * @brief GPU broadcast gradient reduction.
         * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
         * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
         */
        void bcast_sum_to_gpu(const Tensor&, const Tensor&, Tensor&);

    }
}

#include "ElementwiseG.hpp"
