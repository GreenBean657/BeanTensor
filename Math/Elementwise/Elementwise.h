/**
 * @file Elementwise.h
 * @author GreenBean
 * @brief Entry point for elementwise arithmetic ops with broadcasting (cpu vs gpu):
 * add, sub, mul, div, pow (tensor-tensor and tensor-scalar), exp, log (unary).
 * Equality/comparison lives in Math/Compare (owned by the metrics mission).
 * @version 0.3
 * @date 2026-10-28
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;

    /**
     * @brief Binary elementwise operation selector (tensor-tensor and tensor-scalar).
     */
    enum class ElementwiseOp : uint8_t {
        Add,
        Sub,
        Mul,
        Div,
        Pow
    };

    /**
     * Float kinds only — integer exp/log is rejected with ErrorHandling::NotImplemented.
     * @brief Unary elementwise operation selector.
     */
    enum class ElementwiseUnaryOp : uint8_t {
        Exp,
        Log,
        Sigmoid,
        Tanh
    };

    namespace detail {
        /**
         * Fixed 4 entries == BT_TENSOR_MAX_DIMS (the macro lives in Tensor.h, which
         * includes this header, so it cannot be referenced here; a static_assert in
         * ElementwiseShared.h pins the match). a_strides/b_strides are ELEMENT
         * strides of each operand re-expressed in output coordinates: 0 on
         * broadcast dims, the operand's own stride otherwise. exact means both
         * operands share the output shape, so both element offsets equal the flat
         * index. Trivially copyable: passed to GPU kernels by value.
         * @brief Broadcast execution plan shared by the CPU and GPU backends.
         */
        struct ElementwiseBcast {
            size_t out_shape[4];
            size_t out_strides[4];
            size_t a_strides[4];
            size_t b_strides[4];
            size_t out_numel;
            size_t out_ndim;
            bool exact;
        };

        /**
         * @brief CPU tensor-tensor elementwise backend: out = a (op) b with NumPy-style broadcasting.
         */
        void elementwise_cpu(const Tensor&, const Tensor&, const Tensor&, ElementwiseOp, const ElementwiseBcast&);
        /**
         * @brief GPU tensor-tensor elementwise backend: out = a (op) b with NumPy-style broadcasting.
         */
        void elementwise_gpu(const Tensor&, const Tensor&, const Tensor&, ElementwiseOp, const ElementwiseBcast&);
        /**
         * @brief CPU tensor-scalar elementwise backend: out[i] = a[i] (op) scalar, scalar narrowed into a's dtype.
         */
        void elementwise_scalar_cpu(const Tensor&, const float64_t&, const Tensor&, ElementwiseOp);
        /**
         * @brief GPU tensor-scalar elementwise backend: out[i] = a[i] (op) scalar, scalar narrowed into a's dtype.
         */
        void elementwise_scalar_gpu(const Tensor&, const float64_t&, const Tensor&, ElementwiseOp);
        /**
         * @brief CPU unary elementwise backend: out[i] = fn(a[i]) (exp/log, float kinds only).
         */
        void elementwise_unary_cpu(const Tensor&, const Tensor&, ElementwiseUnaryOp);
        /**
         * @brief GPU unary elementwise backend: out[i] = fn(a[i]) (exp/log, float kinds only).
         */
        void elementwise_unary_gpu(const Tensor&, const Tensor&, ElementwiseUnaryOp);
    }
}

#include "Elementwise.hpp"
