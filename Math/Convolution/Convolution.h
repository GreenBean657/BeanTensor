/**
 * @file Convolution.h
 * @author GreenBean
 * @brief Public entry point for Conv2d: forward declarations for the CPU/GPU
 * im2col, col2im, and conv2d backends, plus the Conv2dG backward trampoline.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * Precomputed conv2d dimensions used by all helpers.
         * @brief Conv2d shape bundle.
         */
        struct Conv2dShape;

        /**
         * Zero-fill a tensor buffer.
         * @brief Zero a tensor buffer.
         */
        void conv_zero_tensor(const Tensor& t);

        /**
         * CPU im2col: materializes the {M, C*R*S} column matrix from an NCHW
         * input. Padding is applied as zeros; out-of-bounds positions write 0.
         * @brief CPU im2col data movement.
         * @param x Input tensor {N,C,H,W} on CPU.
         * @param col Output tensor {M, C*R*S} on CPU.
         * @param stride Stride along H and W.
         * @param padding Zero padding on all sides.
         * @param R Kernel height.
         * @param S Kernel width.
         */
        void im2col_cpu(const Tensor& x, const Tensor& col, uint32_t stride, uint32_t padding, size_t R, size_t S);

        /**
         * CPU col2im: folds the {M, C*R*S} column matrix into an NCHW dx,
         * accumulating overlapping positions.
         * @brief CPU col2im data movement.
         * @param col Column matrix {M, C*R*S} on CPU.
         * @param dx Output tensor {N,C,H,W} on CPU.
         * @param stride Stride along H and W.
         * @param padding Zero padding on all sides.
         * @param R Kernel height.
         * @param S Kernel width.
         */
        void col2im_cpu(const Tensor& col, const Tensor& dx, uint32_t stride, uint32_t padding, size_t R, size_t S);

        /**
         * CPU conv2d forward: validates, im2col, GEMM, bias + NCHW scatter.
         * @brief CPU Conv2d forward backend.
         * @param x Input {N,C,H,W} on CPU.
         * @param weight Weight {C,R,S,K} on CPU.
         * @param bias Optional bias {K} on CPU (nullptr for bias-less).
         * @param out Preallocated output {N,K,Hout,Wout} on CPU.
         * @param stride Stride along H and W.
         * @param padding Zero padding on all sides.
         */
        void conv2d_cpu(const Tensor& x, const Tensor& weight, const Tensor* bias, const Tensor& out, uint32_t stride, uint32_t padding);

        // Declared unconditionally (like matmul_gpu in Matmul.h): Tensor.h
        // befriends these in all build flavors; the definitions exist only
        // under the backend macros below.
        void im2col_gpu(const Tensor& x, const Tensor& col, uint32_t stride, uint32_t padding, size_t R, size_t S);
        void col2im_gpu(const Tensor& col, const Tensor& dx, uint32_t stride, uint32_t padding, size_t R, size_t S);
        void conv2d_gpu(const Tensor& x, const Tensor& weight, const Tensor* bias, const Tensor& out, uint32_t stride, uint32_t padding);

#if defined(USE_CUDA)
        /**
         * GPU im2col on CUDA.
         * @brief CUDA im2col data movement.
         */
        void im2col_gpu(const Tensor& x, const Tensor& col, uint32_t stride, uint32_t padding, size_t R, size_t S);
        /**
         * GPU col2im on CUDA.
         * @brief CUDA col2im data movement.
         */
        void col2im_gpu(const Tensor& col, const Tensor& dx, uint32_t stride, uint32_t padding, size_t R, size_t S);
        /**
         * GPU conv2d forward on CUDA.
         * @brief CUDA Conv2d forward backend.
         */
        void conv2d_gpu(const Tensor& x, const Tensor& weight, const Tensor* bias, const Tensor& out, uint32_t stride, uint32_t padding);
#elif defined(USE_HIP)
        /**
         * GPU im2col on HIP.
         * @brief HIP im2col data movement.
         */
        void im2col_gpu(const Tensor& x, const Tensor& col, uint32_t stride, uint32_t padding, size_t R, size_t S);
        /**
         * GPU col2im on HIP.
         * @brief HIP col2im data movement.
         */
        void col2im_gpu(const Tensor& col, const Tensor& dx, uint32_t stride, uint32_t padding, size_t R, size_t S);
        /**
         * GPU conv2d forward on HIP.
         * @brief HIP Conv2d forward backend.
         */
        void conv2d_gpu(const Tensor& x, const Tensor& weight, const Tensor* bias, const Tensor& out, uint32_t stride, uint32_t padding);
#endif

        /**
         * Backward trampoline for Conv2d. Recomputes im2col(x) from _inputs[0],
         * materializes dyT {M,K}, then accumulates dW, db, and dx using the
         * existing GEMM helpers and transpose_last2.
         * @brief Conv2d backward trampoline.
         * @param output_grad Gradient of the conv2d output, in grad_dtype.
         * @param inputs Input tensors of the forward op: (x, weight, [bias]).
         * @param n_inputs 2 for bias-less, 3 for bias.
         * @param grad_dtype The calling backward()'s assigned gradient dtype.
         */
        void conv2d_g_node(Node*, Tensor* output_grad, Tensor** inputs, size_t n_inputs, DataTypes::DType grad_dtype);

        /**
         * dtype-dispatching col2im host helper.
         * @brief col2im accumulation by dtype.
         */
        void col2im_dispatch(const Tensor& col, const Tensor& dx, const Conv2dShape& s, uint32_t stride, uint32_t padding);

        /**
         * dtype-dispatching rowsum over axis 0 of a {M,K} tensor.
         * @brief rowsum helper.
         */
        void rowsum_mk_dispatch(const Tensor& src, const Tensor& out);

        /**
         * dtype-dispatching NCHW bias add.
         * @brief NCHW bias epilogue.
         */
        void bias_add_nchw_dispatch(const Tensor& bias, const Tensor& out);

        /**
         * Pack (stride, padding) into a float64_t for storage in Node::_scalar.
         * @brief Pack conv2d hyperparameters for the backward node.
         */
        [[nodiscard]] float64_t conv2d_pack_stride_padding(uint32_t stride, uint32_t padding);
    }
}
