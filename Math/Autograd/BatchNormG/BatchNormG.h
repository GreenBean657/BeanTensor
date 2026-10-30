/**
 * @file BatchNormG.h
 * @author GreenBean
 * @brief batch_norm backward nodes: the train-mode node recomputes the
 * per-channel statistics from x and inverts the normalization analytically;
 * the eval-mode node reads the running statistics through Node::_saved, the
 * project's first use of that field. The GPU device twins take a POD argument
 * pack (BatchNormGArgs) of offset-applied device pointers, so the CUDA/HIP
 * translation units need no Tensor friend access.
 * @version 0.3
 * @date 2026-10-30
 */
#pragma once

#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * POD argument pack for the GPU backward launchers (CUDA/HIP TUs): raw
         * device pointers with the element offsets already applied by the
         * caller (the node trampolines, which are Tensor friends), so the GPU
         * translation units need no Tensor friend access. running_mean/
         * running_var are read only by the eval launcher (nullptr in train
         * mode, where the statistics are recomputed from x); dx/dgamma/dbeta
         * are nullptr for parents that do not require grad.
         * @brief Device-side argument pack for the batch_norm GPU backward.
         */
        struct BatchNormGArgs {
            DataTypes::DType in_dtype;   // x/gamma/running-buffer storage dtype.
            DataTypes::DType grad_dtype; // dy/dx/dgamma/dbeta dtype.
            const void* x;
            const void* dy;
            const void* gamma;
            const void* running_mean;
            const void* running_var;
            void* dx;
            void* dgamma;
            void* dbeta;
            size_t channels;
            size_t inner;
            size_t count;
            float64_t eps;
        };

        /**
         * GPU train-mode batch_norm backward: recomputes the per-channel
         * statistics from x, then dgamma = sum(dy * x_hat), dbeta = sum(dy)
         * and the pinned dx element formula (BatchNormGShared.h). Runs the
         * kernels on the default stream and stream-syncs before returning
         * (the node trampoline drains everything before accumulate_grad).
         * @brief GPU train-mode batch_norm backward.
         */
        void batch_norm_g_train_gpu(const BatchNormGArgs&);

        /**
         * GPU eval-mode batch_norm backward: per-channel mean/invstd come from
         * the saved running buffers (graph constants), dgamma/dbeta from one
         * reduction pass over dy and dy * x_hat, dx = (gamma * invstd) * dy
         * with no train correction terms. Runs the kernels on the default
         * stream and stream-syncs before returning.
         * @brief GPU eval-mode batch_norm backward.
         */
        void batch_norm_g_eval_gpu(const BatchNormGArgs&);

        /**
         * Backward trampoline for train-mode batch_norm; matches
         * Node::_backward_fn's signature. The leading Node* is the owning node
         * (its _scalar carries eps); the trailing DType is the calling
         * backward()'s assigned grad dtype. inputs is (x, gamma, beta).
         */
        void batch_norm_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);

        /**
         * Backward trampoline for eval-mode batch_norm; matches
         * Node::_backward_fn's signature. Running stats are graph constants
         * AND implicit parents via Node::_saved — the tensors _saved[0]/
         * _saved[1] point to must outlive backward(). The leading Node* is the
         * owning node (its _scalar carries eps, its _saved[0]/_saved[1] point
         * at the forward's running_mean/running_var); the trailing DType is
         * the calling backward()'s assigned grad dtype. inputs is
         * (x, gamma, beta).
         */
        void batch_norm_g_eval_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}
