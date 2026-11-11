/**
 * @file AvgPool2d.h
 * @author GreenBean
 * @brief Entry point for AvgPool2d (cpu vs gpu).
 * @version 0.1
 * @date 2026-11-10
 */
#pragma once

#include <utility>

#include "Intrinsics/Native/DTypeConversions.h"
#include "Tensor/DTypes.h"

namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
    namespace detail {

        /**
         * Pack two uint32_t parameters into the Node's float64_t scalar slot. The
         * backward node unpacks them to recover kernel and stride.
         * @brief Pack AvgPool2d kernel and stride into a float64_t bit pattern.
         */
        [[nodiscard]] inline float64_t pack_avgpool2d_params(const uint32_t kernel, const uint32_t stride) {
            const auto bits = (static_cast<uint64_t>(kernel) << 32) | static_cast<uint64_t>(stride);
            return Intrinsics::detail::bit_cast_memcpy<float64_t>(bits);
        }

        /**
         * @brief Unpack AvgPool2d kernel and stride from a float64_t bit pattern.
         */
        [[nodiscard]] inline std::pair<uint32_t, uint32_t> unpack_avgpool2d_params(const float64_t scalar) {
            const auto bits = Intrinsics::detail::bit_cast_memcpy<uint64_t>(scalar);
            return {static_cast<uint32_t>(bits >> 32), static_cast<uint32_t>(bits & 0xFFFFFFFFu)};
        }

        /**
         * @brief CPU AvgPool2d backend: NCHW rank-4 -> NCHW rank-4, arithmetic mean.
         */
        void avg_pool2d_cpu(const Tensor&, Tensor&, uint32_t, uint32_t);
        /**
         * @brief GPU AvgPool2d backend: NCHW rank-4 -> NCHW rank-4, arithmetic mean.
         */
        void avg_pool2d_gpu(const Tensor&, Tensor&, uint32_t, uint32_t);

        /**
         * @brief CPU AvgPool2d backward backend: dx[window] += dy / (kernel²).
         */
        void avg_pool2d_g_cpu(const Tensor&, Tensor&, uint32_t, uint32_t);
        /**
         * @brief GPU AvgPool2d backward backend: dx[window] += dy / (kernel²).
         */
        void avg_pool2d_g_gpu(const Tensor&, Tensor&, uint32_t, uint32_t);

        /**
         * @brief Backward trampoline for AvgPool2d; matches Node::_backward_fn's signature.
         */
        void avg_pool2d_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        /**
         * @brief Signature bridge between wire_node (mutable output_grad) and avg_pool2d_g_node.
         */
        void avg_pool2d_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
    }
}

#include "AvgPool2d.hpp"
