/**
 * @file AvgPool2dGCPU.cpp
 * @author GreenBean
 * @brief AvgPool2d backward trampoline (avg_pool2d_g_node): dx[window] += dy /
 * (kernel²). The delta is built in the assigned grad_dtype (or Float32 when the
 * assigned dtype is not a float kind), then boundary-cast back to grad_dtype.
 * @version 0.1
 * @date 2026-11-10
 */
#include <vector>

#include "Math/Autograd/AvgPool2dG/AvgPool2dG.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Math/Spatial/AvgPool2d/AvgPool2d.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace elementwise_shared;

    /**
     * inputs are (x). Only x receives a gradient. The kernel adds dy/(kernel²) to
     * every element of the source window, using the kernel/stride packed into the
     * owning Node's _scalar slot.
     * @brief AvgPool2d backward implementation (equal-share scatter).
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1 or the device is neither CPU nor GPU.
     */
    void avg_pool2d_g_node(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        const auto [kernel, stride] = unpack_avgpool2d_params(node->_scalar);
        const auto work = is_float_kind(grad_dtype)
            ? grad_dtype
            : DataTypes::DType::Float32;
        auto dy_cast = std::optional<Tensor>{};
        const Tensor* dyw = output_grad;
        if (output_grad->dtype != work) {
            dy_cast = grad_converted_copy(*output_grad, work);
            dyw = &*dy_cast;
        }
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), work, x.device, false);
        if (x.device == Device::CPU) {
            avg_pool2d_g_cpu(*dyw, dx, kernel, stride);
        } else if (x.device == Device::GPU) {
            avg_pool2d_g_gpu(*dyw, dx, kernel, stride);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        grad_convert_inplace(dx, grad_dtype);
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype);
        }
        x.accumulate_grad(std::move(dx));
    }

    /**
     * Backward trampoline wired into Node::_backward_fn. wire_node's function-pointer
     * type takes a mutable output_grad while avg_pool2d_g_node only reads it; this
     * bridge forwards the call unchanged.
     * @brief Backward trampoline for AvgPool2d, matching Node::_backward_fn's signature.
     */
    void avg_pool2d_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        avg_pool2d_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
