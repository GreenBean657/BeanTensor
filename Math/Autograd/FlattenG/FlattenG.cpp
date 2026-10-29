/**
 * @file FlattenG.cpp
 * @author GreenBean
 * @brief Flatten backward trampoline: dx = reshape(dy, x.shape()).
 * @version 0.1
 * @date 2026-10-28
 */
#include <vector>

#include "Math/Autograd/FlattenG/FlattenG.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {

    /**
     * Reshape the upstream gradient back to the input's shape. dy may be
     * device-resident, so we deep-copy it once, create an x-shaped tensor in
     * grad_dtype, and re-point that tensor at the cloned buffer (flat element
     * order is identical; the cloned buffer is owned by dx so it survives as
     * x.grad).
     * @brief Flatten backward trampoline.
     * @param output_grad Gradient of the forward output, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1.
     */
    void flatten_g_node([[maybe_unused]] Node* node, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        auto dx_buf = const_cast<Tensor&>(*output_grad).clone(true);
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        dx.kill_cpu_data();
        dx.kill_gpu_data();
        dx.data = dx_buf.data;
        dx.owns_data = true;
        dx_buf.data = nullptr;
        dx_buf.owns_data = false;
        x.accumulate_grad(std::move(dx));
    }

    void flatten_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        flatten_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
