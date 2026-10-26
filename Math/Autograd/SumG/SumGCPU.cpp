/**
 * @file SumGCPU.cpp
 * @author GreenBean
 * @brief sum backward trampoline (sum_g_node): dx = the scalar seed broadcast
 * to x's shape. The delta is built directly in grad_dtype from device-agnostic
 * tensor ops, so no dedicated cpu/gpu backends exist for this node.
 * @version 0.1
 * @date 2026-10-25
 */
#include <utility>
#include <vector>

#include "Math/Autograd/SumG/SumG.h"
#include "Math/Autograd/GradDType.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {

    /**
     * inputs is (x). A full-tensor sum's local derivative is 1 for every
     * element, so dx = output_grad (the scalar seed) broadcast to x's shape,
     * built as a ones tensor (allocated in grad_dtype) times the seed. No
     * boundary cast is needed: output_grad already carries grad_dtype
     * (backward()'s contract) and the ones operand is allocated in it, so the
     * delta leaves the node in the assigned dtype by construction (GradDType.h,
     * the single conversion rule, is honored through the tensor ops' own
     * dtype contracts). The multiply dispatches on x's device — no device code
     * here. Syncs the tensors it reads.
     * @brief Backward trampoline for sum.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1.
     */
    void sum_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        auto ones = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        ones.set_one();
        auto dx = ones.mul(*output_grad);
        dx.sync();
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
        }
        x.accumulate_grad(std::move(dx));
    }
}
