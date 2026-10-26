/**
 * @file MeanGCPU.cpp
 * @author GreenBean
 * @brief mean backward trampoline (mean_g_node): dx = (the scalar seed /
 * numel) broadcast to x's shape. The delta is built directly in grad_dtype
 * from device-agnostic tensor ops, so no dedicated cpu/gpu backends exist for
 * this node.
 * @version 0.1
 * @date 2026-10-25
 */
#include <utility>
#include <vector>

#include "Math/Autograd/MeanG/MeanG.h"
#include "Math/Autograd/GradDType.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {

    /**
     * inputs is (x). A full-tensor mean's local derivative is 1/numel for
     * every element, so dx = output_grad (the scalar seed) divided by numel,
     * broadcast to x's shape as a ones tensor (allocated in grad_dtype) times
     * the scaled seed. The scalar division narrows 1/numel's divisor into
     * grad_dtype under the tensor-scalar op's own contract; no further
     * boundary cast is needed because output_grad already carries grad_dtype
     * (backward()'s contract) and both node-local tensors are allocated in it,
     * so the delta leaves the node in the assigned dtype by construction.
     * The ops dispatch on x's device — no device code here. Syncs the tensors
     * it reads.
     * @brief Backward trampoline for mean.
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1.
     */
    void mean_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        auto scaled = output_grad->div(static_cast<float64_t>(x.numel));
        auto ones = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        ones.set_one();
        auto dx = ones.mul(scaled);
        dx.sync();
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
        }
        x.accumulate_grad(std::move(dx));
    }
}
