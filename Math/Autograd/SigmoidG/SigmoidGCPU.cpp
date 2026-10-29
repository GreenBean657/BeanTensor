/**
 * @file SigmoidGCPU.cpp
 * @author GreenBean
 * @brief Backward trampoline for sigmoid: recompute s = sigmoid(x) and
 * deliver dx = dy * s * (1-s).
 * @version 0.1
 * @date 2026-10-28
 */
#include <optional>

#include "Helpers/Errors.h"
#include "Math/Autograd/ElementwiseG/ElementwiseG.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/SigmoidG/SigmoidG.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace elementwise_shared;

    void sigmoid_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        const auto work = is_float_kind(grad_dtype)
            ? grad_dtype
            : DataTypes::DType::Float32;
        const auto xw = ew_operand_copy(x, work);
        auto dy_cast = std::optional<Tensor>{};
        const Tensor* dyw = output_grad;
        if (output_grad->dtype != work) {
            dy_cast = grad_converted_copy(*output_grad, work);
            dyw = &*dy_cast;
        }
        auto s = xw.sigmoid();
        auto dx = dyw->mul(s).mul(s.mul(float64_t{-1.0}).add(float64_t{1.0}));
        ew_deliver_delta(std::move(dx), x, work, grad_dtype);
    }
}
