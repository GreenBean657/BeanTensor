/**
 * @file TanhGCPU.cpp
 * @author GreenBean
 * @brief Backward trampoline for tanh: recompute t = tanh(x) and
 * deliver dx = dy * (1 - t*t).
 * @version 0.1
 * @date 2026-10-28
 */
#include <optional>

#include "Helpers/Errors.h"
#include "Math/Autograd/ElementwiseG/ElementwiseG.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/TanhG/TanhG.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace elementwise_shared;

    void tanh_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
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
        auto t = xw.tanh();
        // Named intermediates: graph ops on rvalues are deleted (the node
        // parents must outlive backward()).
        auto t2 = t.mul(t);
        auto neg = t2.mul(float64_t{-1.0});
        // tanh' = 1 - t^2: no extra factor of t.
        auto dx = dyw->mul(neg.add(float64_t{1.0}));
        ew_deliver_delta(std::move(dx), x, work, grad_dtype);
    }
}
