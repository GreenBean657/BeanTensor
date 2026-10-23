/**
 * @file Loss.cpp
 * @author GreenBean
 * @brief Definition of the BeanTensor::Loss::mse free function. Lives in its
 * own TU (like the Tensor method definitions live in the per-op CPU backends)
 * so Math/Loss/Loss.h only needs a forward declaration of Tensor and stays
 * includable from Tensor.h.
 * @version 0.1
 * @date 2026-10-23
 */
#include "Math/Loss/Loss.h"

#include "Tensor/Tensor.h"

namespace BeanTensor::Loss {

    /**
     * Thin forwarder to the Tensor method; the whole contract (shape/dtype/
     * device validation, backward node wiring, CPU/GPU dispatch) lives there.
     * @brief Mean squared error loss (free function form).
     * @param pred Predictions; the only operand that can receive a gradient.
     * @param target Targets; never receive a gradient.
     * @return Shape-{1} tensor holding the mean loss, in pred's dtype/device.
     */
    Tensors::Tensor mse(const Tensors::Tensor& pred, const Tensors::Tensor& target) {
        return pred.mse(target);
    }}
