#pragma once
#include "Helper/Broadcast/Broadcast.h"
#include "Tensor/Tensor.h"
#include "Tensor/Dtypes.h"

using namespace BeanTensor;

namespace BeanTensor::Ops::detail {
    inline void add_kernel_launcher(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor) {
        //TODO: Return a Tensor
        if (BeanTensor::Casting::dtype_is_int(t1.dtype)) {

            return;
        }

#ifdef USE_NPU_STRIXHALO

#endif

    }

}



namespace BeanTensor::Ops {

    [[nodiscard]] inline Tensors::Tensor add(const Tensors::Tensor& t1, const Tensors::Tensor& t2) {

        assert(t1.device == t2.device);
        size_t out_shape[BT_TENSOR_MAX_DIMS]{};
        size_t out_ndims{0};
        const bool req_grad = (t1.requires_grad || t2.requires_grad);
        assert(Helpers::broadcast_compatible(t1.shape, t1.ndim, t2.shape,t2.ndim , out_shape, &out_ndims));

        Tensors::Tensor ret = Tensors::make_tensor(
         out_shape,
         out_ndims,
         Casting::DType::Float32,
         t1.device,
         req_grad
        );

        if (req_grad) {
            auto* node = new Tensors::Node();
            node->inputs[0] = const_cast<Tensors::Tensor*>(&t1);
            node->inputs[1] = const_cast<Tensors::Tensor*>(&t2);
            node->n_inputs = 2;
            node->backward_fn = [](Tensors::Tensor* output_grad, Tensors::Tensor** inputs, const size_t n_inputs) {
                for (size_t i = 0; i < n_inputs; ++i) {
                    if (!inputs[i]->requires_grad) continue;

                    if (inputs[i]->grad == nullptr) {
                        inputs[i]->grad = new Tensors::Tensor(Tensors::make_tensor(
                            inputs[i]->shape,
                            inputs[i]->ndim,
                            inputs[i]->dtype,
                            inputs[i]->device,
                            false
                        ));
                        // TODO: replace with contiguous recovery
                        assert(inputs[i]->is_contiguous && "std::memcpy attempted on contiguous memory buffer");
                        std::memcpy(inputs[i]->grad->data, output_grad->data, output_grad->nbytes);
                    } else {
                        // TODO: replace with _add_kernel_launcher
                        detail::add_kernel_launcher(*inputs[i], *output_grad, *inputs[i]->grad);
                    }
                }
            };
        }
        detail::add_kernel_launcher(t1, t2, ret);
        return ret;
    }

}

