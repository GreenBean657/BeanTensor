/**
 * @file Conv2dGCPU.cpp
 * @author GreenBean
 * @brief Backward trampoline for Conv2d. Recomputes im2col(x), materializes
 * dyT {M,K}, then uses stock GEMM helpers and transpose_last2 for dW, db, dx.
 * @version 0.2
 * @date 2026-10-28
 */
#include "Math/Convolution/Conv2dG.h"
#include "Math/Convolution/Convolution.h"
#include "Math/Convolution/ConvolutionShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Math/LinearAlgebra/MatMul/Matmul.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {

    namespace {
        inline uint64_t pack_stride_padding(const uint32_t stride, const uint32_t padding) {
            return (static_cast<uint64_t>(stride) << 32) | static_cast<uint64_t>(padding);
        }

        inline std::pair<uint32_t, uint32_t> unpack_stride_padding(const float64_t scalar) {
            const auto packed = Intrinsics::detail::bit_cast_memcpy<uint64_t>(scalar);
            return {static_cast<uint32_t>(packed >> 32), static_cast<uint32_t>(packed & 0xFFFFFFFFu)};
        }
    }

    void conv2d_g_node(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 2 && n_inputs != 3) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        auto& weight = *inputs[1];
        Tensor* bias = (n_inputs == 3) ? inputs[2] : nullptr;

        const auto [stride, padding] = unpack_stride_padding(node->_scalar);
        const auto x_shape = std::vector<size_t>(x.shape, x.shape + x.ndim);
        const auto w_shape = std::vector<size_t>(weight.shape, weight.shape + weight.ndim);
        const auto s = make_conv2d_shape(x_shape, w_shape, stride, padding);

        const auto deps = output_grad->pending_tasks;
        Tensor* const x_ptr = &x;
        Tensor* const weight_ptr = &weight;

        const auto worker = [node, output_grad, x_ptr, weight_ptr, bias, grad_dtype, deps,
                             x_shape, w_shape, stride, padding, s] {
            auto& x = *x_ptr;
            auto& weight = *weight_ptr;
            auto& threadMgr = Threading::GetCPUThreadPool();
            threadMgr.HelpWait(deps);

            if (x.device == Device::GPU) {
#if defined(USE_CUDA)
                Tensor* local_inputs[3] = {&x, &weight, bias};
                conv2d_g_node_gpu(node, output_grad, local_inputs, bias != nullptr ? 3u : 2u, grad_dtype);
                return;
#elif defined(USE_HIP)
                Tensor* local_inputs[3] = {&x, &weight, bias};
                conv2d_g_node_gpu(node, output_grad, local_inputs, bias != nullptr ? 3u : 2u, grad_dtype);
                return;
#else
                throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(Device::GPU));
#endif
            }

            auto col = Tensor(std::vector<size_t>{s.M, s.CRS}, x.dtype, x.device, false);
            im2col_cpu(x, col, stride, padding, s.R, s.S);

            auto dyT = Tensor(std::vector<size_t>{s.M, s.K}, output_grad->dtype, output_grad->device, false);
            nchw_to_mk_kernel(static_cast<const uint8_t*>(output_grad->data) + output_grad->offset * DataTypes::dtype_bytes(output_grad->dtype),
                              static_cast<uint8_t*>(dyT.data) + dyT.offset * DataTypes::dtype_bytes(dyT.dtype),
                              s, static_cast<size_t>(DataTypes::dtype_bytes(dyT.dtype)));

            if (weight.requires_grad) {
                auto colT_shape = std::vector<size_t>{s.CRS, s.M};
                auto colT = Tensor(colT_shape, col.dtype, col.device, false);
                if (col.device == Device::CPU) {
                    transpose_last2_cpu(col, colT);
                } else if (col.device == Device::GPU) {
                    transpose_last2_gpu(col, colT);
                } else [[unlikely]] {
                    throw ErrorHandling::UnReachableCode();
                }
                grad_convert_inplace(colT, grad_dtype);
                grad_convert_inplace(dyT, grad_dtype);
                auto dW2d = Tensor(std::vector<size_t>{s.CRS, s.K}, grad_dtype, x.device, false);
                if (x.device == Device::CPU) {
                    matmul_cpu(colT, dyT, dW2d);
                } else if (x.device == Device::GPU) {
                    matmul_gpu(colT, dyT, dW2d);
                } else [[unlikely]] {
                    throw ErrorHandling::UnReachableCode();
                }
                auto dW = dW2d.reshape(std::vector<size_t>{s.C, s.R, s.S, s.K});
                if (weight.node != nullptr) {
                    grad_probe_hit(dW.dtype);
                }
                weight.accumulate_grad(std::move(dW));
            }

            if (bias != nullptr && bias->requires_grad) {
                grad_convert_inplace(dyT, grad_dtype);
                auto db = Tensor(std::vector<size_t>{s.K}, grad_dtype, x.device, false);
                rowsum_mk_dispatch(dyT, db);
                if (bias->node != nullptr) {
                    grad_probe_hit(db.dtype);
                }
                bias->accumulate_grad(std::move(db));
            }

            if (x.requires_grad) {
                auto weight2d = [&] {
                    Tensor t(std::vector<size_t>{s.CRS, s.K}, weight.dtype, weight.device, false);
                    if (t.device == Device::CPU) {
                        t.kill_cpu_data();
                    } else if (t.device == Device::GPU) {
                        t.kill_gpu_data();
                    }
                    t.data = weight.data;
                    t.offset = weight.offset;
                    t.owns_data = false;
                    return t;
                }();
                auto weight2dT_shape = std::vector<size_t>{s.K, s.CRS};
                auto weight2dT = Tensor(weight2dT_shape, weight2d.dtype, weight2d.device, false);
                if (weight2d.device == Device::CPU) {
                    transpose_last2_cpu(weight2d, weight2dT);
                } else if (weight2d.device == Device::GPU) {
                    transpose_last2_gpu(weight2d, weight2dT);
                } else [[unlikely]] {
                    throw ErrorHandling::UnReachableCode();
                }
                grad_convert_inplace(weight2dT, grad_dtype);
                grad_convert_inplace(dyT, grad_dtype);
                auto dcol = Tensor(std::vector<size_t>{s.M, s.CRS}, grad_dtype, x.device, false);
                if (x.device == Device::CPU) {
                    matmul_cpu(dyT, weight2dT, dcol);
                } else if (x.device == Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
                    matmul_gpu(dyT, weight2dT, dcol);
#else
                    throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
                } else [[unlikely]] {
                    throw ErrorHandling::UnReachableCode();
                }
                auto dx = Tensor(x_shape, grad_dtype, x.device, false);
                if (x.device == Device::CPU) {
                    col2im_cpu(dcol, dx, stride, padding, s.R, s.S);
                } else if (x.device == Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
                    col2im_gpu(dcol, dx, stride, padding, s.R, s.S);
#else
                    throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
                } else [[unlikely]] {
                    throw ErrorHandling::UnReachableCode();
                }
                if (x.node != nullptr) {
                    grad_probe_hit(dx.dtype);
                }
                x.accumulate_grad(std::move(dx));
            }
        };

        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        x.give_future(future);
        weight.give_future(future);
        output_grad->give_future(future);
        if (bias != nullptr) {
            bias->give_future(future);
        }
    }

    float64_t conv2d_pack_stride_padding(const uint32_t stride, const uint32_t padding) {
        return Intrinsics::detail::bit_cast_memcpy<float64_t>(pack_stride_padding(stride, padding));
    }
}
