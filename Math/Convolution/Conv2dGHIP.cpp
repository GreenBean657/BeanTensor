/**
 * @file Conv2dGHIP.cpp
 * @author GreenBean
 * @brief HIP implementation of the Conv2d backward trampoline.
 * @version 0.3
 * @date 2026-10-28
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <vector>

#include "Math/Convolution/Conv2dG.h"
#include "Math/Convolution/ConvolutionHipShared.h"
#include "Math/Convolution/Convolution.h"
#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Math/LinearAlgebra/MatMul/Matmul.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace convolution_gpu;
    using namespace autograd_shared;

    namespace {
        __host__ void launch_rowsum_mk(const uint8_t* src, uint8_t* out, const size_t M, const size_t K, const DataTypes::DType dtype) {
            if (K == 0) return;
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: rowsum_mk_kernel<float32_t><<<grid_for(K), BT_GPU_THREADS>>>(reinterpret_cast<const float32_t*>(src), reinterpret_cast<float32_t*>(out), M, K); break;
                case DType::Float64: rowsum_mk_kernel<float64_t><<<grid_for(K), BT_GPU_THREADS>>>(reinterpret_cast<const float64_t*>(src), reinterpret_cast<float64_t*>(out), M, K); break;
                case DType::Float16: rowsum_mk_kernel<float16_t><<<grid_for(K), BT_GPU_THREADS>>>(reinterpret_cast<const float16_t*>(src), reinterpret_cast<float16_t*>(out), M, K); break;
                case DType::BFloat16: rowsum_mk_kernel<bfloat16_t><<<grid_for(K), BT_GPU_THREADS>>>(reinterpret_cast<const bfloat16_t*>(src), reinterpret_cast<bfloat16_t*>(out), M, K); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }
    }

    __host__ void conv2d_g_node_gpu(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 2 && n_inputs != 3) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        auto& weight = *inputs[1];
        Tensor* bias = (n_inputs == 3) ? inputs[2] : nullptr;

        const auto packed = Intrinsics::detail::bit_cast_memcpy<uint64_t>(node->_scalar);
        const auto stride = static_cast<uint32_t>(packed >> 32);
        const auto padding = static_cast<uint32_t>(packed & 0xFFFFFFFFu);

        const auto x_shape = std::vector<size_t>(x.shape, x.shape + x.ndim);
        const auto w_shape = std::vector<size_t>(weight.shape, weight.shape + weight.ndim);
        const auto s = convolution_gpu::make_shape(x_shape, w_shape, x.dtype, stride, padding);
        const auto eb = DataTypes::dtype_bytes(x.dtype);

        auto col = Tensor(std::vector<size_t>{s.M, s.CRS}, x.dtype, Device::GPU, false);
        auto dyT = Tensor(std::vector<size_t>{s.M, s.K}, output_grad->dtype, Device::GPU, false);

        const auto* x_ptr = static_cast<const uint8_t*>(x.data) + x.offset * eb;
        auto* col_ptr = static_cast<uint8_t*>(col.data) + col.offset * eb;
        const auto* dy_ptr = static_cast<const uint8_t*>(output_grad->data) + output_grad->offset * DataTypes::dtype_bytes(output_grad->dtype);
        auto* dyT_ptr = static_cast<uint8_t*>(dyT.data) + dyT.offset * DataTypes::dtype_bytes(dyT.dtype);

        const auto total_col = s.M * s.CRS;
        if (total_col > 0) {
            im2col_kernel<<<grid_for(total_col), BT_GPU_THREADS>>>(x_ptr, col_ptr, s, stride, padding, eb);
        }
        const auto total_dy = s.N * s.K * s.Hout * s.Wout;
        if (total_dy > 0) {
            nchw_to_mk_kernel<<<grid_for(total_dy), BT_GPU_THREADS>>>(dy_ptr, dyT_ptr, s, DataTypes::dtype_bytes(dyT.dtype));
        }

        if (weight.requires_grad) {
            auto colT = Tensor(std::vector<size_t>{s.CRS, s.M}, col.dtype, Device::GPU, false);
            transpose_last2_gpu(col, colT);
            grad_convert_inplace(colT, grad_dtype);
            grad_convert_inplace(dyT, grad_dtype);
            auto dW2d = Tensor(std::vector<size_t>{s.CRS, s.K}, grad_dtype, Device::GPU, false);
            matmul_gpu(colT, dyT, dW2d);
            auto dW = dW2d.reshape(std::vector<size_t>{s.C, s.R, s.S, s.K});
            if (weight.node != nullptr) {
                grad_probe_hit(dW.dtype);
            }
            weight.accumulate_grad(std::move(dW));
        }

        if (bias != nullptr && bias->requires_grad) {
            grad_convert_inplace(dyT, grad_dtype);
            auto db = Tensor(std::vector<size_t>{s.K}, grad_dtype, Device::GPU, false);
            const auto* dyT_ptr2 = static_cast<const uint8_t*>(dyT.data) + dyT.offset * DataTypes::dtype_bytes(dyT.dtype);
            auto* db_ptr = static_cast<uint8_t*>(db.data) + db.offset * DataTypes::dtype_bytes(db.dtype);
            launch_rowsum_mk(dyT_ptr2, db_ptr, s.M, s.K, grad_dtype);
            if (bias->node != nullptr) {
                grad_probe_hit(db.dtype);
            }
            bias->accumulate_grad(std::move(db));
        }

        if (x.requires_grad) {
            auto weight2d = [&] {
                Tensor t(std::vector<size_t>{s.CRS, s.K}, weight.dtype, Device::GPU, false);
                t.kill_gpu_data();
                t.data = weight.data;
                t.offset = weight.offset;
                t.owns_data = false;
                return t;
            }();
            auto weight2dT = Tensor(std::vector<size_t>{s.K, s.CRS}, weight2d.dtype, Device::GPU, false);
            transpose_last2_gpu(weight2d, weight2dT);
            grad_convert_inplace(weight2dT, grad_dtype);
            grad_convert_inplace(dyT, grad_dtype);
            auto dcol = Tensor(std::vector<size_t>{s.M, s.CRS}, grad_dtype, Device::GPU, false);
            matmul_gpu(dyT, weight2dT, dcol);

            const auto dcol_deps = dcol.pending_tasks;
            auto& threadMgr = Threading::GetCPUThreadPool();
            threadMgr.HelpWait(dcol_deps);

            auto dx = Tensor(x_shape, grad_dtype, Device::GPU, false);
            col2im_gpu(dcol, dx, stride, padding, s.R, s.S);
            if (x.node != nullptr) {
                grad_probe_hit(dx.dtype);
            }
            x.accumulate_grad(std::move(dx));
        }
    }
}
#endif
