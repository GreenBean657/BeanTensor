/**
 * @file OpCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for fused Conv2d forward.
 * @version 0.3
 * @date 2026-10-28
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <limits>
#include <vector>

#include "Math/Convolution/Conv2d/Op.h"
#include "Math/Convolution/ConvolutionCudaShared.h"
#include "Math/LinearAlgebra/MatMul/Matmul.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace convolution_gpu;

    namespace {
        __host__ void launch_bias_add_nchw(const uint8_t* b_data, uint8_t* o_data, const convolution_gpu::Conv2dShape& s) {
            const auto total = s.N * s.K * s.Hout * s.Wout;
            if (total == 0) return;
            using DType = DataTypes::DType;
            switch (s.dtype) {
                case DType::Float32: bias_add_nchw_kernel<float32_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const float32_t*>(b_data), reinterpret_cast<float32_t*>(o_data), s); break;
                case DType::Float64: bias_add_nchw_kernel<float64_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const float64_t*>(b_data), reinterpret_cast<float64_t*>(o_data), s); break;
                case DType::Float16: bias_add_nchw_kernel<float16_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const float16_t*>(b_data), reinterpret_cast<float16_t*>(o_data), s); break;
                case DType::BFloat16: bias_add_nchw_kernel<bfloat16_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const bfloat16_t*>(b_data), reinterpret_cast<bfloat16_t*>(o_data), s); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        }
    }

    __host__ void conv2d_gpu(const Tensor& x, const Tensor& weight, const Tensor* bias, const Tensor& out,
                             const uint32_t stride, const uint32_t padding) {
        if (x.device != Device::GPU || weight.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        if (bias != nullptr && bias->device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(bias->device));
        }
        const auto x_shape = std::vector<size_t>(x.shape, x.shape + x.ndim);
        const auto w_shape = std::vector<size_t>(weight.shape, weight.shape + weight.ndim);
        const auto s = convolution_gpu::make_shape(x_shape, w_shape, x.dtype, stride, padding);

        const auto eb = DataTypes::dtype_bytes(x.dtype);

        const auto* const x_data = static_cast<const uint8_t*>(x.data);
        const auto* const weight_data = static_cast<const uint8_t*>(weight.data);
        auto* const out_data = static_cast<uint8_t*>(out.data);
        const auto x_offset = x.offset;
        const auto weight_offset = weight.offset;
        const auto out_offset = out.offset;
        const auto has_bias = bias != nullptr;
        const uint8_t* bias_data = nullptr;
        size_t bias_offset = 0;
        if (has_bias) {
            bias_data = static_cast<const uint8_t*>(bias->data);
            bias_offset = bias->offset;
        }

        const auto worker = [x_data, x_offset, weight_data, weight_offset, out_data, out_offset,
                             bias_data, bias_offset, has_bias, s, eb, stride, padding] {
            auto col = Tensor(std::vector<size_t>{s.M, s.CRS}, s.dtype, Device::GPU, false);
            auto gemm_out = Tensor(std::vector<size_t>{s.M, s.K}, s.dtype, Device::GPU, false);

            const auto* x_ptr = x_data + x_offset * eb;
            auto* col_ptr = static_cast<uint8_t*>(col.data) + col.offset * eb;
            const auto total_col = s.M * s.CRS;
            if (total_col > 0) {
                im2col_kernel<<<grid_for(total_col), BT_GPU_THREADS>>>(x_ptr, col_ptr, s, stride, padding, eb);
            }

            auto weight2d = [&] {
                Tensor t(std::vector<size_t>{s.CRS, s.K}, s.dtype, Device::GPU, false);
                t.kill_gpu_data();
                t.data = const_cast<uint8_t*>(weight_data);
                t.offset = weight_offset;
                t.owns_data = false;
                return t;
            }();

            matmul_gpu(col, weight2d, gemm_out);

            const auto gemm_deps = gemm_out.pending_tasks;
            auto& threadMgr = Threading::GetCPUThreadPool();
            threadMgr.HelpWait(gemm_deps);

            const auto* gemm_src = static_cast<const uint8_t*>(gemm_out.data) + gemm_out.offset * eb;
            auto* out_ptr = out_data + out_offset * eb;
            const auto total_out = s.N * s.K * s.Hout * s.Wout;
            if (total_out > 0) {
                mk_to_nchw_kernel<<<grid_for(total_out), BT_GPU_THREADS>>>(gemm_src, out_ptr, s, eb);
            }

            if (has_bias) {
                const auto* b_data = bias_data + bias_offset * eb;
                launch_bias_add_nchw(b_data, out_ptr, s);
            }

            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };

        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        x.give_future(future);
        weight.give_future(future);
        out.give_future(future);
        if (bias != nullptr) {
            bias->give_future(future);
        }
    }
}
#endif
