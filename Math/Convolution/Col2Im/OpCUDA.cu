/**
 * @file OpCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for col2im.
 * @version 0.1
 * @date 2026-10-28
 */
#if defined(USE_CUDA)
#include "Math/Convolution/Col2Im/Op.h"
#include "Math/Convolution/ConvolutionCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace convolution_gpu;

    __host__ void col2im_gpu(const Tensor& col, const Tensor& dx, const uint32_t stride, const uint32_t padding,
                             const size_t R, const size_t S) {
        if (col.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(col.device));
        }
        const auto dx_shape = std::vector<size_t>(dx.shape, dx.shape + dx.ndim);
        const auto s = make_shape(dx_shape, std::vector<size_t>{dx_shape[1], R, S, size_t{1}}, dx.dtype, stride, padding);
        const auto* col_data = static_cast<const uint8_t*>(col.data) + col.offset * DataTypes::dtype_bytes(col.dtype);
        auto* dx_data = static_cast<uint8_t*>(dx.data) + dx.offset * DataTypes::dtype_bytes(dx.dtype);
        CUDA_CHECK_ERROR(cudaMemset(dx_data, 0, dx.numel * DataTypes::dtype_bytes(dx.dtype)));

        const auto worker = [=] {
            launch_col2im(col_data, dx_data, s, stride, padding);
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        col.give_future(future);
        dx.give_future(future);
    }
}
#endif
