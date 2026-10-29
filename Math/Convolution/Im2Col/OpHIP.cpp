/**
 * @file OpHIP.cpp
 * @author GreenBean
 * @brief HIP backend for im2col.
 * @version 0.1
 * @date 2026-10-28
 */
#if defined(USE_HIP)
#include "Math/Convolution/Im2Col/Op.h"
#include "Math/Convolution/ConvolutionHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace convolution_gpu;

    __host__ void im2col_gpu(const Tensor& x, const Tensor& col, const uint32_t stride, const uint32_t padding,
                             const size_t R, const size_t S) {
        if (x.device != Device::GPU || col.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        const auto s = make_shape(std::vector<size_t>(x.shape, x.shape + x.ndim), std::vector<size_t>{x.shape[1], R, S, size_t{1}}, x.dtype, stride, padding);
        const auto eb = DataTypes::dtype_bytes(x.dtype);
        const auto* x_data = static_cast<const uint8_t*>(x.data) + x.offset * eb;
        auto* col_data = static_cast<uint8_t*>(col.data) + col.offset * eb;

        const auto worker = [=] {
            const auto total = s.M * s.CRS;
            if (total > 0) {
                im2col_kernel<<<grid_for(total), BT_GPU_THREADS>>>(x_data, col_data, s, stride, padding, eb);
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        x.give_future(future);
        col.give_future(future);
    }
}
#endif
