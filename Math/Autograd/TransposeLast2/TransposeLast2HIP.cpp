/**
 * @file TransposeLast2HIP.cpp
 * @author GreenBean
 * @brief HIP backend for the transpose_last2 op (GEMM_G operand
 * materialization; dtype-agnostic last-two-dims transpose).
 * Divergent twin of Math/Autograd/TransposeLast2/TransposeLast2CUDA.cu — the
 * two backends are written separately per project rule (no HIP-only
 * unification), but must stay behaviorally identical.
 * @version 0.1
 * @date 2026-10-20
 */
#if defined(USE_HIP)
#include <cstdint>
#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Math/Autograd/AutogradHipShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
        /**
         * Dtype-agnostic last-two-dims transpose: element bits are moved as an
         * unsigned word of the dtype's width (E), so no value is interpreted.
         * Launched as grid_for(numel) blocks of BT_GPU_THREADS threads, grid-stride
         * over the flat input index.
         * @brief Batched transpose kernel (out[b, j, i] = in[b, i, j]).
         * @param in Input data, batch slabs of rows x cols, numel elements.
         * @param out Output data, batch slabs of cols x rows, numel elements.
         * @param rows Second-to-last dimension of in.
         * @param cols Last dimension of in.
         * @param numel Total element count (batch * rows * cols).
         */
        template<typename E>
        __global__ void transpose_kernel(const E* __restrict__ in, E* __restrict__ out,
                                         const size_t rows, const size_t cols, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            const auto slab = rows * cols;
            for (size_t idx = tid; idx < numel; idx += stride) {
                const auto batch = idx / slab;
                const auto rem = idx - batch * slab;
                const auto i = rem / cols;
                const auto j = rem - i * cols;
                out[batch * slab + j * rows + i] = in[idx];
            }
        }

        /**
         * Skips the launch when numel is 0. The launch is asynchronous; the caller syncs.
         * @brief Host-side transpose launch (E = unsigned word of the element width).
         * @param in Input data (type-erased).
         * @param out Output data (type-erased).
         * @param rows Second-to-last dimension of in.
         * @param cols Last dimension of in.
         * @param numel Total element count.
         */
        template<typename E>
        __host__ void run_transpose(const void* in, void* out, const size_t rows, const size_t cols, const size_t numel) {
            if (numel > 0) {
                transpose_kernel<E><<<grid_for(numel), BT_GPU_THREADS>>>(static_cast<const E*>(in), static_cast<E*>(out), rows, cols, numel);
            }
        }
    }

    /**
     * Dtype-agnostic: bits are moved as an unsigned word of the element byte width
     * (1, 2, 4 or 8), so no value is interpreted. Batched over all leading dimensions.
     * Queues the launch on the CPU thread pool (High priority) and returns; the shared
     * future is given to both tensors, and errors surface at sync().
     * @brief GPU transpose of the last two dimensions (out[..., j, i] = t[..., i, j]).
     * @param t Input tensor.
     * @param out Output tensor with the last two dimensions swapped.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if t or out is not on the GPU.
     * @throws ErrorHandling::UnReachableCode (from the worker, at sync) for an element width other than 1, 2, 4 or 8 bytes.
     */
    __host__ void transpose_last2_gpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto eb = DataTypes::dtype_bytes(t.dtype);
        const auto* const in_data = static_cast<const uint8_t*>(t.data) + t.offset * eb;
        auto* const out_data = static_cast<uint8_t*>(out.data) + out.offset * eb;
        const auto rows = t.shape[t.ndim - 2];
        const auto cols = t.shape[t.ndim - 1];
        const auto numel = t.numel;
        /**
         * Captures raw pointers, element byte width, rows, cols and numel by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on the element byte width (1, 2, 4 or 8) to run_transpose over the matching unsigned word type, then calls hipStreamSynchronize(nullptr) so the future completes only once out is written.
         * Any other byte width throws ErrorHandling::UnReachableCode.
         * Runs on a CPU thread pool thread (High priority) via the async pool; the host call has already returned. Any exception thrown here (including a failed HIP_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Async pool worker that launches the transpose kernel for the element width and syncs the stream.
         */
        const auto worker = [=] {
            switch (eb) {
                case 1: run_transpose<uint8_t>(in_data, out_data, rows, cols, numel); break;
                case 2: run_transpose<uint16_t>(in_data, out_data, rows, cols, numel); break;
                case 4: run_transpose<uint32_t>(in_data, out_data, rows, cols, numel); break;
                case 8: run_transpose<uint64_t>(in_data, out_data, rows, cols, numel); break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        out.give_future(future);
    }
}
#endif
