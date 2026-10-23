/**
 * @file ElementwiseGCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the elementwise backward broadcast gradient reduction
 * (bcast_sum_to_gpu): one thread per operand element enumerates the output
 * indices reducing into it (the device twin of the host for_each_preimage —
 * the two must stay behaviorally identical) and sums them in the dtype
 * family's contract width. Divergent twin of
 * Math/Autograd/ElementwiseG/ElementwiseGHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must
 * stay behaviorally identical: same fp32/fp64 compute widths, same CPU-unaccel
 * RNE narrowing twins, same wrap integer adds, same fixed reduction order.
 * @version 0.1
 * @date 2026-10-23
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Autograd/ElementwiseG/ElementwiseG.h"
#include "Math/Autograd/ElementwiseG/ElementwiseGShared.h"
#include "Math/Autograd/AutogradCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
        /**
         * Device twin of the host for_each_preimage in ElementwiseGShared.h:
         * invokes f(i) for every output flat index i whose broadcast mapping
         * lands on operand element j, in the same fixed mixed-radix order.
         * @brief Enumerate the output indices reducing into operand element j.
         * @param plan Broadcast-reduction plan.
         * @param j Operand flat index.
         * @param f Callable invoked with each output flat index.
         */
        template<typename F>
        __device__ __forceinline__ void dev_for_each_preimage(const BcastReducePlan& p, const size_t j, F f) {
            size_t op_strides_c[3] = {1, 1, 1};
            for (size_t od = p.op_ndim - 1; od-- > 0;) {
                op_strides_c[od] = op_strides_c[od + 1] * p.op_shape[od + 1];
            }
            size_t coord[3] = {0, 0, 0};
            auto rem = j;
            for (auto od = p.op_ndim; od-- > 0;) {
                coord[od] = rem / op_strides_c[od];
                rem -= coord[od] * op_strides_c[od];
            }
            size_t free_dim[3] = {0, 0, 0};
            size_t free_ext[3] = {1, 1, 1};
            auto k = size_t{0};
            auto base = size_t{0};
            const auto lead = p.out_ndim - p.op_ndim;
            for (auto d = size_t{0}; d < p.out_ndim; ++d) {
                if (d < lead) {
                    free_dim[k] = d;
                    free_ext[k] = p.out_shape[d];
                    ++k;
                } else {
                    const auto od = d - lead;
                    if (p.op_shape[od] == 1) {
                        free_dim[k] = d;
                        free_ext[k] = p.out_shape[d];
                        ++k;
                    } else {
                        base += coord[od] * p.out_strides[d];
                    }
                }
            }
            size_t div[3] = {1, 1, 1};
            for (auto m = k; m-- > 0;) {
                if (m + 1 < k) {
                    div[m] = div[m + 1] * free_ext[m + 1];
                }
            }
            auto total = size_t{1};
            for (auto m = size_t{0}; m < k; ++m) {
                total *= free_ext[m];
            }
            for (auto t = size_t{0}; t < total; ++t) {
                auto r = t;
                auto idx = base;
                for (auto m = size_t{0}; m < k; ++m) {
                    const auto cc = r / div[m];
                    r -= cc * div[m];
                    idx += cc * p.out_strides[free_dim[m]];
                }
                f(idx);
            }
        }

        /**
         * Sums delta over the preimage of operand element j and stores the
         * result with the dtype family's contract: floats in the storage
         * width (fixed enumeration order — bit-identical to the CPU scalar
         * path); fp16/bf16 in fp32, narrowed RNE once; packed dtypes per lane
         * in fp32; integers wrap through the unsigned twin; int4x2 wraps per
         * nibble.
         * @brief Reduce delta into one operand element (device).
         * @param delta Out-shaped delta elements.
         * @param plan Broadcast-reduction plan.
         * @param j Operand flat index.
         * @return The reduced gradient element.
         */
        template<typename T>
        __device__ __forceinline__ T dev_bcast_sum_elem(const T* delta, const BcastReducePlan& p, const size_t j) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                auto acc = T{0};
                dev_for_each_preimage(p, j, [&](const size_t idx) { acc += delta[idx]; });
                return acc;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                auto acc = float32_t{0.0f};
                dev_for_each_preimage(p, j, [&](const size_t idx) { acc += __half2float(delta[idx]); });
                return __float2half_rn(acc);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                auto acc = float32_t{0.0f};
                dev_for_each_preimage(p, j, [&](const size_t idx) { acc += __bfloat162float(delta[idx]); });
                return dev_narrow_bf16(acc);
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                auto lo = float32_t{0.0f};
                auto hi = float32_t{0.0f};
                dev_for_each_preimage(p, j, [&](const size_t idx) {
                    const auto v = delta[idx];
                    lo += __half2float(__low2half(v));
                    hi += __half2float(__high2half(v));
                });
                return __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                auto lo = float32_t{0.0f};
                auto hi = float32_t{0.0f};
                dev_for_each_preimage(p, j, [&](const size_t idx) {
                    const auto v = delta[idx];
                    lo += __bfloat162float(__low2bfloat16(v));
                    hi += __bfloat162float(__high2bfloat16(v));
                });
                return __halves2bfloat162(dev_narrow_bf16(lo), dev_narrow_bf16(hi));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                auto lo = int32_t{0};
                auto hi = int32_t{0};
                dev_for_each_preimage(p, j, [&](const size_t idx) {
                    lo += dev_int4_lo(delta[idx]);
                    hi += dev_int4_hi(delta[idx]);
                });
                return dev_pack_int4(lo, hi);
            } else {
                using U = std::make_unsigned_t<T>;
                auto acc = U{0};
                dev_for_each_preimage(p, j, [&](const size_t idx) { acc = static_cast<U>(acc + static_cast<U>(delta[idx])); });
                return static_cast<T>(acc);
            }
        }

        /**
         * One thread per operand element (each is written by exactly one
         * thread, so no atomics are needed); grid-stride over op_numel.
         * @brief Broadcast gradient reduction kernel (all dtypes).
         * @param dx Output gradient elements, op_numel of them (written).
         * @param delta Out-shaped delta elements.
         * @param plan Broadcast-reduction plan (trivially copyable).
         * @param op_numel Operand element count (storage elements, so packed dtypes count pairs).
         */
        template<typename T>
        __global__ void bcast_sum_to_kernel(T* __restrict__ dx, const T* __restrict__ delta, const BcastReducePlan plan, const size_t op_numel) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < op_numel; j += stride) {
                dx[j] = dev_bcast_sum_elem<T>(delta, plan, j);
            }
        }

        /**
         * Skips the launch when op_numel is 0; the kernel is asynchronous, the caller syncs.
         * @brief Host-side broadcast gradient reduction launch.
         * @param dx Output base pointer, offset already applied.
         * @param delta Delta base pointer, offset already applied.
         * @param op_numel Operand element count.
         * @param plan Broadcast-reduction plan.
         */
        template<typename T>
        __host__ void run_bcast_sum_to(T* dx, const T* delta, const size_t op_numel, const BcastReducePlan& plan) {
            if (op_numel > 0) {
                bcast_sum_to_kernel<T><<<grid_for(op_numel), BT_GPU_THREADS>>>(dx, delta, plan, op_numel);
            }
        }
    }

    /**
     * Queues the launch on the CUDA stream via the CPU thread pool (High
     * priority) and returns; the shared future is given to both tensors, and
     * errors surface at sync(). The reduction plan is built on the host (same
     * builder the CPU backend uses) and passed to the kernel by value. One
     * thread per operand element reproduces the CPU scalar path's fixed
     * reduction order bit-exactly.
     * @brief GPU broadcast gradient reduction (all dtypes).
     * @param delta Out-shaped delta tensor, contiguous.
     * @param op The broadcast operand (provides the target shape/strides).
     * @param dx Output gradient tensor, op-shaped (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void bcast_sum_to_gpu(const Tensor& delta, const Tensor& op, Tensor& dx) {
        if (delta.device != Device::GPU || op.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(delta.device));
        }
        const auto* const delta_data = delta.data;
        auto* const dx_data = dx.data;
        const auto delta_offset = delta.offset;
        const auto dx_offset = dx.offset;
        const auto dtype = delta.dtype;
        const auto numel = dx.numel;
        const auto plan = delta.build_bcast_reduce_plan(op);
        /**
         * Captures raw pointers, offsets, dtype, numel and the plan by value, so the lambda stays valid after the enclosing call returns.
         * Dispatches on dtype to run_bcast_sum_to, which launches the broadcast gradient reduction kernel on the default stream, then calls cudaStreamSynchronize(nullptr) so the future completes only once dx is written.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * Runs on a CPU thread pool thread (High priority); the host call has already returned. Any exception thrown here (including a failed CUDA_CHECK_ERROR) is captured in the shared future and rethrown at sync().
         * @brief Pool worker that launches the bcast_sum_to kernel for the tensor dtype and syncs the stream.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32: run_bcast_sum_to<float32_t>(static_cast<float32_t*>(dx_data) + dx_offset, static_cast<const float32_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Float64: run_bcast_sum_to<float64_t>(static_cast<float64_t*>(dx_data) + dx_offset, static_cast<const float64_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Float16: run_bcast_sum_to<float16_t>(static_cast<float16_t*>(dx_data) + dx_offset, static_cast<const float16_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::BFloat16: run_bcast_sum_to<bfloat16_t>(static_cast<bfloat16_t*>(dx_data) + dx_offset, static_cast<const bfloat16_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Float16x2: run_bcast_sum_to<float16x2_t>(static_cast<float16x2_t*>(dx_data) + dx_offset, static_cast<const float16x2_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::BFloat16x2: run_bcast_sum_to<bfloat16x2_t>(static_cast<bfloat16x2_t*>(dx_data) + dx_offset, static_cast<const bfloat16x2_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Int8: run_bcast_sum_to<int8_t>(static_cast<int8_t*>(dx_data) + dx_offset, static_cast<const int8_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::UInt8: run_bcast_sum_to<uint8_t>(static_cast<uint8_t*>(dx_data) + dx_offset, static_cast<const uint8_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Int16: run_bcast_sum_to<int16_t>(static_cast<int16_t*>(dx_data) + dx_offset, static_cast<const int16_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::UInt16: run_bcast_sum_to<uint16_t>(static_cast<uint16_t*>(dx_data) + dx_offset, static_cast<const uint16_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Int32: run_bcast_sum_to<int32_t>(static_cast<int32_t*>(dx_data) + dx_offset, static_cast<const int32_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::UInt32: run_bcast_sum_to<uint32_t>(static_cast<uint32_t*>(dx_data) + dx_offset, static_cast<const uint32_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Int64: run_bcast_sum_to<int64_t>(static_cast<int64_t*>(dx_data) + dx_offset, static_cast<const int64_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::UInt64: run_bcast_sum_to<uint64_t>(static_cast<uint64_t*>(dx_data) + dx_offset, static_cast<const uint64_t*>(delta_data) + delta_offset, numel, plan); break;
                case DType::Int4x2: run_bcast_sum_to<int4x2_t>(static_cast<int4x2_t*>(dx_data) + dx_offset, static_cast<const int4x2_t*>(delta_data) + delta_offset, numel, plan); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        delta.give_future(future);
        dx.give_future(future);
    }
}
#endif
