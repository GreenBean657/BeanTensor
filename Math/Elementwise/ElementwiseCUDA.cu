/**
 * @file ElementwiseCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for elementwise arithmetic ops with broadcasting.
 * The kernels and device helpers mirror the CPU backend bit-exactly (AGENTS.md
 * parity invariant): same fp32/fp64 compute widths, same CPU-unaccel RNE
 * narrowing twins, same wrap integer ops, same div-by-zero/min rules.
 * Divergent twin of Math/Elementwise/ElementwiseHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical.
 * @version 0.1.1
 * @date 2026-10-22
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Elementwise/Elementwise.h"
#include "Math/Elementwise/ElementwiseCudaShared.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace elementwise_kernels;
    using namespace elementwise_shared;

    namespace {
        /**
         * Grid-stride over the broadcast output; per-element offsets come from
         * the device twin of the host bcast_offsets loop.
         * @brief Broadcast binary elementwise kernel (all dtypes).
         * @param out Output elements, plan.out_numel of them.
         * @param a Left operand elements.
         * @param b Right operand elements.
         * @param plan Broadcast execution plan (trivially copyable).
         */
        template<typename T, ElementwiseOp Op>
        __global__ void ew_binary_kernel(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                                         const ElementwiseBcast plan) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < plan.out_numel; i += stride) {
                size_t ao;
                size_t bo;
                dev_bcast_offsets(plan, i, ao, bo);
                out[i] = dev_apply_any<T, Op>(a[ao], b[bo]);
            }
        }

        /**
         * @brief Tensor-scalar elementwise kernel (all dtypes); the scalar was already narrowed on the host.
         * @param out Output elements, numel of them.
         * @param a Input elements.
         * @param scalar Scalar operand, already narrowed into T.
         * @param numel Total element count of in/out.
         */
        template<typename T, ElementwiseOp Op>
        __global__ void ew_scalar_kernel(T* __restrict__ out, const T* __restrict__ a, const T scalar, const size_t numel) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += stride) {
                out[i] = dev_apply_any<T, Op>(a[i], scalar);
            }
        }

        /**
         * @brief Unary elementwise kernel (float kinds); expf/logf mirror std::exp/std::log.
         * @param out Output elements, numel of them.
         * @param in Input elements.
         * @param numel Total element count of in/out.
         */
        template<typename T, ElementwiseUnaryOp Op>
        __global__ void ew_unary_kernel(T* __restrict__ out, const T* __restrict__ in, const size_t numel) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < numel; i += stride) {
                out[i] = dev_apply_any_unary<T, Op>(in[i]);
            }
        }

        /**
         * Skips the launch for empty outputs; the kernel is asynchronous, the caller syncs.
         * @brief Host-side broadcast binary launch.
         * @param out Output base pointer, offset already applied.
         * @param a Left base pointer, offset already applied.
         * @param b Right base pointer, offset already applied.
         * @param plan Broadcast execution plan.
         */
        template<typename T, ElementwiseOp Op>
        __host__ void run_ew_binary(T* out, const T* a, const T* b, const ElementwiseBcast& plan) {
            if (plan.out_numel > 0) {
                ew_binary_kernel<T, Op><<<grid_for(plan.out_numel), BT_GPU_THREADS>>>(out, a, b, plan);
            }
        }

        /**
         * The scalar is narrowed on the host through the SAME narrow_scalar helper the
         * CPU backend uses, so the PERMISSIVE convert contract is identical by construction.
         * @brief Host-side tensor-scalar launch.
         * @param out Output base pointer, offset already applied.
         * @param a Input base pointer, offset already applied.
         * @param scalar Scalar operand (float64_t).
         * @param numel Total element count of in/out.
         */
        template<typename T, ElementwiseOp Op>
        __host__ void run_ew_scalar(T* out, const T* a, const float64_t scalar, const size_t numel) {
            if (numel > 0) {
                ew_scalar_kernel<T, Op><<<grid_for(numel), BT_GPU_THREADS>>>(out, a, narrow_scalar<T>(scalar), numel);
            }
        }

        /**
         * Skips the launch for empty outputs; the kernel is asynchronous, the caller syncs.
         * @brief Host-side unary launch.
         * @param out Output base pointer, offset already applied.
         * @param in Input base pointer, offset already applied.
         * @param numel Total element count of in/out.
         */
        template<typename T, ElementwiseUnaryOp Op>
        __host__ void run_ew_unary(T* out, const T* in, const size_t numel) {
            if (numel > 0) {
                ew_unary_kernel<T, Op><<<grid_for(numel), BT_GPU_THREADS>>>(out, in, numel);
            }
        }
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the kernel. The future lands
     * on all three tensors so teardown drains it.
     * @brief GPU tensor-tensor elementwise: out = a (op) b with broadcasting.
     * @param a Left operand, GPU resident, contiguous.
     * @param b Right operand, GPU resident, contiguous.
     * @param out Output tensor, broadcast shape, a's dtype.
     * @param op Operation selector.
     * @param plan Broadcast execution plan built by the Tensor method.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void elementwise_gpu(const Tensor& a, const Tensor& b, const Tensor& out, const ElementwiseOp op, const ElementwiseBcast& plan) {
        if (a.device != Device::GPU || b.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(a.device));
        }
        const auto* const a_data = a.data;
        const auto* const b_data = b.data;
        auto* const out_data = out.data;
        const auto a_offset = a.offset;
        const auto b_offset = b.offset;
        const auto out_offset = out.offset;
        const auto dtype = a.dtype;
        // Raw pointers and the plan are captured by value: the future lands on
        // all three tensors, so buffer teardown always drains this worker first.
        /**
         * Dispatches on dtype to run_ew_binary, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the broadcast binary elementwise launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (op) {
                case ElementwiseOp::Add:
                    switch (dtype) {
                        case DType::Float32: run_ew_binary<float32_t, ElementwiseOp::Add>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, static_cast<const float32_t*>(b_data) + b_offset, plan); break;
                        case DType::Float64: run_ew_binary<float64_t, ElementwiseOp::Add>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, static_cast<const float64_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16: run_ew_binary<float16_t, ElementwiseOp::Add>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, static_cast<const float16_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16: run_ew_binary<bfloat16_t, ElementwiseOp::Add>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16x2: run_ew_binary<float16x2_t, ElementwiseOp::Add>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, static_cast<const float16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16x2: run_ew_binary<bfloat16x2_t, ElementwiseOp::Add>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::Int8: run_ew_binary<int8_t, ElementwiseOp::Add>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, static_cast<const int8_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt8: run_ew_binary<uint8_t, ElementwiseOp::Add>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, static_cast<const uint8_t*>(b_data) + b_offset, plan); break;
                        case DType::Int16: run_ew_binary<int16_t, ElementwiseOp::Add>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, static_cast<const int16_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt16: run_ew_binary<uint16_t, ElementwiseOp::Add>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, static_cast<const uint16_t*>(b_data) + b_offset, plan); break;
                        case DType::Int32: run_ew_binary<int32_t, ElementwiseOp::Add>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, static_cast<const int32_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt32: run_ew_binary<uint32_t, ElementwiseOp::Add>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, static_cast<const uint32_t*>(b_data) + b_offset, plan); break;
                        case DType::Int64: run_ew_binary<int64_t, ElementwiseOp::Add>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, static_cast<const int64_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt64: run_ew_binary<uint64_t, ElementwiseOp::Add>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, static_cast<const uint64_t*>(b_data) + b_offset, plan); break;
                        case DType::Int4x2: run_ew_binary<int4x2_t, ElementwiseOp::Add>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, static_cast<const int4x2_t*>(b_data) + b_offset, plan); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Sub:
                    switch (dtype) {
                        case DType::Float32: run_ew_binary<float32_t, ElementwiseOp::Sub>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, static_cast<const float32_t*>(b_data) + b_offset, plan); break;
                        case DType::Float64: run_ew_binary<float64_t, ElementwiseOp::Sub>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, static_cast<const float64_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16: run_ew_binary<float16_t, ElementwiseOp::Sub>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, static_cast<const float16_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16: run_ew_binary<bfloat16_t, ElementwiseOp::Sub>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16x2: run_ew_binary<float16x2_t, ElementwiseOp::Sub>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, static_cast<const float16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16x2: run_ew_binary<bfloat16x2_t, ElementwiseOp::Sub>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::Int8: run_ew_binary<int8_t, ElementwiseOp::Sub>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, static_cast<const int8_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt8: run_ew_binary<uint8_t, ElementwiseOp::Sub>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, static_cast<const uint8_t*>(b_data) + b_offset, plan); break;
                        case DType::Int16: run_ew_binary<int16_t, ElementwiseOp::Sub>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, static_cast<const int16_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt16: run_ew_binary<uint16_t, ElementwiseOp::Sub>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, static_cast<const uint16_t*>(b_data) + b_offset, plan); break;
                        case DType::Int32: run_ew_binary<int32_t, ElementwiseOp::Sub>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, static_cast<const int32_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt32: run_ew_binary<uint32_t, ElementwiseOp::Sub>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, static_cast<const uint32_t*>(b_data) + b_offset, plan); break;
                        case DType::Int64: run_ew_binary<int64_t, ElementwiseOp::Sub>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, static_cast<const int64_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt64: run_ew_binary<uint64_t, ElementwiseOp::Sub>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, static_cast<const uint64_t*>(b_data) + b_offset, plan); break;
                        case DType::Int4x2: run_ew_binary<int4x2_t, ElementwiseOp::Sub>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, static_cast<const int4x2_t*>(b_data) + b_offset, plan); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Mul:
                    switch (dtype) {
                        case DType::Float32: run_ew_binary<float32_t, ElementwiseOp::Mul>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, static_cast<const float32_t*>(b_data) + b_offset, plan); break;
                        case DType::Float64: run_ew_binary<float64_t, ElementwiseOp::Mul>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, static_cast<const float64_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16: run_ew_binary<float16_t, ElementwiseOp::Mul>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, static_cast<const float16_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16: run_ew_binary<bfloat16_t, ElementwiseOp::Mul>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16x2: run_ew_binary<float16x2_t, ElementwiseOp::Mul>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, static_cast<const float16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16x2: run_ew_binary<bfloat16x2_t, ElementwiseOp::Mul>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::Int8: run_ew_binary<int8_t, ElementwiseOp::Mul>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, static_cast<const int8_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt8: run_ew_binary<uint8_t, ElementwiseOp::Mul>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, static_cast<const uint8_t*>(b_data) + b_offset, plan); break;
                        case DType::Int16: run_ew_binary<int16_t, ElementwiseOp::Mul>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, static_cast<const int16_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt16: run_ew_binary<uint16_t, ElementwiseOp::Mul>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, static_cast<const uint16_t*>(b_data) + b_offset, plan); break;
                        case DType::Int32: run_ew_binary<int32_t, ElementwiseOp::Mul>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, static_cast<const int32_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt32: run_ew_binary<uint32_t, ElementwiseOp::Mul>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, static_cast<const uint32_t*>(b_data) + b_offset, plan); break;
                        case DType::Int64: run_ew_binary<int64_t, ElementwiseOp::Mul>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, static_cast<const int64_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt64: run_ew_binary<uint64_t, ElementwiseOp::Mul>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, static_cast<const uint64_t*>(b_data) + b_offset, plan); break;
                        case DType::Int4x2: run_ew_binary<int4x2_t, ElementwiseOp::Mul>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, static_cast<const int4x2_t*>(b_data) + b_offset, plan); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Div:
                    switch (dtype) {
                        case DType::Float32: run_ew_binary<float32_t, ElementwiseOp::Div>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, static_cast<const float32_t*>(b_data) + b_offset, plan); break;
                        case DType::Float64: run_ew_binary<float64_t, ElementwiseOp::Div>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, static_cast<const float64_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16: run_ew_binary<float16_t, ElementwiseOp::Div>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, static_cast<const float16_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16: run_ew_binary<bfloat16_t, ElementwiseOp::Div>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16x2: run_ew_binary<float16x2_t, ElementwiseOp::Div>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, static_cast<const float16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16x2: run_ew_binary<bfloat16x2_t, ElementwiseOp::Div>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::Int8: run_ew_binary<int8_t, ElementwiseOp::Div>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, static_cast<const int8_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt8: run_ew_binary<uint8_t, ElementwiseOp::Div>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, static_cast<const uint8_t*>(b_data) + b_offset, plan); break;
                        case DType::Int16: run_ew_binary<int16_t, ElementwiseOp::Div>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, static_cast<const int16_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt16: run_ew_binary<uint16_t, ElementwiseOp::Div>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, static_cast<const uint16_t*>(b_data) + b_offset, plan); break;
                        case DType::Int32: run_ew_binary<int32_t, ElementwiseOp::Div>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, static_cast<const int32_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt32: run_ew_binary<uint32_t, ElementwiseOp::Div>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, static_cast<const uint32_t*>(b_data) + b_offset, plan); break;
                        case DType::Int64: run_ew_binary<int64_t, ElementwiseOp::Div>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, static_cast<const int64_t*>(b_data) + b_offset, plan); break;
                        case DType::UInt64: run_ew_binary<uint64_t, ElementwiseOp::Div>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, static_cast<const uint64_t*>(b_data) + b_offset, plan); break;
                        case DType::Int4x2: run_ew_binary<int4x2_t, ElementwiseOp::Div>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, static_cast<const int4x2_t*>(b_data) + b_offset, plan); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Pow:
                    switch (dtype) {
                        case DType::Float32: run_ew_binary<float32_t, ElementwiseOp::Pow>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, static_cast<const float32_t*>(b_data) + b_offset, plan); break;
                        case DType::Float64: run_ew_binary<float64_t, ElementwiseOp::Pow>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, static_cast<const float64_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16: run_ew_binary<float16_t, ElementwiseOp::Pow>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, static_cast<const float16_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16: run_ew_binary<bfloat16_t, ElementwiseOp::Pow>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, static_cast<const bfloat16_t*>(b_data) + b_offset, plan); break;
                        case DType::Float16x2: run_ew_binary<float16x2_t, ElementwiseOp::Pow>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, static_cast<const float16x2_t*>(b_data) + b_offset, plan); break;
                        case DType::BFloat16x2: run_ew_binary<bfloat16x2_t, ElementwiseOp::Pow>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, static_cast<const bfloat16x2_t*>(b_data) + b_offset, plan); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        a.give_future(future);
        b.give_future(future);
        out.give_future(future);
    }

    /**
     * Queues one pool worker that narrows the scalar into a's dtype (the same
     * host helper the CPU backend uses) and launches the kernel. The future
     * lands on both tensors so teardown drains it.
     * @brief GPU tensor-scalar elementwise: out[i] = a[i] (op) scalar.
     * @param a Input tensor, GPU resident, contiguous.
     * @param scalar Scalar operand (float64_t, narrowed into a's dtype).
     * @param out Output tensor, same shape as a, a's dtype.
     * @param op Operation selector.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void elementwise_scalar_gpu(const Tensor& a, const float64_t& scalar, const Tensor& out, const ElementwiseOp op) {
        if (a.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(a.device));
        }
        const auto* const a_data = a.data;
        auto* const out_data = out.data;
        const auto a_offset = a.offset;
        const auto out_offset = out.offset;
        const auto dtype = a.dtype;
        const auto numel = a.numel;
        // Raw pointers are captured by value: the future lands on both
        // tensors, so buffer teardown always drains this worker first.
        /**
         * Dispatches on dtype to run_ew_scalar, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the tensor-scalar elementwise launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (op) {
                case ElementwiseOp::Add:
                    switch (dtype) {
                        case DType::Float32: run_ew_scalar<float32_t, ElementwiseOp::Add>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float64: run_ew_scalar<float64_t, ElementwiseOp::Add>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16: run_ew_scalar<float16_t, ElementwiseOp::Add>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16: run_ew_scalar<bfloat16_t, ElementwiseOp::Add>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16x2: run_ew_scalar<float16x2_t, ElementwiseOp::Add>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16x2: run_ew_scalar<bfloat16x2_t, ElementwiseOp::Add>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int8: run_ew_scalar<int8_t, ElementwiseOp::Add>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt8: run_ew_scalar<uint8_t, ElementwiseOp::Add>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int16: run_ew_scalar<int16_t, ElementwiseOp::Add>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt16: run_ew_scalar<uint16_t, ElementwiseOp::Add>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int32: run_ew_scalar<int32_t, ElementwiseOp::Add>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt32: run_ew_scalar<uint32_t, ElementwiseOp::Add>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int64: run_ew_scalar<int64_t, ElementwiseOp::Add>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt64: run_ew_scalar<uint64_t, ElementwiseOp::Add>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int4x2: run_ew_scalar<int4x2_t, ElementwiseOp::Add>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, scalar, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Sub:
                    switch (dtype) {
                        case DType::Float32: run_ew_scalar<float32_t, ElementwiseOp::Sub>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float64: run_ew_scalar<float64_t, ElementwiseOp::Sub>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16: run_ew_scalar<float16_t, ElementwiseOp::Sub>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16: run_ew_scalar<bfloat16_t, ElementwiseOp::Sub>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16x2: run_ew_scalar<float16x2_t, ElementwiseOp::Sub>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16x2: run_ew_scalar<bfloat16x2_t, ElementwiseOp::Sub>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int8: run_ew_scalar<int8_t, ElementwiseOp::Sub>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt8: run_ew_scalar<uint8_t, ElementwiseOp::Sub>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int16: run_ew_scalar<int16_t, ElementwiseOp::Sub>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt16: run_ew_scalar<uint16_t, ElementwiseOp::Sub>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int32: run_ew_scalar<int32_t, ElementwiseOp::Sub>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt32: run_ew_scalar<uint32_t, ElementwiseOp::Sub>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int64: run_ew_scalar<int64_t, ElementwiseOp::Sub>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt64: run_ew_scalar<uint64_t, ElementwiseOp::Sub>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int4x2: run_ew_scalar<int4x2_t, ElementwiseOp::Sub>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, scalar, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Mul:
                    switch (dtype) {
                        case DType::Float32: run_ew_scalar<float32_t, ElementwiseOp::Mul>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float64: run_ew_scalar<float64_t, ElementwiseOp::Mul>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16: run_ew_scalar<float16_t, ElementwiseOp::Mul>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16: run_ew_scalar<bfloat16_t, ElementwiseOp::Mul>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16x2: run_ew_scalar<float16x2_t, ElementwiseOp::Mul>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16x2: run_ew_scalar<bfloat16x2_t, ElementwiseOp::Mul>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int8: run_ew_scalar<int8_t, ElementwiseOp::Mul>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt8: run_ew_scalar<uint8_t, ElementwiseOp::Mul>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int16: run_ew_scalar<int16_t, ElementwiseOp::Mul>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt16: run_ew_scalar<uint16_t, ElementwiseOp::Mul>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int32: run_ew_scalar<int32_t, ElementwiseOp::Mul>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt32: run_ew_scalar<uint32_t, ElementwiseOp::Mul>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int64: run_ew_scalar<int64_t, ElementwiseOp::Mul>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt64: run_ew_scalar<uint64_t, ElementwiseOp::Mul>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int4x2: run_ew_scalar<int4x2_t, ElementwiseOp::Mul>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, scalar, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Div:
                    switch (dtype) {
                        case DType::Float32: run_ew_scalar<float32_t, ElementwiseOp::Div>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float64: run_ew_scalar<float64_t, ElementwiseOp::Div>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16: run_ew_scalar<float16_t, ElementwiseOp::Div>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16: run_ew_scalar<bfloat16_t, ElementwiseOp::Div>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16x2: run_ew_scalar<float16x2_t, ElementwiseOp::Div>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16x2: run_ew_scalar<bfloat16x2_t, ElementwiseOp::Div>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int8: run_ew_scalar<int8_t, ElementwiseOp::Div>(static_cast<int8_t*>(out_data) + out_offset, static_cast<const int8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt8: run_ew_scalar<uint8_t, ElementwiseOp::Div>(static_cast<uint8_t*>(out_data) + out_offset, static_cast<const uint8_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int16: run_ew_scalar<int16_t, ElementwiseOp::Div>(static_cast<int16_t*>(out_data) + out_offset, static_cast<const int16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt16: run_ew_scalar<uint16_t, ElementwiseOp::Div>(static_cast<uint16_t*>(out_data) + out_offset, static_cast<const uint16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int32: run_ew_scalar<int32_t, ElementwiseOp::Div>(static_cast<int32_t*>(out_data) + out_offset, static_cast<const int32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt32: run_ew_scalar<uint32_t, ElementwiseOp::Div>(static_cast<uint32_t*>(out_data) + out_offset, static_cast<const uint32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int64: run_ew_scalar<int64_t, ElementwiseOp::Div>(static_cast<int64_t*>(out_data) + out_offset, static_cast<const int64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::UInt64: run_ew_scalar<uint64_t, ElementwiseOp::Div>(static_cast<uint64_t*>(out_data) + out_offset, static_cast<const uint64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Int4x2: run_ew_scalar<int4x2_t, ElementwiseOp::Div>(static_cast<int4x2_t*>(out_data) + out_offset, static_cast<const int4x2_t*>(a_data) + a_offset, scalar, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseOp::Pow:
                    switch (dtype) {
                        case DType::Float32: run_ew_scalar<float32_t, ElementwiseOp::Pow>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float64: run_ew_scalar<float64_t, ElementwiseOp::Pow>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16: run_ew_scalar<float16_t, ElementwiseOp::Pow>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16: run_ew_scalar<bfloat16_t, ElementwiseOp::Pow>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::Float16x2: run_ew_scalar<float16x2_t, ElementwiseOp::Pow>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        case DType::BFloat16x2: run_ew_scalar<bfloat16x2_t, ElementwiseOp::Pow>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, scalar, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        a.give_future(future);
        out.give_future(future);
    }

    /**
     * Queues one pool worker that dispatches on dtype and launches the kernel. The future lands
     * on both tensors so teardown drains it.
     * @brief GPU unary elementwise: out[i] = fn(a[i]) (exp/log, float kinds only).
     * @param a Input tensor, GPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as a.
     * @param op Operation selector.
     * @note Throws DirectFunctionCallOnWrongDataSource when any tensor is not on the GPU.
     */
    __host__ void elementwise_unary_gpu(const Tensor& a, const Tensor& out, const ElementwiseUnaryOp op) {
        if (a.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(a.device));
        }
        const auto* const a_data = a.data;
        auto* const out_data = out.data;
        const auto a_offset = a.offset;
        const auto out_offset = out.offset;
        const auto dtype = a.dtype;
        const auto numel = a.numel;
        // Raw pointers are captured by value: the future lands on both
        // tensors, so buffer teardown always drains this worker first.
        /**
         * Dispatches on dtype to run_ew_unary, then syncs the default stream so the future only
         * completes once the buffer writes are done.
         * @brief Pool worker for the unary elementwise launch.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (op) {
                case ElementwiseUnaryOp::Exp:
                    switch (dtype) {
                        case DType::Float32: run_ew_unary<float32_t, ElementwiseUnaryOp::Exp>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, numel); break;
                        case DType::Float64: run_ew_unary<float64_t, ElementwiseUnaryOp::Exp>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, numel); break;
                        case DType::Float16: run_ew_unary<float16_t, ElementwiseUnaryOp::Exp>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, numel); break;
                        case DType::BFloat16: run_ew_unary<bfloat16_t, ElementwiseUnaryOp::Exp>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, numel); break;
                        case DType::Float16x2: run_ew_unary<float16x2_t, ElementwiseUnaryOp::Exp>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, numel); break;
                        case DType::BFloat16x2: run_ew_unary<bfloat16x2_t, ElementwiseUnaryOp::Exp>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                case ElementwiseUnaryOp::Log:
                    switch (dtype) {
                        case DType::Float32: run_ew_unary<float32_t, ElementwiseUnaryOp::Log>(static_cast<float32_t*>(out_data) + out_offset, static_cast<const float32_t*>(a_data) + a_offset, numel); break;
                        case DType::Float64: run_ew_unary<float64_t, ElementwiseUnaryOp::Log>(static_cast<float64_t*>(out_data) + out_offset, static_cast<const float64_t*>(a_data) + a_offset, numel); break;
                        case DType::Float16: run_ew_unary<float16_t, ElementwiseUnaryOp::Log>(static_cast<float16_t*>(out_data) + out_offset, static_cast<const float16_t*>(a_data) + a_offset, numel); break;
                        case DType::BFloat16: run_ew_unary<bfloat16_t, ElementwiseUnaryOp::Log>(static_cast<bfloat16_t*>(out_data) + out_offset, static_cast<const bfloat16_t*>(a_data) + a_offset, numel); break;
                        case DType::Float16x2: run_ew_unary<float16x2_t, ElementwiseUnaryOp::Log>(static_cast<float16x2_t*>(out_data) + out_offset, static_cast<const float16x2_t*>(a_data) + a_offset, numel); break;
                        case DType::BFloat16x2: run_ew_unary<bfloat16x2_t, ElementwiseUnaryOp::Log>(static_cast<bfloat16x2_t*>(out_data) + out_offset, static_cast<const bfloat16x2_t*>(a_data) + a_offset, numel); break;
                        [[unlikely]] default: throw ErrorHandling::NotImplemented();
                    }
                    break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            // The kernel is in-flight when the launch returns; sync so the
            // future only completes once the buffer writes are done.
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        a.give_future(future);
        out.give_future(future);
    }
}
#endif
