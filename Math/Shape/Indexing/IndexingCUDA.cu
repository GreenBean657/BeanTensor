/**
 * @file IndexingCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the indexing ops (index_select, gather, cat).
 * Divergent twin of Math/Shape/Indexing/IndexingHIP.cpp — the two backends are
 * written separately per project rule (no HIP-only unification), but must stay
 * behaviorally identical. The ops are pure byte movement: no kernels, just
 * device-to-device slab/element copies; index tensors are staged to the host
 * so bounds checks run where they can throw.
 * @version 0.2
 * @date 2026-10-22
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "Math/Shape/Indexing/Indexing.h"
#include "Math/Shape/Indexing/IndexingShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace indexing_shared;

    /**
     * Runs on the CPU thread pool at high priority; the future lands on all three tensors.
     * @brief GPU index_select along axis (host-staged indices, D2D slab copies).
     * @param t Input tensor (contiguous).
     * @param idx Rank-1 integer index tensor (contiguous).
     * @param out Fresh output tensor, t's shape with shape[axis] = idx.numel.
     * @param axis Axis selected along.
     */
    __host__ void index_select_gpu(const Tensor& t, const Tensor& idx, const Tensor& out, const size_t axis) {
        if (t.device != Device::GPU || idx.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = static_cast<const uint8_t*>(t.data);
        const auto* const idx_data = static_cast<const uint8_t*>(idx.data);
        auto* const out_data = static_cast<uint8_t*>(out.data);
        const auto t_offset = t.offset;
        const auto idx_offset = idx.offset;
        const auto idx_dtype = idx.dtype;
        const auto idx_nbytes = idx.nbytes;
        const auto elem = DataTypes::dtype_bytes(t.dtype);
        const auto outer = span_product(t.shape, 0, axis);
        const auto extent = t.shape[axis];
        const auto inner = span_product(t.shape, axis + 1, t.ndim);
        const auto picks = idx.numel;
        /**
         * Raw pointers and scalars are captured by value: the future lands on the tensors, so buffer
         * teardown always drains this worker first. Index values are staged host-side (the staging
         * cudaMemcpy syncs the default stream), bounds-checked there, then each (outer, pick) pair
         * issues one D2D slab copy of inner elements.
         * @brief Pool worker for the GPU index_select.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            std::vector<uint8_t> staged(idx_nbytes);
            if (idx_nbytes != 0) {
                CUDA_CHECK_ERROR(cudaMemcpy(staged.data(), idx_data + idx_offset * DataTypes::dtype_bytes(idx_dtype), idx_nbytes, cudaMemcpyDeviceToHost));
            }
            const auto* src = t_data + t_offset * elem;
            auto run = [&]<typename I>(const I* iptr) {
                for (auto o = size_t{0}; o < outer; ++o) {
                    for (auto j = size_t{0}; j < picks; ++j) {
                        const auto k = checked_axis_index(iptr, j, extent);
                        CUDA_CHECK_ERROR(cudaMemcpy(out_data + (o * picks + j) * inner * elem, src + (o * extent + k) * inner * elem, inner * elem, cudaMemcpyDeviceToDevice));
                    }
                }
            };
            using DType = DataTypes::DType;
            switch (idx_dtype) {
                case DType::Int8: run(reinterpret_cast<const int8_t*>(staged.data())); break;
                case DType::UInt8: run(staged.data()); break;
                case DType::Int16: run(reinterpret_cast<const int16_t*>(staged.data())); break;
                case DType::UInt16: run(reinterpret_cast<const uint16_t*>(staged.data())); break;
                case DType::Int32: run(reinterpret_cast<const int32_t*>(staged.data())); break;
                case DType::UInt32: run(reinterpret_cast<const uint32_t*>(staged.data())); break;
                case DType::Int64: run(reinterpret_cast<const int64_t*>(staged.data())); break;
                case DType::UInt64: run(reinterpret_cast<const uint64_t*>(staged.data())); break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        idx.give_future(future);
        out.give_future(future);
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on all three tensors.
     * @brief GPU gather along axis (host-staged indices, per-element D2D copies).
     * @param t Input tensor (contiguous).
     * @param idx Integer index tensor, same rank as t (contiguous).
     * @param out Fresh output tensor with idx's shape and t's dtype.
     * @param axis Axis gathered along.
     */
    __host__ void gather_gpu(const Tensor& t, const Tensor& idx, const Tensor& out, const size_t axis) {
        if (t.device != Device::GPU || idx.device != Device::GPU || out.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
        }
        const auto* const t_data = static_cast<const uint8_t*>(t.data);
        const auto* const idx_data = static_cast<const uint8_t*>(idx.data);
        auto* const out_data = static_cast<uint8_t*>(out.data);
        const auto t_offset = t.offset;
        const auto idx_offset = idx.offset;
        const auto idx_dtype = idx.dtype;
        const auto idx_nbytes = idx.nbytes;
        const auto elem = DataTypes::dtype_bytes(t.dtype);
        const auto extent = t.shape[axis];
        const auto ndim = t.ndim;
        const auto numel = out.numel;
        auto t_dims = std::vector(t.dims, t.dims + BT_TENSOR_MAX_DIMS);
        const auto out_dims = std::vector(out.dims, out.dims + BT_TENSOR_MAX_DIMS);
        /**
         * Same staging contract as index_select's worker; every output element
         * is one D2D copy (correctness-first: gather has no slab structure).
         * @brief Pool worker for the GPU gather.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            std::vector<uint8_t> staged(idx_nbytes);
            if (idx_nbytes != 0) {
                CUDA_CHECK_ERROR(cudaMemcpy(staged.data(), idx_data + idx_offset * DataTypes::dtype_bytes(idx_dtype), idx_nbytes, cudaMemcpyDeviceToHost));
            }
            auto run = [&]<typename I>(const I* iptr) {
                for (auto flat = size_t{0}; flat < numel; ++flat) {
                    const auto sel = checked_axis_index(iptr, flat, extent);
                    auto rem = flat;
                    auto src_off = t_offset;
                    for (auto d = size_t{0}; d < ndim; ++d) {
                        const auto coord = rem / out_dims[d];
                        rem %= out_dims[d];
                        src_off += (d == axis ? sel : coord) * t_dims[d];
                    }
                    CUDA_CHECK_ERROR(cudaMemcpy(out_data + flat * elem, t_data + src_off * elem, elem, cudaMemcpyDeviceToDevice));
                }
            };
            using DType = DataTypes::DType;
            switch (idx_dtype) {
                case DType::Int8: run(reinterpret_cast<const int8_t*>(staged.data())); break;
                case DType::UInt8: run(staged.data()); break;
                case DType::Int16: run(reinterpret_cast<const int16_t*>(staged.data())); break;
                case DType::UInt16: run(reinterpret_cast<const uint16_t*>(staged.data())); break;
                case DType::Int32: run(reinterpret_cast<const int32_t*>(staged.data())); break;
                case DType::UInt32: run(reinterpret_cast<const uint32_t*>(staged.data())); break;
                case DType::Int64: run(reinterpret_cast<const int64_t*>(staged.data())); break;
                case DType::UInt64: run(reinterpret_cast<const uint64_t*>(staged.data())); break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        t.give_future(future);
        idx.give_future(future);
        out.give_future(future);
    }

    /**
     * Runs on the CPU thread pool at high priority; the future lands on the output and every input.
     * @brief GPU concatenation along axis (D2D slab copies per input per outer position).
     * @param tensors Input list (validated by Tensor::cat).
     * @param out Fresh output tensor, summed extent along axis.
     * @param axis Axis concatenated along.
     */
    __host__ void cat_gpu(const std::vector<Tensor>& tensors, const Tensor& out, const size_t axis) {
        for (const auto& t : tensors) {
            if (t.device != Device::GPU || out.device != Device::GPU) {
                throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(t.device));
            }
        }
        auto* const out_data = static_cast<uint8_t*>(out.data);
        const auto elem = DataTypes::dtype_bytes(out.dtype);
        const auto outer = span_product(out.shape, 0, axis);
        const auto inner = span_product(out.shape, axis + 1, out.ndim);
        const auto out_extent = out.shape[axis];
        // (source slab base, extent along axis) per input, captured by value.
        std::vector<std::pair<const uint8_t*, size_t>> inputs;
        inputs.reserve(tensors.size());
        for (const auto& t : tensors) {
            inputs.emplace_back(static_cast<const uint8_t*>(t.data) + t.offset * elem, t.shape[axis]);
        }
        /**
         * @brief Pool worker for the GPU cat.
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            auto running = size_t{0};
            for (const auto& [src, extent] : inputs) {
                for (auto o = size_t{0}; o < outer; ++o) {
                    CUDA_CHECK_ERROR(cudaMemcpy(out_data + (o * out_extent + running) * inner * elem, src + o * extent * inner * elem, extent * inner * elem, cudaMemcpyDeviceToDevice));
                }
                running += extent;
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        for (const auto& t : tensors) {
            t.give_future(future);
        }
        out.give_future(future);
    }
}
#endif
