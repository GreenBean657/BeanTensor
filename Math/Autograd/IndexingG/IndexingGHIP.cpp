/**
 * @file IndexingGHIP.cpp
 * @author GreenBean
 * @brief HIP backend for the indexing/view backward ops (index_select, gather,
 * cat, slice). Divergent twin of Math/Autograd/IndexingG/IndexingGCUDA.cu —
 * the two backends are written separately per project rule, but must stay
 * behaviorally identical. index_select/gather run one deterministic scatter
 * kernel: each dx element is owned by a single thread that scans the index
 * axis in ascending pick order (no atomics), so duplicate-index accumulation
 * is bit-identical to the CPU backend. cat/slice are pure byte movement: D2D
 * slab copies from pool workers, mirroring the forward indexing ops.
 * @version 0.1
 * @date 2026-11-14
 */
#if defined(USE_HIP)
#include <array>
#include <cstdint>
#include <cstring>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "Math/Autograd/AutogradHipShared.h"
#include "Math/Autograd/IndexingG/IndexingG.h"
#include "Math/Shape/Indexing/IndexingShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;

    namespace {
        /**
         * Little-endian byte read of one index element; the forward pass
         * already bounds-checked the index tensor. All supported targets are
         * little-endian.
         * @brief Read one index element without a typed pointer.
         * @param p First byte of the index element.
         * @param width Element width in bytes (1, 2, 4 or 8).
         * @return The index value, zero-extended to uint64_t.
         */
        __device__ __forceinline__ uint64_t dev_read_index_le(const uint8_t* p, const size_t width) {
            auto v = uint64_t{0};
            for (auto b = size_t{0}; b < width; ++b) {
                v |= static_cast<uint64_t>(p[b]) << (8 * b);
            }
            return v;
        }

        /**
         * w += d under the grad_acc dtype contract: fp32/fp64 storage-width
         * add, halfs widen to fp32 and narrow RNE, packed per lane, integers
         * wrap through the unsigned twin, int4x2 per nibble. Matches the CPU
         * indexing_g_shared::acc_element.
         * @brief Per-element gradient accumulation (all dtypes).
         * @param w Accumulator element, updated in place.
         * @param d Delta element added into w.
         */
        template<typename T>
        __device__ __forceinline__ void dev_acc(T& w, const T d) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                w = w + d;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                w = __float2half_rn(__half2float(w) + __half2float(d));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                w = dev_narrow_bf16(__bfloat162float(w) + __bfloat162float(d));
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                const auto r_lo = __float2half_rn(__half2float(__low2half(w)) + __half2float(__low2half(d)));
                const auto r_hi = __float2half_rn(__half2float(__high2half(w)) + __half2float(__high2half(d)));
                w = __halves2half2(r_lo, r_hi);
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                const auto r_lo = dev_narrow_bf16(__bfloat162float(__low2bfloat16(w)) + __bfloat162float(__low2bfloat16(d)));
                const auto r_hi = dev_narrow_bf16(__bfloat162float(__high2bfloat16(w)) + __bfloat162float(__high2bfloat16(d)));
                w = __halves2bfloat162(r_lo, r_hi);
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                w = dev_pack_int4(dev_int4_lo(w) + dev_int4_lo(d), dev_int4_hi(w) + dev_int4_hi(d));
            } else {
                using U = std::make_unsigned_t<T>;
                w = static_cast<T>(static_cast<U>(static_cast<U>(w) + static_cast<U>(d)));
            }
        }

        /**
         * One thread per dx flat element: scan the picks ascending and add
         * every dy slab whose index selects this element. No atomics — the
         * per-element ascending scan is the determinism contract.
         * @brief index_select backward kernel (grid-stride over dx elements).
         * @param dy Upstream gradient, out shape (picks along the axis).
         * @param idx Index elements, view offset already applied.
         * @param idx_width Index element width in bytes.
         * @param dx Zeroed gradient of the forward input.
         * @param extent Forward input's extent along the axis.
         * @param inner Product of the extents past the axis.
         * @param picks Number of indices.
         * @param numel dx element count.
         */
        template<typename T>
        __global__ void index_select_g_kernel(const T* __restrict__ dy, const uint8_t* __restrict__ idx, const size_t idx_width,
                                              T* __restrict__ dx, const size_t extent, const size_t inner, const size_t picks, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t f = tid; f < numel; f += stride) {
                const auto i = f % inner;
                const auto k = (f / inner) % extent;
                const auto o = f / (inner * extent);
                for (auto j = size_t{0}; j < picks; ++j) {
                    if (dev_read_index_le(idx + j * idx_width, idx_width) == k) {
                        dev_acc(dx[f], dy[(o * picks + j) * inner + i]);
                    }
                }
            }
        }

        /**
         * Fixed-size by-value geometry for the gather backward kernel (rank
         * is capped at BT_TENSOR_MAX_DIMS).
         * @brief By-value shape/stride bundle for gather_g_kernel.
         */
        struct GatherGSizes {
            size_t dx_shape[BT_TENSOR_MAX_DIMS];
            size_t dx_strides[BT_TENSOR_MAX_DIMS];
            size_t dy_strides[BT_TENSOR_MAX_DIMS];
            size_t idx_axis_extent;
            size_t axis;
            size_t ndim;
        };

        /**
         * One thread per dx flat element: decompose to coordinates, scan the
         * index axis ascending at the matching off-axis coordinates, add every
         * dy element whose index equals this element's axis coordinate.
         * @brief gather backward kernel (grid-stride over dx elements).
         * @param dy Upstream gradient, index tensor's shape.
         * @param idx Index elements, view offset already applied.
         * @param idx_width Index element width in bytes.
         * @param dx Zeroed gradient of the forward input.
         * @param sz Shape/stride bundle.
         * @param numel dx element count.
         */
        template<typename T>
        __global__ void gather_g_kernel(const T* __restrict__ dy, const uint8_t* __restrict__ idx, const size_t idx_width,
                                        T* __restrict__ dx, const GatherGSizes sz, const size_t numel) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t f = tid; f < numel; f += stride) {
                auto base = size_t{0};
                auto axis_coord = size_t{0};
                for (auto d = size_t{0}; d < sz.ndim; ++d) {
                    const auto coord = (f / sz.dx_strides[d]) % sz.dx_shape[d];
                    if (d == sz.axis) {
                        axis_coord = coord;
                    } else {
                        base += coord * sz.dy_strides[d];
                    }
                }
                for (auto j = size_t{0}; j < sz.idx_axis_extent; ++j) {
                    const auto dj = base + j * sz.dy_strides[sz.axis];
                    if (dev_read_index_le(idx + dj * idx_width, idx_width) == axis_coord) {
                        dev_acc(dx[f], dy[dj]);
                    }
                }
            }
        }

        /**
         * Skips the launch when numel is 0.
         * @brief Host-side index_select backward launch.
         */
        template<typename T>
        __host__ void run_index_select_g(const T* dy, const uint8_t* idx, const size_t idx_width, T* dx,
                                         const size_t extent, const size_t inner, const size_t picks, const size_t numel) {
            if (numel > 0) {
                index_select_g_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(dy, idx, idx_width, dx, extent, inner, picks, numel);
            }
        }

        /**
         * Skips the launch when numel is 0.
         * @brief Host-side gather backward launch.
         */
        template<typename T>
        __host__ void run_gather_g(const T* dy, const uint8_t* idx, const size_t idx_width, T* dx, const GatherGSizes& sz, const size_t numel) {
            if (numel > 0) {
                gather_g_kernel<T><<<grid_for(numel), BT_GPU_THREADS>>>(dy, idx, idx_width, dx, sz, numel);
            }
        }

        /**
         * Plain-data parameter bundle for the scatter backward kernels: the
         * wrappers are Tensor friends and extract the raw pointers/geometry,
         * this helper (and its pool worker) never touches Tensor internals.
         * @brief By-value scatter-launch parameters.
         */
        struct ScatterGParams {
            const void* dy;
            size_t dy_offset;
            const void* idx;
            size_t idx_offset;
            size_t idx_width;
            void* dx;
            size_t dx_offset;
            DataTypes::DType dtype;
            size_t numel;
            size_t extent; // index_select: forward input's extent along the axis
            size_t inner; // index_select: product of the extents past the axis
            size_t picks; // index_select: number of indices
            GatherGSizes sz; // gather geometry
            bool gather;
        };

        /**
         * dx is pre-zeroed by the caller; the dtype switch dispatches the
         * kernel and the stream sync makes the future complete only once the
         * writes have landed.
         * @brief Launch one scatter backward kernel; return the completion future.
         * @param p Scatter-launch parameters (plain data).
         * @return Shared future completing after the stream sync.
         */
        __host__ std::shared_future<void> launch_scatter_g(const ScatterGParams& p) {
            /**
             * Captures the plain-data parameters by value, so the lambda stays
             * valid after the enclosing call returns. The stream sync makes the
             * shared future complete only once the kernel writes are visible.
             * @brief Async pool worker that launches the scatter backward kernel and syncs the stream.
             */
            const auto worker = [p] {
                const auto* idx_base = static_cast<const uint8_t*>(p.idx) + p.idx_offset * p.idx_width;
                using DType = DataTypes::DType;
                /**
                 * Dispatches the typed kernel for one concrete data dtype.
                 * @brief Per-dtype kernel dispatch inside the worker.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto run = [&]<typename T>() {
                    const auto* dyp = static_cast<const T*>(p.dy) + p.dy_offset;
                    auto* dxp = static_cast<T*>(p.dx) + p.dx_offset;
                    if (p.gather) {
                        run_gather_g(dyp, idx_base, p.idx_width, dxp, p.sz, p.numel);
                    } else {
                        run_index_select_g(dyp, idx_base, p.idx_width, dxp, p.extent, p.inner, p.picks, p.numel);
                    }
                };
                switch (p.dtype) {
                    case DType::Float32: run.template operator()<float32_t>(); break;
                    case DType::Float64: run.template operator()<float64_t>(); break;
                    case DType::Float16: run.template operator()<float16_t>(); break;
                    case DType::BFloat16: run.template operator()<bfloat16_t>(); break;
                    case DType::Float16x2: run.template operator()<float16x2_t>(); break;
                    case DType::BFloat16x2: run.template operator()<bfloat16x2_t>(); break;
                    case DType::Int8: run.template operator()<int8_t>(); break;
                    case DType::UInt8: run.template operator()<uint8_t>(); break;
                    case DType::Int16: run.template operator()<int16_t>(); break;
                    case DType::UInt16: run.template operator()<uint16_t>(); break;
                    case DType::Int32: run.template operator()<int32_t>(); break;
                    case DType::UInt32: run.template operator()<uint32_t>(); break;
                    case DType::Int64: run.template operator()<int64_t>(); break;
                    case DType::UInt64: run.template operator()<uint64_t>(); break;
                    case DType::Int4x2: run.template operator()<int4x2_t>(); break;
                    [[unlikely]] default: throw ErrorHandling::NotImplemented();
                }
                // The kernel is in-flight when the launch returns; sync so the
                // future only completes once the buffer writes are done.
                HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));
            };
            auto& threadMgr = Threading::GetCPUThreadPool();
            return threadMgr.submit(worker, Threading::TaskPriority::High).share();
        }
    }

    /**
     * Queues the scatter kernel on the CPU thread pool (High priority) and
     * returns; the shared future lands on all three tensors.
     * @brief GPU index_select backward (deterministic scatter-add kernel).
     * @param dy Upstream gradient (contiguous), the forward output's shape.
     * @param idx Rank-1 index tensor used by the forward pass.
     * @param dx Pre-zeroed gradient of the forward input (contiguous).
     * @param axis Axis the forward pass selected along.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     */
    __host__ void index_select_g_gpu(const Tensor& dy, const Tensor& idx, const Tensor& dx, const size_t axis) {
        if (dy.device != Device::GPU || idx.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dy.device));
        }
        ScatterGParams p{};
        p.dy = dy.data;
        p.dy_offset = dy.offset;
        p.idx = idx.data;
        p.idx_offset = idx.offset;
        p.idx_width = DataTypes::dtype_bytes(idx.dtype);
        p.dx = dx.data;
        p.dx_offset = dx.offset;
        p.dtype = dy.dtype;
        p.numel = dx.numel;
        p.extent = dx.shape[axis];
        p.inner = indexing_shared::span_product(dx.shape, axis + 1, dx.ndim);
        p.picks = idx.numel;
        p.gather = false;
        const auto future = launch_scatter_g(p);
        dy.give_future(future);
        idx.give_future(future);
        dx.give_future(future);
    }

    /**
     * Same pool/future contract as index_select_g_gpu.
     * @brief GPU gather backward (deterministic scatter-add kernel).
     * @param dy Upstream gradient (contiguous), the index tensor's shape.
     * @param idx Index tensor used by the forward pass.
     * @param dx Pre-zeroed gradient of the forward input (contiguous).
     * @param axis Axis the forward pass gathered along.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     */
    __host__ void gather_g_gpu(const Tensor& dy, const Tensor& idx, const Tensor& dx, const size_t axis) {
        if (dy.device != Device::GPU || idx.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dy.device));
        }
        ScatterGParams p{};
        p.dy = dy.data;
        p.dy_offset = dy.offset;
        p.idx = idx.data;
        p.idx_offset = idx.offset;
        p.idx_width = DataTypes::dtype_bytes(idx.dtype);
        p.dx = dx.data;
        p.dx_offset = dx.offset;
        p.dtype = dy.dtype;
        p.numel = dx.numel;
        p.gather = true;
        const auto ndim = dx.ndim;
        auto running = size_t{1};
        for (auto d = ndim; d-- > 0;) {
            p.sz.dx_strides[d] = running;
            running *= dx.shape[d];
            p.sz.dx_shape[d] = dx.shape[d];
        }
        running = 1;
        for (auto d = ndim; d-- > 0;) {
            p.sz.dy_strides[d] = running;
            running *= dy.shape[d];
        }
        p.sz.idx_axis_extent = idx.shape[axis];
        p.sz.axis = axis;
        p.sz.ndim = ndim;
        const auto future = launch_scatter_g(p);
        dy.give_future(future);
        idx.give_future(future);
        dx.give_future(future);
    }

    /**
     * Pure byte movement: one D2D slab copy per outer row, mirroring the
     * forward cat worker. The future lands on dy and dx.
     * @brief GPU cat backward (extract one input's slab from dy).
     * @param dy Upstream gradient (contiguous), the cat output's shape.
     * @param dx Fresh gradient of one input (contiguous), filled entirely.
     * @param axis Axis the forward pass concatenated along.
     * @param axis_offset This input's start offset along the axis in dy.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     */
    __host__ void cat_g_gpu(const Tensor& dy, const Tensor& dx, const size_t axis, const size_t axis_offset) {
        if (dy.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dy.device));
        }
        const auto* const dy_data = static_cast<const uint8_t*>(dy.data);
        auto* const dx_data = static_cast<uint8_t*>(dx.data);
        const auto dy_offset = dy.offset;
        const auto dx_offset = dx.offset;
        const auto elem = DataTypes::dtype_bytes(dy.dtype);
        const auto outer = indexing_shared::span_product(dy.shape, 0, axis);
        const auto inner = indexing_shared::span_product(dy.shape, axis + 1, dy.ndim);
        const auto extent = dx.shape[axis];
        const auto out_extent = dy.shape[axis];
        /**
         * Raw pointers and scalars are captured by value: the future lands on
         * both tensors, so teardown always drains this worker first.
         * @brief Pool worker for the GPU cat backward (D2D slab copies).
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            const auto* src = dy_data + dy_offset * elem;
            auto* dst = dx_data + dx_offset * elem;
            for (auto o = size_t{0}; o < outer; ++o) {
                HIP_CHECK_ERROR(hipMemcpy(dst + o * extent * inner * elem, src + (o * out_extent + axis_offset) * inner * elem, extent * inner * elem, hipMemcpyDeviceToDevice));
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        dx.give_future(future);
    }

    /**
     * Pure byte movement: dy's contiguous inner runs are D2D-copied into the
     * recorded box of the zeroed dx. dx is pre-zeroed by the caller; the box
     * coordinates come from the metadata ranges, never from view pointers.
     * The future lands on dy and dx.
     * @brief GPU slice backward (embed dy's box into the zeroed dx).
     * @param dy Upstream gradient (contiguous), the slice view's shape.
     * @param dx Pre-zeroed gradient of the forward input (contiguous).
     * @param meta Recorded slice metadata (source shape + per-dim ranges).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     */
    __host__ void slice_g_gpu(const Tensor& dy, const Tensor& dx, const NodeMeta& meta) {
        if (dy.device != Device::GPU || dx.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(dy.device));
        }
        const auto* const dy_data = static_cast<const uint8_t*>(dy.data);
        auto* const dx_data = static_cast<uint8_t*>(dx.data);
        const auto dy_offset = dy.offset;
        const auto dx_offset = dx.offset;
        const auto elem = DataTypes::dtype_bytes(dy.dtype);
        const auto ndim = meta.src_ndim;
        std::array<size_t, BT_TENSOR_MAX_DIMS> dx_strides{};
        std::array<size_t, BT_TENSOR_MAX_DIMS> view_shape{};
        auto running = size_t{1};
        for (auto d = ndim; d-- > 0;) {
            dx_strides[d] = running;
            running *= meta.src_shape[d];
        }
        for (auto d = size_t{0}; d < ndim; ++d) {
            view_shape[d] = meta.ranges[d][1] - meta.ranges[d][0];
        }
        std::array<size_t, BT_TENSOR_MAX_DIMS> run_strides{};
        running = 1;
        for (auto d = ndim; d-- > 0;) {
            if (d + 1 == ndim) {
                continue;
            }
            run_strides[d] = running;
            running *= view_shape[d];
        }
        std::array<size_t, BT_TENSOR_MAX_DIMS> starts{};
        for (auto d = size_t{0}; d < ndim; ++d) {
            starts[d] = meta.ranges[d][0];
        }
        const auto rows = dy.numel == 0 ? size_t{0} : dy.numel / view_shape[ndim - 1];
        const auto inner = view_shape[ndim - 1];
        /**
         * Raw pointers, geometry and the recorded ranges are captured by
         * value: the future lands on both tensors, so teardown always drains
         * this worker first.
         * @brief Pool worker for the GPU slice backward (D2D box copies).
         * @note Lambda subcall. Self-execution prohibited.
         */
        const auto worker = [=] {
            const auto* src = dy_data + dy_offset * elem;
            auto* dst = dx_data + dx_offset * elem;
            for (auto r = size_t{0}; r < rows; ++r) {
                auto dst_off = starts[ndim - 1];
                for (auto d = size_t{0}; d + 1 < ndim; ++d) {
                    const auto coord = (r / run_strides[d]) % view_shape[d];
                    dst_off += (coord + starts[d]) * dx_strides[d];
                }
                HIP_CHECK_ERROR(hipMemcpy(dst + dst_off * elem, src + r * inner * elem, inner * elem, hipMemcpyDeviceToDevice));
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        dy.give_future(future);
        dx.give_future(future);
    }
}
#endif
