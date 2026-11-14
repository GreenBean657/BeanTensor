/**
 * @file IndexingGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for the indexing/view backward ops (index_select, gather,
 * cat, slice) and their node trampolines. index_select/gather scatter-add the
 * upstream gradient into a zeroed dx: each dx element is computed by one
 * worker scanning the index axis in ascending pick order, so duplicate-index
 * accumulation is deterministic and bit-identical across chunkings (and to the
 * GPU kernels, which use the same per-element ascending scan). cat extracts
 * each input's slab from dy; slice embeds dy's box into a zeroed dx at the
 * recorded ranges. cat/slice are pure byte movement (dtype-agnostic memcpy);
 * the scatter ops dispatch per dtype under the grad_acc accumulation contract
 * (IndexingGShared.h::acc_element).
 * @version 0.1
 * @date 2026-11-14
 */
#include <array>
#include <cstring>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/IndexingG/IndexingG.h"
#include "Math/Autograd/IndexingG/IndexingGShared.h"
#include "Math/Shape/Indexing/IndexingShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace indexing_g_shared;

    namespace {
        /**
         * One dx element's contribution: scan the picks ascending, accumulate
         * dy slabs whose index selects this element.
         * @brief CPU index_select backward over the dx element range [begin, end).
         * @param dy Upstream gradient (contiguous), out shape.
         * @param idx Index elements, view offset already applied.
         * @param idx_width Index element width in bytes.
         * @param dx Zeroed gradient of the forward input (contiguous).
         * @param extent Forward input's extent along the axis.
         * @param inner Product of the extents past the axis.
         * @param picks Number of indices.
         * @param begin First dx flat element processed.
         * @param end One past the last dx flat element processed.
         */
        template<typename T>
        void index_select_g_range(const T* dy, const uint8_t* idx, const size_t idx_width, T* dx,
                                  const size_t extent, const size_t inner, const size_t picks,
                                  const size_t begin, const size_t end) {
            for (auto f = begin; f < end; ++f) {
                const auto i = f % inner;
                const auto k = (f / inner) % extent;
                const auto o = f / (inner * extent);
                for (auto j = size_t{0}; j < picks; ++j) {
                    if (read_index_le(idx + j * idx_width, idx_width) == k) {
                        acc_element(&dx[f], dy[(o * picks + j) * inner + i]);
                    }
                }
            }
        }

        /**
         * One dx element's contribution: scan the index axis ascending over the
         * matching off-axis coordinates, accumulate dy elements whose index
         * equals this element's axis coordinate.
         * @brief CPU gather backward over the dx element range [begin, end).
         * @param dy Upstream gradient (contiguous), index tensor's shape.
         * @param idx Index elements, view offset already applied.
         * @param idx_width Index element width in bytes.
         * @param dx Zeroed gradient of the forward input (contiguous).
         * @param dx_shape Forward input's shape.
         * @param dx_strides Contiguous strides of the forward input's shape.
         * @param dy_strides Contiguous strides of the index/output shape.
         * @param ndim Rank of the forward input.
         * @param axis Axis gathered along.
         * @param idx_axis_extent Index tensor's extent along the axis.
         * @param begin First dx flat element processed.
         * @param end One past the last dx flat element processed.
         */
        template<typename T>
        void gather_g_range(const T* dy, const uint8_t* idx, const size_t idx_width, T* dx,
                            const std::array<size_t, BT_TENSOR_MAX_DIMS>& dx_shape,
                            const std::array<size_t, BT_TENSOR_MAX_DIMS>& dx_strides,
                            const std::array<size_t, BT_TENSOR_MAX_DIMS>& dy_strides,
                            const size_t ndim, const size_t axis, const size_t idx_axis_extent,
                            const size_t begin, const size_t end) {
            for (auto f = begin; f < end; ++f) {
                auto base = size_t{0};
                auto axis_coord = size_t{0};
                for (auto d = size_t{0}; d < ndim; ++d) {
                    const auto coord = (f / dx_strides[d]) % dx_shape[d];
                    if (d == axis) {
                        axis_coord = coord;
                    } else {
                        base += coord * dy_strides[d];
                    }
                }
                for (auto j = size_t{0}; j < idx_axis_extent; ++j) {
                    const auto dj = base + j * dy_strides[axis];
                    if (read_index_le(idx + dj * idx_width, idx_width) == axis_coord) {
                        acc_element(&dx[f], dy[dj]);
                    }
                }
            }
        }

        /**
         * @brief Contiguous strides of a shape.
         * @param shape Shape array.
         * @param ndim Rank.
         * @return Strides in row-major order.
         */
        [[nodiscard]] std::array<size_t, BT_TENSOR_MAX_DIMS> contiguous_strides(const size_t* shape, const size_t ndim) {
            std::array<size_t, BT_TENSOR_MAX_DIMS> strides{};
            auto running = size_t{1};
            for (auto d = ndim; d-- > 0;) {
                strides[d] = running;
                running *= shape[d];
            }
            return strides;
        }
    }

    /**
     * The scatter is chunked over dx's flat elements; chunks write disjoint
     * elements, so the threaded result is bit-identical to the serial one.
     * @brief CPU index_select backward (scatter-add of dy slabs).
     * @param dy Upstream gradient (contiguous), the forward output's shape.
     * @param idx Rank-1 index tensor used by the forward pass.
     * @param dx Pre-zeroed gradient of the forward input (contiguous).
     * @param axis Axis the forward pass selected along.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void index_select_g_cpu(const Tensor& dy, const Tensor& idx, const Tensor& dx, const size_t axis) {
        if (dy.device != Device::CPU || idx.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        const auto idx_width = DataTypes::dtype_bytes(idx.dtype);
        const auto* idx_base = static_cast<const uint8_t*>(idx.data) + idx.offset * idx_width;
        const auto extent = dx.shape[axis];
        const auto inner = indexing_shared::span_product(dx.shape, axis + 1, dx.ndim);
        const auto picks = idx.numel;
        const auto numel = dx.numel;
        using DType = DataTypes::DType;
        /**
         * Dispatches once on dtype, then chunks the dx element range across the
         * pool via autograd_shared::launch_elementwise (futures land on dy, idx
         * and dx).
         * @brief dtype dispatch + threaded launch for index_select_g_cpu.
         * @note Lambda subcall. Self-execution prohibited.
         */
        auto run = [&]<typename T>() {
            const auto* dyp = static_cast<const T*>(dy.data) + dy.offset;
            auto* dxp = static_cast<T*>(dx.data) + dx.offset;
            autograd_shared::launch_elementwise([=](const size_t b, const size_t e) {
                index_select_g_range(dyp, idx_base, idx_width, dxp, extent, inner, picks, b, e);
            }, numel, dy, idx, dx);
        };
        switch (dy.dtype) {
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
    }

    /**
     * Same chunking contract as index_select_g_cpu.
     * @brief CPU gather backward (scatter-add of dy elements).
     * @param dy Upstream gradient (contiguous), the index tensor's shape.
     * @param idx Index tensor used by the forward pass.
     * @param dx Pre-zeroed gradient of the forward input (contiguous).
     * @param axis Axis the forward pass gathered along.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void gather_g_cpu(const Tensor& dy, const Tensor& idx, const Tensor& dx, const size_t axis) {
        if (dy.device != Device::CPU || idx.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        const auto idx_width = DataTypes::dtype_bytes(idx.dtype);
        const auto* idx_base = static_cast<const uint8_t*>(idx.data) + idx.offset * idx_width;
        const auto ndim = dx.ndim;
        const auto dx_shape = [&] {
            std::array<size_t, BT_TENSOR_MAX_DIMS> a{};
            std::copy_n(dx.shape, ndim, a.begin());
            return a;
        }();
        const auto dx_strides = contiguous_strides(dx.shape, ndim);
        const auto dy_strides = contiguous_strides(dy.shape, ndim);
        const auto idx_axis_extent = idx.shape[axis];
        const auto numel = dx.numel;
        using DType = DataTypes::DType;
        /**
         * Dispatches once on dtype, then chunks the dx element range across the
         * pool via autograd_shared::launch_elementwise (futures land on dy, idx
         * and dx).
         * @brief dtype dispatch + threaded launch for gather_g_cpu.
         * @note Lambda subcall. Self-execution prohibited.
         */
        auto run = [&]<typename T>() {
            const auto* dyp = static_cast<const T*>(dy.data) + dy.offset;
            auto* dxp = static_cast<T*>(dx.data) + dx.offset;
            autograd_shared::launch_elementwise([=](const size_t b, const size_t e) {
                gather_g_range(dyp, idx_base, idx_width, dxp, dx_shape, dx_strides, dy_strides, ndim, axis, idx_axis_extent, b, e);
            }, numel, dy, idx, dx);
        };
        switch (dy.dtype) {
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
    }

    /**
     * Pure byte movement: each outer row of the input's slab is one memcpy out
     * of dy. Chunked over the outer rows; chunks write disjoint dx regions.
     * @brief CPU cat backward (extract one input's slab from dy).
     * @param dy Upstream gradient (contiguous), the cat output's shape.
     * @param dx Fresh gradient of one input (contiguous), filled entirely.
     * @param axis Axis the forward pass concatenated along.
     * @param axis_offset This input's start offset along the axis in dy.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     */
    void cat_g_cpu(const Tensor& dy, const Tensor& dx, const size_t axis, const size_t axis_offset) {
        if (dy.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        const auto elem = DataTypes::dtype_bytes(dy.dtype);
        const auto outer = indexing_shared::span_product(dy.shape, 0, axis);
        const auto inner = indexing_shared::span_product(dy.shape, axis + 1, dy.ndim);
        const auto extent = dx.shape[axis];
        const auto out_extent = dy.shape[axis];
        const auto* src = static_cast<const uint8_t*>(dy.data) + dy.offset * elem;
        auto* dst = static_cast<uint8_t*>(dx.data) + dx.offset * elem;
        autograd_shared::launch_elementwise([=](const size_t b, const size_t e) {
            for (auto o = b; o < e; ++o) {
                std::memcpy(dst + o * extent * inner * elem, src + (o * out_extent + axis_offset) * inner * elem, extent * inner * elem);
            }
        }, outer, dy, dx, dx);
    }

    /**
     * Pure byte movement: dy's contiguous inner runs memcpy into the recorded
     * box of the zeroed dx. Both buffers are fresh and contiguous; the box
     * coordinates come from the metadata ranges, never from view pointers.
     * Chunked over dy's inner-run rows; chunks write disjoint dx regions.
     * @brief CPU slice backward (embed dy's box into the zeroed dx).
     * @param dy Upstream gradient (contiguous), the slice view's shape.
     * @param dx Pre-zeroed gradient of the forward input (contiguous).
     * @param meta Recorded slice metadata (source shape + per-dim ranges).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     */
    void slice_g_cpu(const Tensor& dy, const Tensor& dx, const NodeMeta& meta) {
        if (dy.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        if (dy.numel == 0) {
            return;
        }
        const auto elem = DataTypes::dtype_bytes(dy.dtype);
        const auto ndim = meta.src_ndim;
        const auto dx_strides = contiguous_strides(meta.src_shape, ndim);
        std::array<size_t, BT_TENSOR_MAX_DIMS> view_shape{};
        for (auto d = size_t{0}; d < ndim; ++d) {
            view_shape[d] = meta.ranges[d][1] - meta.ranges[d][0];
        }
        const auto inner = view_shape[ndim - 1];
        const auto rows = dy.numel / inner;
        // r indexes inner runs, so the outer coordinates decompose with
        // strides in RUN units (the last dim is the run itself).
        std::array<size_t, BT_TENSOR_MAX_DIMS> run_strides{};
        auto run_stride = size_t{1};
        for (auto d = ndim; d-- > 0;) {
            if (d + 1 == ndim) {
                continue;
            }
            run_strides[d] = run_stride;
            run_stride *= view_shape[d];
        }
        std::array<size_t, BT_TENSOR_MAX_DIMS> starts{};
        for (auto d = size_t{0}; d < ndim; ++d) {
            starts[d] = meta.ranges[d][0];
        }
        const auto* src = static_cast<const uint8_t*>(dy.data) + dy.offset * elem;
        auto* dst = static_cast<uint8_t*>(dx.data) + dx.offset * elem;
        autograd_shared::launch_elementwise([=](const size_t b, const size_t e) {
            for (auto r = b; r < e; ++r) {
                auto dst_off = starts[ndim - 1];
                for (auto d = size_t{0}; d + 1 < ndim; ++d) {
                    const auto coord = (r / run_strides[d]) % view_shape[d];
                    dst_off += (coord + starts[d]) * dx_strides[d];
                }
                std::memcpy(dst + dst_off * elem, src + r * inner * elem, inner * elem);
            }
        }, rows, dy, dx, dx);
    }

    // --------------------------- node trampolines ---------------------------

    /**
     * dx is built in grad_dtype with the forward input's shape, zeroed, filled
     * by the device backend from the saved index tensor, then accumulated into
     * the input's grad (which boundary-casts into a pinned leaf buffer's dtype
     * via the single rule in GradDType.h). output_grad already carries
     * grad_dtype.
     * @brief Backward trampoline for index_select.
     * @param node Owning node (metadata + saved index tensor).
     * @param output_grad Gradient of the forward output, in grad_dtype.
     * @param inputs Input tensors of the forward op (just the source tensor).
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if the node shape is not an index_select node.
     */
    void index_select_g_node(Node* node, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1 || node == nullptr || node->_meta == nullptr || node->_meta->tag != NodeMetaTag::IndexSelect || node->_n_saved != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        const auto& idx = *node->_saved[0];
        output_grad->sync();
        x.sync();
        idx.sync();
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        dx.set_zero();
        dx.sync(); // the zero-fill must land before the scatter writes
        if (x.device == Device::CPU) {
            index_select_g_cpu(*output_grad, idx, dx, node->_meta->axis);
        } else if (x.device == Device::GPU) {
            index_select_g_gpu(*output_grad, idx, dx, node->_meta->axis);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
        }
        x.accumulate_grad(std::move(dx));
    }

    void index_select_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        index_select_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }

    /**
     * Same accumulation contract as index_select_g_node, elementwise.
     * @brief Backward trampoline for gather.
     * @param node Owning node (metadata + saved index tensor).
     * @param output_grad Gradient of the forward output, in grad_dtype.
     * @param inputs Input tensors of the forward op (just the source tensor).
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if the node shape is not a gather node.
     */
    void gather_g_node(Node* node, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1 || node == nullptr || node->_meta == nullptr || node->_meta->tag != NodeMetaTag::Gather || node->_n_saved != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        const auto& idx = *node->_saved[0];
        output_grad->sync();
        x.sync();
        idx.sync();
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        dx.set_zero();
        dx.sync(); // the zero-fill must land before the scatter writes
        if (x.device == Device::CPU) {
            gather_g_cpu(*output_grad, idx, dx, node->_meta->axis);
        } else if (x.device == Device::GPU) {
            gather_g_gpu(*output_grad, idx, dx, node->_meta->axis);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
        }
        x.accumulate_grad(std::move(dx));
    }

    void gather_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        gather_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }

    /**
     * Walks the direct _inputs plus the metadata's overflow parents in list
     * order; each input that requires grad receives a fresh grad_dtype delta
     * holding its axis slab of dy. The running axis offset advances over every
     * input, grad or not.
     * @brief Backward trampoline for cat.
     * @param node Owning node (axis metadata + overflow parents).
     * @param output_grad Gradient of the forward output, in grad_dtype.
     * @param inputs The first BT_TENSOR_MAX_PARENTS forward inputs.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if the node shape is not a cat node.
     */
    void cat_g_node(Node* node, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (node == nullptr || node->_meta == nullptr || node->_meta->tag != NodeMetaTag::Cat) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        const auto axis = node->_meta->axis;
        output_grad->sync();
        auto running = size_t{0};
        /**
         * Extracts one input's slab and accumulates it; advances the running
         * axis offset over every input regardless of its requires_grad.
         * @brief Per-input delivery for cat_g_node.
         * @param in One forward input tensor.
         */
        auto deliver = [&](Tensor& in) {
            const auto extent = in.shape[axis];
            if (in.requires_grad) {
                in.sync();
                auto dx = Tensor(std::vector(in.shape, in.shape + in.ndim), grad_dtype, in.device, false);
                if (in.device == Device::CPU) {
                    cat_g_cpu(*output_grad, dx, axis, running);
                } else if (in.device == Device::GPU) {
                    cat_g_gpu(*output_grad, dx, axis, running);
                } else [[unlikely]] {
                    throw ErrorHandling::UnReachableCode();
                }
                if (in.node != nullptr) {
                    grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
                }
                in.accumulate_grad(std::move(dx));
            }
            running += extent;
        };
        for (auto i = size_t{0}; i < n_inputs; ++i) {
            deliver(*inputs[i]);
        }
        for (auto i = size_t{0}; i < node->_meta->n_extra_parents; ++i) {
            deliver(*node->_meta->extra_parents[i]);
        }
    }

    void cat_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        cat_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }

    /**
     * dx is the zeroed, input-shaped grad_dtype tensor; the backend embeds
     * dy's box at the recorded ranges. Non-contiguous forward views need no
     * special casing: both gradient buffers are fresh and contiguous, and the
     * box coordinates come from the metadata.
     * @brief Backward trampoline for slice (split reuses it per chunk).
     * @param node Owning node (range metadata).
     * @param output_grad Gradient of the slice view, in grad_dtype.
     * @param inputs Input tensors of the forward op (just the source tensor).
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if the node shape is not a slice node.
     */
    void slice_g_node(Node* node, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1 || node == nullptr || node->_meta == nullptr || node->_meta->tag != NodeMetaTag::Slice) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), grad_dtype, x.device, false);
        dx.set_zero();
        dx.sync(); // the zero-fill must land before the box embed writes
        if (x.device == Device::CPU) {
            slice_g_cpu(*output_grad, dx, *node->_meta);
        } else if (x.device == Device::GPU) {
            slice_g_gpu(*output_grad, dx, *node->_meta);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype); // test hook: an intermediate received a delta
        }
        x.accumulate_grad(std::move(dx));
    }

    void slice_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        slice_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
