/**
 * @file IndexingCPU.cpp
 * @author GreenBean
 * @brief CPU backend for the indexing ops (index_select, gather, cat). Hosts
 * the Tensor::index_select / Tensor::gather / Tensor::cat method definitions
 * (mirrors how Tensor::sum lives in SumCPU.cpp). All three are pure byte
 * movement: dtype-agnostic slab/element copies, so no per-dtype kernels exist.
 * @version 0.1
 * @date 2026-10-22
 */
#include <cstdint>
#include <cstring>
#include <vector>

#include "Math/Shape/Indexing/Indexing.h"
#include "Math/Shape/Indexing/IndexingShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace indexing_shared;

    /**
     * Indices are read per dtype (bounds-checked); the copies are byte slabs,
     * one memcpy per (outer slab, picked index).
     * @brief CPU index_select along axis.
     * @param t Input tensor (contiguous).
     * @param idx Rank-1 integer index tensor (contiguous).
     * @param out Fresh output tensor, t's shape with shape[axis] = idx.numel.
     * @param axis Axis selected along.
     */
    void index_select_cpu(const Tensor& t, const Tensor& idx, Tensor& out, const size_t axis) {
        if (t.device != Device::CPU || idx.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        const auto elem = DataTypes::dtype_bytes(t.dtype);
        const auto outer = span_product(t.shape, 0, axis);
        const auto extent = t.shape[axis];
        const auto inner = span_product(t.shape, axis + 1, t.ndim);
        const auto picks = idx.numel;
        const auto* src = static_cast<const uint8_t*>(t.data) + t.offset * elem;
        const auto* idx_base = static_cast<const uint8_t*>(idx.data) + idx.offset * DataTypes::dtype_bytes(idx.dtype);
        auto* dst = static_cast<uint8_t*>(out.data);

        /**
         * @brief Slab copy for one concrete index element type.
         * @param iptr Index elements, view offset already applied.
         * @note Lambda subcall. Self-execution prohibited.
         */
        auto run = [&]<typename I>(const I* iptr) {
            for (auto o = size_t{0}; o < outer; ++o) {
                for (auto j = size_t{0}; j < picks; ++j) {
                    const auto k = checked_axis_index(iptr, j, extent);
                    std::memcpy(dst + (o * picks + j) * inner * elem, src + (o * extent + k) * inner * elem, inner * elem);
                }
            }
        };
        using DType = DataTypes::DType;
        switch (idx.dtype) {
            case DType::Int8: run(reinterpret_cast<const int8_t*>(idx_base)); break;
            case DType::UInt8: run(reinterpret_cast<const uint8_t*>(idx_base)); break;
            case DType::Int16: run(reinterpret_cast<const int16_t*>(idx_base)); break;
            case DType::UInt16: run(reinterpret_cast<const uint16_t*>(idx_base)); break;
            case DType::Int32: run(reinterpret_cast<const int32_t*>(idx_base)); break;
            case DType::UInt32: run(reinterpret_cast<const uint32_t*>(idx_base)); break;
            case DType::Int64: run(reinterpret_cast<const int64_t*>(idx_base)); break;
            case DType::UInt64: run(reinterpret_cast<const uint64_t*>(idx_base)); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * out flat position decomposes via the output strides; idx shares out's
     * shape and layout, so the same flat position reads the selecting index.
     * @brief CPU gather along axis.
     * @param t Input tensor (contiguous).
     * @param idx Integer index tensor, same rank as t (contiguous).
     * @param out Fresh output tensor with idx's shape and t's dtype.
     * @param axis Axis gathered along.
     */
    void gather_cpu(const Tensor& t, const Tensor& idx, Tensor& out, const size_t axis) {
        if (t.device != Device::CPU || idx.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        const auto elem = DataTypes::dtype_bytes(t.dtype);
        const auto extent = t.shape[axis];
        const auto ndim = t.ndim;
        const auto* src = static_cast<const uint8_t*>(t.data);
        const auto* idx_base = static_cast<const uint8_t*>(idx.data) + idx.offset * DataTypes::dtype_bytes(idx.dtype);
        auto* dst = static_cast<uint8_t*>(out.data);

        /**
         * @brief Elementwise gather for one concrete index element type.
         * @param iptr Index elements, view offset already applied.
         * @note Lambda subcall. Self-execution prohibited.
         */
        auto run = [&]<typename I>(const I* iptr) {
            for (auto flat = size_t{0}; flat < out.numel; ++flat) {
                const auto sel = checked_axis_index(iptr, flat, extent);
                auto rem = flat;
                auto src_off = t.offset;
                for (auto d = size_t{0}; d < ndim; ++d) {
                    const auto coord = rem / out.dims[d];
                    rem %= out.dims[d];
                    src_off += (d == axis ? sel : coord) * t.dims[d];
                }
                std::memcpy(dst + flat * elem, src + src_off * elem, elem);
            }
        };
        using DType = DataTypes::DType;
        switch (idx.dtype) {
            case DType::Int8: run(reinterpret_cast<const int8_t*>(idx_base)); break;
            case DType::UInt8: run(reinterpret_cast<const uint8_t*>(idx_base)); break;
            case DType::Int16: run(reinterpret_cast<const int16_t*>(idx_base)); break;
            case DType::UInt16: run(reinterpret_cast<const uint16_t*>(idx_base)); break;
            case DType::Int32: run(reinterpret_cast<const int32_t*>(idx_base)); break;
            case DType::UInt32: run(reinterpret_cast<const uint32_t*>(idx_base)); break;
            case DType::Int64: run(reinterpret_cast<const int64_t*>(idx_base)); break;
            case DType::UInt64: run(reinterpret_cast<const uint64_t*>(idx_base)); break;
            [[unlikely]] default: throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * Every input contributes one contiguous slab per outer position.
     * @brief CPU concatenation along axis.
     * @param tensors Input list (validated by Tensor::cat).
     * @param out Fresh output tensor, summed extent along axis.
     * @param axis Axis concatenated along.
     */
    void cat_cpu(const std::vector<Tensor>& tensors, Tensor& out, const size_t axis) {
        for (const auto& t : tensors) {
            if (t.device != Device::CPU || out.device != Device::CPU) {
                throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
            }
        }
        const auto elem = DataTypes::dtype_bytes(out.dtype);
        const auto outer = span_product(out.shape, 0, axis);
        const auto inner = span_product(out.shape, axis + 1, out.ndim);
        const auto out_extent = out.shape[axis];
        auto* dst = static_cast<uint8_t*>(out.data);
        auto running = size_t{0};
        for (const auto& t : tensors) {
            const auto extent = t.shape[axis];
            const auto* src = static_cast<const uint8_t*>(t.data) + t.offset * elem;
            for (auto o = size_t{0}; o < outer; ++o) {
                std::memcpy(dst + (o * out_extent + running) * inner * elem, src + o * extent * inner * elem, extent * inner * elem);
            }
            running += extent;
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Validates axis, rank, dtype, device and contiguity eagerly, syncs both
     * operands, then routes to the backend. The output never joins the
     * autograd graph (no backward node exists for indexing yet), so it is
     * built with requires_grad=false.
     * @brief Select slabs along an axis by index.
     * @param axis Axis to select along.
     * @param indices Rank-1 integer index tensor on the same device.
     * @return New tensor, this tensor's shape with shape[axis] = indices.numel.
     */
    Tensor Tensor::index_select(const size_t axis, const Tensor& indices) const {
        if (axis >= this->ndim) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (indices.ndim != 1) {
            throw ErrorHandling::MismatchedShapes("index_select requires a rank-1 index tensor");
        }
        if (this->device != indices.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!detail::indexing_shared::is_integer_index_dtype(indices.dtype)) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (!this->is_contiguous || !indices.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        indices.sync();
        std::vector<size_t> out_shape(this->shape, this->shape + this->ndim);
        out_shape[axis] = indices.numel;
        auto out = Tensor(out_shape, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::index_select_cpu(*this, indices, out, axis);
        } else if (this->device == Device::GPU) {
            detail::index_select_gpu(*this, indices, out, axis);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    /**
     * @brief Gather elements along an axis by an index tensor.
     * @param axis Axis to gather along.
     * @param indices Integer index tensor with the same rank as this tensor;
     * its extents must not exceed this tensor's on the non-axis dimensions.
     * @return New tensor with the indices' shape and this tensor's dtype.
     */
    Tensor Tensor::gather(const size_t axis, const Tensor& indices) const {
        if (axis >= this->ndim) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (indices.ndim != this->ndim) {
            throw ErrorHandling::MismatchedShapes("gather requires an index tensor of the same rank");
        }
        for (auto d = size_t{0}; d < this->ndim; ++d) {
            if (d != axis && indices.shape[d] > this->shape[d]) {
                throw ErrorHandling::MismatchedShapes("gather index extents must not exceed the input's off-axis");
            }
        }
        if (this->device != indices.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!detail::indexing_shared::is_integer_index_dtype(indices.dtype)) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (!this->is_contiguous || !indices.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        indices.sync();
        auto out = Tensor(std::vector<size_t>(indices.shape, indices.shape + indices.ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::gather_cpu(*this, indices, out, axis);
        } else if (this->device == Device::GPU) {
            detail::gather_gpu(*this, indices, out, axis);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    /**
     * @brief Concatenate a list of tensors along an axis.
     * @param tensors Non-empty list; equal rank/dtype/device, equal shapes off-axis.
     * @param axis Axis to concatenate along.
     * @return New contiguous tensor with the summed extent along axis.
     */
    Tensor Tensor::cat(const std::vector<Tensor>& tensors, const size_t axis) {
        if (tensors.empty()) {
            throw ErrorHandling::MismatchedShapes("cat requires at least one tensor");
        }
        const auto& first = tensors.front();
        if (axis >= first.ndim) {
            throw ErrorHandling::IndexOutOfRange();
        }
        for (const auto& t : tensors) {
            if (t.ndim != first.ndim) {
                throw ErrorHandling::MismatchedShapes("cat requires equal ranks");
            }
            if (t.dtype != first.dtype) {
                throw ErrorHandling::MismatchedDTypes();
            }
            if (t.device != first.device) {
                throw ErrorHandling::MismatchedDevices();
            }
            if (!t.is_contiguous) {
                throw ErrorHandling::NotImplemented();
            }
            for (auto d = size_t{0}; d < first.ndim; ++d) {
                if (d != axis && t.shape[d] != first.shape[d]) {
                    throw ErrorHandling::MismatchedShapes("cat requires equal shapes except along the concatenation axis");
                }
            }
            t.sync();
        }
        std::vector<size_t> out_shape(first.shape, first.shape + first.ndim);
        out_shape[axis] = 0;
        for (const auto& t : tensors) {
            out_shape[axis] += t.shape[axis];
        }
        auto out = Tensor(out_shape, first.dtype, first.device, false);
        if (first.device == Device::CPU) {
            detail::cat_cpu(tensors, out, axis);
        } else if (first.device == Device::GPU) {
            detail::cat_gpu(tensors, out, axis);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }
}
