/**
 * @file Slice.cpp
 * @author GreenBean
 * @brief Tensor::slice / Tensor::split: strided same-buffer views over
 * per-dimension [start, end) ranges. When the source requires grad each view
 * joins the graph with a slice_g node whose ranges ride in the node's owned
 * NodeMeta — split's chunks therefore reuse the slice backward verbatim.
 * @version 0.2
 * @date 2026-11-14
 */
#include <string>
#include <utility>
#include <vector>

#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors {

    /**
     * Metadata-only (like reshape): the result is a view over this tensor's
     * buffer with bumped offset and inherited strides, so slicing a
     * non-leading dimension yields a non-contiguous view (ops requiring
     * contiguous operands reject it with NotImplemented, matching the rest of
     * the framework). Works identically on CPU and GPU — no data moves.
     * Autograd: when this tensor requires grad the view joins the graph with a
     * slice_g node; the recorded ranges live in the node's owned NodeMeta.
     * @brief View into per-dimension [start, end) ranges of this tensor.
     * @param ranges One [start, end) range per dimension (rank must match).
     * @return View tensor sharing this tensor's buffer.
     * @throws ErrorHandling::MismatchedShapes Range count differs from the rank.
     * @throws ErrorHandling::IndexOutOfRange start > end or end > shape[d].
     */
    Tensor Tensor::slice(const std::vector<std::pair<size_t, size_t>>& ranges) {
        sync();
        if (ranges.size() != this->ndim) {
            throw ErrorHandling::MismatchedShapes("slice requires one [start, end) range per dimension");
        }
        std::vector<size_t> new_shape;
        new_shape.reserve(this->ndim);
        auto new_offset = this->offset;
        for (auto d = size_t{0}; d < this->ndim; ++d) {
            const auto& [start, end] = ranges[d];
            if (start > end || end > this->shape[d]) {
                throw ErrorHandling::IndexOutOfRange();
            }
            new_offset += start * this->dims[d];
            new_shape.push_back(end - start);
        }
        // Build a same-buffer view (same dance as reshape): the constructor
        // allocates a fresh buffer; release it while owns_data is still true,
        // then re-point at ours with the accumulated offset and our strides.
        auto view = Tensor(new_shape, this->dtype, this->device, this->requires_grad);
        view.kill_cpu_data();
        view.kill_gpu_data();
        view.owns_data = false;
        view.data = this->data;
        view.offset = new_offset;
        std::copy_n(this->dims, BT_TENSOR_MAX_DIMS, view.dims);
        // The view counts as contiguous only when the inherited strides match
        // the contiguous strides of its own shape (e.g. slicing leading
        // dimensions). Empty views have no layout to violate.
        auto contiguous = true;
        auto expected = size_t{1};
        for (auto d = this->ndim; d-- > 0;) {
            if (view.dims[d] != expected) {
                contiguous = false;
                break;
            }
            expected *= new_shape[d];
        }
        view.is_contiguous = contiguous || view.numel == 0;
        this->children.emplace_back(&view);
        view.view_parents.emplace_back(this);
        view.wire_node(detail::slice_g_bridge, *this);
        if (view.node != nullptr) {
            auto* meta = new NodeMeta{};
            meta->tag = NodeMetaTag::Slice;
            std::copy_n(this->shape, BT_TENSOR_MAX_DIMS, meta->src_shape);
            meta->src_ndim = this->ndim;
            for (auto d = size_t{0}; d < this->ndim; ++d) {
                meta->ranges[d][0] = ranges[d].first;
                meta->ranges[d][1] = ranges[d].second;
            }
            view.node->_meta = meta;
        }
        return view;
    }

    /**
     * @brief Split into equal view chunks along an axis (leading-axis chunks
     * round-trip through cat; inner-axis chunks are non-contiguous and cat
     * rejects those, per the contiguity contract in Tensor.h).
     * @param axis Axis to split along.
     * @param sections Number of equal chunks; shape[axis] must be divisible.
     * @return One slice view per chunk, in axis order.
     * @throws ErrorHandling::IndexOutOfRange axis out of bounds.
     * @throws ErrorHandling::MismatchedShapes sections is 0 or does not divide shape[axis].
     */
    std::vector<Tensor> Tensor::split(const size_t axis, const size_t sections) {
        if (axis >= this->ndim) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (sections == 0 || this->shape[axis] % sections != 0) {
            throw ErrorHandling::MismatchedShapes("split requires a nonzero section count dividing shape[axis] (" + std::to_string(this->shape[axis]) + " vs " + std::to_string(sections) + ")");
        }
        const auto chunk = this->shape[axis] / sections;
        std::vector<Tensor> pieces;
        pieces.reserve(sections);
        for (auto i = size_t{0}; i < sections; ++i) {
            std::vector<std::pair<size_t, size_t>> ranges(this->ndim);
            for (auto d = size_t{0}; d < this->ndim; ++d) {
                ranges[d] = {size_t{0}, this->shape[d]};
            }
            ranges[axis] = {i * chunk, (i + 1) * chunk};
            pieces.emplace_back(this->slice(ranges));
        }
        return pieces;
    }
}
