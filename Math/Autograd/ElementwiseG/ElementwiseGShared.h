/**
 * @file ElementwiseGShared.h
 * @author GreenBean
 * @brief Host-side plan shared by the elementwise backward backends
 * (CPU/HIP/CUDA): the broadcast-reduce execution plan built by
 * Tensor::build_bcast_reduce_plan and the preimage enumerator that maps each
 * operand element back to the output indices reducing into it. The GPU
 * backends carry a device-side twin of the enumerator (device code cannot
 * call this host inline); the two must stay behaviorally identical.
 * Header-only: everything here is inline.
 * @version 0.1
 * @date 2026-10-23
 */
#pragma once

#include <array>
#include <cstddef>

#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {

    /**
     * Fixed 3 entries == BT_TENSOR_MAX_DIMS. `this` of
     * Tensor::build_bcast_reduce_plan is the out-shaped delta; `op` is the
     * broadcast operand being reduced into. An operand dim of size 1 (or a
     * dim missing entirely, when the operand ranks below the output) is
     * "free": every output coordinate along that dim maps to the operand's
     * single element and the reduction sums over the whole extent.
     * Trivially copyable: passed to GPU kernels by value.
     * @brief Broadcast-reduction execution plan for one operand.
     */
    struct BcastReducePlan {
        size_t out_shape[3];
        size_t out_strides[3];
        size_t op_shape[3];
        size_t out_ndim;
        size_t op_ndim;
    };

    /**
     * Decomposes the operand flat index j into operand coordinates, then
     * walks the Cartesian product of the free (broadcast) dims to enumerate,
     * in increasing mixed-radix order, every output flat index whose
     * broadcast mapping lands on j. f is invoked once per such index; the
     * enumeration order is fixed (independent of chunking), which is what
     * makes the float reduction bit-identical across thread counts and
     * against the GPU twin. Every output index is enumerated exactly once
     * across all j, so the reduction is a partition of the delta.
     * @brief Invoke f(i) for every output index i reducing into operand element j.
     * @param plan Broadcast-reduction plan.
     * @param j Operand flat index.
     * @param f Callable invoked with each output flat index.
     */
    template<typename F>
    inline void for_each_preimage(const BcastReducePlan& p, const size_t j, F&& f) {
        // Contiguous strides of the operand, from its shape.
        auto op_strides_c = std::array<size_t, 3>{size_t{1}, size_t{1}, size_t{1}};
        for (auto od = p.op_ndim - 1; od-- > 0;) {
            op_strides_c[od] = op_strides_c[od + 1] * p.op_shape[od + 1];
        }
        auto coord = std::array<size_t, 3>{size_t{0}, size_t{0}, size_t{0}};
        auto rem = j;
        for (auto od = p.op_ndim; od-- > 0;) {
            coord[od] = rem / op_strides_c[od];
            rem -= coord[od] * op_strides_c[od];
        }
        // Free dims (left-padded missing dims + size-1 dims) and the fixed
        // output-offset contribution of j's coordinates.
        auto free_dim = std::array<size_t, 3>{size_t{0}, size_t{0}, size_t{0}};
        auto free_ext = std::array<size_t, 3>{size_t{1}, size_t{1}, size_t{1}};
        auto k = size_t{0};
        auto base = size_t{0};
        const auto lead = p.out_ndim - p.op_ndim;
        for (auto d = size_t{0}; d < p.out_ndim; ++d) {
            if (d < lead) {
                free_dim[k] = d;
                free_ext[k] = p.out_shape[d];
                ++k;
            } else if (
                const auto od = d - lead;
                p.op_shape[od] == 1
            ) {
                free_dim[k] = d;
                free_ext[k] = p.out_shape[d];
                ++k;
            } else {
                base += coord[od] * p.out_strides[d];
            }
        }
        // Mixed-radix divisors: div[m] = product of free_ext[n] for n > m.
        auto div = std::array<size_t, 3>{size_t{1}, size_t{1}, size_t{1}};
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
}
