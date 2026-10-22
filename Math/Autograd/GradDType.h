/**
 * @file GradDType.h
 * @author GreenBean
 * @brief The single conversion rule for gradients crossing a dtype boundary
 * under backward(DType), plus the intermediate-delta probe (a test hook).
 * backward() makes the gradient dtype a per-call parameter: the seed and every
 * node-to-node delta ARE the assigned dtype, and every cast a backward node
 * performs targets that assigned dtype through the helpers below — never a
 * literal dtype. Header-only: everything here is inline.
 * @version 0.1
 * @date 2026-10-22
 */
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"
#include "Intrinsics/ConversionMethods.h"

namespace BeanTensor::Tensors::detail {

    // --------------------------- the single grad-conversion rule ---------------------------

    /**
     * Both helpers funnel into Tensor::convert_dtype under PERMISSIVE — this is
     * the ONE place gradient dtype-crossing semantics are defined; backward
     * nodes and accumulate_grad reference it, never re-derive it. The verified
     * contract (Intrinsics/Dispatcher.h scalar_convert, PERMISSIVE):
     * float -> float narrows round-to-nearest-even, overflow clamps to the
     * target range, NaN stays NaN; float -> int truncates toward zero, clamps
     * to the target's [lowest, max], NaN -> 0; int -> int follows the
     * scalar_convert clamp contract; packed dtypes convert per lane
     * (unpacked -> packed duplicates the value into both lanes, packed ->
     * unpacked keeps the low lane).
     */

    /**
     * Deep-copies t and converts the copy to dst under the single
     * grad-conversion rule; blocks until the conversion has landed. Used at
     * node boundaries where the source tensor (an operand or the upstream
     * delta) must not be modified.
     * @brief Boundary cast: copy of t in dtype dst.
     * @param t Source tensor (not modified).
     * @param dst Target dtype — the assigned grad dtype (or a pinned leaf buffer's dtype).
     * @return A new tensor holding t's values in dtype dst.
     */
    [[nodiscard]] inline Tensor grad_converted_copy(const Tensor& t, const DataTypes::DType& dst) {
        // clone() is non-const only because it queues onto t's pool; it never
        // mutates t's observable state.
        auto out = const_cast<Tensor&>(t).clone(true);
        out.convert_dtype(dst, ConversionClampMethod::PERMISSIVE);
        out.sync();
        return out;
    }

    /**
     * Converts t to dst in place under the single grad-conversion rule (a
     * no-op when the dtypes already match) and blocks until it has landed.
     * Used on node-local scratch (transposes, freshly computed deltas) and on
     * deltas arriving at a pinned leaf grad buffer.
     * @brief Boundary cast: t converted to dtype dst in place.
     * @param t Tensor to convert (node-local scratch or an incoming delta).
     * @param dst Target dtype — the assigned grad dtype (or a pinned leaf buffer's dtype).
     */
    inline void grad_convert_inplace(Tensor& t, const DataTypes::DType& dst) {
        t.convert_dtype(dst, ConversionClampMethod::PERMISSIVE);
        t.sync();
    }

    // --------------------------- intermediate-delta probe (test hook) ---------------------------

    /**
     * Test hook, always on: grad_probe_hit records that a backward node
     * delivered a delta of the recorded dtype into an INTERMEDIATE
     * (node-owning) tensor — proof that node-to-node deltas actually carry the
     * assigned grad dtype, which leaf-grad inspection alone cannot show.
     * One slot per DType enumerator (15 today), rounded up.
     */
    inline std::array<std::atomic<uint64_t>, 16> g_grad_dtype_probe{};

    /**
     * @brief Record one intermediate-delta delivery of dtype dt (called by backward node fns).
     * @param dt Dtype of the delta that was delivered.
     */
    inline void grad_probe_hit(const DataTypes::DType& dt) {
        if (
            const auto idx = static_cast<size_t>(dt);
            idx < g_grad_dtype_probe.size()
        ) {
            g_grad_dtype_probe[idx].fetch_add(1, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Zero every probe counter.
     */
    inline void grad_probe_reset() {
        for (auto& c : g_grad_dtype_probe) {
            c.store(0, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Read the probe counter for a dtype.
     * @param dt Dtype to query.
     * @return Number of intermediate deltas recorded in dt since the last reset.
     */
    [[nodiscard]] inline uint64_t grad_probe_count(const DataTypes::DType& dt) {
        if (
            const auto idx = static_cast<size_t>(dt);
            idx < g_grad_dtype_probe.size()
        ) {
            return g_grad_dtype_probe[idx].load(std::memory_order_relaxed);
        }
        return 0;
    }
}
