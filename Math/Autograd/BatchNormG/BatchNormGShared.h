/**
 * @file BatchNormGShared.h
 * @author GreenBean
 * @brief Pinned arithmetic-order contract for the batch_norm backward. The
 * train-mode formulas below are the exact analytic inverse of the forward
 * pinned in Math/Normalization/BatchNorm/BatchNormShared.h, with M the
 * BIASED per-channel element count (per_channel_count); the eval-mode
 * formulas further below treat the running statistics as graph constants
 * delivered through Node::_saved. Header-only and scalar-only (no Tensor
 * access), so every backend TU — CPU, CUDA device code, HIP device code —
 * can include it. Every backend must perform exactly the operations pinned
 * here, in this order.
 * @version 0.2
 * @date 2026-10-30
 */
#pragma once

#include <cstddef>

namespace BeanTensor::Tensors::detail::batch_norm_g_shared {

    // --- Pinned backward arithmetic-order contract ---
    //
    //   Pass 1 recomputes per-channel mean and invstd from x with the
    //   forward's accumulation rules (BatchNormShared.h, unchanged).
    //   Pass 2 computes per-channel sum_dy = sum(dy) and
    //   sum_dyx = sum(dy * x_hat) with x_hat = (x - mean) * invstd.
    //   Then:
    //     dgamma[c] = sum_dyx[c]
    //     dbeta[c]  = sum_dy[c]
    //     dx[i] = (gamma[c] * invstd[c] / M) * (M * dy[i] - sum_dy[c] - x_hat[i] * sum_dyx[c])
    //
    // Float-reduction accumulation order in the passes is backend-defined
    // (the mean/sum exemption); the dx element formula itself is bit-exact
    // across backends for fixed per-channel statistics.

    /**
     * Pinned train-mode dx element: dx = (gamma*invstd/M) * (M*dy - sum_dy -
     * x_hat*sum_dyx), each step rounded in W, no reassociation.
     * @brief One input-gradient element of train-mode batch_norm backward.
     * @param gamma Channel scale widened to W.
     * @param invstd Per-channel invstd in W (recomputed from x).
     * @param m Per-channel element count M, as W.
     * @param dy Upstream gradient element widened to W.
     * @param sum_dy Per-channel sum of dy in W.
     * @param x_hat Normalized input element (x - mean) * invstd in W.
     * @param sum_dyx Per-channel sum of dy * x_hat in W.
     * @return dx in W (the caller narrows once into the grad dtype).
     */
    template<typename W>
    [[nodiscard]] W dx_element(const W gamma, const W invstd, const W m,
                               const W dy, const W sum_dy, const W x_hat, const W sum_dyx) {
        return (gamma * invstd / m) * (m * dy - sum_dy - x_hat * sum_dyx);
    }

    // --- Pinned eval backward arithmetic-order contract ---
    //
    //   Per-channel mean and invstd come from the SAVED running buffers
    //   (Node::_saved[0] = running_mean, _saved[1] = running_var), read live
    //   at backward() time — running stats are graph constants in eval
    //   backward, so the train correction terms vanish:
    //     invstd[c] = 1 / sqrt(running_var[c] + eps)   (inv_std, BatchNormShared.h)
    //     x_hat[i] = (x[i] - running_mean[c]) * invstd[c]
    //     dgamma[c] = sum(dy * x_hat)   (reduction order backend-defined)
    //     dbeta[c]  = sum(dy)           (reduction order backend-defined)
    //     dx[i] = (gamma[c] * invstd[c]) * dy[i]
    //
    // Float-reduction accumulation order in the sums is backend-defined (the
    // mean/sum exemption); the dx element formula itself is bit-exact across
    // backends for fixed running statistics.

    /**
     * Pinned eval-mode dx element: dx = (gamma * invstd) * dy, each step
     * rounded in W, no reassociation — no train correction terms because the
     * running statistics are graph constants.
     * @brief One input-gradient element of eval-mode batch_norm backward.
     * @param gamma Channel scale widened to W.
     * @param invstd Per-channel invstd in W (from the saved running_var + eps).
     * @param dy Upstream gradient element widened to W.
     * @return dx in W (the caller narrows once into the grad dtype).
     */
    template<typename W>
    [[nodiscard]] W eval_dx_element(const W gamma, const W invstd, const W dy) {
        return (gamma * invstd) * dy;
    }
}
