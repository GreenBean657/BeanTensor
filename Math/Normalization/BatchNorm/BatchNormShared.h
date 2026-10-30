/**
 * @file BatchNormShared.h
 * @author GreenBean
 * @brief Per-channel indexing helpers and the pinned arithmetic-order contract
 * for batch normalization. Header-only and scalar-only (no Tensor access), so
 * every backend TU — CPU, CUDA device code, HIP device code — can include it.
 * Every backend must perform exactly the operations pinned here, in this
 * order, so results are bit-identical across backends for fixed inputs.
 * @version 0.1
 * @date 2026-10-30
 */
#pragma once

#include <cmath>
#include <cstddef>

namespace BeanTensor::Tensors::detail::batch_norm_shared {

    /**
     * Channel layout of a contiguous rank-2 (N, C) or rank-4 (N, C, H, W)
     * tensor: the flat index i decomposes as
     *   i = n * (channels * inner) + c * inner + j
     * with n the batch index, c the channel, j the within-channel offset.
     * inner is 1 for rank 2 and H*W for rank 4.
     * @brief Channel index of a flat element index.
     * @param i Flat element index.
     * @param inner Within-channel-per-batch extent (1 or H*W).
     * @param channels Channel count C.
     * @return The channel c of element i.
     */
    [[nodiscard]] inline size_t channel_of_flat(const size_t i, const size_t inner, const size_t channels) {
        return (i / inner) % channels;
    }

    /**
     * Per-channel element count M = numel / channels (N for rank 2, N*H*W for
     * rank 4). This is the BIASED estimator divisor — batch of one gives M = 1
     * and a variance contribution of exactly 0, never a 0/0 NaN.
     * @brief Per-channel element count.
     * @param numel Total element count.
     * @param channels Channel count C.
     * @return M, the biased divisor.
     */
    [[nodiscard]] inline size_t per_channel_count(const size_t numel, const size_t channels) {
        return numel / channels;
    }

    // --- Pinned arithmetic-order contract (verbatim from the plan's Global Constraints) ---
    //
    //   invstd = 1/sqrt(var + eps); x_hat = (x - mean) * invstd; y = x_hat * gamma[c] + beta[c].
    //   Running-stat blend, evaluated as three separately rounded steps, momentum
    //   narrowed once into the working precision, no FMA contraction:
    //     a = running * (1 - m); b = batch_stat * m; new_running = a + b.
    //
    // Working precision W: fp64 for fp64 input, fp32 otherwise (fp16/bf16
    // included — no fp64 kernel math for half inputs). eps and momentum are
    // float64_t at the API and narrowed once into W. A half-precision running
    // buffer stores the blend narrowed (RNE) once on store.
    //
    // Float-reduction accumulation order is backend-defined (the mean/sum
    // exemption); everything else above is bit-exact across backends for
    // fixed inputs. The biased variance form below is pinned with it.

    /**
     * Variance estimator: BIASED everywhere (divide by M), from the phase-1
     * accumulators. Computational form: mean = sum / M; var = sumsq / M - mean * mean,
     * both rounded in W. Bit-exact across backends for fixed inputs (not part
     * of the float-order exemption: the phase-1 merges are pinned in chunk
     * order per backend).
     * @brief Biased per-channel variance from the mean and the sum-of-squares.
     * @param mean Per-channel mean (sum / M) in W.
     * @param sumsq Per-channel sum-of-squares in W.
     * @param m Per-channel element count M, as W.
     * @return The biased variance in W.
     */
    template<typename W>
    [[nodiscard]] W biased_variance(const W mean, const W sumsq, const W m) {
        return sumsq / m - mean * mean;
    }

    /**
     * Pinned normalization statistic: invstd = 1/sqrt(var + eps), computed in W.
     * @brief Per-channel inverse standard deviation.
     * @param var Biased per-channel variance in W.
     * @param eps Epsilon narrowed once into W.
     * @return invstd in W.
     */
    template<typename W>
    [[nodiscard]] W inv_std(const W var, const W eps) {
        return W{1} / std::sqrt(var + eps);
    }

    /**
     * Pinned output element: x_hat = (x - mean) * invstd; y = x_hat * gamma[c] + beta[c],
     * each step rounded in W.
     * @brief One normalized output element.
     * @param x Input element widened to W.
     * @param mean Per-channel mean in W.
     * @param invstd Per-channel invstd in W.
     * @param gamma Channel scale widened to W.
     * @param beta Channel shift widened to W.
     * @return y in W (the caller narrows once into the storage dtype).
     */
    template<typename W>
    [[nodiscard]] W normalize(const W x, const W mean, const W invstd, const W gamma, const W beta) {
        const auto x_hat = (x - mean) * invstd;
        return x_hat * gamma + beta;
    }

    /**
     * Pinned running-stat blend: three separately rounded steps in W, no FMA
     * contraction, momentum narrowed once into W.
     * @brief Blend one running buffer element with its batch statistic.
     * @param running Old buffer element widened to W.
     * @param batch_stat Batch statistic (mean or biased variance) in W.
     * @param m Momentum narrowed once into W.
     * @return The blended value in W (the caller narrows once into the buffer dtype).
     */
    template<typename W>
    [[nodiscard]] W blend_running(const W running, const W batch_stat, const W m) {
        const auto a = running * (W{1} - m);
        const auto b = batch_stat * m;
        return a + b;
    }
}
