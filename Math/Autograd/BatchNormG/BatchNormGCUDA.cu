/**
 * @file BatchNormGCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the batch_norm backward launchers. Mirrors the CPU
 * backward (Math/Autograd/BatchNormG/BatchNormGCPU.cpp): the train launcher
 * recomputes per-channel mean/invstd from x with the forward's accumulation
 * rules (per-channel-block partials merged by a shared-memory tree — the
 * float-order exemption covers this merge order), then one reduction pass
 * computes per-channel sum(dy) and sum(dy * x_hat), then dgamma =
 * sum(dy * x_hat), dbeta = sum(dy) and the pinned dx element formula; the
 * eval launcher reads per-channel mean/invstd from the saved running buffers
 * (graph constants) and uses the pinned eval dx element with no train
 * correction terms. Every non-reduction expression reproduces
 * Math/Autograd/BatchNormG/BatchNormGShared.h and
 * Math/Normalization/BatchNorm/BatchNormShared.h verbatim (the dev_* twins
 * below are expression-for-expression copies; -fmad=false on this TU bars
 * contraction). Delta buffers carry grad_dtype by construction, narrowed
 * from the working precision once on store. Divergent twin of the Task 7 HIP
 * backend — same fp32/fp64 compute widths, same RNE narrowing.
 * @version 0.2
 * @date 2026-10-30
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Autograd/AutogradCudaShared.h"
#include "Math/Autograd/BatchNormG/BatchNormG.h"
#include "Math/Normalization/BatchNorm/BatchNormShared.h"

#include "Helpers/Errors.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_gpu;
    using batch_norm_shared::per_channel_count;

    namespace {
        /**
         * Identity when the storage matches the working precision, device
         * widen for fp16/bf16 lanes, explicit conversion otherwise (dy arrives
         * in grad_dtype and converts into W — narrowing included, e.g. an
         * fp64 delta against fp32 storage). Device twin of the CPU backward's
         * to_working (BatchNormGCPU.cpp).
         * @brief Widen one storage element into the working precision W.
         * @tparam W Working precision (fp32 unless the input storage is fp64).
         * @tparam T Storage element type.
         * @param v Element.
         * @return The element in W.
         */
        template<typename W, typename T>
        __device__ __forceinline__ W dev_to_working(const T v) {
            if constexpr (std::is_same_v<W, T>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return static_cast<W>(__half2float(v));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return static_cast<W>(__bfloat162float(v));
            } else {
                return static_cast<W>(v);
            }
        }

        /**
         * Identity when the working precision matches the storage, RNE narrow
         * for fp16/bf16 lanes (bf16 through the CPU-matching flush), explicit
         * narrowing conversion otherwise (RNE fp64 -> fp32, the same result
         * the PERMISSIVE grad-conversion rule produces). Device twin of the
         * CPU backward's from_working.
         * @brief Narrow one working-precision value into the storage dtype T.
         * @tparam T Storage element type.
         * @tparam W Working precision.
         * @param v Value in W.
         * @return The value narrowed into T.
         */
        template<typename T, typename W>
        __device__ __forceinline__ T dev_from_working(const W v) {
            if constexpr (std::is_same_v<W, T>) {
                return v;
            } else if constexpr (std::is_same_v<T, float16_t>) {
                return __float2half_rn(static_cast<float32_t>(v));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                return dev_narrow_bf16(static_cast<float32_t>(v));
            } else {
                return static_cast<T>(v);
            }
        }

        /**
         * Verbatim device twin of batch_norm_shared::biased_variance.
         * @brief Biased per-channel variance from the mean and the sum-of-squares.
         * @param mean Per-channel mean (sum / M) in W.
         * @param sumsq Per-channel sum-of-squares in W.
         * @param m Per-channel element count M, as W.
         * @return The biased variance in W.
         */
        template<typename W>
        __device__ __forceinline__ W dev_biased_variance(const W mean, const W sumsq, const W m) {
            return sumsq / m - mean * mean;
        }

        /**
         * Verbatim device twin of batch_norm_shared::inv_std.
         * @brief Per-channel inverse standard deviation.
         * @param var Biased per-channel variance in W.
         * @param eps Epsilon narrowed once into W.
         * @return invstd in W.
         */
        template<typename W>
        __device__ __forceinline__ W dev_inv_std(const W var, const W eps) {
            return W{1} / sqrt(var + eps);
        }

        /**
         * Verbatim device twin of batch_norm_g_shared::dx_element.
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
        __device__ __forceinline__ W dev_dx_element(const W gamma, const W invstd, const W m,
                                                    const W dy, const W sum_dy, const W x_hat, const W sum_dyx) {
            return (gamma * invstd / m) * (m * dy - sum_dy - x_hat * sum_dyx);
        }

        /**
         * Verbatim device twin of batch_norm_g_shared::eval_dx_element.
         * @brief One input-gradient element of eval-mode batch_norm backward.
         * @param gamma Channel scale widened to W.
         * @param invstd Per-channel invstd in W (from the saved running_var + eps).
         * @param dy Upstream gradient element widened to W.
         * @return dx in W (the caller narrows once into the grad dtype).
         */
        template<typename W>
        __device__ __forceinline__ W dev_eval_dx_element(const W gamma, const W invstd, const W dy) {
            return (gamma * invstd) * dy;
        }

        /**
         * Train pass 1: recompute per-channel mean/invstd from x with the
         * forward's accumulation rules — one block per channel, block-strided
         * per-thread partials of sum and sum-of-squares merged by a
         * shared-memory tree, thread 0 computes the pinned mean and
         * invstd = inv_std(biased_variance(mean, sumsq, m), eps). Flat element
         * of channel c at per-channel offset k:
         * (k / inner) * (channels * inner) + c * inner + (k % inner).
         * @brief Train-mode per-channel statistics recompute kernel.
         * @param x Input elements.
         * @param means Out: per-channel mean in W (channels elements).
         * @param invstds Out: per-channel invstd in W (channels elements).
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param channels Channel count C (also the grid size).
         * @param count Total element count.
         * @param m Per-channel element count M, as W.
         * @param eps_w Epsilon narrowed once into W.
         */
        template<typename T, typename W>
        __global__ void bng_train_stats_kernel(const T* __restrict__ x, W* __restrict__ means, W* __restrict__ invstds,
                                               const size_t inner, const size_t channels, const size_t count,
                                               const W m, const W eps_w) {
            extern __shared__ uint8_t s_base[];
            auto* const s_sum = reinterpret_cast<W*>(s_base);
            auto* const s_sq = s_sum + blockDim.x;
            const size_t c = blockIdx.x;
            const auto tid = threadIdx.x;
            const auto per_channel = count / channels;
            auto sum = W{0};
            auto sumsq = W{0};
            for (size_t k = tid; k < per_channel; k += blockDim.x) {
                const auto flat = (k / inner) * (channels * inner) + c * inner + (k % inner);
                const auto w = dev_to_working<W>(x[flat]);
                sum += w;
                sumsq += w * w;
            }
            s_sum[tid] = sum;
            s_sq[tid] = sumsq;
            __syncthreads();
            for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_sum[tid] += s_sum[tid + s];
                    s_sq[tid] += s_sq[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const auto mean = s_sum[0] / m;
                means[c] = mean;
                invstds[c] = dev_inv_std(dev_biased_variance(mean, s_sq[0], m), eps_w);
            }
        }

        /**
         * Eval statistics: per-channel mean/invstd from the saved running
         * buffers (graph constants, read live at backward() time) in W, the
         * pinned inv_std. Grid-stride over the channels.
         * @brief Eval-mode per-channel statistics kernel.
         * @param rm Saved running mean buffer (read-only).
         * @param rv Saved running variance buffer (read-only).
         * @param means Out: per-channel mean in W (channels elements).
         * @param invstds Out: per-channel invstd in W (channels elements).
         * @param eps_w Epsilon narrowed once into W.
         * @param channels Channel count C.
         */
        template<typename T, typename W>
        __global__ void bng_eval_stats_kernel(const T* __restrict__ rm, const T* __restrict__ rv,
                                              W* __restrict__ means, W* __restrict__ invstds,
                                              const W eps_w, const size_t channels) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t c = threadIdx.x + blockIdx.x * blockDim.x; c < channels; c += stride) {
                const auto var = dev_to_working<W>(rv[c]);
                means[c] = dev_to_working<W>(rm[c]);
                invstds[c] = dev_inv_std(var, eps_w);
            }
        }

        /**
         * One block per channel: block-strided per-thread partials of dy and
         * dy * x_hat (x_hat = (x - mean) * invstd with the channel's
         * statistics), merged by a shared-memory tree; thread 0 stores the
         * per-channel sums in W and, when requested, dgamma = sum(dy * x_hat)
         * and dbeta = sum(dy) narrowed once into grad_dtype. Shared by the
         * train and eval launchers (they differ only in where means/invstds
         * came from).
         * @brief Per-channel sum(dy)/sum(dy*x_hat) reduction kernel.
         * @param x Input elements.
         * @param dy Upstream gradient elements, in grad_dtype.
         * @param means Per-channel mean in W.
         * @param invstds Per-channel invstd in W.
         * @param sum_dy Out: per-channel sum of dy in W.
         * @param sum_dyx Out: per-channel sum of dy * x_hat in W.
         * @param dg Out: dgamma in grad_dtype, or nullptr when not required.
         * @param db Out: dbeta in grad_dtype, or nullptr when not required.
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param channels Channel count C (also the grid size).
         * @param count Total element count.
         */
        template<typename T, typename W, typename G>
        __global__ void bng_dysums_kernel(const T* __restrict__ x, const G* __restrict__ dy,
                                          const W* __restrict__ means, const W* __restrict__ invstds,
                                          W* __restrict__ sum_dy, W* __restrict__ sum_dyx,
                                          G* __restrict__ dg, G* __restrict__ db,
                                          const size_t inner, const size_t channels, const size_t count) {
            extern __shared__ uint8_t s_base[];
            auto* const s_dy = reinterpret_cast<W*>(s_base);
            auto* const s_dyx = s_dy + blockDim.x;
            const size_t c = blockIdx.x;
            const auto tid = threadIdx.x;
            const auto per_channel = count / channels;
            auto acc_dy = W{0};
            auto acc_dyx = W{0};
            for (size_t k = tid; k < per_channel; k += blockDim.x) {
                const auto flat = (k / inner) * (channels * inner) + c * inner + (k % inner);
                const auto dyw = dev_to_working<W>(dy[flat]);
                const auto x_hat = (dev_to_working<W>(x[flat]) - means[c]) * invstds[c];
                acc_dy += dyw;
                acc_dyx += dyw * x_hat;
            }
            s_dy[tid] = acc_dy;
            s_dyx[tid] = acc_dyx;
            __syncthreads();
            for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_dy[tid] += s_dy[tid + s];
                    s_dyx[tid] += s_dyx[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                sum_dy[c] = s_dy[0];
                sum_dyx[c] = s_dyx[0];
                if (dg != nullptr) {
                    dg[c] = dev_from_working<G>(s_dyx[0]);
                }
                if (db != nullptr) {
                    db[c] = dev_from_working<G>(s_dy[0]);
                }
            }
        }

        /**
         * Grid-stride over every element: c = channel_of_flat(i), dx[i] =
         * dx_element(widen(gamma[c]), invstds[c], m, widen(dy[i]), sum_dy[c],
         * x_hat, sum_dyx[c]) with x_hat = (widen(x[i]) - means[c]) *
         * invstds[c], narrowed once into grad_dtype on store.
         * @brief Train-mode dx kernel (pinned dx element formula).
         * @param x Input elements.
         * @param dy Upstream gradient elements, in grad_dtype.
         * @param g Per-channel scale.
         * @param means Per-channel mean in W (recomputed from x).
         * @param invstds Per-channel invstd in W (recomputed from x).
         * @param sum_dy Per-channel sum of dy in W.
         * @param sum_dyx Per-channel sum of dy * x_hat in W.
         * @param dx Out: input gradient in grad_dtype.
         * @param m Per-channel element count M, as W.
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param channels Channel count C.
         * @param count Total element count.
         */
        template<typename T, typename W, typename G>
        __global__ void bng_train_dx_kernel(const T* __restrict__ x, const G* __restrict__ dy, const T* __restrict__ g,
                                            const W* __restrict__ means, const W* __restrict__ invstds,
                                            const W* __restrict__ sum_dy, const W* __restrict__ sum_dyx,
                                            G* __restrict__ dx, const W m,
                                            const size_t inner, const size_t channels, const size_t count) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += stride) {
                const auto c = (i / inner) % channels;
                const auto dyw = dev_to_working<W>(dy[i]);
                const auto x_hat = (dev_to_working<W>(x[i]) - means[c]) * invstds[c];
                dx[i] = dev_from_working<G>(dev_dx_element(dev_to_working<W>(g[c]), invstds[c], m,
                                                           dyw, sum_dy[c], x_hat, sum_dyx[c]));
            }
        }

        /**
         * Grid-stride over every element: c = channel_of_flat(i), dx[i] =
         * eval_dx_element(widen(gamma[c]), invstds[c], widen(dy[i])) — no
         * train correction terms, the running statistics are graph constants —
         * narrowed once into grad_dtype on store.
         * @brief Eval-mode dx kernel (pinned eval dx element formula).
         * @param dy Upstream gradient elements, in grad_dtype.
         * @param g Per-channel scale.
         * @param invstds Per-channel invstd in W (from the saved running_var + eps).
         * @param dx Out: input gradient in grad_dtype.
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param channels Channel count C.
         * @param count Total element count.
         */
        template<typename T, typename W, typename G>
        __global__ void bng_eval_dx_kernel(const G* __restrict__ dy, const T* __restrict__ g,
                                           const W* __restrict__ invstds, G* __restrict__ dx,
                                           const size_t inner, const size_t channels, const size_t count) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += stride) {
                const auto c = (i / inner) % channels;
                dx[i] = dev_from_working<G>(dev_eval_dx_element(dev_to_working<W>(g[c]), invstds[c],
                                                                dev_to_working<W>(dy[i])));
            }
        }

        /**
         * eps narrows once into the working precision here (the pinned single
         * narrowing), M through per_channel_count. The means/invstds/sum
         * scratch is launcher-local alloc/free; the free syncs the device, and
         * the explicit stream sync leaves every requested delta complete
         * before the launcher returns (the node trampoline accumulates right
         * after).
         * @brief Host-side CUDA train-mode batch_norm backward for one storage/grad type pair.
         * @tparam T Input storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         * @tparam G Grad storage element type (grad_dtype).
         * @param a Device argument pack (offset-applied device pointers).
         */
        template<typename T, typename W, typename G>
        __host__ void run_batch_norm_g_train(const BatchNormGArgs& a) {
            const auto m = static_cast<W>(per_channel_count(a.count, a.channels));
            const auto eps_w = static_cast<W>(a.eps);
            W* scratch = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&scratch, 4 * a.channels * sizeof(W)));
            auto* const means = scratch;
            auto* const invstds = scratch + a.channels;
            auto* const sum_dy = scratch + 2 * a.channels;
            auto* const sum_dyx = scratch + 3 * a.channels;
            const auto* const x_data = static_cast<const T*>(a.x);
            const auto* const dy_data = static_cast<const G*>(a.dy);
            const auto* const g_data = static_cast<const T*>(a.gamma);
            const auto stats_shared = 2 * BT_GPU_THREADS * sizeof(W);
            bng_train_stats_kernel<T, W><<<a.channels, BT_GPU_THREADS, stats_shared>>>(
                x_data, means, invstds, a.inner, a.channels, a.count, m, eps_w);
            bng_dysums_kernel<T, W, G><<<a.channels, BT_GPU_THREADS, stats_shared>>>(
                x_data, dy_data, means, invstds, sum_dy, sum_dyx,
                static_cast<G*>(a.dgamma), static_cast<G*>(a.dbeta), a.inner, a.channels, a.count);
            if (a.dx != nullptr) {
                bng_train_dx_kernel<T, W, G><<<grid_for(a.count), BT_GPU_THREADS>>>(
                    x_data, dy_data, g_data, means, invstds, sum_dy, sum_dyx, static_cast<G*>(a.dx), m,
                    a.inner, a.channels, a.count);
            }
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(scratch));
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        }

        /**
         * Per-channel mean/invstd come from the saved running buffers; the
         * dy-sums pass only runs when dgamma or dbeta is requested (the eval
         * dx element has no train correction terms and needs neither sum).
         * Same scratch/sync discipline as the train launcher.
         * @brief Host-side CUDA eval-mode batch_norm backward for one storage/grad type pair.
         * @tparam T Input storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         * @tparam G Grad storage element type (grad_dtype).
         * @param a Device argument pack (offset-applied device pointers).
         */
        template<typename T, typename W, typename G>
        __host__ void run_batch_norm_g_eval(const BatchNormGArgs& a) {
            const auto eps_w = static_cast<W>(a.eps);
            W* scratch = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&scratch, 4 * a.channels * sizeof(W)));
            auto* const means = scratch;
            auto* const invstds = scratch + a.channels;
            auto* const sum_dy = scratch + 2 * a.channels;
            auto* const sum_dyx = scratch + 3 * a.channels;
            const auto* const x_data = static_cast<const T*>(a.x);
            const auto* const dy_data = static_cast<const G*>(a.dy);
            const auto* const g_data = static_cast<const T*>(a.gamma);
            const auto* const rm_data = static_cast<const T*>(a.running_mean);
            const auto* const rv_data = static_cast<const T*>(a.running_var);
            bng_eval_stats_kernel<T, W><<<grid_for(a.channels), BT_GPU_THREADS>>>(
                rm_data, rv_data, means, invstds, eps_w, a.channels);
            if (a.dgamma != nullptr || a.dbeta != nullptr) {
                const auto stats_shared = 2 * BT_GPU_THREADS * sizeof(W);
                bng_dysums_kernel<T, W, G><<<a.channels, BT_GPU_THREADS, stats_shared>>>(
                    x_data, dy_data, means, invstds, sum_dy, sum_dyx,
                    static_cast<G*>(a.dgamma), static_cast<G*>(a.dbeta), a.inner, a.channels, a.count);
            }
            if (a.dx != nullptr) {
                bng_eval_dx_kernel<T, W, G><<<grid_for(a.count), BT_GPU_THREADS>>>(
                    dy_data, g_data, invstds, static_cast<G*>(a.dx), a.inner, a.channels, a.count);
            }
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(scratch));
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        }
    }

    /**
     * in_dtype picks the working precision (fp64 storage accumulates in fp64,
     * everything else in fp32); grad_dtype picks the delta storage. An
     * unsupported dtype throws ErrorHandling::NotImplemented (the forward only
     * supports float inputs, so grad_dtype outside the float set is refused).
     * @brief GPU train-mode batch_norm backward.
     * @param args Device argument pack (offset-applied device pointers; running buffers unused).
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    __host__ void batch_norm_g_train_gpu(const BatchNormGArgs& args) {
        auto run = [&]<typename T, typename W>() {
            auto run_g = [&]<typename G>() {
                run_batch_norm_g_train<T, W, G>(args);
            };
            switch (args.grad_dtype) {
                case DataTypes::DType::Float32: run_g.template operator()<float32_t>(); break;
                case DataTypes::DType::Float64: run_g.template operator()<float64_t>(); break;
                case DataTypes::DType::Float16: run_g.template operator()<float16_t>(); break;
                case DataTypes::DType::BFloat16: run_g.template operator()<bfloat16_t>(); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        };
        switch (args.in_dtype) {
            case DataTypes::DType::Float32: run.template operator()<float32_t, float32_t>(); break;
            case DataTypes::DType::Float64: run.template operator()<float64_t, float64_t>(); break;
            case DataTypes::DType::Float16: run.template operator()<float16_t, float32_t>(); break;
            case DataTypes::DType::BFloat16: run.template operator()<bfloat16_t, float32_t>(); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * Same dtype ladders as the train launcher; the running buffers in the
     * argument pack supply the per-channel statistics.
     * @brief GPU eval-mode batch_norm backward.
     * @param args Device argument pack (offset-applied device pointers, running buffers included).
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    __host__ void batch_norm_g_eval_gpu(const BatchNormGArgs& args) {
        auto run = [&]<typename T, typename W>() {
            auto run_g = [&]<typename G>() {
                run_batch_norm_g_eval<T, W, G>(args);
            };
            switch (args.grad_dtype) {
                case DataTypes::DType::Float32: run_g.template operator()<float32_t>(); break;
                case DataTypes::DType::Float64: run_g.template operator()<float64_t>(); break;
                case DataTypes::DType::Float16: run_g.template operator()<float16_t>(); break;
                case DataTypes::DType::BFloat16: run_g.template operator()<bfloat16_t>(); break;
                [[unlikely]] default: throw ErrorHandling::NotImplemented();
            }
        };
        switch (args.in_dtype) {
            case DataTypes::DType::Float32: run.template operator()<float32_t, float32_t>(); break;
            case DataTypes::DType::Float64: run.template operator()<float64_t, float64_t>(); break;
            case DataTypes::DType::Float16: run.template operator()<float16_t, float32_t>(); break;
            case DataTypes::DType::BFloat16: run.template operator()<bfloat16_t, float32_t>(); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }
}
#endif
