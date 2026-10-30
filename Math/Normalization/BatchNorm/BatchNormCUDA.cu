/**
 * @file BatchNormCUDA.cu
 * @author GreenBean
 * @brief CUDA backend for the batch normalization forward. Mirrors the CPU
 * two-phase structure (Math/Normalization/BatchNorm/BatchNormCPU.cpp): train
 * mode runs a per-channel-block stats kernel (block-strided partials merged
 * by a shared-memory tree — the float-order exemption covers this merge
 * order) whose thread 0 computes the pinned mean/variance/invstd and applies
 * the pinned three-step running-stat blend in place, then a grid-stride
 * normalize kernel writes the output; eval mode reads per-channel
 * mean/invstd from the running buffers (invstd = 1/sqrt(var + eps) in the
 * working precision) and runs the same normalize kernel, leaving the buffers
 * untouched. Every non-reduction expression reproduces
 * Math/Normalization/BatchNorm/BatchNormShared.h verbatim (the dev_* twins
 * below are expression-for-expression copies; -fmad=false on this TU bars
 * contraction). Divergent twin of the Task 7 HIP backend — same fp32/fp64
 * compute widths, same RNE narrowing.
 * @version 0.2
 * @date 2026-10-30
 */
#if defined(USE_CUDA)
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Math/Normalization/BatchNorm/BatchNorm.h"
#include "Math/Normalization/BatchNorm/BatchNormShared.h"
#include "Math/Reductions/ReductionsCudaShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_gpu;
    using batch_norm_shared::per_channel_count;

    namespace {
        /**
         * Identity when the storage matches the working precision, device
         * widen for fp16/bf16 lanes, explicit conversion otherwise (e.g. fp32
         * into an fp64 working precision). Device twin of the CPU forward's
         * to_working (BatchNormCPU.cpp).
         * @brief Widen one storage element into the working precision W.
         * @tparam W Working precision (fp32 unless the storage is fp64).
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
         * narrowing conversion otherwise. Device twin of the CPU forward's
         * from_working.
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
         * Verbatim device twin of batch_norm_shared::normalize.
         * @brief One normalized output element.
         * @param x Input element widened to W.
         * @param mean Per-channel mean in W.
         * @param invstd Per-channel invstd in W.
         * @param gamma Channel scale widened to W.
         * @param beta Channel shift widened to W.
         * @return y in W (the caller narrows once into the storage dtype).
         */
        template<typename W>
        __device__ __forceinline__ W dev_normalize(const W x, const W mean, const W invstd, const W gamma, const W beta) {
            const auto x_hat = (x - mean) * invstd;
            return x_hat * gamma + beta;
        }

        /**
         * Verbatim device twin of batch_norm_shared::blend_running.
         * @brief Blend one running buffer element with its batch statistic.
         * @param running Old buffer element widened to W.
         * @param batch_stat Batch statistic (mean or biased variance) in W.
         * @param m Momentum narrowed once into W.
         * @return The blended value in W (the caller narrows once into the buffer dtype).
         */
        template<typename W>
        __device__ __forceinline__ W dev_blend_running(const W running, const W batch_stat, const W m) {
            const auto a = running * (W{1} - m);
            const auto b = batch_stat * m;
            return a + b;
        }

        /**
         * One block per channel: block-strided per-thread partials of sum and
         * sum-of-squares over the channel's M elements, merged by a
         * shared-memory tree (the merge order is the backend-defined float
         * reduction order); thread 0 then computes the pinned mean/var/invstd
         * and applies the pinned three-step running-stat blend in place, so
         * the buffer update is ordered before everything the same stream runs
         * next. Flat element of channel c at per-channel offset k:
         * (k / inner) * (channels * inner) + c * inner + (k % inner) — the
         * channel_of_flat layout, inverted.
         * @brief Train-mode per-channel statistics + running-stat blend kernel.
         * @param x Input elements.
         * @param means Out: per-channel mean in W (channels elements).
         * @param invstds Out: per-channel invstd in W (channels elements).
         * @param rm Running mean buffer, blended in place.
         * @param rv Running variance buffer (biased), blended in place.
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param channels Channel count C (also the grid size).
         * @param count Total element count.
         * @param m Per-channel element count M, as W.
         * @param eps_w Epsilon narrowed once into W.
         * @param momentum_w Momentum narrowed once into W.
         */
        template<typename T, typename W>
        __global__ void bn_train_stats_kernel(const T* __restrict__ x, W* __restrict__ means, W* __restrict__ invstds,
                                              T* __restrict__ rm, T* __restrict__ rv,
                                              const size_t inner, const size_t channels, const size_t count,
                                              const W m, const W eps_w, const W momentum_w) {
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
                const auto var = dev_biased_variance(mean, s_sq[0], m);
                means[c] = mean;
                invstds[c] = dev_inv_std(var, eps_w);
                rm[c] = dev_from_working<T>(dev_blend_running(dev_to_working<W>(rm[c]), mean, momentum_w));
                rv[c] = dev_from_working<T>(dev_blend_running(dev_to_working<W>(rv[c]), var, momentum_w));
            }
        }

        /**
         * Per-channel mean/invstd are read from the running buffers in W (the
         * pinned inv_std), grid-stride over the channels. No reduction phase
         * and no buffer writes — eval leaves the running buffers untouched.
         * @brief Eval-mode per-channel statistics kernel.
         * @param rm Running mean buffer (read-only).
         * @param rv Running variance buffer (read-only).
         * @param means Out: per-channel mean in W (channels elements).
         * @param invstds Out: per-channel invstd in W (channels elements).
         * @param eps_w Epsilon narrowed once into W.
         * @param channels Channel count C.
         */
        template<typename T, typename W>
        __global__ void bn_eval_stats_kernel(const T* __restrict__ rm, const T* __restrict__ rv,
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
         * Grid-stride over every element: c = channel_of_flat(i), y =
         * normalize(widen(x[i]), means[c], invstds[c], widen(gamma[c]),
         * widen(beta[c])), narrowed once into the storage dtype on store.
         * @brief Normalization kernel (train and eval share it).
         * @param x Input elements.
         * @param g Per-channel scale.
         * @param b Per-channel shift.
         * @param o Output elements.
         * @param means Per-channel mean in W.
         * @param invstds Per-channel invstd in W.
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param channels Channel count C.
         * @param count Total element count.
         */
        template<typename T, typename W>
        __global__ void bn_normalize_kernel(const T* __restrict__ x, const T* __restrict__ g, const T* __restrict__ b,
                                            T* __restrict__ o, const W* __restrict__ means, const W* __restrict__ invstds,
                                            const size_t inner, const size_t channels, const size_t count) {
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += stride) {
                const auto c = (i / inner) % channels;
                const auto xv = dev_to_working<W>(x[i]);
                const auto y = dev_normalize(xv, means[c], invstds[c],
                                             dev_to_working<W>(g[c]), dev_to_working<W>(b[c]));
                o[i] = dev_from_working<T>(y);
            }
        }

        /**
         * eps/momentum narrow once into the working precision here (the pinned
         * single narrowing), M is computed through per_channel_count. The
         * means/invstds scratch is worker-local alloc/free; the free syncs the
         * device, and the explicit stream sync leaves the output and the
         * blended buffers complete before the worker returns.
         * @brief Host-side CUDA batch normalization forward for one storage type.
         * @tparam T Storage element type.
         * @tparam W Working precision: fp64 for fp64 storage, fp32 otherwise.
         * @param x_data Input elements (offset applied).
         * @param g_data Per-channel scale (offset applied).
         * @param b_data Per-channel shift (offset applied).
         * @param o_data Output elements (offset applied).
         * @param rm_data Running mean buffer (offset applied).
         * @param rv_data Running variance buffer (offset applied).
         * @param training Train mode recomputes batch stats and blends the buffers; eval mode reads them.
         * @param channels Channel count C.
         * @param inner Within-channel-per-batch extent (1 or H*W).
         * @param count Total element count.
         * @param momentum Blend factor (train mode only), narrowed once into W.
         * @param eps Epsilon inside invstd = 1/sqrt(var + eps), narrowed once into W.
         */
        template<typename T, typename W>
        __host__ void run_batch_norm(const T* const x_data, const T* const g_data, const T* const b_data,
                                     T* const o_data, T* const rm_data, T* const rv_data,
                                     const bool training,
                                     const size_t channels, const size_t inner, const size_t count,
                                     const float64_t momentum, const float64_t eps) {
            const auto m = static_cast<W>(per_channel_count(count, channels));
            const auto eps_w = static_cast<W>(eps);
            const auto momentum_w = static_cast<W>(momentum);
            W* stats = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&stats, 2 * channels * sizeof(W)));
            auto* const means = stats;
            auto* const invstds = stats + channels;
            const auto stats_shared = 2 * BT_GPU_THREADS * sizeof(W);
            if (training) {
                bn_train_stats_kernel<T, W><<<channels, BT_GPU_THREADS, stats_shared>>>(
                    x_data, means, invstds, rm_data, rv_data, inner, channels, count, m, eps_w, momentum_w);
            } else {
                bn_eval_stats_kernel<T, W><<<grid_for(channels), BT_GPU_THREADS>>>(
                    rm_data, rv_data, means, invstds, eps_w, channels);
            }
            bn_normalize_kernel<T, W><<<grid_for(count), BT_GPU_THREADS>>>(
                x_data, g_data, b_data, o_data, means, invstds, inner, channels, count);
            CUDA_CHECK_ERROR(cudaGetLastError());
            CUDA_CHECK_ERROR(cudaFree(stats));
            CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
        }
    }

    /**
     * Dispatches on dtype to the typed runner: fp64 accumulates in fp64,
     * everything else in fp32 (fp16/bf16 lanes widen/narrow on device, RNE on
     * store). Queues the kernel sequence on the default stream via the CPU
     * thread pool (High priority) and returns; the shared future lands on
     * every tensor the kernels touch (the running buffers included — the
     * train-mode blend mutates them, and a destroy-before-sync must drain the
     * worker first), so errors surface at sync(). Raw pointers are captured
     * by value for that reason.
     * @brief GPU batch normalization forward.
     * @param x Contiguous rank-2/4 input, float dtype, GPU device.
     * @param gamma Rank-1 per-channel scale, x's dtype.
     * @param beta Rank-1 per-channel shift, x's dtype.
     * @param out Preallocated output, x's shape/dtype/device.
     * @param running_mean Rank-1 running mean buffer; blended in train mode, read-only in eval mode.
     * @param running_var Rank-1 running variance buffer (biased); blended in train mode, read-only in eval mode.
     * @param training Train mode uses batch statistics and updates the buffers; eval mode normalizes with the running statistics.
     * @param momentum Blend factor, narrowed once into the working precision (train mode only).
     * @param eps Epsilon inside invstd = 1/sqrt(var + eps), narrowed once into the working precision.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the GPU.
     * @throws ErrorHandling::NotImplemented (from the worker, at sync) for an unsupported dtype.
     */
    __host__ void batch_norm_gpu(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                                 const Tensor& out, Tensor& running_mean, Tensor& running_var,
                                 const bool training, const float64_t momentum, const float64_t eps) {
        if (x.device != Device::GPU || gamma.device != Device::GPU || beta.device != Device::GPU ||
            out.device != Device::GPU || running_mean.device != Device::GPU || running_var.device != Device::GPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("GPU", stringify_device_source(x.device));
        }
        const auto channels = x.shape[1];
        const auto inner = x.numel / (x.shape[0] * channels);
        const auto count = x.numel;
        const auto dtype = x.dtype;
        const auto* const x_ptr = x.data;
        const auto* const g_ptr = gamma.data;
        const auto* const b_ptr = beta.data;
        auto* const o_ptr = out.data;
        auto* const rm_ptr = running_mean.data;
        auto* const rv_ptr = running_var.data;
        const auto x_off = x.offset;
        const auto g_off = gamma.offset;
        const auto b_off = beta.offset;
        const auto o_off = out.offset;
        const auto rm_off = running_mean.offset;
        const auto rv_off = running_var.offset;
        /**
         * Raw pointers, offsets, dtype, sizes and hyperparameters are captured
         * by value: the future lands on every tensor, so buffer teardown
         * always drains this worker first. Dispatches on dtype to the typed
         * runner, which launches the stats (+blend, train mode) and normalize
         * kernels on the default stream and stream-syncs, so the future
         * completes only once the output and the blended buffers are written.
         * An unsupported dtype throws ErrorHandling::NotImplemented.
         * @brief Pool worker for the GPU batch normalization forward.
         */
        const auto worker = [=] {
            using DType = DataTypes::DType;
            switch (dtype) {
                case DType::Float32:
                    run_batch_norm<float32_t, float32_t>(
                        static_cast<const float32_t*>(x_ptr) + x_off, static_cast<const float32_t*>(g_ptr) + g_off,
                        static_cast<const float32_t*>(b_ptr) + b_off, static_cast<float32_t*>(o_ptr) + o_off,
                        static_cast<float32_t*>(rm_ptr) + rm_off, static_cast<float32_t*>(rv_ptr) + rv_off,
                        training, channels, inner, count, momentum, eps);
                    break;
                case DType::Float64:
                    run_batch_norm<float64_t, float64_t>(
                        static_cast<const float64_t*>(x_ptr) + x_off, static_cast<const float64_t*>(g_ptr) + g_off,
                        static_cast<const float64_t*>(b_ptr) + b_off, static_cast<float64_t*>(o_ptr) + o_off,
                        static_cast<float64_t*>(rm_ptr) + rm_off, static_cast<float64_t*>(rv_ptr) + rv_off,
                        training, channels, inner, count, momentum, eps);
                    break;
                case DType::Float16:
                    run_batch_norm<float16_t, float32_t>(
                        static_cast<const float16_t*>(x_ptr) + x_off, static_cast<const float16_t*>(g_ptr) + g_off,
                        static_cast<const float16_t*>(b_ptr) + b_off, static_cast<float16_t*>(o_ptr) + o_off,
                        static_cast<float16_t*>(rm_ptr) + rm_off, static_cast<float16_t*>(rv_ptr) + rv_off,
                        training, channels, inner, count, momentum, eps);
                    break;
                case DType::BFloat16:
                    run_batch_norm<bfloat16_t, float32_t>(
                        static_cast<const bfloat16_t*>(x_ptr) + x_off, static_cast<const bfloat16_t*>(g_ptr) + g_off,
                        static_cast<const bfloat16_t*>(b_ptr) + b_off, static_cast<bfloat16_t*>(o_ptr) + o_off,
                        static_cast<bfloat16_t*>(rm_ptr) + rm_off, static_cast<bfloat16_t*>(rv_ptr) + rv_off,
                        training, channels, inner, count, momentum, eps);
                    break;
                [[unlikely]] default:
                    throw ErrorHandling::NotImplemented();
            }
        };
        auto& threadMgr = Threading::GetCPUThreadPool();
        const auto future = threadMgr.submit(worker, Threading::TaskPriority::High).share();
        x.give_future(future);
        gamma.give_future(future);
        beta.give_future(future);
        out.give_future(future);
        running_mean.give_future(future);
        running_var.give_future(future);
    }
}
#endif
