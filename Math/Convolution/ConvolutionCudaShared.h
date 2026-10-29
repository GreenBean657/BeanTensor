/**
 * @file ConvolutionCudaShared.h
 * @author GreenBean
 * @brief CUDA device kernels and host launchers shared by Conv2d forward/backward.
 * Divergent twin of ConvolutionHipShared.h.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#if defined(USE_CUDA)

#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"
#include "Helpers/CPUThreading.h"

namespace BeanTensor::Tensors::detail::convolution_gpu {
    constexpr size_t BT_GPU_THREADS = 256;
    constexpr size_t BT_GPU_MAX_BLOCKS = 1024;

    [[nodiscard]] inline __host__ size_t grid_for(const size_t numel) {
        const auto blocks = (numel + BT_GPU_THREADS - 1) / BT_GPU_THREADS;
        return blocks < 1 ? 1 : blocks > BT_GPU_MAX_BLOCKS ? BT_GPU_MAX_BLOCKS : blocks;
    }

    struct Conv2dShape {
        size_t N;
        size_t C;
        size_t H;
        size_t W;
        size_t R;
        size_t S;
        size_t K;
        size_t Hout;
        size_t Wout;
        size_t M;
        size_t CRS;
        DataTypes::DType dtype;
    };

    inline __host__ __device__ size_t conv_out_size(const size_t in_size, const size_t kernel, const uint32_t padding, const uint32_t stride) {
        return (in_size + 2u * padding >= kernel)
            ? (in_size + 2u * padding - kernel) / stride + 1u
            : 0;
    }

    inline __host__ Conv2dShape make_shape(const std::vector<size_t>& x_shape, const std::vector<size_t>& w_shape,
                                           const DataTypes::DType dtype, const uint32_t stride, const uint32_t padding) {
        const auto N = x_shape[0];
        const auto C = x_shape[1];
        const auto H = x_shape[2];
        const auto W = x_shape[3];
        const auto R = w_shape[1];
        const auto S = w_shape[2];
        const auto K = w_shape[3];
        const auto Hout = conv_out_size(H, R, padding, stride);
        const auto Wout = conv_out_size(W, S, padding, stride);
        const auto M = N * Hout * Wout;
        const auto CRS = C * R * S;
        return {N, C, H, W, R, S, K, Hout, Wout, M, CRS, dtype};
    }

    inline __device__ size_t nchw_offset(const Conv2dShape& s, const size_t n, const size_t c, const size_t h, const size_t w) {
        return ((n * s.C + c) * s.H + h) * s.W + w;
    }

    inline __device__ size_t nchw_out_offset(const Conv2dShape& s, const size_t n, const size_t k, const size_t ho, const size_t wo) {
        return ((n * s.K + k) * s.Hout + ho) * s.Wout + wo;
    }

    inline __device__ size_t mk_offset(const Conv2dShape& s, const size_t m, const size_t k) {
        return m * s.K + k;
    }

    inline __device__ size_t col_offset(const Conv2dShape& s, const size_t m, const size_t col_idx) {
        return m * s.CRS + col_idx;
    }

    namespace {
        __global__ void im2col_kernel(const uint8_t* __restrict__ x, uint8_t* __restrict__ col,
                                      const Conv2dShape s, const uint32_t stride, const uint32_t padding,
                                      const size_t eb) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride_grid = blockDim.x * gridDim.x;
            const size_t total = s.M * s.CRS;
            for (size_t idx = tid; idx < total; idx += stride_grid) {
                const auto m = idx / s.CRS;
                const auto col_idx = idx % s.CRS;
                const auto n = m / (s.Hout * s.Wout);
                const auto rem = m % (s.Hout * s.Wout);
                const auto ho = rem / s.Wout;
                const auto wo = rem % s.Wout;
                const auto c = col_idx / (s.R * s.S);
                const auto r = (col_idx / s.S) % s.R;
                const auto ss = col_idx % s.S;
                const auto ih = static_cast<int64_t>(ho * stride + r) - static_cast<int64_t>(padding);
                const auto iw = static_cast<int64_t>(wo * stride + ss) - static_cast<int64_t>(padding);
                if (ih >= 0 && ih < static_cast<int64_t>(s.H) && iw >= 0 && iw < static_cast<int64_t>(s.W)) {
                    const auto xi = nchw_offset(s, n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)) * eb;
                    const auto ci = col_offset(s, m, col_idx) * eb;
                    for (size_t b = 0; b < eb; ++b) {
                        col[ci + b] = x[xi + b];
                    }
                } else {
                    const auto ci = col_offset(s, m, col_idx) * eb;
                    for (size_t b = 0; b < eb; ++b) {
                        col[ci + b] = 0;
                    }
                }
            }
        }

        __global__ void mk_to_nchw_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ out,
                                          const Conv2dShape s, const size_t eb) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride_grid = blockDim.x * gridDim.x;
            const size_t total = s.N * s.K * s.Hout * s.Wout;
            for (size_t idx = tid; idx < total; idx += stride_grid) {
                const auto wo = idx % s.Wout;
                const auto tmp = idx / s.Wout;
                const auto ho = tmp % s.Hout;
                const auto tmp2 = tmp / s.Hout;
                const auto k = tmp2 % s.K;
                const auto n = tmp2 / s.K;
                const auto m = (n * s.Hout + ho) * s.Wout + wo;
                const auto src_off = mk_offset(s, m, k) * eb;
                const auto out_off = nchw_out_offset(s, n, k, ho, wo) * eb;
                for (size_t b = 0; b < eb; ++b) {
                    out[out_off + b] = src[src_off + b];
                }
            }
        }

        __global__ void nchw_to_mk_kernel(const uint8_t* __restrict__ dy, uint8_t* __restrict__ dyT,
                                          const Conv2dShape s, const size_t eb) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride_grid = blockDim.x * gridDim.x;
            const size_t total = s.N * s.K * s.Hout * s.Wout;
            for (size_t idx = tid; idx < total; idx += stride_grid) {
                const auto wo = idx % s.Wout;
                const auto tmp = idx / s.Wout;
                const auto ho = tmp % s.Hout;
                const auto tmp2 = tmp / s.Hout;
                const auto k = tmp2 % s.K;
                const auto n = tmp2 / s.K;
                const auto m = (n * s.Hout + ho) * s.Wout + wo;
                const auto dy_off = nchw_out_offset(s, n, k, ho, wo) * eb;
                const auto dt_off = mk_offset(s, m, k) * eb;
                for (size_t b = 0; b < eb; ++b) {
                    dyT[dt_off + b] = dy[dy_off + b];
                }
            }
        }

        template<typename T>
        __global__ void col2im_kernel(const T* __restrict__ col, T* __restrict__ dx,
                                      const Conv2dShape s, const uint32_t stride, const uint32_t padding) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride_grid = blockDim.x * gridDim.x;
            const size_t total = s.N * s.C * s.H * s.W;
            for (size_t idx = tid; idx < total; idx += stride_grid) {
                const auto w = idx % s.W;
                const auto tmp = idx / s.W;
                const auto h = tmp % s.H;
                const auto tmp2 = tmp / s.H;
                const auto c = tmp2 % s.C;
                const auto n = tmp2 / s.C;
                T acc = T{0};
                for (auto r = size_t{0}; r < s.R; ++r) {
                    for (auto ss = size_t{0}; ss < s.S; ++ss) {
                        const auto ho_num = static_cast<int64_t>(h) + static_cast<int64_t>(padding) - static_cast<int64_t>(r);
                        const auto wo_num = static_cast<int64_t>(w) + static_cast<int64_t>(padding) - static_cast<int64_t>(ss);
                        if (ho_num % static_cast<int64_t>(stride) != 0 || wo_num % static_cast<int64_t>(stride) != 0) {
                            continue;
                        }
                        const auto ho = ho_num / static_cast<int64_t>(stride);
                        const auto wo = wo_num / static_cast<int64_t>(stride);
                        if (ho < 0 || ho >= static_cast<int64_t>(s.Hout) || wo < 0 || wo >= static_cast<int64_t>(s.Wout)) {
                            continue;
                        }
                        const auto m = (n * s.Hout + static_cast<size_t>(ho)) * s.Wout + static_cast<size_t>(wo);
                        const auto col_idx = ((c * s.R + r) * s.S + ss);
                        acc = static_cast<T>(acc + col[col_offset(s, m, col_idx)]);
                    }
                }
                dx[nchw_offset(s, n, c, h, w)] = acc;
            }
        }

        template<typename T>
        __global__ void rowsum_mk_kernel(const T* __restrict__ src, T* __restrict__ out,
                                         const size_t M, const size_t K) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride_grid = blockDim.x * gridDim.x;
            for (size_t k = tid; k < K; k += stride_grid) {
                T acc = T{0};
                for (size_t m = 0; m < M; ++m) {
                    acc = static_cast<T>(acc + src[m * K + k]);
                }
                out[k] = acc;
            }
        }

        template<typename T>
        __global__ void bias_add_nchw_kernel(const T* __restrict__ bias, T* __restrict__ out,
                                             const Conv2dShape s) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t stride_grid = blockDim.x * gridDim.x;
            const size_t total = s.N * s.K * s.Hout * s.Wout;
            for (size_t idx = tid; idx < total; idx += stride_grid) {
                const auto wo = idx % s.Wout;
                const auto tmp = idx / s.Wout;
                const auto ho = tmp % s.Hout;
                const auto tmp2 = tmp / s.Hout;
                const auto k = tmp2 % s.K;
                const auto n = tmp2 / s.K;
                const auto oi = nchw_out_offset(s, n, k, ho, wo);
                out[oi] = static_cast<T>(out[oi] + bias[k]);
            }
        }
    }

    inline __host__ void launch_col2im(const uint8_t* col, uint8_t* dx, const Conv2dShape& s,
                                const uint32_t stride, const uint32_t padding) {
        const auto total = s.N * s.C * s.H * s.W;
        if (total == 0) return;
        using DType = DataTypes::DType;
        switch (s.dtype) {
            case DType::Float32: col2im_kernel<float32_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const float32_t*>(col), reinterpret_cast<float32_t*>(dx), s, stride, padding); break;
            case DType::Float64: col2im_kernel<float64_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const float64_t*>(col), reinterpret_cast<float64_t*>(dx), s, stride, padding); break;
            case DType::Float16: col2im_kernel<float16_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const float16_t*>(col), reinterpret_cast<float16_t*>(dx), s, stride, padding); break;
            case DType::BFloat16: col2im_kernel<bfloat16_t><<<grid_for(total), BT_GPU_THREADS>>>(reinterpret_cast<const bfloat16_t*>(col), reinterpret_cast<bfloat16_t*>(dx), s, stride, padding); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    template<typename T>
    __host__ void launch_rowsum_mk(const T* src, T* out, const size_t M, const size_t K) {
        if (K == 0) return;
        rowsum_mk_kernel<T><<<grid_for(K), BT_GPU_THREADS>>>(src, out, M, K);
    }
}

#endif
