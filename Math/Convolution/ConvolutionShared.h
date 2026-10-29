/**
 * @file ConvolutionShared.h
 * @author GreenBean
 * @brief Host-side and CPU shared kernels for im2col, col2im, NCHW<->{M,K}
 * scatter/gather, and rowsum. The loops are scalar (no SIMD) because these
 * ops are memory-bound data movement; threading comes from the launcher.
 * @version 0.1
 * @date 2026-10-28
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <vector>

#include "Tensor/DTypes.h"
#include "Tensor/Tensor.h"
#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Math/Autograd/AutogradShared.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    /**
     * Shape metadata bundle used by all conv2d helpers.
     * @brief Precomputed conv2d dimensions.
     */
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
    };

    /**
     * Compute Hout/Wout with floored integer division per PyTorch.
     * @brief Output spatial size for 2-D convolution.
     */
    inline size_t conv_out_size(const size_t in_size, const size_t kernel, const uint32_t padding, const uint32_t stride) {
        if (stride == 0) {
            return 0;
        }
        return (in_size + 2u * padding >= kernel)
            ? (in_size + 2u * padding - kernel) / stride + 1u
            : 0;
    }

    namespace {
        /**
         * Threaded elementwise launcher used by the convolution data-movement
         * kernels. Small inputs run inline; otherwise work is chunked across
         * the CPU thread pool and futures are attached to the touched tensors.
         * @brief Threaded launcher for memory-bound convolution helpers.
         * @param worker Callable (begin, end) over an element range.
         * @param numel Total element count.
         * @param tensors Pointers to every tensor the workers read/write (futures receiver).
         */
        template<typename Worker>
        void conv_launch_elementwise(Worker&& worker, const size_t numel, const std::vector<const Tensor*>& tensors) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= numel) {
                    break;
                }
                futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
            }
            for (const auto t : tensors) {
                if (t != nullptr) {
                    t->give_futures(futures);
                }
            }
        }

        inline Conv2dShape make_conv2d_shape(const std::vector<size_t>& x_shape, const std::vector<size_t>& w_shape, const uint32_t stride, const uint32_t padding) {
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
            return {N, C, H, W, R, S, K, Hout, Wout, M, CRS};
        }

        /**
         * Flat NCHW index for the input tensor {N,C,H,W}.
         * @brief Linear offset for x[n,c,h,w].
         */
        inline size_t nchw_offset(const Conv2dShape& s, const size_t n, const size_t c, const size_t h, const size_t w) {
            return ((n * s.C + c) * s.H + h) * s.W + w;
        }

        /**
         * Flat NCHW index for the output tensor {N,K,Hout,Wout}.
         * @brief Linear offset for out[n,k,ho,wo].
         */
        inline size_t nchw_out_offset(const Conv2dShape& s, const size_t n, const size_t k, const size_t ho, const size_t wo) {
            return ((n * s.K + k) * s.Hout + ho) * s.Wout + wo;
        }

        /**
         * Flat {M,K} index.
         * @brief Linear offset for mk[m,k].
         */
        inline size_t mk_offset(const Conv2dShape& s, const size_t m, const size_t k) {
            return m * s.K + k;
        }

        /**
         * Flat column matrix index.
         * @brief Linear offset for col[m,col_idx].
         */
        inline size_t col_offset(const Conv2dShape& s, const size_t m, const size_t col_idx) {
            return m * s.CRS + col_idx;
        }
    }

    /**
     * Zero-fill a tensor's data buffer (including its offset). Used to pre-zero
     * the column matrix and dx before col2im accumulation.
     * @brief Zero a tensor buffer.
     * @param t Tensor to zero.
     */
    inline void conv_zero_tensor(const Tensor& t) {
        const auto bytes = t.numel * DataTypes::dtype_bytes(t.dtype);
        if (bytes > 0) {
            std::memset(static_cast<uint8_t*>(t.data) + t.offset * DataTypes::dtype_bytes(t.dtype), 0, bytes);
        }
    }

    /**
     * CPU im2col kernel: for each (m, col_idx) compute the source NCHW
     * coordinate and copy the element byte-wise; out-of-bounds writes zeros.
     * The output is pre-zeroed by the caller.
     * @brief Scalar im2col data movement.
     * @param x Input tensor data pointer (with offset applied).
     * @param col Output column matrix pointer.
     * @param s Precomputed conv2d dimensions.
     * @param stride Stride along H and W.
     * @param padding Zero padding on all sides.
     * @param eb Element size in bytes.
     */
    inline void im2col_kernel(const uint8_t* x, uint8_t* col, const Conv2dShape& s,
                              const uint32_t stride, const uint32_t padding, const size_t eb) {
        for (auto m = size_t{0}; m < s.M; ++m) {
            const auto n = m / (s.Hout * s.Wout);
            const auto rem = m % (s.Hout * s.Wout);
            const auto ho = rem / s.Wout;
            const auto wo = rem % s.Wout;
            for (auto col_idx = size_t{0}; col_idx < s.CRS; ++col_idx) {
                const auto c = col_idx / (s.R * s.S);
                const auto r = (col_idx / s.S) % s.R;
                const auto ss = col_idx % s.S;
                const auto ih = static_cast<int64_t>(ho * stride + r) - static_cast<int64_t>(padding);
                const auto iw = static_cast<int64_t>(wo * stride + ss) - static_cast<int64_t>(padding);
                if (
                    ih >= 0 && ih < static_cast<int64_t>(s.H) &&
                    iw >= 0 && iw < static_cast<int64_t>(s.W)
                ) {
                    const auto xi = nchw_offset(s, n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)) * eb;
                    const auto ci = col_offset(s, m, col_idx) * eb;
                    std::memcpy(col + ci, x + xi, eb);
                }
            }
        }
    }

    /**
     * CPU col2im accumulation: dx starts zeroed, then every (m, col_idx)
     * contribution is added to the corresponding NCHW position.
     * @brief Scalar col2im data movement for type T.
     */
    template<typename T>
    void col2im_typed(const T* col, T* dx, const Conv2dShape& s,
                      const uint32_t stride, const uint32_t padding) {
        for (auto n = size_t{0}; n < s.N; ++n) {
            for (auto c = size_t{0}; c < s.C; ++c) {
                for (auto h = size_t{0}; h < s.H; ++h) {
                    for (auto w = size_t{0}; w < s.W; ++w) {
                        auto acc = T{0};
                        for (auto r = size_t{0}; r < s.R; ++r) {
                            for (auto ss = size_t{0}; ss < s.S; ++ss) {
                                const auto ho_num = static_cast<int64_t>(h) + static_cast<int64_t>(padding) - static_cast<int64_t>(r);
                                const auto wo_num = static_cast<int64_t>(w) + static_cast<int64_t>(padding) - static_cast<int64_t>(ss);
                                if (
                                    ho_num % static_cast<int64_t>(stride) != 0 ||
                                    wo_num % static_cast<int64_t>(stride) != 0
                                ) {
                                    continue;
                                }
                                const auto ho = static_cast<int64_t>(ho_num / static_cast<int64_t>(stride));
                                const auto wo = static_cast<int64_t>(wo_num / static_cast<int64_t>(stride));
                                if (
                                    ho < 0 || ho >= static_cast<int64_t>(s.Hout) ||
                                    wo < 0 || wo >= static_cast<int64_t>(s.Wout)
                                ) {
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
            }
        }
    }

    /**
     * CPU nchw_to_mk: scatter NCHW dy into contiguous {M,K} row-major dyT.
     * @brief NCHW -> {M,K} transpose-into-scratch.
     */
    inline void nchw_to_mk_kernel(const uint8_t* dy, uint8_t* dyT, const Conv2dShape& s, const size_t eb) {
        for (auto n = size_t{0}; n < s.N; ++n) {
            for (auto k = size_t{0}; k < s.K; ++k) {
                for (auto ho = size_t{0}; ho < s.Hout; ++ho) {
                    for (auto wo = size_t{0}; wo < s.Wout; ++wo) {
                        const auto m = (n * s.Hout + ho) * s.Wout + wo;
                        const auto dy_off = nchw_out_offset(s, n, k, ho, wo) * eb;
                        const auto dt_off = mk_offset(s, m, k) * eb;
                        std::memcpy(dyT + dt_off, dy + dy_off, eb);
                    }
                }
            }
        }
    }

    /**
     * CPU mk_to_nchw: scatter {M,K} row-major src into NCHW out.
     * @brief {M,K} -> NCHW scatter epilogue.
     */
    inline void mk_to_nchw_kernel(const uint8_t* src, uint8_t* out, const Conv2dShape& s, const size_t eb) {
        for (auto n = size_t{0}; n < s.N; ++n) {
            for (auto k = size_t{0}; k < s.K; ++k) {
                for (auto ho = size_t{0}; ho < s.Hout; ++ho) {
                    for (auto wo = size_t{0}; wo < s.Wout; ++wo) {
                        const auto m = (n * s.Hout + ho) * s.Wout + wo;
                        const auto src_off = mk_offset(s, m, k) * eb;
                        const auto out_off = nchw_out_offset(s, n, k, ho, wo) * eb;
                        std::memcpy(out + out_off, src + src_off, eb);
                    }
                }
            }
        }
    }

    /**
     * CPU rowsum over axis 0 of a {M,K} tensor.
     * @brief Sum each column of a row-major 2-D matrix.
     */
    template<typename T>
    void rowsum_mk_typed(const T* src, T* out, const size_t M, const size_t K) {
        for (auto k = size_t{0}; k < K; ++k) {
            auto acc = T{0};
            for (auto m = size_t{0}; m < M; ++m) {
                acc = static_cast<T>(acc + src[m * K + k]);
            }
            out[k] = acc;
        }
    }

    /**
     * Dispatch col2im by dtype. Float kinds use native add; half kinds widen to
     * fp32 and narrow RNE; integer kinds wrap. Packed and int4x2 are left as
     * NotImplemented for the backward path.
     * @brief dtype-dispatching col2im host helper.
     */
    inline void col2im_dispatch(const Tensor& col, const Tensor& dx, const Conv2dShape& s,
                                const uint32_t stride, const uint32_t padding) {
        conv_zero_tensor(dx);
        using DType = DataTypes::DType;
        switch (col.dtype) {
            case DType::Float32: {
                col2im_typed(static_cast<const float32_t*>(col.data) + col.offset,
                             static_cast<float32_t*>(dx.data) + dx.offset, s, stride, padding);
                break;
            }
            case DType::Float64: {
                col2im_typed(static_cast<const float64_t*>(col.data) + col.offset,
                             static_cast<float64_t*>(dx.data) + dx.offset, s, stride, padding);
                break;
            }
            case DType::Float16: {
                const auto* c = static_cast<const float16_t*>(col.data) + col.offset;
                auto* d = static_cast<float16_t*>(dx.data) + dx.offset;
                for (auto n = size_t{0}; n < s.N; ++n) {
                    for (auto c_idx = size_t{0}; c_idx < s.C; ++c_idx) {
                        for (auto h = size_t{0}; h < s.H; ++h) {
                            for (auto w = size_t{0}; w < s.W; ++w) {
                                auto acc = float32_t{0.0f};
                                for (auto r = size_t{0}; r < s.R; ++r) {
                                    for (auto ss = size_t{0}; ss < s.S; ++ss) {
                                        const auto ho_num = static_cast<int64_t>(h) + static_cast<int64_t>(padding) - static_cast<int64_t>(r);
                                        const auto wo_num = static_cast<int64_t>(w) + static_cast<int64_t>(padding) - static_cast<int64_t>(ss);
                                        if (
                                            ho_num % static_cast<int64_t>(stride) != 0 ||
                                            wo_num % static_cast<int64_t>(stride) != 0
                                        ) {
                                            continue;
                                        }
                                        const auto ho = ho_num / static_cast<int64_t>(stride);
                                        const auto wo = wo_num / static_cast<int64_t>(stride);
                                        if (
                                            ho < 0 || ho >= static_cast<int64_t>(s.Hout) ||
                                            wo < 0 || wo >= static_cast<int64_t>(s.Wout)
                                        ) {
                                            continue;
                                        }
                                        const auto m = (n * s.Hout + static_cast<size_t>(ho)) * s.Wout + static_cast<size_t>(wo);
                                        const auto col_idx = ((c_idx * s.R + r) * s.S + ss);
                                        acc += widen_lane(c[col_offset(s, m, col_idx)]);
                                    }
                                }
                                d[nchw_offset(s, n, c_idx, h, w)] = narrow_lane<float16_t>(acc);
                            }
                        }
                    }
                }
                break;
            }
            case DType::BFloat16: {
                const auto* c = static_cast<const bfloat16_t*>(col.data) + col.offset;
                auto* d = static_cast<bfloat16_t*>(dx.data) + dx.offset;
                for (auto n = size_t{0}; n < s.N; ++n) {
                    for (auto c_idx = size_t{0}; c_idx < s.C; ++c_idx) {
                        for (auto h = size_t{0}; h < s.H; ++h) {
                            for (auto w = size_t{0}; w < s.W; ++w) {
                                auto acc = float32_t{0.0f};
                                for (auto r = size_t{0}; r < s.R; ++r) {
                                    for (auto ss = size_t{0}; ss < s.S; ++ss) {
                                        const auto ho_num = static_cast<int64_t>(h) + static_cast<int64_t>(padding) - static_cast<int64_t>(r);
                                        const auto wo_num = static_cast<int64_t>(w) + static_cast<int64_t>(padding) - static_cast<int64_t>(ss);
                                        if (
                                            ho_num % static_cast<int64_t>(stride) != 0 ||
                                            wo_num % static_cast<int64_t>(stride) != 0
                                        ) {
                                            continue;
                                        }
                                        const auto ho = ho_num / static_cast<int64_t>(stride);
                                        const auto wo = wo_num / static_cast<int64_t>(stride);
                                        if (
                                            ho < 0 || ho >= static_cast<int64_t>(s.Hout) ||
                                            wo < 0 || wo >= static_cast<int64_t>(s.Wout)
                                        ) {
                                            continue;
                                        }
                                        const auto m = (n * s.Hout + static_cast<size_t>(ho)) * s.Wout + static_cast<size_t>(wo);
                                        const auto col_idx = ((c_idx * s.R + r) * s.S + ss);
                                        acc += widen_lane(c[col_offset(s, m, col_idx)]);
                                    }
                                }
                                d[nchw_offset(s, n, c_idx, h, w)] = narrow_lane<bfloat16_t>(acc);
                            }
                        }
                    }
                }
                break;
            }
            default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Add bias {K} to an NCHW tensor {N,K,H,W}: out[n,k,h,w] += bias[k].
     * @brief NCHW bias epilogue.
     */
    template<typename T>
    void bias_add_nchw_typed(const T* bias, T* out, const Conv2dShape& s) {
        for (auto n = size_t{0}; n < s.N; ++n) {
            for (auto k = size_t{0}; k < s.K; ++k) {
                for (auto ho = size_t{0}; ho < s.Hout; ++ho) {
                    for (auto wo = size_t{0}; wo < s.Wout; ++wo) {
                        const auto idx = nchw_offset(s, n, k, ho, wo);
                        out[idx] = static_cast<T>(out[idx] + bias[k]);
                    }
                }
            }
        }
    }

    /**
     * Dispatch NCHW bias add by dtype.
     * @brief dtype-dispatching NCHW bias epilogue.
     */
    inline void bias_add_nchw_dispatch(const Tensor& bias, const Tensor& out) {
        const auto s = make_conv2d_shape(
            std::vector<size_t>(out.shape, out.shape + out.ndim),
            std::vector<size_t>{out.shape[1], 1, 1, out.shape[1]},
            1, 0);
        using DType = DataTypes::DType;
        switch (out.dtype) {
            case DType::Float32: {
                bias_add_nchw_typed(static_cast<const float32_t*>(bias.data) + bias.offset,
                                    static_cast<float32_t*>(out.data) + out.offset, s);
                break;
            }
            case DType::Float64: {
                bias_add_nchw_typed(static_cast<const float64_t*>(bias.data) + bias.offset,
                                    static_cast<float64_t*>(out.data) + out.offset, s);
                break;
            }
            case DType::Float16: {
                const auto* b = static_cast<const float16_t*>(bias.data) + bias.offset;
                auto* o = static_cast<float16_t*>(out.data) + out.offset;
                for (auto n = size_t{0}; n < s.N; ++n) {
                    for (auto k = size_t{0}; k < s.K; ++k) {
                        const auto bval = widen_lane(b[k]);
                        for (auto ho = size_t{0}; ho < s.Hout; ++ho) {
                            for (auto wo = size_t{0}; wo < s.Wout; ++wo) {
                                const auto idx = nchw_offset(s, n, k, ho, wo);
                                o[idx] = narrow_lane<float16_t>(widen_lane(o[idx]) + bval);
                            }
                        }
                    }
                }
                break;
            }
            case DType::BFloat16: {
                const auto* b = static_cast<const bfloat16_t*>(bias.data) + bias.offset;
                auto* o = static_cast<bfloat16_t*>(out.data) + out.offset;
                for (auto n = size_t{0}; n < s.N; ++n) {
                    for (auto k = size_t{0}; k < s.K; ++k) {
                        const auto bval = widen_lane(b[k]);
                        for (auto ho = size_t{0}; ho < s.Hout; ++ho) {
                            for (auto wo = size_t{0}; wo < s.Wout; ++wo) {
                                const auto idx = nchw_offset(s, n, k, ho, wo);
                                o[idx] = narrow_lane<bfloat16_t>(widen_lane(o[idx]) + bval);
                            }
                        }
                    }
                }
                break;
            }
            default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Dispatch rowsum over axis 0 by dtype.
     * @brief dtype-dispatching rowsum host helper.
     */
    inline void rowsum_mk_dispatch(const Tensor& src, const Tensor& out) {
        const auto M = src.shape[0];
        const auto K = src.shape[1];
        using DType = DataTypes::DType;
        switch (src.dtype) {
            case DType::Float32: {
                rowsum_mk_typed(static_cast<const float32_t*>(src.data) + src.offset,
                                static_cast<float32_t*>(out.data) + out.offset, M, K);
                break;
            }
            case DType::Float64: {
                rowsum_mk_typed(static_cast<const float64_t*>(src.data) + src.offset,
                                static_cast<float64_t*>(out.data) + out.offset, M, K);
                break;
            }
            case DType::Float16: {
                const auto* s = static_cast<const float16_t*>(src.data) + src.offset;
                auto* o = static_cast<float16_t*>(out.data) + out.offset;
                for (auto k = size_t{0}; k < K; ++k) {
                    auto acc = float32_t{0.0f};
                    for (auto m = size_t{0}; m < M; ++m) {
                        acc += widen_lane(s[m * K + k]);
                    }
                    o[k] = narrow_lane<float16_t>(acc);
                }
                break;
            }
            case DType::BFloat16: {
                const auto* s = static_cast<const bfloat16_t*>(src.data) + src.offset;
                auto* o = static_cast<bfloat16_t*>(out.data) + out.offset;
                for (auto k = size_t{0}; k < K; ++k) {
                    auto acc = float32_t{0.0f};
                    for (auto m = size_t{0}; m < M; ++m) {
                        acc += widen_lane(s[m * K + k]);
                    }
                    o[k] = narrow_lane<bfloat16_t>(acc);
                }
                break;
            }
            default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}
