/**
 * @file MaxPool2dCPU.cpp
 * @author GreenBean
 * @brief CPU backend for MaxPool2d forward/backward, plus the Tensor::max_pool2d
 * method definition and the MaxPool2G backward trampoline.
 * @version 0.1
 * @date 2026-10-28
 */
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Elementwise/ElementwiseShared.h"
#include "Math/Spatial/MaxPool2d/MaxPool2d.h"
#include "Math/Spatial/MaxPool2d/MaxPool2dKernels.h"
#include "Math/Spatial/SpatialShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace spatial_shared;
    using namespace elementwise_shared;

    namespace {
        // --------------------------- forward kernels ---------------------------

        /**
         * Window max for scalar float/integer dtypes. First max wins; NaN beats
         * every non-NaN value and the first NaN wins.
         * @brief Scalar MaxPool2d forward window loop.
         * @param in Input pointer (offset applied by caller).
         * @param out Output pointer (offset applied by caller).
         * @param N Batch size.
         * @param C Channels.
         * @param H Input height.
         * @param W Input width.
         * @param Hout Output height.
         * @param Wout Output width.
         * @param kernel Window height/width (square).
         * @param stride Step between adjacent windows.
         */
        template<typename T>
        void max_pool2d_window_scalar(const T* in, T* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                T best_v;
                if constexpr (std::is_floating_point_v<T>) {
                    best_v = -std::numeric_limits<T>::infinity();
                } else {
                    best_v = std::numeric_limits<T>::lowest();
                }
                auto best_h = size_t{0};
                auto best_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto v = in[((n * C + c) * H + h) * W + w];
                        if constexpr (std::is_floating_point_v<T>) {
                            if (__builtin_isnan(v)) {
                                if (!__builtin_isnan(best_v)) {
                                    best_v = v;
                                    best_h = h;
                                    best_w = w;
                                }
                            } else if (__builtin_isnan(best_v) || v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        } else {
                            if (v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                }
                out[idx] = in[((n * C + c) * H + best_h) * W + best_w];
            }
        }

        /**
         * Window max for fp16/bf16 lanes. Widen to fp32 for compare, first max wins.
         * @brief Scalar MaxPool2d forward window loop for half dtypes.
         */
        template<typename Half>
        void max_pool2d_window_half(const Half* in, Half* out,
                                    const size_t N, const size_t C,
                                    const size_t H, const size_t W,
                                    const size_t Hout, const size_t Wout,
                                    const size_t kernel, const size_t stride,
                                    const size_t begin, const size_t end) {
            static_cast<void>(N);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto best_v = -std::numeric_limits<float32_t>::infinity();
                auto best_h = size_t{0};
                auto best_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint16_t>(in[((n * C + c) * H + h) * W + w]);
                        const auto v = widen_lane(Intrinsics::detail::bit_cast_memcpy<Half>(bits));
                        if (__builtin_isnan(v)) {
                            if (!__builtin_isnan(best_v)) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        } else if (__builtin_isnan(best_v) || v > best_v) {
                            best_v = v;
                            best_h = h;
                            best_w = w;
                        }
                    }
                }
                out[idx] = in[((n * C + c) * H + best_h) * W + best_w];
            }
        }

        /**
         * Window max for packed fp16x2/bf16x2 dtypes. Each lane is treated
         * independently; the per-lane winners may come from different spatial
         * positions, matching the per-lane max convention.
         * @brief Scalar MaxPool2d forward window loop for packed x2 dtypes.
         */
        template<typename P>
        void max_pool2d_window_packed(const P* in, P* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            using Lane = packed_lane_t<P>;
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto best_lo_v = -std::numeric_limits<float32_t>::infinity();
                auto best_hi_v = -std::numeric_limits<float32_t>::infinity();
                auto best_lo_h = size_t{0};
                auto best_lo_w = size_t{0};
                auto best_hi_h = size_t{0};
                auto best_hi_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[((n * C + c) * H + h) * W + w]);
                        const auto lo = widen_lane(lane_lo<Lane>(bits));
                        const auto hi = widen_lane(lane_hi<Lane>(bits));
                        if (__builtin_isnan(lo)) {
                            if (!__builtin_isnan(best_lo_v)) {
                                best_lo_v = lo;
                                best_lo_h = h;
                                best_lo_w = w;
                            }
                        } else if (__builtin_isnan(best_lo_v) || lo > best_lo_v) {
                            best_lo_v = lo;
                            best_lo_h = h;
                            best_lo_w = w;
                        }
                        if (__builtin_isnan(hi)) {
                            if (!__builtin_isnan(best_hi_v)) {
                                best_hi_v = hi;
                                best_hi_h = h;
                                best_hi_w = w;
                            }
                        } else if (__builtin_isnan(best_hi_v) || hi > best_hi_v) {
                            best_hi_v = hi;
                            best_hi_h = h;
                            best_hi_w = w;
                        }
                    }
                }
                const auto lo_store = static_cast<uint16_t>(Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[((n * C + c) * H + best_lo_h) * W + best_lo_w]) & 0xFFFFu);
                const auto hi_store = static_cast<uint16_t>(Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[((n * C + c) * H + best_hi_h) * W + best_hi_w]) >> 16);
                out[idx] = pack_lanes<P>(lo_store, hi_store);
            }
        }

        /**
         * Window max for int4x2. Per-nibble compare, first max wins per nibble.
         * @brief Scalar MaxPool2d forward window loop for int4x2.
         */
        void max_pool2d_window_int4x2(const int4x2_t* in, int4x2_t* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto best_lo = int8_t{-8};
                auto best_hi = int8_t{-8};
                auto best_lo_h = size_t{0};
                auto best_lo_w = size_t{0};
                auto best_hi_h = size_t{0};
                auto best_hi_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto lo = int4_lo(in[((n * C + c) * H + h) * W + w]);
                        const auto hi = int4_hi(in[((n * C + c) * H + h) * W + w]);
                        if (lo > best_lo) {
                            best_lo = lo;
                            best_lo_h = h;
                            best_lo_w = w;
                        }
                        if (hi > best_hi) {
                            best_hi = hi;
                            best_hi_h = h;
                            best_hi_w = w;
                        }
                    }
                }
                const auto lo_byte = static_cast<uint8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(in[((n * C + c) * H + best_lo_h) * W + best_lo_w]) & 0x0Fu);
                const auto hi_byte = static_cast<uint8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(in[((n * C + c) * H + best_hi_h) * W + best_hi_w]) & 0xF0u);
                out[idx] = Intrinsics::detail::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>(lo_byte | hi_byte));
            }
        }

        /**
         * @brief Function pointer type shared by every MaxPool2d forward window kernel.
         */
        template<typename T>
        using MaxPool2dWindowFn = void(*)(const T*, T*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

        // --------------------------- backward kernels ---------------------------

        /**
         * Backward routing for scalar float/integer dtypes. Recompute the first-max
         * argmax for each window and add dy there.
         * @brief Scalar MaxPool2d backward window loop.
         */
        template<typename T>
        void max_pool2d_g_window_scalar(const T* dy, const T* x, T* dx,
                                        const size_t N, const size_t C,
                                        const size_t H, const size_t W,
                                        const size_t Hout, const size_t Wout,
                                        const size_t kernel, const size_t stride,
                                        const size_t begin, const size_t end) {
            static_cast<void>(N);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                T best_v;
                if constexpr (std::is_floating_point_v<T>) {
                    best_v = -std::numeric_limits<T>::infinity();
                } else {
                    best_v = std::numeric_limits<T>::lowest();
                }
                auto best_h = size_t{0};
                auto best_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto v = x[((n * C + c) * H + h) * W + w];
                        if constexpr (std::is_floating_point_v<T>) {
                            if (__builtin_isnan(v)) {
                                if (!__builtin_isnan(best_v)) {
                                    best_v = v;
                                    best_h = h;
                                    best_w = w;
                                }
                            } else if (__builtin_isnan(best_v) || v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        } else {
                            if (v > best_v) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                }
                dx[((n * C + c) * H + best_h) * W + best_w] += dy[idx];
            }
        }

        /**
         * @brief Zero bit pattern for a half-precision lane type.
         */
        template<typename H>
        [[nodiscard]] H zero_bits() {
            return Intrinsics::detail::bit_cast_memcpy<H>(uint16_t{0});
        }

        /**
         * Backward routing for fp16/bf16. Widen to fp32 for argmax, route dy bits.
         * @brief Scalar MaxPool2d backward window loop for half dtypes.
         */
        template<typename Half>
        void max_pool2d_g_window_half(const Half* dy, const Half* x, Half* dx,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto best_v = -std::numeric_limits<float32_t>::infinity();
                auto best_h = size_t{0};
                auto best_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint16_t>(x[((n * C + c) * H + h) * W + w]);
                        const auto v = widen_lane(Intrinsics::detail::bit_cast_memcpy<Half>(bits));
                        if (__builtin_isnan(v)) {
                            if (!__builtin_isnan(best_v)) {
                                best_v = v;
                                best_h = h;
                                best_w = w;
                            }
                        } else if (__builtin_isnan(best_v) || v > best_v) {
                            best_v = v;
                            best_h = h;
                            best_w = w;
                        }
                    }
                }
                const auto in_idx = ((n * C + c) * H + best_h) * W + best_w;
                const auto dx_bits = Intrinsics::detail::bit_cast_memcpy<uint16_t>(dx[in_idx]);
                const auto dy_bits = Intrinsics::detail::bit_cast_memcpy<uint16_t>(dy[idx]);
                const auto sum = widen_lane(Intrinsics::detail::bit_cast_memcpy<Half>(dx_bits)) + widen_lane(Intrinsics::detail::bit_cast_memcpy<Half>(dy_bits));
                dx[in_idx] = narrow_lane<Half>(sum);
            }
        }

        /**
         * Backward routing for packed fp16x2/bf16x2. Per-lane argmax, per-lane dy route.
         * @brief Scalar MaxPool2d backward window loop for packed x2 dtypes.
         */
        template<typename P>
        void max_pool2d_g_window_packed(const P* dy, const P* x, P* dx,
                                        const size_t N, const size_t C,
                                        const size_t H, const size_t W,
                                        const size_t Hout, const size_t Wout,
                                        const size_t kernel, const size_t stride,
                                        const size_t begin, const size_t end) {
            static_cast<void>(N);
            using Lane = packed_lane_t<P>;
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto best_lo_v = -std::numeric_limits<float32_t>::infinity();
                auto best_hi_v = -std::numeric_limits<float32_t>::infinity();
                auto best_lo_h = size_t{0};
                auto best_lo_w = size_t{0};
                auto best_hi_h = size_t{0};
                auto best_hi_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(x[((n * C + c) * H + h) * W + w]);
                        const auto lo = widen_lane(lane_lo<Lane>(bits));
                        const auto hi = widen_lane(lane_hi<Lane>(bits));
                        if (__builtin_isnan(lo)) {
                            if (!__builtin_isnan(best_lo_v)) {
                                best_lo_v = lo;
                                best_lo_h = h;
                                best_lo_w = w;
                            }
                        } else if (__builtin_isnan(best_lo_v) || lo > best_lo_v) {
                            best_lo_v = lo;
                            best_lo_h = h;
                            best_lo_w = w;
                        }
                        if (__builtin_isnan(hi)) {
                            if (!__builtin_isnan(best_hi_v)) {
                                best_hi_v = hi;
                                best_hi_h = h;
                                best_hi_w = w;
                            }
                        } else if (__builtin_isnan(best_hi_v) || hi > best_hi_v) {
                            best_hi_v = hi;
                            best_hi_h = h;
                            best_hi_w = w;
                        }
                    }
                }
                const auto out_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[idx]);
                const auto lo_in_idx = ((n * C + c) * H + best_lo_h) * W + best_lo_w;
                const auto hi_in_idx = ((n * C + c) * H + best_hi_h) * W + best_hi_w;

                const auto dx_lo_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dx[lo_in_idx]);
                const auto dx_lo = widen_lane(lane_lo<Lane>(dx_lo_bits));
                const auto dy_lo = widen_lane(lane_lo<Lane>(out_bits));
                const auto sum_lo = dx_lo + dy_lo;
                const auto lo_keep = static_cast<uint16_t>(dx_lo_bits >> 16);
                dx[lo_in_idx] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<Lane>(sum_lo)), lo_keep);

                const auto dx_hi_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dx[hi_in_idx]);
                const auto dx_hi = widen_lane(lane_hi<Lane>(dx_hi_bits));
                const auto dy_hi = widen_lane(lane_hi<Lane>(out_bits));
                const auto sum_hi = dx_hi + dy_hi;
                const auto hi_keep = static_cast<uint16_t>(dx_hi_bits & 0xFFFFu);
                dx[hi_in_idx] = pack_lanes<P>(hi_keep, Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<Lane>(sum_hi)));
            }
        }

        /**
         * Backward routing for int4x2. Per-nibble argmax, per-nibble dy route.
         * @brief Scalar MaxPool2d backward window loop for int4x2.
         */
        void max_pool2d_g_window_int4x2(const int4x2_t* dy, const int4x2_t* x, int4x2_t* dx,
                                        const size_t N, const size_t C,
                                        const size_t H, const size_t W,
                                        const size_t Hout, const size_t Wout,
                                        const size_t kernel, const size_t stride,
                                        const size_t begin, const size_t end) {
            static_cast<void>(N);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto best_lo = int8_t{-8};
                auto best_hi = int8_t{-8};
                auto best_lo_h = size_t{0};
                auto best_lo_w = size_t{0};
                auto best_hi_h = size_t{0};
                auto best_hi_w = size_t{0};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto h = h0 + kh;
                        const auto w = w0 + kw;
                        const auto lo = int4_lo(x[((n * C + c) * H + h) * W + w]);
                        const auto hi = int4_hi(x[((n * C + c) * H + h) * W + w]);
                        if (lo > best_lo) {
                            best_lo = lo;
                            best_lo_h = h;
                            best_lo_w = w;
                        }
                        if (hi > best_hi) {
                            best_hi = hi;
                            best_hi_h = h;
                            best_hi_w = w;
                        }
                    }
                }
                const auto out_lo = int4_lo(dy[idx]);
                const auto out_hi = int4_hi(dy[idx]);
                const auto lo_in_idx = ((n * C + c) * H + best_lo_h) * W + best_lo_w;
                const auto hi_in_idx = ((n * C + c) * H + best_hi_h) * W + best_hi_w;
                const auto dx_lo = int4_lo(dx[lo_in_idx]);
                const auto dx_hi = int4_hi(dx[hi_in_idx]);
                dx[lo_in_idx] = pack_int4(dx_lo + out_lo, int4_hi(dx[lo_in_idx]));
                dx[hi_in_idx] = pack_int4(int4_lo(dx[hi_in_idx]), dx_hi + out_hi);
            }
        }

        /**
         * @brief Function pointer type shared by every MaxPool2d backward window kernel.
         */
        template<typename T>
        using MaxPool2dGWindowFn = void(*)(const T*, const T*, T*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
    }

    /**
     * Pack two uint32_t parameters into the Node's float64_t scalar slot. The
     * backward node unpacks them to recover kernel and stride.
     * @brief Pack kernel and stride into a float64_t bit pattern.
     */
    [[nodiscard]] inline float64_t pack_kernel_stride(const uint32_t kernel, const uint32_t stride) {
        const auto bits = (static_cast<uint64_t>(kernel) << 32) | static_cast<uint64_t>(stride);
        return Intrinsics::detail::bit_cast_memcpy<float64_t>(bits);
    }

    /**
     * @brief Unpack kernel and stride from a float64_t bit pattern.
     */
    [[nodiscard]] inline std::pair<uint32_t, uint32_t> unpack_kernel_stride(const float64_t scalar) {
        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint64_t>(scalar);
        return {static_cast<uint32_t>(bits >> 32), static_cast<uint32_t>(bits & 0xFFFFFFFFu)};
    }

    /**
     * MaxPool2d forward CPU backend. Validates device, dispatches on dtype, and
     * runs the window-loop kernel. Work is queued on the thread pool and drained
     * before returning.
     * @brief CPU MaxPool2d forward.
     * @param t Input tensor, rank-4 NCHW, CPU resident, contiguous.
     * @param out Output tensor, rank-4 NCHW, same dtype/device.
     * @param kernel Window height/width.
     * @param stride Step between adjacent windows.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void max_pool2d_cpu(const Tensor& t, Tensor& out, const uint32_t kernel, const uint32_t stride) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto N = t.shape[0];
        const auto C = t.shape[1];
        const auto H = t.shape[2];
        const auto W = t.shape[3];
        const auto Hout = out.shape[2];
        const auto Wout = out.shape[3];
        const auto in_ptr = static_cast<const uint8_t*>(t.data) + t.offset * DataTypes::dtype_bytes(t.dtype);
        auto* const out_ptr = static_cast<uint8_t*>(out.data) + out.offset * DataTypes::dtype_bytes(out.dtype);
        switch (t.dtype) {
            case DType::Float32: {
                const auto* in = reinterpret_cast<const float32_t*>(in_ptr);
                auto* o = reinterpret_cast<float32_t*>(out_ptr);
                const MaxPool2dWindowFn<float32_t> kernel_fn = Math::Spatial::max_pool2d_avx512_available()
                    ? Math::Spatial::max_pool2d_f32_window_avx512
                    : Math::Spatial::max_pool2d_avx2_available() ? Math::Spatial::max_pool2d_f32_window_avx2 : max_pool2d_window_scalar<float32_t>;
                launch_elementwise([=](const size_t begin, const size_t end) { kernel_fn(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float32_t), t, t, out);
                break;
            }
            case DType::Float64: {
                const auto* in = reinterpret_cast<const float64_t*>(in_ptr);
                auto* o = reinterpret_cast<float64_t*>(out_ptr);
                const MaxPool2dWindowFn<float64_t> kernel_fn = Math::Spatial::max_pool2d_avx512_available()
                    ? Math::Spatial::max_pool2d_f64_window_avx512
                    : Math::Spatial::max_pool2d_avx2_available() ? Math::Spatial::max_pool2d_f64_window_avx2 : max_pool2d_window_scalar<float64_t>;
                launch_elementwise([=](const size_t begin, const size_t end) { kernel_fn(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float64_t), t, t, out);
                break;
            }
            case DType::Float16: {
                const auto* in = reinterpret_cast<const float16_t*>(in_ptr);
                auto* o = reinterpret_cast<float16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_half<float16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float16_t), t, t, out);
                break;
            }
            case DType::BFloat16: {
                const auto* in = reinterpret_cast<const bfloat16_t*>(in_ptr);
                auto* o = reinterpret_cast<bfloat16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_half<bfloat16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(bfloat16_t), t, t, out);
                break;
            }
            case DType::Float16x2: {
                const auto* in = reinterpret_cast<const float16x2_t*>(in_ptr);
                auto* o = reinterpret_cast<float16x2_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_packed<float16x2_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float16x2_t), t, t, out);
                break;
            }
            case DType::BFloat16x2: {
                const auto* in = reinterpret_cast<const bfloat16x2_t*>(in_ptr);
                auto* o = reinterpret_cast<bfloat16x2_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_packed<bfloat16x2_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(bfloat16x2_t), t, t, out);
                break;
            }
            case DType::Int8: {
                const auto* in = reinterpret_cast<const int8_t*>(in_ptr);
                auto* o = reinterpret_cast<int8_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<int8_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int8_t), t, t, out);
                break;
            }
            case DType::UInt8: {
                const auto* in = reinterpret_cast<const uint8_t*>(in_ptr);
                auto* o = reinterpret_cast<uint8_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<uint8_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint8_t), t, t, out);
                break;
            }
            case DType::Int16: {
                const auto* in = reinterpret_cast<const int16_t*>(in_ptr);
                auto* o = reinterpret_cast<int16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<int16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int16_t), t, t, out);
                break;
            }
            case DType::UInt16: {
                const auto* in = reinterpret_cast<const uint16_t*>(in_ptr);
                auto* o = reinterpret_cast<uint16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<uint16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint16_t), t, t, out);
                break;
            }
            case DType::Int32: {
                const auto* in = reinterpret_cast<const int32_t*>(in_ptr);
                auto* o = reinterpret_cast<int32_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<int32_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int32_t), t, t, out);
                break;
            }
            case DType::UInt32: {
                const auto* in = reinterpret_cast<const uint32_t*>(in_ptr);
                auto* o = reinterpret_cast<uint32_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<uint32_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint32_t), t, t, out);
                break;
            }
            case DType::Int64: {
                const auto* in = reinterpret_cast<const int64_t*>(in_ptr);
                auto* o = reinterpret_cast<int64_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<int64_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int64_t), t, t, out);
                break;
            }
            case DType::UInt64: {
                const auto* in = reinterpret_cast<const uint64_t*>(in_ptr);
                auto* o = reinterpret_cast<uint64_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_scalar<uint64_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint64_t), t, t, out);
                break;
            }
            case DType::Int4x2: {
                const auto* in = reinterpret_cast<const int4x2_t*>(in_ptr);
                auto* o = reinterpret_cast<int4x2_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { max_pool2d_window_int4x2(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int4x2_t), t, t, out);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * MaxPool2d backward CPU backend. Validates device, zeroes dx, dispatches on
     * dtype, and runs the argmax-recompute routing kernel.
     * @brief CPU MaxPool2d backward.
     * @param dy Upstream gradient tensor, rank-4 NCHW.
     * @param x Forward input tensor, rank-4 NCHW.
     * @param dx Output gradient tensor, rank-4 NCHW (written).
     * @param kernel Window height/width.
     * @param stride Step between adjacent windows.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void max_pool2d_g_cpu(const Tensor& dy, const Tensor& x, Tensor& dx, const uint32_t kernel, const uint32_t stride) {
        if (dy.device != Device::CPU || x.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        using DType = DataTypes::DType;
        const auto N = x.shape[0];
        const auto C = x.shape[1];
        const auto H = x.shape[2];
        const auto W = x.shape[3];
        const auto Hout = dy.shape[2];
        const auto Wout = dy.shape[3];
        const auto x_ptr = static_cast<const uint8_t*>(x.data) + x.offset * DataTypes::dtype_bytes(x.dtype);
        const auto dy_ptr = static_cast<const uint8_t*>(dy.data) + dy.offset * DataTypes::dtype_bytes(dy.dtype);
        auto* const dx_ptr = static_cast<uint8_t*>(dx.data) + dx.offset * DataTypes::dtype_bytes(dx.dtype);
        dx.set_zero();
        // Overlapping windows (stride < kernel) can route multiple output elements to
        // the same input element. Keep those writes serial by running inline; the
        // scalar/packed/int4x2 kernels then accumulate safely via RMW.
        const auto run_backward = [&](const auto& worker, const size_t out_bytes) {
            const auto numel = dy.numel;
            if (stride < kernel) {
                worker(0, numel);
            } else {
                launch_elementwise(worker, numel, out_bytes, x, dy, dx);
            }
        };
        switch (x.dtype) {
            case DType::Float32: {
                const auto* dy_in = reinterpret_cast<const float32_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const float32_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<float32_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<float32_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float32_t));
                break;
            }
            case DType::Float64: {
                const auto* dy_in = reinterpret_cast<const float64_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const float64_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<float64_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<float64_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float64_t));
                break;
            }
            case DType::Float16: {
                const auto* dy_in = reinterpret_cast<const float16_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const float16_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<float16_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_half<float16_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float16_t));
                break;
            }
            case DType::BFloat16: {
                const auto* dy_in = reinterpret_cast<const bfloat16_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const bfloat16_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<bfloat16_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_half<bfloat16_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(bfloat16_t));
                break;
            }
            case DType::Float16x2: {
                const auto* dy_in = reinterpret_cast<const float16x2_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const float16x2_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<float16x2_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_packed<float16x2_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float16x2_t));
                break;
            }
            case DType::BFloat16x2: {
                const auto* dy_in = reinterpret_cast<const bfloat16x2_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const bfloat16x2_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<bfloat16x2_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_packed<bfloat16x2_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(bfloat16x2_t));
                break;
            }
            case DType::Int8: {
                const auto* dy_in = reinterpret_cast<const int8_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const int8_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<int8_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<int8_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(int8_t));
                break;
            }
            case DType::UInt8: {
                const auto* dy_in = reinterpret_cast<const uint8_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const uint8_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<uint8_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<uint8_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(uint8_t));
                break;
            }
            case DType::Int16: {
                const auto* dy_in = reinterpret_cast<const int16_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const int16_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<int16_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<int16_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(int16_t));
                break;
            }
            case DType::UInt16: {
                const auto* dy_in = reinterpret_cast<const uint16_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const uint16_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<uint16_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<uint16_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(uint16_t));
                break;
            }
            case DType::Int32: {
                const auto* dy_in = reinterpret_cast<const int32_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const int32_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<int32_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<int32_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(int32_t));
                break;
            }
            case DType::UInt32: {
                const auto* dy_in = reinterpret_cast<const uint32_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const uint32_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<uint32_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<uint32_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(uint32_t));
                break;
            }
            case DType::Int64: {
                const auto* dy_in = reinterpret_cast<const int64_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const int64_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<int64_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<int64_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(int64_t));
                break;
            }
            case DType::UInt64: {
                const auto* dy_in = reinterpret_cast<const uint64_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const uint64_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<uint64_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_scalar<uint64_t>(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(uint64_t));
                break;
            }
            case DType::Int4x2: {
                const auto* dy_in = reinterpret_cast<const int4x2_t*>(dy_ptr);
                const auto* x_in = reinterpret_cast<const int4x2_t*>(x_ptr);
                auto* dx_out = reinterpret_cast<int4x2_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { max_pool2d_g_window_int4x2(dy_in, x_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(int4x2_t));
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * 2D max pooling over the spatial dimensions of a rank-4 NCHW tensor. The
     * output shape is {N, C, Hout, Wout} with Hout = floor((H - kernel) / stride) + 1
     * and similarly for Wout; no padding is applied. The first occurrence of a
     * repeated maximum wins in both forward and backward. Errors are thrown
     * synchronously for invalid geometry; the compute is queued asynchronously and
     * materializes at out.sync().
     * @brief 2D max pooling (NCHW).
     * @param kernel Window height and width.
     * @param stride Step between adjacent windows.
     * @return New rank-4 tensor of the same dtype/device as this.
     * @throws ErrorHandling::MismatchedShapes Input is not rank-4, kernel > H/W,
     *         kernel == 0, or stride == 0.
     * @throws ErrorHandling::NotImplemented Non-contiguous operand.
     */
    Tensor Tensor::max_pool2d(const uint32_t kernel, const uint32_t stride) const {
        if (this->ndim != 4) {
            throw ErrorHandling::MismatchedShapes("max_pool2d requires rank-4 input (NCHW)");
        }
        if (kernel == 0) {
            throw ErrorHandling::MismatchedShapes("max_pool2d kernel must be non-zero");
        }
        if (stride == 0) {
            throw ErrorHandling::MismatchedShapes("max_pool2d stride must be non-zero");
        }
        const auto H = this->shape[2];
        const auto W = this->shape[3];
        if (kernel > H || kernel > W) {
            throw ErrorHandling::MismatchedShapes("max_pool2d kernel larger than input spatial size");
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        const auto Hout = (H - kernel) / stride + 1;
        const auto Wout = (W - kernel) / stride + 1;
        auto out = Tensor(std::vector{this->shape[0], this->shape[1], Hout, Wout}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::max_pool2d_cpu(*this, out, kernel, stride);
        } else if (this->device == Device::GPU) {
            detail::max_pool2d_gpu(*this, out, kernel, stride);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (this->requires_grad) {
            out.wire_node(detail::max_pool2d_g_bridge, *this);
            out.node->_scalar = detail::pack_kernel_stride(kernel, stride);
        }
        return out;
    }
}

namespace BeanTensor::Tensors::detail {

    /**
     * inputs are (x). Only x receives a gradient. The kernel recomputes the window
     * argmax from x and routes dy there, using the kernel/stride packed into the
     * owning Node's _scalar slot.
     * @brief MaxPool2d backward implementation (recomputes argmax and routes dy).
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1 or the device is neither CPU nor GPU.
     */
    void max_pool2d_g_node(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        const auto [kernel, stride] = unpack_kernel_stride(node->_scalar);
        const auto work = is_float_kind(grad_dtype)
            ? grad_dtype
            : DataTypes::DType::Float32;
        const auto xw = grad_converted_copy(x, work);
        auto dy_cast = std::optional<Tensor>{};
        const Tensor* dyw = output_grad;
        if (output_grad->dtype != work) {
            dy_cast = grad_converted_copy(*output_grad, work);
            dyw = &*dy_cast;
        }
        auto dx = Tensor(std::vector(x.shape, x.shape + x.ndim), work, x.device, false);
        if (x.device == Device::CPU) {
            max_pool2d_g_cpu(*dyw, xw, dx, kernel, stride);
        } else if (x.device == Device::GPU) {
            max_pool2d_g_gpu(*dyw, xw, dx, kernel, stride);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        grad_convert_inplace(dx, grad_dtype);
        if (x.node != nullptr) {
            grad_probe_hit(dx.dtype);
        }
        x.accumulate_grad(std::move(dx));
    }

    /**
     * Backward trampoline wired into Node::_backward_fn. wire_node's function-pointer
     * type takes a mutable output_grad while max_pool2d_g_node only reads it; this
     * bridge forwards the call unchanged.
     * @brief Backward trampoline for MaxPool2d, matching Node::_backward_fn's signature.
     */
    void max_pool2d_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        max_pool2d_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
