/**
 * @file AvgPool2dCPU.cpp
 * @author GreenBean
 * @brief CPU backend for AvgPool2d forward/backward, plus the Tensor::avg_pool2d
 * method definitions and the AvgPool2dG backward trampoline.
 * @version 0.1
 * @date 2026-11-10
 */
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/AvgPool2dG/AvgPool2dG.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Spatial/AvgPool2d/AvgPool2d.h"
#include "Math/Spatial/AvgPool2d/AvgPool2dKernels.h"
#include "Math/Spatial/SpatialShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace spatial_shared;

    namespace {
        // --------------------------- forward kernels ---------------------------

        /**
         * Window mean for scalar float/integer dtypes. Sum then divide by kernel²;
         * integer sums accumulate in fp64 and narrow with clamp/truncate toward zero.
         * @brief Scalar AvgPool2d forward window loop.
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
        void avg_pool2d_window_scalar(const T* in, T* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            const auto count = static_cast<float64_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                if constexpr (std::is_floating_point_v<T>) {
                    auto total = T{0};
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            total += in[((n * C + c) * H + h0 + kh) * W + w0 + kw];
                        }
                    }
                    out[idx] = total / static_cast<T>(count);
                } else {
                    auto total = float64_t{0.0};
                    for (auto kh = size_t{0}; kh < kernel; ++kh) {
                        for (auto kw = size_t{0}; kw < kernel; ++kw) {
                            total += static_cast<float64_t>(in[((n * C + c) * H + h0 + kh) * W + w0 + kw]);
                        }
                    }
                    constexpr auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                    constexpr auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
                    const auto mean = total / count;
                    if (__builtin_isnan(mean)) {
                        out[idx] = T{0};
                    } else if (mean <= lo) {
                        out[idx] = std::numeric_limits<T>::lowest();
                    } else if (mean >= hi) {
                        out[idx] = std::numeric_limits<T>::max();
                    } else {
                        out[idx] = static_cast<T>(mean);
                    }
                }
            }
        }

        /**
         * Window mean for fp16/bf16 lanes. Widen to fp32 for sum and divide, then
         * narrow RNE back to the storage dtype.
         * @brief Scalar AvgPool2d forward window loop for half dtypes.
         */
        template<typename Half>
        void avg_pool2d_window_half(const Half* in, Half* out,
                                    const size_t N, const size_t C,
                                    const size_t H, const size_t W,
                                    const size_t Hout, const size_t Wout,
                                    const size_t kernel, const size_t stride,
                                    const size_t begin, const size_t end) {
            static_cast<void>(N);
            const auto count = static_cast<float32_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto total = float32_t{0.0f};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        total += widen_lane(in[((n * C + c) * H + h0 + kh) * W + w0 + kw]);
                    }
                }
                out[idx] = narrow_lane<Half>(total / count);
            }
        }

        /**
         * Window mean for packed fp16x2/bf16x2 dtypes. Each lane is treated
         * independently; per-lane fp32 sum/divide, then pack.
         * @brief Scalar AvgPool2d forward window loop for packed x2 dtypes.
         */
        template<typename P>
        void avg_pool2d_window_packed(const P* in, P* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            using Lane = packed_lane_t<P>;
            const auto count = static_cast<float32_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto total_lo = float32_t{0.0f};
                auto total_hi = float32_t{0.0f};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[((n * C + c) * H + h0 + kh) * W + w0 + kw]);
                        total_lo += widen_lane(lane_lo<Lane>(bits));
                        total_hi += widen_lane(lane_hi<Lane>(bits));
                    }
                }
                const auto lo = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<Lane>(total_lo / count));
                const auto hi = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<Lane>(total_hi / count));
                out[idx] = pack_lanes<P>(lo, hi);
            }
        }

        /**
         * Window mean for int4x2. Per-nibble sum in fp32, divide, clamp to [-8, 7].
         * @brief Scalar AvgPool2d forward window loop for int4x2.
         */
        void avg_pool2d_window_int4x2(const int4x2_t* in, int4x2_t* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            const auto count = static_cast<float32_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                auto total_lo = float32_t{0.0f};
                auto total_hi = float32_t{0.0f};
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto v = in[((n * C + c) * H + h0 + kh) * W + w0 + kw];
                        total_lo += static_cast<float32_t>(int4_lo(v));
                        total_hi += static_cast<float32_t>(int4_hi(v));
                    }
                }
                const auto mean_lo = static_cast<int8_t>(std::clamp(total_lo / count, -8.0f, 7.0f));
                const auto mean_hi = static_cast<int8_t>(std::clamp(total_hi / count, -8.0f, 7.0f));
                out[idx] = pack_int4(mean_lo, mean_hi);
            }
        }

        /**
         * @brief Function pointer type shared by every AvgPool2d forward window kernel.
         */
        template<typename T>
        using AvgPool2dWindowFn = void(*)(const T*, T*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

        // --------------------------- backward kernels ---------------------------

        /**
         * Backward routing for scalar float dtypes. dx[window] += dy / (kernel²).
         * Overlapping windows accumulate through read-modify-write; the caller
         * serializes this kernel when stride < kernel.
         * @brief Scalar AvgPool2d backward window loop.
         */
        template<typename T>
        void avg_pool2d_g_window_scalar(const T* dy, T* dx,
                                        const size_t N, const size_t C,
                                        const size_t H, const size_t W,
                                        const size_t Hout, const size_t Wout,
                                        const size_t kernel, const size_t stride,
                                        const size_t begin, const size_t end) {
            static_cast<void>(N);
            const auto scale = T{1} / static_cast<T>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                const auto inc = dy[idx] * scale;
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        dx[((n * C + c) * H + h0 + kh) * W + w0 + kw] += inc;
                    }
                }
            }
        }

        /**
         * Backward routing for fp16/bf16. Widen to fp32 for accumulation, narrow
         * back to the storage dtype on every read-modify-write.
         * @brief Scalar AvgPool2d backward window loop for half dtypes.
         */
        template<typename Half>
        void avg_pool2d_g_window_half(const Half* dy, Half* dx,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
            static_cast<void>(N);
            const auto scale = float32_t{1.0f} / static_cast<float32_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                const auto inc = widen_lane(dy[idx]) * scale;
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto in_idx = ((n * C + c) * H + h0 + kh) * W + w0 + kw;
                        const auto sum = widen_lane(dx[in_idx]) + inc;
                        dx[in_idx] = narrow_lane<Half>(sum);
                    }
                }
            }
        }

        /**
         * Backward routing for packed fp16x2/bf16x2. Per-lane widen, accumulate,
         * and narrow; lanes are independent.
         * @brief Scalar AvgPool2d backward window loop for packed x2 dtypes.
         */
        template<typename P>
        void avg_pool2d_g_window_packed(const P* dy, P* dx,
                                        const size_t N, const size_t C,
                                        const size_t H, const size_t W,
                                        const size_t Hout, const size_t Wout,
                                        const size_t kernel, const size_t stride,
                                        const size_t begin, const size_t end) {
            static_cast<void>(N);
            using Lane = packed_lane_t<P>;
            const auto scale = float32_t{1.0f} / static_cast<float32_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                const auto out_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[idx]);
                const auto inc_lo = widen_lane(lane_lo<Lane>(out_bits)) * scale;
                const auto inc_hi = widen_lane(lane_hi<Lane>(out_bits)) * scale;
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto in_idx = ((n * C + c) * H + h0 + kh) * W + w0 + kw;
                        const auto dx_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dx[in_idx]);
                        const auto sum_lo = widen_lane(lane_lo<Lane>(dx_bits)) + inc_lo;
                        const auto sum_hi = widen_lane(lane_hi<Lane>(dx_bits)) + inc_hi;
                        const auto lo = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<Lane>(sum_lo));
                        const auto hi = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<Lane>(sum_hi));
                        dx[in_idx] = pack_lanes<P>(lo, hi);
                    }
                }
            }
        }

        /**
         * Backward routing for int4x2. Per-nibble fp32 accumulation, clamped to
         * the signed nibble range.
         * @brief Scalar AvgPool2d backward window loop for int4x2.
         */
        void avg_pool2d_g_window_int4x2(const int4x2_t* dy, int4x2_t* dx,
                                        const size_t N, const size_t C,
                                        const size_t H, const size_t W,
                                        const size_t Hout, const size_t Wout,
                                        const size_t kernel, const size_t stride,
                                        const size_t begin, const size_t end) {
            static_cast<void>(N);
            const auto scale = float32_t{1.0f} / static_cast<float32_t>(kernel * kernel);
            for (auto idx = begin; idx < end; ++idx) {
                const auto ow = idx % Wout;
                const auto oh = (idx / Wout) % Hout;
                const auto c = (idx / (Wout * Hout)) % C;
                const auto n = idx / (Wout * Hout * C);
                const auto h0 = oh * stride;
                const auto w0 = ow * stride;
                const auto inc_lo = static_cast<float32_t>(int4_lo(dy[idx])) * scale;
                const auto inc_hi = static_cast<float32_t>(int4_hi(dy[idx])) * scale;
                for (auto kh = size_t{0}; kh < kernel; ++kh) {
                    for (auto kw = size_t{0}; kw < kernel; ++kw) {
                        const auto in_idx = ((n * C + c) * H + h0 + kh) * W + w0 + kw;
                        const auto sum_lo = static_cast<float32_t>(int4_lo(dx[in_idx])) + inc_lo;
                        const auto sum_hi = static_cast<float32_t>(int4_hi(dx[in_idx])) + inc_hi;
                        const auto v_lo = static_cast<int8_t>(std::clamp(sum_lo, -8.0f, 7.0f));
                        const auto v_hi = static_cast<int8_t>(std::clamp(sum_hi, -8.0f, 7.0f));
                        dx[in_idx] = pack_int4(v_lo, v_hi);
                    }
                }
            }
        }

        /**
         * @brief Function pointer type shared by every AvgPool2d backward window kernel.
         */
        template<typename T>
        using AvgPool2dGWindowFn = void(*)(const T*, T*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
    }

    /**
     * AvgPool2d forward CPU backend. Validates device, dispatches on dtype, and
     * runs the window-loop kernel. Work is queued on the thread pool and drained
     * before returning.
     * @brief CPU AvgPool2d forward.
     * @param t Input tensor, rank-4 NCHW, CPU resident, contiguous.
     * @param out Output tensor, rank-4 NCHW, same dtype/device.
     * @param kernel Window height/width.
     * @param stride Step between adjacent windows.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void avg_pool2d_cpu(const Tensor& t, Tensor& out, const uint32_t kernel, const uint32_t stride) {
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
                const AvgPool2dWindowFn<float32_t> kernel_fn = Math::Spatial::avg_pool2d_avx512_available()
                    ? Math::Spatial::avg_pool2d_f32_window_avx512
                    : Math::Spatial::avg_pool2d_avx2_available() ? Math::Spatial::avg_pool2d_f32_window_avx2 : avg_pool2d_window_scalar<float32_t>;
                launch_elementwise([=](const size_t begin, const size_t end) { kernel_fn(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float32_t), t, t, out);
                break;
            }
            case DType::Float64: {
                const auto* in = reinterpret_cast<const float64_t*>(in_ptr);
                auto* o = reinterpret_cast<float64_t*>(out_ptr);
                const AvgPool2dWindowFn<float64_t> kernel_fn = Math::Spatial::avg_pool2d_avx512_available()
                    ? Math::Spatial::avg_pool2d_f64_window_avx512
                    : Math::Spatial::avg_pool2d_avx2_available() ? Math::Spatial::avg_pool2d_f64_window_avx2 : avg_pool2d_window_scalar<float64_t>;
                launch_elementwise([=](const size_t begin, const size_t end) { kernel_fn(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float64_t), t, t, out);
                break;
            }
            case DType::Float16: {
                const auto* in = reinterpret_cast<const float16_t*>(in_ptr);
                auto* o = reinterpret_cast<float16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_half<float16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float16_t), t, t, out);
                break;
            }
            case DType::BFloat16: {
                const auto* in = reinterpret_cast<const bfloat16_t*>(in_ptr);
                auto* o = reinterpret_cast<bfloat16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_half<bfloat16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(bfloat16_t), t, t, out);
                break;
            }
            case DType::Float16x2: {
                const auto* in = reinterpret_cast<const float16x2_t*>(in_ptr);
                auto* o = reinterpret_cast<float16x2_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_packed<float16x2_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(float16x2_t), t, t, out);
                break;
            }
            case DType::BFloat16x2: {
                const auto* in = reinterpret_cast<const bfloat16x2_t*>(in_ptr);
                auto* o = reinterpret_cast<bfloat16x2_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_packed<bfloat16x2_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(bfloat16x2_t), t, t, out);
                break;
            }
            case DType::Int8: {
                const auto* in = reinterpret_cast<const int8_t*>(in_ptr);
                auto* o = reinterpret_cast<int8_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<int8_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int8_t), t, t, out);
                break;
            }
            case DType::UInt8: {
                const auto* in = reinterpret_cast<const uint8_t*>(in_ptr);
                auto* o = reinterpret_cast<uint8_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<uint8_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint8_t), t, t, out);
                break;
            }
            case DType::Int16: {
                const auto* in = reinterpret_cast<const int16_t*>(in_ptr);
                auto* o = reinterpret_cast<int16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<int16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int16_t), t, t, out);
                break;
            }
            case DType::UInt16: {
                const auto* in = reinterpret_cast<const uint16_t*>(in_ptr);
                auto* o = reinterpret_cast<uint16_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<uint16_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint16_t), t, t, out);
                break;
            }
            case DType::Int32: {
                const auto* in = reinterpret_cast<const int32_t*>(in_ptr);
                auto* o = reinterpret_cast<int32_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<int32_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int32_t), t, t, out);
                break;
            }
            case DType::UInt32: {
                const auto* in = reinterpret_cast<const uint32_t*>(in_ptr);
                auto* o = reinterpret_cast<uint32_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<uint32_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint32_t), t, t, out);
                break;
            }
            case DType::Int64: {
                const auto* in = reinterpret_cast<const int64_t*>(in_ptr);
                auto* o = reinterpret_cast<int64_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<int64_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int64_t), t, t, out);
                break;
            }
            case DType::UInt64: {
                const auto* in = reinterpret_cast<const uint64_t*>(in_ptr);
                auto* o = reinterpret_cast<uint64_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_scalar<uint64_t>(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(uint64_t), t, t, out);
                break;
            }
            case DType::Int4x2: {
                const auto* in = reinterpret_cast<const int4x2_t*>(in_ptr);
                auto* o = reinterpret_cast<int4x2_t*>(out_ptr);
                launch_elementwise([=](const size_t begin, const size_t end) { avg_pool2d_window_int4x2(in, o, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, out.numel, out.numel * sizeof(int4x2_t), t, t, out);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * AvgPool2d backward CPU backend. Validates device, zeroes dx, dispatches on
     * dtype, and runs the equal-share routing kernel.
     * @brief CPU AvgPool2d backward.
     * @param dy Upstream gradient tensor, rank-4 NCHW.
     * @param dx Output gradient tensor, rank-4 NCHW (written).
     * @param kernel Window height/width.
     * @param stride Step between adjacent windows.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void avg_pool2d_g_cpu(const Tensor& dy, Tensor& dx, const uint32_t kernel, const uint32_t stride) {
        if (dy.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        using DType = DataTypes::DType;
        const auto N = dx.shape[0];
        const auto C = dx.shape[1];
        const auto H = dx.shape[2];
        const auto W = dx.shape[3];
        const auto Hout = dy.shape[2];
        const auto Wout = dy.shape[3];
        const auto dy_ptr = static_cast<const uint8_t*>(dy.data) + dy.offset * DataTypes::dtype_bytes(dy.dtype);
        auto* const dx_ptr = static_cast<uint8_t*>(dx.data) + dx.offset * DataTypes::dtype_bytes(dx.dtype);
        dx.set_zero();
        // Overlapping windows (stride < kernel) can add to the same input element
        // from multiple output elements. Keep those writes serial by running inline;
        // the scalar/packed/int4x2 kernels then accumulate safely via RMW.
        const auto run_backward = [&](const auto& worker, const size_t out_bytes) {
            const auto numel = dy.numel;
            if (stride < kernel) {
                worker(0, numel);
            } else {
                launch_elementwise(worker, numel, out_bytes, dy, dx, dx);
            }
        };
        switch (dy.dtype) {
            case DType::Float32: {
                const auto* dy_in = reinterpret_cast<const float32_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<float32_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_scalar<float32_t>(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float32_t));
                break;
            }
            case DType::Float64: {
                const auto* dy_in = reinterpret_cast<const float64_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<float64_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_scalar<float64_t>(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float64_t));
                break;
            }
            case DType::Float16: {
                const auto* dy_in = reinterpret_cast<const float16_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<float16_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_half<float16_t>(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float16_t));
                break;
            }
            case DType::BFloat16: {
                const auto* dy_in = reinterpret_cast<const bfloat16_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<bfloat16_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_half<bfloat16_t>(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(bfloat16_t));
                break;
            }
            case DType::Float16x2: {
                const auto* dy_in = reinterpret_cast<const float16x2_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<float16x2_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_packed<float16x2_t>(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(float16x2_t));
                break;
            }
            case DType::BFloat16x2: {
                const auto* dy_in = reinterpret_cast<const bfloat16x2_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<bfloat16x2_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_packed<bfloat16x2_t>(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(bfloat16x2_t));
                break;
            }
            case DType::Int4x2: {
                const auto* dy_in = reinterpret_cast<const int4x2_t*>(dy_ptr);
                auto* dx_out = reinterpret_cast<int4x2_t*>(dx_ptr);
                run_backward([=](const size_t begin, const size_t end) { avg_pool2d_g_window_int4x2(dy_in, dx_out, N, C, H, W, Hout, Wout, kernel, stride, begin, end); }, dx.numel * sizeof(int4x2_t));
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
     * 2D average pooling over the spatial dimensions of a rank-4 NCHW tensor.
     * Output shape is {N, C, Hout, Wout} with Hout = floor((H - kernel) / stride) + 1
     * and Wout = floor((W - kernel) / stride) + 1; no padding. Each output element
     * is the arithmetic mean of the kernel×kernel input window. Errors are thrown
     * synchronously for invalid geometry; the compute is queued asynchronously and
     * materializes at out.sync().
     * @brief 2D average pooling (square kernel, no padding).
     * @param kernel Window height and width.
     * @param stride Step between adjacent windows.
     * @return New rank-4 tensor of the same dtype/device as this.
     * @throws ErrorHandling::MismatchedShapes Input is not rank-4, kernel > H/W,
     *         kernel == 0, or stride == 0.
     * @throws ErrorHandling::NotImplemented Non-contiguous operand.
     */
    Tensor Tensor::avg_pool2d(const uint32_t kernel) const & {
        return this->avg_pool2d(kernel, kernel);
    }

    Tensor Tensor::avg_pool2d(const uint32_t kernel, const uint32_t stride) const & {
        if (this->ndim != 4) {
            throw ErrorHandling::MismatchedShapes("avg_pool2d requires rank-4 input (NCHW)");
        }
        if (kernel == 0) {
            throw ErrorHandling::MismatchedShapes("avg_pool2d kernel must be non-zero");
        }
        if (stride == 0) {
            throw ErrorHandling::MismatchedShapes("avg_pool2d stride must be non-zero");
        }
        const auto H = this->shape[2];
        const auto W = this->shape[3];
        if (kernel > H || kernel > W) {
            throw ErrorHandling::MismatchedShapes("avg_pool2d kernel larger than input spatial size");
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        const auto Hout = (H - kernel) / stride + 1;
        const auto Wout = (W - kernel) / stride + 1;
        auto out = Tensor(std::vector{this->shape[0], this->shape[1], Hout, Wout}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::avg_pool2d_cpu(*this, out, kernel, stride);
        } else if (this->device == Device::GPU) {
            detail::avg_pool2d_gpu(*this, out, kernel, stride);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (this->requires_grad) {
            out.wire_node(detail::avg_pool2d_g_bridge, *this);
            out.node->_scalar = detail::pack_avgpool2d_params(kernel, stride);
        }
        return out;
    }
}
