/**
 * @file AvgPool2dAVX512.cpp
 * @author GreenBean
 * @brief AVX512-compiled scalar window kernels for AvgPool2d fp32/fp64 forward.
 * The op's 2D window access pattern does not vectorize simply, so this TU
 * provides the same scalar loop as the generic CPU path but compiled with AVX512
 * flags for ABI consistency.
 * @version 0.1
 * @date 2026-11-10
 */
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Helpers/Hardware.h"
#include "Math/Spatial/AvgPool2d/AvgPool2dKernels.h"

namespace BeanTensor::Math::Spatial {

    bool avg_pool2d_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    namespace {
        template<typename T>
        void window_scalar(const T* in, T* out,
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
    }

    void avg_pool2d_f32_window_avx512(const float32_t* in, float32_t* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
        window_scalar(in, out, N, C, H, W, Hout, Wout, kernel, stride, begin, end);
    }

    void avg_pool2d_f64_window_avx512(const float64_t* in, float64_t* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
        window_scalar(in, out, N, C, H, W, Hout, Wout, kernel, stride, begin, end);
    }
}
