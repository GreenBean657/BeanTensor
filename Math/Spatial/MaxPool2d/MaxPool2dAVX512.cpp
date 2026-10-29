/**
 * @file MaxPool2dAVX512.cpp
 * @author GreenBean
 * @brief AVX512-compiled scalar window kernels for MaxPool2d fp32/fp64 forward.
 * The op's 2D window access pattern does not vectorize simply, so this TU
 * provides the same scalar loop as the generic CPU path but compiled with AVX512
 * flags for ABI consistency.
 * @version 0.1
 * @date 2026-10-28
 */
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Helpers/Hardware.h"
#include "Math/Spatial/MaxPool2d/MaxPool2dKernels.h"

namespace BeanTensor::Math::Spatial {

    bool max_pool2d_avx512_available() {
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
    }

    void max_pool2d_f32_window_avx512(const float32_t* in, float32_t* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
        window_scalar(in, out, N, C, H, W, Hout, Wout, kernel, stride, begin, end);
    }

    void max_pool2d_f64_window_avx512(const float64_t* in, float64_t* out,
                                      const size_t N, const size_t C,
                                      const size_t H, const size_t W,
                                      const size_t Hout, const size_t Wout,
                                      const size_t kernel, const size_t stride,
                                      const size_t begin, const size_t end) {
        window_scalar(in, out, N, C, H, W, Hout, Wout, kernel, stride, begin, end);
    }
}
