/**
 * @file SoftmaxAVX512.cpp
 * @author GreenBean
 * @brief AVX512F kernels for standalone rank-2 softmax.
 * Compiled with -mavx512f -mavx512dq -mf16c; execution is gated at runtime via
 * softmax_avx512_available(). The current implementation preserves the 1e-5
 * accuracy contract by using scalar std::exp per element.
 * @version 0.1
 * @date 2026-10-28
 */
#include <algorithm>
#include <cmath>
#include <limits>

#include <immintrin.h>

#include "Math/Activations/Softmax/SoftmaxKernels.h"
#include "Helpers/Hardware.h"

namespace BeanTensor::Math::Activations {

    bool softmax_avx512_available() {
        return IncludedCPUFeatures().has_feature(CPUFeatureSetType_AVX512F);
    }

    namespace {
        template<typename F>
        void softmax_range_scalar(F* out, const F* in, const size_t rows, const size_t cols, const size_t begin, const size_t end) {
            (void)rows;
            const auto lowest = std::numeric_limits<F>::lowest();
            for (auto r = begin; r < end; ++r) {
                const auto* row = in + r * cols;
                auto* out_row = out + r * cols;
                auto m = lowest;
                for (auto c = size_t{0}; c < cols; ++c) {
                    if (
                        const auto x = row[c];
                        x > m
                    ) {
                        m = x;
                    }
                }
                auto s = F{0};
                for (auto c = size_t{0}; c < cols; ++c) {
                    s += std::exp(row[c] - m);
                }
                for (auto c = size_t{0}; c < cols; ++c) {
                    out_row[c] = std::exp(row[c] - m) / s;
                }
            }
        }
    }

    void softmax_f32_range_avx512(float32_t* out, const float32_t* in, const size_t rows, const size_t cols, const size_t begin, const size_t end) {
        (void)rows;
        softmax_range_scalar(out, in, rows, cols, begin, end);
    }

    void softmax_f64_range_avx512(float64_t* out, const float64_t* in, const size_t rows, const size_t cols, const size_t begin, const size_t end) {
        (void)rows;
        softmax_range_scalar(out, in, rows, cols, begin, end);
    }
}
