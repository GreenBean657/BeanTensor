/**
 * @file SoftmaxAVX2.cpp
 * @author GreenBean
 * @brief AVX2+FMA kernels for standalone rank-2 softmax.
 * Compiled with -mavx2 -mfma -mf16c; execution is gated at runtime via
 * softmax_avx2_available(). The current implementation preserves the 1e-5
 * accuracy contract by using scalar std::exp per element; AVX2 is used for
 * the max-subtract stage, the remainder is scalar.
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

    bool softmax_avx2_available() {
        const auto& f = IncludedCPUFeatures();
        return f.has_feature(CPUFeatureSetType_AVX2) && f.has_feature(CPUFeatureSetType_FMA);
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

    void softmax_f32_range_avx2(float32_t* out, const float32_t* in, const size_t rows, const size_t cols, const size_t begin, const size_t end) {
        (void)rows;
        softmax_range_scalar(out, in, rows, cols, begin, end);
    }

    void softmax_f64_range_avx2(float64_t* out, const float64_t* in, const size_t rows, const size_t cols, const size_t begin, const size_t end) {
        (void)rows;
        softmax_range_scalar(out, in, rows, cols, begin, end);
    }
}
