/**
 * @file AutogradCPU.cpp
 * @author GreenBean
 * @brief CPU backend for the autograd backward ops: gemm_g, bias_add_g,
 * relu_g, softmax_ce_g, plus the shared gradient plumbing (grad_acc,
 * transpose_last2) and the Tensor graph API (backward/zero_grad/wire_node).
 * Hosts the Tensor method definitions (mirrors how Tensor::matmul lives in
 * MatmulCPU.cpp).
 * @version 0.1
 * @date 2026-10-18
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "Math/Autograd/Autograd.h"
#include "Math/Autograd/AutogradKernels.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        /**
         * Dispatches at compile time: fp16 and bf16 each use their own unaccel conversion.
         * @brief Widen one fp16/bf16 lane to fp32 (unaccel pathway, untracked).
         * @param v Half-width lane value.
         * @return The lane widened to fp32.
         */
        template<typename H>
        [[nodiscard]] float32_t widen_lane(const H& v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp16_to_fp32_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_bf16_to_fp32(v);
            }
        }

        /**
         * Dispatches at compile time: fp16 and bf16 each use their own unaccel conversion.
         * @brief Narrow one fp32 accumulator to fp16/bf16 (unaccel pathway, untracked/PERMISSIVE).
         * @param v FP32 accumulator value.
         * @return The value narrowed to H, RNE.
         */
        template<typename H>
        [[nodiscard]] H narrow_lane(const float32_t v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp32_to_fp16_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_fp32_to_bf16_untracked(v);
            }
        }

        /**
         * Only float16x2_t and bfloat16x2_t are specialized; any other P fails to compile.
         * @brief Lane type of a packed dtype (float16x2_t -> float16_t, bfloat16x2_t -> bfloat16_t).
         */
        template<typename P> struct packed_lane;
        /**
         * Explicit specialization for the packed fp16 pair: each lane is a float16_t.
         * @brief packed_lane specialization mapping float16x2_t to its lane type float16_t.
         */
        template<> struct packed_lane<float16x2_t> { using type = float16_t; };
        /**
         * Explicit specialization for the packed bf16 pair: each lane is a bfloat16_t.
         * @brief packed_lane specialization mapping bfloat16x2_t to its lane type bfloat16_t.
         */
        template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
        /**
         * Resolves to packed_lane<P>::type, so it only compiles for float16x2_t and bfloat16x2_t.
         * @brief Alias for the lane type of packed dtype P.
         */
        template<typename P> using packed_lane_t = typename packed_lane<P>::type;

        /**
         * Reinterprets the low 16 bits through memcpy, never a pointer cast.
         * @brief Low lane of a packed value, via its bit pattern.
         * @param bits Packed 32-bit pattern, (lo, hi).
         * @return The low 16 bits as lane type H.
         */
        template<typename H>
        [[nodiscard]] H lane_lo(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu));
        }

        /**
         * Reinterprets the high 16 bits through memcpy, never a pointer cast.
         * @brief High lane of a packed value, via its bit pattern.
         * @param bits Packed 32-bit pattern, (lo, hi).
         * @return The high 16 bits as lane type H.
         */
        template<typename H>
        [[nodiscard]] H lane_hi(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16));
        }

        /**
         * @brief Pack two narrowed lanes back into a packed dtype value.
         * @param lo Bit pattern of the low lane.
         * @param hi Bit pattern of the high lane.
         * @return Packed value P with lo in the low half, hi in the high half.
         */
        template<typename P>
        [[nodiscard]] P pack_lanes(const uint16_t lo, const uint16_t hi) {
            return Intrinsics::detail::bit_cast_memcpy<P>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
        }

        /**
         * Packed dtypes (float16x2_t/bfloat16x2_t) yield 32 zero bits, half dtypes 16.
         * @brief Bit-zero value of a half/packed dtype.
         * @return A value of type H whose bit pattern is all zero (+0.0 for the float dtypes).
         */
        template<typename H>
        [[nodiscard]] H zero_bits() {
            if constexpr (std::is_same_v<H, float16x2_t> || std::is_same_v<H, bfloat16x2_t>) {
                return Intrinsics::detail::bit_cast_memcpy<H>(uint32_t{0});
            } else {
                return Intrinsics::detail::bit_cast_memcpy<H>(uint16_t{0});
            }
        }

        // --------------------------- int4x2 helpers ---------------------------

        /**
         * Low 4 bits, sign-extended from bit 3.
         * @brief Sign-extended low nibble of a packed int4 pair.
         * @param v Packed int4 pair.
         * @return Low nibble in [-8, 7].
         */
        [[nodiscard]] inline int32_t int4_lo(const int4x2_t v) {
            const auto n = static_cast<int32_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
            return n >= 8 ? n - 16 : n;
        }

        /**
         * High 4 bits, sign-extended from bit 3.
         * @brief Sign-extended high nibble of a packed int4 pair.
         * @param v Packed int4 pair.
         * @return High nibble in [-8, 7].
         */
        [[nodiscard]] inline int32_t int4_hi(const int4x2_t v) {
            const auto n = static_cast<int32_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) >> 4);
            return n >= 8 ? n - 16 : n;
        }

        /**
         * Only the low 4 bits of each input survive, so out-of-range values wrap.
         * @brief Pack two signed values into an int4 pair (modular, low 4 bits each).
         * @param lo Value for the low nibble.
         * @param hi Value for the high nibble.
         * @return Packed int4 pair.
         */
        [[nodiscard]] inline int4x2_t pack_int4(const int32_t lo, const int32_t hi) {
            return Intrinsics::detail::bit_cast_memcpy<int4x2_t>(
                static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
        }

        // --------------------------- grad_acc kernels ---------------------------

        /**
         * Single-rounding add - bit-identical to the AVX kernels.
         * @brief Scalar range grad accumulation (fp32/fp64).
         * @param w Accumulated gradient buffer, updated in place (w += d).
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void grad_acc_range_scalar(T* w, const T* d, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = w[i] + d[i];
            }
        }

        /**
         * Half-dtype accumulation: widen to fp32, add, narrow RNE (unaccel
         * pathway) - same conversion-pathway semantics as the forward kernels.
         * @brief Scalar range grad accumulation (fp16/bf16, fp32 compute).
         * @param w Accumulated gradient buffer, updated in place.
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void grad_acc_range_half(H* w, const H* d, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = narrow_lane<H>(widen_lane(w[i]) + widen_lane(d[i]));
            }
        }

        /**
         * Packed-dtype accumulation: per-lane widen/add/narrow, repack on store.
         * @brief Scalar range grad accumulation (fp16x2/bf16x2, per-lane fp32 compute).
         * @param w Accumulated gradient buffer, updated in place.
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void grad_acc_range_packed(P* w, const P* d, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto w_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(w[i]);
                const auto d_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(d[i]);
                const auto lo = narrow_lane<H>(widen_lane(lane_lo<H>(w_bits)) + widen_lane(lane_lo<H>(d_bits)));
                const auto hi = narrow_lane<H>(widen_lane(lane_hi<H>(w_bits)) + widen_lane(lane_hi<H>(d_bits)));
                w[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                     Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Integer accumulation: modular (wrap) add through the unsigned twin -
         * the same wrap contract as the integer sum/bias_add kernels.
         * @brief Scalar range grad accumulation (all integer dtypes, wrap add).
         * @param w Accumulated gradient buffer, updated in place.
         * @param d Incoming gradient delta.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void grad_acc_range_int(T* w, const T* d, const size_t begin, const size_t end) {
            using U = std::make_unsigned_t<T>;
            for (auto i = begin; i < end; ++i) {
                w[i] = static_cast<T>(static_cast<U>(static_cast<U>(w[i]) + static_cast<U>(d[i])));
            }
        }

        /**
         * Int4x2 accumulation: per-nibble wrap add, repack on store.
         * @brief Scalar range grad accumulation (int4x2, per-nibble wrap).
         * @param w Accumulated gradient buffer (one element = two nibbles), updated in place.
         * @param d Incoming gradient delta.
         * @param begin First storage element processed.
         * @param end One past the last storage element processed.
         */
        void grad_acc_range_int4x2(int4x2_t* w, const int4x2_t* d, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = pack_int4(int4_lo(w[i]) + int4_lo(d[i]),
                                 int4_hi(w[i]) + int4_hi(d[i]));
            }
        }

        // --------------------------- relu_g kernels ---------------------------

        /**
         * dx[i] = x[i] > 0 ? dy[i] : 0 - a select, not a multiply: NaN in x
         * fails the comparison and zeroes the lane; NaN in a selected dy lane
         * propagates. Bit-identical to the AVX mask/blend kernels.
         * @brief Scalar range relu backward (fp32/fp64).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void relu_g_range_scalar(T* dx, const T* x, const T* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                dx[i] = x[i] > T{0} ? dy[i] : T{0};
            }
        }

        /**
         * Half-dtype relu backward: the comparison widens x to fp32 (NaN fails
         * it); the selected lane is stored as-is - no arithmetic, no rounding.
         * @brief Scalar range relu backward (fp16/bf16).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void relu_g_range_half(H* dx, const H* x, const H* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                dx[i] = widen_lane(x[i]) > 0.0f ? dy[i] : zero_bits<H>();
            }
        }

        /**
         * Packed-dtype relu backward: per-lane widened compare, per-lane select.
         * @brief Scalar range relu backward (fp16x2/bf16x2).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void relu_g_range_packed(P* dx, const P* x, const P* dy, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto x_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(x[i]);
                const auto d_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[i]);
                const auto lo = widen_lane(lane_lo<H>(x_bits)) > 0.0f
                    ? static_cast<uint16_t>(d_bits & 0xFFFFu) : uint16_t{0};
                const auto hi = widen_lane(lane_hi<H>(x_bits)) > 0.0f
                    ? static_cast<uint16_t>(d_bits >> 16) : uint16_t{0};
                dx[i] = pack_lanes<P>(lo, hi);
            }
        }

        /**
         * Integer relu backward: signed dtypes select on x > 0; unsigned dtypes
         * copy dy through (matches forward relu's unsigned passthrough).
         * @brief Scalar range relu backward (all integer dtypes).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void relu_g_range_int(T* dx, const T* x, const T* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                if constexpr (std::is_signed_v<T>) {
                    dx[i] = x[i] > T{0} ? dy[i] : T{0};
                } else {
                    dx[i] = dy[i];
                }
            }
        }

        /**
         * Int4x2 relu backward: per-nibble select on x > 0, repack on store.
         * @brief Scalar range relu backward (int4x2, per-nibble).
         * @param dx Output gradient buffer (written, not accumulated).
         * @param x Forward input of the relu.
         * @param dy Upstream gradient.
         * @param begin First storage element processed.
         * @param end One past the last storage element processed.
         */
        void relu_g_range_int4x2(int4x2_t* dx, const int4x2_t* x, const int4x2_t* dy, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto lo = int4_lo(x[i]) > 0 ? int4_lo(dy[i]) : 0;
                const auto hi = int4_hi(x[i]) > 0 ? int4_hi(dy[i]) : 0;
                dx[i] = pack_int4(lo, hi);
            }
        }

        // --------------------------- bias_add_g kernels ---------------------------

        /**
         * Per-column sums of dy over a row range, in the storage width.
         * @brief Scalar column-sum partial (fp32/fp64).
         * @param partial Per-slot column accumulator, cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols.
         * @param cols Last-dimension size.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename T>
        void bias_add_g_colsum_scalar(T* partial, const T* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    partial[c] = partial[c] + dy[r * cols + c];
                }
            }
        }

        /**
         * Half-dtype column sums: widen each element to fp32, accumulate in
         * fp32; narrowed RNE once at store time.
         * @brief Scalar column-sum partial (fp16/bf16, fp32 accumulate).
         * @param partial Per-slot fp32 column accumulator, cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols.
         * @param cols Last-dimension size.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename H>
        void bias_add_g_colsum_half(float32_t* partial, const H* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    partial[c] = partial[c] + widen_lane(dy[r * cols + c]);
                }
            }
        }

        /**
         * Packed-dtype column sums: each element's two lanes are consecutive
         * columns of a 2*cols-wide fp32 partial.
         * @brief Scalar column-sum partial (fp16x2/bf16x2, per-lane fp32 accumulate).
         * @param partial Per-slot fp32 column accumulator, 2*cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols packed elements.
         * @param cols Last-dimension size in packed elements.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename P>
        void bias_add_g_colsum_packed(float32_t* partial, const P* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            using H = packed_lane_t<P>;
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(dy[r * cols + c]);
                    partial[2 * c] = partial[2 * c] + widen_lane(lane_lo<H>(bits));
                    partial[2 * c + 1] = partial[2 * c + 1] + widen_lane(lane_hi<H>(bits));
                }
            }
        }

        /**
         * Integer column sums: modular (wrap) accumulation in the unsigned
         * twin, bit-identical on every backend and chunking.
         * @brief Scalar column-sum partial (all integer dtypes, wrap).
         * @param partial Per-slot column accumulator, cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols.
         * @param cols Last-dimension size.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        template<typename T>
        void bias_add_g_colsum_int(T* partial, const T* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            using U = std::make_unsigned_t<T>;
            auto* acc = reinterpret_cast<U*>(partial);
            const auto* in = reinterpret_cast<const U*>(dy);
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    acc[c] = static_cast<U>(acc[c] + in[r * cols + c]);
                }
            }
        }

        /**
         * Int4x2 column sums: per-nibble int32 partials (2*cols wide); the
         * modular wrap lands at store time via pack_int4's low-4-bit mask,
         * equivalent to per-add wrap.
         * @brief Scalar column-sum partial (int4x2, per-nibble).
         * @param partial Per-slot int32 nibble accumulator, 2*cols elements; added to, not zeroed.
         * @param dy Upstream gradient, row-major rows x cols packed elements.
         * @param cols Last-dimension size in packed elements.
         * @param rbegin First row summed.
         * @param rend One past the last row summed.
         */
        void bias_add_g_colsum_int4x2(int32_t* partial, const int4x2_t* dy, const size_t cols, const size_t rbegin, const size_t rend) {
            for (auto r = rbegin; r < rend; ++r) {
                for (auto c = size_t{0}; c < cols; ++c) {
                    partial[2 * c] += int4_lo(dy[r * cols + c]);
                    partial[2 * c + 1] += int4_hi(dy[r * cols + c]);
                }
            }
        }

        /**
         * Merge the per-slot partials and narrow into dbias: floats in the
         * accumulator width (halfs RNE via the unaccel pathway), packed dtypes
         * per lane, integers wrap, int4x2 wraps per nibble. Float merge order
         * is chunking-defined - not guaranteed bit-identical across tiers.
         * @brief Merge column-sum partials into the bias gradient.
         * @param dbias Output bias gradient (overwritten); packed dtypes receive cols_eff / 2 elements.
         * @param partials Per-slot partials, slots x cols_eff row-major.
         * @param slots Number of partial slots to merge.
         * @param cols_eff Effective column count (2 * cols for packed dtypes).
         */
        template<typename T, typename Acc>
        void bias_add_g_store(T* dbias, const std::vector<Acc>& partials, const size_t slots, const size_t cols_eff) {
            for (auto c = size_t{0}; c < cols_eff; ++c) {
                if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
                    if ((c & 1) != 0) {
                        continue;
                    }
                    using H = packed_lane_t<T>;
                    auto s_lo = Acc{0};
                    auto s_hi = Acc{0};
                    for (auto slot = size_t{0}; slot < slots; ++slot) {
                        s_lo = s_lo + partials[slot * cols_eff + c];
                        s_hi = s_hi + partials[slot * cols_eff + c + 1];
                    }
                    const auto lo = narrow_lane<H>(s_lo);
                    const auto hi = narrow_lane<H>(s_hi);
                    dbias[c >> 1] = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                                  Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
                } else if constexpr (std::is_integral_v<T>) {
                    // Wrap-merge through the unsigned twin - signed overflow is UB.
                    using U = std::make_unsigned_t<T>;
                    auto s = U{0};
                    for (auto slot = size_t{0}; slot < slots; ++slot) {
                        s = static_cast<U>(s + static_cast<U>(partials[slot * cols_eff + c]));
                    }
                    dbias[c] = static_cast<T>(s);
                } else {
                    auto s = Acc{0};
                    for (auto slot = size_t{0}; slot < slots; ++slot) {
                        s = s + partials[slot * cols_eff + c];
                    }
                    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                        dbias[c] = s;
                    } else {
                        dbias[c] = narrow_lane<T>(s);
                    }
                }
            }
        }

        /**
         * Int4x2 store: merge int32 nibble partials and wrap-pack each pair.
         * @brief Merge int4x2 column-sum partials into the bias gradient.
         * @param dbias Output bias gradient (overwritten), cols packed elements.
         * @param partials Per-slot int32 nibble partials, slots x (2 * cols) row-major.
         * @param slots Number of partial slots to merge.
         * @param cols Last-dimension size in packed elements.
         */
        void bias_add_g_store_int4x2(int4x2_t* dbias, const std::vector<int32_t>& partials, const size_t slots, const size_t cols) {
            for (auto c = size_t{0}; c < cols; ++c) {
                auto lo = int32_t{0};
                auto hi = int32_t{0};
                for (auto slot = size_t{0}; slot < slots; ++slot) {
                    lo += partials[slot * 2 * cols + 2 * c];
                    hi += partials[slot * 2 * cols + 2 * c + 1];
                }
                dbias[c] = pack_int4(lo, hi);
            }
        }

        // --------------------------- softmax_ce_g kernels ---------------------------

        /**
         * @brief Widen class j of a row to fp32 (fp32/fp16/bf16/packed x2 rows;
         * packed dtypes unpack each element's lanes as consecutive classes).
         * @param row Logits row.
         * @param j Class index (lane-granular for packed dtypes).
         * @return Class j widened to fp32.
         */
        template<typename T>
        [[nodiscard]] float32_t widen_class_f32(const T* row, const size_t j) {
            if constexpr (std::is_same_v<T, float32_t>) {
                return row[j];
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return widen_lane(row[j]);
            } else {
                using H = packed_lane_t<T>;
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(row[j >> 1]);
                return widen_lane((j & 1) != 0 ? lane_hi<H>(bits) : lane_lo<H>(bits));
            }
        }

        /**
         * @brief Widen class j of a row to fp64 (fp64/integer/int4x2 rows;
         * int4x2 unpacks each element's nibbles as consecutive classes).
         * @param row Logits row.
         * @param j Class index (nibble-granular for int4x2).
         * @return Class j widened to fp64.
         */
        template<typename T>
        [[nodiscard]] float64_t widen_class_f64(const T* row, const size_t j) {
            if constexpr (std::is_same_v<T, float64_t>) {
                return row[j];
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>((j & 1) != 0 ? int4_hi(row[j >> 1]) : int4_lo(row[j >> 1]));
            } else {
                return static_cast<float64_t>(row[j]);
            }
        }

        /**
         * @brief Widen the scalar dloss gradient element to the accumulator
         * width. Packed grad dtypes read the low lane (the forward store_mean
         * duplicates the mean into both lanes); int4x2 reads the low nibble.
         * @param g Pointer to the single dloss gradient element, of dtype T.
         * @return The dloss value in the accumulator type Acc.
         */
        template<typename T, typename Acc>
        [[nodiscard]] Acc widen_dloss(const void* g) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return *static_cast<const T*>(g);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return widen_lane(*static_cast<const T*>(g));
            } else if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t>) {
                using H = packed_lane_t<T>;
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(*static_cast<const T*>(g));
                return widen_lane(lane_lo<H>(bits));
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                return static_cast<float64_t>(int4_lo(*static_cast<const int4x2_t*>(g)));
            } else {
                return static_cast<float64_t>(*static_cast<const T*>(g));
            }
        }

        /**
         * Narrow one accumulated gradient value into the logits dtype: floats
         * RNE (halfs via the unaccel pathway), integers clamp to the dtype
         * range and truncate - a NaN gradient stores 0 (no integer NaN),
         * matching store_mean's contract.
         * @brief Store one softmax_ce_g gradient element.
         * @param v_raw Accumulated gradient value.
         * @return The value narrowed to T.
         */
        template<typename T, typename Acc>
        [[nodiscard]] T narrow_grad(const Acc v_raw) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return v_raw;
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return narrow_lane<T>(v_raw);
            } else {
                auto v = static_cast<float64_t>(v_raw);
                if (std::isnan(v)) {
                    v = 0.0;
                }
                const auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
                const auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
                if (v <= lo) {
                    v = lo;
                } else if (v >= hi) {
                    v = hi;
                }
                return static_cast<T>(v);
            }
        }

        /**
         * Row softmax gradient: dx_j = (exp(x_j - m) / s - (j == target)) *
         * dloss_scaled, with m/s recomputed exactly like the forward pass (max
         * ignores NaN; a NaN logit poisons s and thus the whole row).
         * @brief Scalar row softmax_ce backward (non-packed dtypes).
         * @param row Logits row.
         * @param out_row Output gradient row (written).
         * @param cols Number of classes in the row.
         * @param target Target class index, already bounds-checked.
         * @param dloss_scaled Upstream loss gradient divided by the row count.
         */
        template<typename T, typename Acc>
        void softmax_ce_g_row(const T* row, T* out_row, const size_t cols, const size_t target, const Acc dloss_scaled) {
            /**
             * Captures by reference: row (or the logits row) and the accumulator type Acc come from the enclosing function.
             * Selects at compile time between widen_class_f32 (Acc is float32_t) and widen_class_f64 (otherwise), reading class j of the row.
             * @brief Widen logit j of the row to the accumulator type.
             * @param j Class index within the row, in [0, cols).
             * @return Logit j widened to Acc.
             */
            auto widen = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(row, j);
                } else {
                    return widen_class_f64(row, j);
                }
            };
            auto m = std::numeric_limits<Acc>::lowest();
            for (auto j = size_t{0}; j < cols; ++j) {
                if (
                    const auto x = widen(j);
                    x > m
                ) {
                    m = x;
                }
            }
            auto s = Acc{0};
            for (auto j = size_t{0}; j < cols; ++j) {
                s += std::exp(widen(j) - m);
            }
            for (auto j = size_t{0}; j < cols; ++j) {
                const auto p = std::exp(widen(j) - m) / s;
                const auto g = (p - (j == target ? Acc{1} : Acc{0})) * dloss_scaled;
                out_row[j] = narrow_grad<T, Acc>(g);
            }
        }

        /**
         * Packed/int4x2 row softmax gradient: same math, but each output
         * element packs two consecutive class gradients. int4x2 results are
         * narrowed through int8 (NaN -> 0, clamp, truncate) then clamped to [-8, 7].
         * @brief Scalar row softmax_ce backward (packed x2 / int4x2 dtypes).
         * @param out_row Output gradient row (written), cols_eff / 2 packed elements.
         * @param probs Precomputed softmax probabilities, cols_eff entries.
         * @param cols_eff Effective class count (two per packed element).
         * @param target Target class index, already bounds-checked.
         * @param dloss_scaled Upstream loss gradient divided by the row count.
         */
        template<typename T, typename Acc>
        void softmax_ce_g_row_packed(T* out_row, const Acc* probs, const size_t cols_eff, const size_t target, const Acc dloss_scaled) {
            for (auto e = size_t{0}; e < cols_eff / 2; ++e) {
                const auto j = 2 * e;
                const auto glo = (probs[j] - (j == target ? Acc{1} : Acc{0})) * dloss_scaled;
                const auto ghi = (probs[j + 1] - (j + 1 == target ? Acc{1} : Acc{0})) * dloss_scaled;
                if constexpr (std::is_same_v<T, int4x2_t>) {
                    // Clamp to the signed nibble range [-8, 7] (store_mean's
                    // int4x2 contract), after the int8 NaN->0 / truncate.
                    out_row[e] = pack_int4(std::clamp(static_cast<int32_t>(narrow_grad<int8_t, Acc>(glo)), -8, 7),
                                           std::clamp(static_cast<int32_t>(narrow_grad<int8_t, Acc>(ghi)), -8, 7));
                } else {
                    using H = packed_lane_t<T>;
                    const auto lo = narrow_lane<H>(static_cast<float32_t>(glo));
                    const auto hi = narrow_lane<H>(static_cast<float32_t>(ghi));
                    out_row[e] = pack_lanes<T>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                               Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
                }
            }
        }

        /**
         * Packed variant driver: compute widened probs per class, then pack.
         * @brief Scalar row softmax_ce backward (packed x2 / int4x2 rows).
         * @param row Logits row of packed elements.
         * @param out_row Output gradient row (written).
         * @param cols_eff Effective class count (two per packed element).
         * @param target Target class index, already bounds-checked.
         * @param dloss_scaled Upstream loss gradient divided by the row count.
         */
        template<typename T, typename Acc>
        void softmax_ce_g_row_wide(const T* row, T* out_row, const size_t cols_eff, const size_t target, const Acc dloss_scaled) {
            /**
             * Captures by reference: row and the accumulator type Acc come from the enclosing function.
             * Selects at compile time between widen_class_f32 (Acc is float32_t) and widen_class_f64 (otherwise), reading effective class j (two per packed element).
             * @brief Widen effective class j of the packed row to the accumulator type.
             * @param j Effective class index, in [0, cols_eff).
             * @return Class j widened to Acc.
             */
            auto widen = [&](const size_t j) -> Acc {
                if constexpr (std::is_same_v<Acc, float32_t>) {
                    return widen_class_f32(row, j);
                } else {
                    return widen_class_f64(row, j);
                }
            };
            auto m = std::numeric_limits<Acc>::lowest();
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                if (
                    const auto x = widen(j);
                    x > m
                ) {
                    m = x;
                }
            }
            auto s = Acc{0};
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                s += std::exp(widen(j) - m);
            }
            std::vector<Acc> probs(cols_eff);
            for (auto j = size_t{0}; j < cols_eff; ++j) {
                probs[j] = std::exp(widen(j) - m) / s;
            }
            softmax_ce_g_row_packed<T, Acc>(out_row, probs.data(), cols_eff, target, dloss_scaled);
        }

        /**
         * @brief Read target class index r, bounds-checked against the effective
         * class count. Int4x2 targets use the low nibble (matches the
         * packed->unpacked first-lane conversion contract).
         * @param targets Target buffer, one class index per row.
         * @param r Row whose target is read.
         * @param cols Effective class count; valid targets are in [0, cols).
         * @return The target class index.
         * @throws ErrorHandling::IndexOutOfRange if the target is negative or >= cols.
         */
        template<typename I>
        [[nodiscard]] uint64_t target_index(const I* targets, const size_t r, const size_t cols) {
            int64_t t;
            if constexpr (std::is_same_v<I, int4x2_t>) {
                t = int4_lo(targets[r]);
            } else {
                t = static_cast<int64_t>(targets[r]);
            }
            if (t < 0 || static_cast<uint64_t>(t) >= cols) {
                throw ErrorHandling::IndexOutOfRange();
            }
            return static_cast<uint64_t>(t);
        }

        // --------------------------- launchers ---------------------------

        /**
         * Chunk an elementwise range worker across the CPU thread pool; futures
         * land on every tensor the workers touch so teardown drains them (they
         * hold raw buffer pointers). Small inputs run inline.
         * @brief Threaded elementwise launcher.
         * @param worker Callable (begin, end) over an element range; copied into each task.
         * @param numel Total element count.
         * @param a First tensor the workers touch (receives the futures).
         * @param b Second tensor the workers touch (receives the futures).
         * @param out Output tensor (receives the futures).
         */
        template<typename Worker>
        void launch_elementwise(Worker&& worker, const size_t numel, const Tensor& a, const Tensor& b, const Tensor& out) {
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
                // worker is captured BY VALUE: these futures outlive this call
                // (they land on the tensors), so a reference would dangle.
                futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
            }
            a.give_futures(futures);
            b.give_futures(futures);
            out.give_futures(futures);
        }

        /**
         * Chunk a row-indexed worker (slot, begin, end) across the pool,
         * returning the futures for the caller to drain on its own thread
         * (this runs in a node fn under backward(), never inside a pool
         * worker, so waiting there is legal). Small inputs run inline and return no futures.
         * @brief Threaded row launcher with per-slot partials. Returns the futures for the caller to drain.
         * @param rows Total row count.
         * @param range_fn Callable (slot, begin, end) over a row range; copied into each task.
         * @param t First tensor the workers touch (receives the futures).
         * @param extra Second tensor the workers touch (receives the futures).
         * @return Futures of the submitted chunks; empty if the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows_partial(const size_t rows, RangeFn&& range_fn, const Tensor& t, const Tensor& extra) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, 0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            auto slot = size_t{0};
            for (auto ti = size_t{0}; ti < t_count; ++ti) {
                const auto begin = ti * chunk;
                if (begin >= rows) {
                    break;
                }
                // range_fn is captured by value; its own captures reference the
                // caller's partials, which stay alive until drain() returns.
                futures.push_back(threadMgr.submit([range_fn, slot, begin, end = std::min(begin + chunk, rows)] { range_fn(slot, begin, end); }, Threading::TaskPriority::Normal).share());
                ++slot;
            }
            t.give_futures(futures);
            extra.give_futures(futures);
            return futures;
        }

        /**
         * Chunk a row-indexed worker (begin, end) across the pool; futures land
         * on the touched tensors and are returned for the caller to drain.
         * Small inputs run inline and return no futures.
         * @brief Threaded row launcher without partials. Returns the futures for the caller to drain.
         * @param rows Total row count.
         * @param range_fn Callable (begin, end) over a row range; copied into each task.
         * @param t First tensor the workers touch (receives the futures).
         * @param extra Second tensor the workers touch (receives the futures).
         * @param out Output tensor (receives the futures).
         * @return Futures of the submitted chunks; empty if the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows(const size_t rows, RangeFn&& range_fn, const Tensor& t, const Tensor& extra, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            for (auto ti = size_t{0}; ti < t_count; ++ti) {
                const auto begin = ti * chunk;
                if (begin >= rows) {
                    break;
                }
                futures.push_back(threadMgr.submit([range_fn, begin, end = std::min(begin + chunk, rows)] { range_fn(begin, end); }, Threading::TaskPriority::Normal).share());
            }
            t.give_futures(futures);
            extra.give_futures(futures);
            out.give_futures(futures);
            return futures;
        }

        /**
         * Rethrows the first worker error encountered.
         * @brief Drain futures on the caller thread.
         * @param futures Futures to wait on.
         */
        void drain(const std::vector<std::shared_future<void>>& futures) {
            for (const auto& f : futures) {
                f.get();
            }
        }

        /**
         * Fixed-arity function pointer type of a grad_acc range kernel (scalar or AVX).
         * @brief Range kernel signature for grad accumulation: (w, d, begin, end).
         */
        template<typename T>
        using GradAccScalarFn = void(*)(T*, const T*, size_t, size_t);
        /**
         * Fixed-arity function pointer type of a relu_g range kernel (scalar or AVX).
         * @brief Range kernel signature for relu backward: (dx, x, dy, begin, end).
         */
        template<typename T>
        using ReluGScalarFn = void(*)(T*, const T*, const T*, size_t, size_t);
        /**
         * Fixed-arity function pointer type of a bias_add_g column-sum kernel (scalar or AVX).
         * @brief Column-sum kernel signature: (partial, dy, cols, rbegin, rend).
         */
        template<typename T>
        using ColsumScalarFn = void(*)(T*, const T*, size_t, size_t, size_t);
    }

    /**
     * In-place accumulation g += d over g.numel elements, honoring both tensors' offsets.
     * d must share g's dtype and shape (not checked). Integer dtypes wrap, halfs
     * widen to fp32 and narrow RNE, fp32/fp64 add directly (AVX tiers bit-identical).
     * Work is queued on the CPU pool and returns immediately; futures land on g and d.
     * @brief CPU gradient accumulation, g += d.
     * @param g Accumulated gradient, updated in place.
     * @param d Gradient delta to add.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if either tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void grad_acc_cpu(Tensor& g, const Tensor& d) {
        if (g.device != Device::CPU || d.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(g.device));
        }
        using DType = DataTypes::DType;
        const auto numel = g.numel;
        switch (g.dtype) {
            case DType::Float32: {
                auto* w = static_cast<float32_t*>(g.data) + g.offset;
                const auto* d_ptr = static_cast<const float32_t*>(d.data) + d.offset;
                const GradAccScalarFn<float32_t> kernel = Math::Autograd::autograd_avx512_available()
                    ? Math::Autograd::grad_acc_f32_range_avx512
                    : (Math::Autograd::autograd_avx2_available() ? Math::Autograd::grad_acc_f32_range_avx2 : grad_acc_range_scalar<float32_t>);
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): w, d_ptr and the dispatched kernel.
                 * Calls kernel(w, d_ptr, b, e), where w and d_ptr already include their tensors' offsets.
                 * @brief Worker for grad_acc_cpu, float32_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Float64: {
                auto* w = static_cast<float64_t*>(g.data) + g.offset;
                const auto* d_ptr = static_cast<const float64_t*>(d.data) + d.offset;
                const GradAccScalarFn<float64_t> kernel = Math::Autograd::autograd_avx512_available()
                    ? Math::Autograd::grad_acc_f64_range_avx512
                    : (Math::Autograd::autograd_avx2_available() ? Math::Autograd::grad_acc_f64_range_avx2 : grad_acc_range_scalar<float64_t>);
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): w, d_ptr and the dispatched kernel.
                 * Calls kernel(w, d_ptr, b, e), where w and d_ptr already include their tensors' offsets.
                 * @brief Worker for grad_acc_cpu, float64_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Float16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Lanes widen to fp32, add, and narrow RNE. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the float16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<float16_t*>(g.data) + g.offset, d_ptr = static_cast<const float16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_half<float16_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Lanes widen to fp32, add, and narrow RNE. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the bfloat16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<bfloat16_t*>(g.data) + g.offset, d_ptr = static_cast<const bfloat16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_half<bfloat16_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Each element holds two lanes; both are processed per element. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the float16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<float16x2_t*>(g.data) + g.offset, d_ptr = static_cast<const float16x2_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_packed<float16x2_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Each element holds two lanes; both are processed per element. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the bfloat16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<bfloat16x2_t*>(g.data) + g.offset, d_ptr = static_cast<const bfloat16x2_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_packed<bfloat16x2_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Int8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<int8_t*>(g.data) + g.offset, d_ptr = static_cast<const int8_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int8_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::UInt8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<uint8_t*>(g.data) + g.offset, d_ptr = static_cast<const uint8_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint8_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Int16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<int16_t*>(g.data) + g.offset, d_ptr = static_cast<const int16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int16_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::UInt16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<uint16_t*>(g.data) + g.offset, d_ptr = static_cast<const uint16_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint16_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Int32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<int32_t*>(g.data) + g.offset, d_ptr = static_cast<const int32_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int32_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::UInt32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<uint32_t*>(g.data) + g.offset, d_ptr = static_cast<const uint32_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint32_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Int64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<int64_t*>(g.data) + g.offset, d_ptr = static_cast<const int64_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<int64_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::UInt64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Integer arithmetic with the dtype's wrap/clamp semantics. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the uint64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<uint64_t*>(g.data) + g.offset, d_ptr = static_cast<const uint64_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int<uint64_t>(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (w, d_ptr, offsets already applied); no default capture.
                 * Each element holds two nibbles; both are processed per element. Calls the matching grad_acc range kernel on [b, e).
                 * @brief Worker for grad_acc_cpu, the int4x2_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([w = static_cast<int4x2_t*>(g.data) + g.offset, d_ptr = static_cast<const int4x2_t*>(d.data) + d.offset](const size_t b, const size_t e) { grad_acc_range_int4x2(w, d_ptr, b, e); }, numel, g, d, g);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * dx = x > 0 ? dy : 0 over x.numel elements. NaN in x zeroes the lane; NaN in a
     * selected dy lane propagates. dx is written, not accumulated.
     * Work is queued on the CPU pool and returns immediately; futures land on x, dy and dx.
     * @brief CPU relu backward.
     * @param dy Upstream gradient.
     * @param x Forward input of the relu.
     * @param dx Output gradient (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void relu_g_cpu(const Tensor& dy, const Tensor& x, Tensor& dx) {
        if (dy.device != Device::CPU || x.device != Device::CPU || dx.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        using DType = DataTypes::DType;
        const auto numel = x.numel;
        switch (x.dtype) {
            case DType::Float32: {
                auto* dx_ptr = static_cast<float32_t*>(dx.data) + dx.offset;
                const auto* x_ptr = static_cast<const float32_t*>(x.data) + x.offset;
                const auto* dy_ptr = static_cast<const float32_t*>(dy.data) + dy.offset;
                const ReluGScalarFn<float32_t> kernel = Math::Autograd::autograd_avx512_available()
                    ? Math::Autograd::relu_g_f32_range_avx512
                    : (Math::Autograd::autograd_avx2_available() ? Math::Autograd::relu_g_f32_range_avx2 : relu_g_range_scalar<float32_t>);
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): dx_ptr, x_ptr, dy_ptr and the dispatched kernel.
                 * Calls kernel(dx_ptr, x_ptr, dy_ptr, b, e), where the pointers already include their tensors' offsets.
                 * @brief Worker for relu_g_cpu, float32_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Float64: {
                auto* dx_ptr = static_cast<float64_t*>(dx.data) + dx.offset;
                const auto* x_ptr = static_cast<const float64_t*>(x.data) + x.offset;
                const auto* dy_ptr = static_cast<const float64_t*>(dy.data) + dy.offset;
                const ReluGScalarFn<float64_t> kernel = Math::Autograd::autograd_avx512_available()
                    ? Math::Autograd::relu_g_f64_range_avx512
                    : (Math::Autograd::autograd_avx2_available() ? Math::Autograd::relu_g_f64_range_avx2 : relu_g_range_scalar<float64_t>);
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value (the futures outlive this call): dx_ptr, x_ptr, dy_ptr and the dispatched kernel.
                 * Calls kernel(dx_ptr, x_ptr, dy_ptr, b, e), where the pointers already include their tensors' offsets.
                 * @brief Worker for relu_g_cpu, float64_t path (scalar or AVX kernel).
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Float16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Lanes widen to fp32, select by x > 0, and narrow RNE.
                 * @brief Worker for relu_g_cpu, the float16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const float16_t*>(x.data) + x.offset, dy_ptr = static_cast<const float16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_half<float16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Lanes widen to fp32, select by x > 0, and narrow RNE.
                 * @brief Worker for relu_g_cpu, the bfloat16_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<bfloat16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const bfloat16_t*>(x.data) + x.offset, dy_ptr = static_cast<const bfloat16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_half<bfloat16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Each element holds two lanes; both are processed per element.
                 * @brief Worker for relu_g_cpu, the float16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<float16x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const float16x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const float16x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_packed<float16x2_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Each element holds two lanes; both are processed per element.
                 * @brief Worker for relu_g_cpu, the bfloat16x2_t packed path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<bfloat16x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const bfloat16x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const bfloat16x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_packed<bfloat16x2_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int8_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int8_t*>(x.data) + x.offset, dy_ptr = static_cast<const int8_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int8_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt8: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint8_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint8_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint8_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint8_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint8_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int16_t*>(x.data) + x.offset, dy_ptr = static_cast<const int16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt16: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint16_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint16_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint16_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint16_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint16_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int32_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int32_t*>(x.data) + x.offset, dy_ptr = static_cast<const int32_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int32_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt32: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint32_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint32_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint32_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint32_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint32_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the int64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int64_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int64_t*>(x.data) + x.offset, dy_ptr = static_cast<const int64_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<int64_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::UInt64: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Integer arithmetic with the dtype's wrap/clamp semantics.
                 * @brief Worker for relu_g_cpu, the uint64_t integer path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<uint64_t*>(dx.data) + dx.offset, x_ptr = static_cast<const uint64_t*>(x.data) + x.offset, dy_ptr = static_cast<const uint64_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int<uint64_t>(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Runs on a pool worker (or inline for small inputs) over the element range [b, e). Captures by value via init-captures (dx_ptr, x_ptr, dy_ptr, offsets already applied); no default capture.
                 * Writes dx = x > 0 ? dy : 0 on [b, e). Each element holds two nibbles; both are processed per element.
                 * @brief Worker for relu_g_cpu, the int4x2_t path.
                 * @param b First element index of the range (inclusive).
                 * @param e One past the last element index of the range.
                 */
                launch_elementwise([dx_ptr = static_cast<int4x2_t*>(dx.data) + dx.offset, x_ptr = static_cast<const int4x2_t*>(x.data) + x.offset, dy_ptr = static_cast<const int4x2_t*>(dy.data) + dy.offset](const size_t b, const size_t e) { relu_g_range_int4x2(dx_ptr, x_ptr, dy_ptr, b, e); }, numel, x, dy, dx);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * dbias = per-column sum of dy over all leading rows (last dimension = cols).
     * Per-thread-slot partials are merged at the end, so float results depend on the
     * chunking; halfs accumulate in fp32 and narrow RNE once, integers wrap. dbias is
     * overwritten. A zero-width last dimension returns without touching dbias. Blocks
     * until the workers finish (drains on the caller thread).
     * @brief CPU bias_add backward, column-sum of dy.
     * @param dy Upstream gradient, any rank >= 1.
     * @param dbias Output bias gradient, last-dimension sized (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if either tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported dtype.
     */
    void bias_add_g_cpu(const Tensor& dy, Tensor& dbias) {
        if (dy.device != Device::CPU || dbias.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(dy.device));
        }
        using DType = DataTypes::DType;
        const auto cols = dy.shape[dy.ndim - 1];
        if (cols == 0) {
            return;
        }
        const auto rows = dy.numel / cols;
        const auto slots = static_cast<size_t>(IncludedCPUFeatures().threads);
        switch (dy.dtype) {
            case DType::Float32: {
                const auto* dy_ptr = static_cast<const float32_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * cols, 0.0f);
                const ColsumScalarFn<float32_t> kernel = Math::Autograd::autograd_avx512_available()
                    ? Math::Autograd::bias_add_g_f32_colsum_avx512
                    : (Math::Autograd::autograd_avx2_available() ? Math::Autograd::bias_add_g_f32_colsum_avx2 : bias_add_g_colsum_scalar<float32_t>);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Uses the dispatched kernel.
                 * @brief Worker for bias_add_g_cpu, float32_t path (scalar or AVX kernel).
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { kernel(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float32_t, float32_t>(static_cast<float32_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::Float64: {
                const auto* dy_ptr = static_cast<const float64_t*>(dy.data) + dy.offset;
                std::vector<float64_t> partials(slots * cols, 0.0);
                const ColsumScalarFn<float64_t> kernel = Math::Autograd::autograd_avx512_available()
                    ? Math::Autograd::bias_add_g_f64_colsum_avx512
                    : (Math::Autograd::autograd_avx2_available() ? Math::Autograd::bias_add_g_f64_colsum_avx2 : bias_add_g_colsum_scalar<float64_t>);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Uses the dispatched kernel.
                 * @brief Worker for bias_add_g_cpu, float64_t path (scalar or AVX kernel).
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { kernel(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float64_t, float64_t>(static_cast<float64_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::Float16: {
                const auto* dy_ptr = static_cast<const float16_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Accumulates in fp32.
                 * @brief Worker for bias_add_g_cpu, float16_t path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_half<float16_t>(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float16_t, float32_t>(static_cast<float16_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::BFloat16: {
                const auto* dy_ptr = static_cast<const bfloat16_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Accumulates in fp32.
                 * @brief Worker for bias_add_g_cpu, bfloat16_t path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_half<bfloat16_t>(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<bfloat16_t, float32_t>(static_cast<bfloat16_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            case DType::Float16x2: {
                const auto* dy_ptr = static_cast<const float16x2_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * 2 * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * 2 * cols), so chunks never race; the partials are merged afterwards. Each element carries two columns, so each slot holds 2 * cols fp32 sums.
                 * @brief Worker for bias_add_g_cpu, float16x2_t packed path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_packed<float16x2_t>(partials.data() + slot * 2 * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<float16x2_t, float32_t>(static_cast<float16x2_t*>(dbias.data) + dbias.offset, partials, slots, 2 * cols);
                break;
            }
            case DType::BFloat16x2: {
                const auto* dy_ptr = static_cast<const bfloat16x2_t*>(dy.data) + dy.offset;
                std::vector<float32_t> partials(slots * 2 * cols, 0.0f);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * 2 * cols), so chunks never race; the partials are merged afterwards. Each element carries two columns, so each slot holds 2 * cols fp32 sums.
                 * @brief Worker for bias_add_g_cpu, bfloat16x2_t packed path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_packed<bfloat16x2_t>(partials.data() + slot * 2 * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store<bfloat16x2_t, float32_t>(static_cast<bfloat16x2_t*>(dbias.data) + dbias.offset, partials, slots, 2 * cols);
                break;
            }
            case DType::Int4x2: {
                const auto* dy_ptr = static_cast<const int4x2_t*>(dy.data) + dy.offset;
                std::vector<int32_t> partials(slots * 2 * cols, 0);
                /**
                 * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                 * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * 2 * cols), so chunks never race; the partials are merged afterwards. Each element carries two nibbles, so each slot holds 2 * cols int32 sums.
                 * @brief Worker for bias_add_g_cpu, int4x2_t path.
                 * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                 * @param b First row index of the range (inclusive).
                 * @param e One past the last row index of the range.
                 */
                drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_int4x2(partials.data() + slot * 2 * cols, dy_ptr, cols, b, e); }, dy, dbias));
                bias_add_g_store_int4x2(static_cast<int4x2_t*>(dbias.data) + dbias.offset, partials, slots, cols);
                break;
            }
            default: {
                /**
                 * Captures by reference: slots, cols, rows, dy and dbias from the enclosing function. Generic lambda; T is deduced from dy_ptr.
                 * Allocates zeroed per-slot partials, drains the row-partial launch on the caller thread, then merges the slots into dbias with wrapping integer sums.
                 * @brief Integer-dtype bias_add backward driver for element type T.
                 * @param T Integer element type of dy (int8_t ... uint64_t).
                 * @param dy_ptr Pointer to the first element of dy, offset already applied.
                 */
                auto run_int = [&]<typename T>(const T* dy_ptr) {
                    std::vector<T> partials(slots * cols, T{0});
                    /**
                     * Runs on a pool worker (or inline, slot 0, for small inputs) over the row range [b, e). Captures by reference (partials and dy_ptr outlive it: the caller drains before returning).
                     * Adds the column sums of those rows of dy into its own slot of partials (partials + slot * cols), so chunks never race; the partials are merged afterwards. Integer sums wrap in T.
                     * @brief Worker for bias_add_g_cpu, integer path of type T.
                     * @param slot Index of the per-thread partial buffer this chunk accumulates into.
                     * @param b First row index of the range (inclusive).
                     * @param e One past the last row index of the range.
                     */
                    drain(launch_rows_partial(rows, [&](const size_t slot, const size_t b, const size_t e) { bias_add_g_colsum_int<T>(partials.data() + slot * cols, dy_ptr, cols, b, e); }, dy, dbias));
                    bias_add_g_store<T, T>(static_cast<T*>(dbias.data) + dbias.offset, partials, slots, cols);
                };
                switch (dy.dtype) {
                    case DType::Int8: run_int(static_cast<const int8_t*>(dy.data) + dy.offset); break;
                    case DType::UInt8: run_int(static_cast<const uint8_t*>(dy.data) + dy.offset); break;
                    case DType::Int16: run_int(static_cast<const int16_t*>(dy.data) + dy.offset); break;
                    case DType::UInt16: run_int(static_cast<const uint16_t*>(dy.data) + dy.offset); break;
                    case DType::Int32: run_int(static_cast<const int32_t*>(dy.data) + dy.offset); break;
                    case DType::UInt32: run_int(static_cast<const uint32_t*>(dy.data) + dy.offset); break;
                    case DType::Int64: run_int(static_cast<const int64_t*>(dy.data) + dy.offset); break;
                    case DType::UInt64: run_int(static_cast<const uint64_t*>(dy.data) + dy.offset); break;
                    [[unlikely]] default: throw ErrorHandling::NotImplemented();
                }
                break;
            }
        }
    }

    /**
     * dlogits = (softmax(logits) - onehot(target)) * dloss / rows, row by row, with the
     * softmax recomputed as in the forward pass (a NaN logit poisons its whole row).
     * logits must be 2-D (rows x cols); packed dtypes carry two classes per element.
     * Accumulation is fp32 for fp32/fp16/bf16/packed halfs and fp64 for fp64, integers
     * and int4x2; integer outputs clamp and map NaN to 0. dloss_grad is a single element
     * (packed dtypes read the low lane, int4x2 the low nibble). Blocks until the workers finish.
     * @brief CPU softmax cross-entropy backward.
     * @param dloss_grad Scalar upstream gradient of the loss.
     * @param logits Forward logits, rows x cols.
     * @param targets Per-row target class indices (any integer dtype or int4x2).
     * @param dlogits Output gradient, same shape/dtype as logits (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::IndexOutOfRange if a target is outside [0, effective classes).
     * @throws ErrorHandling::UnReachableCode for a non-integer targets dtype.
     * @throws ErrorHandling::NotImplemented for an unsupported logits dtype.
     */
    void softmax_ce_g_cpu(const Tensor& dloss_grad, const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        if (dloss_grad.device != Device::CPU || logits.device != Device::CPU || targets.device != Device::CPU || dlogits.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(logits.device));
        }
        using DType = DataTypes::DType;
        const auto rows = logits.shape[0];
        const auto cols_store = logits.shape[1];
        const auto packed = logits.dtype == DType::Float16x2 || logits.dtype == DType::BFloat16x2 || logits.dtype == DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        const auto tgt_dtype = targets.dtype;
        const auto* const tgt_data = targets.data;
        const auto tgt_offset = targets.offset;
        /**
         * Captures by reference: rows, cols_eff, cols_store, tgt_dtype, tgt_data, tgt_offset, dy-side tensors from the enclosing function. Generic lambda with explicit template arguments, invoked through operator()<T, Acc> from the logits dtype switch.
         * Widens the scalar upstream gradient, divides it by the row count, then drains a launch_rows over the rows; a (b, e) range is a block of logits rows.
         * @brief Logits-dtype driver for softmax_ce_g_cpu.
         * @param T Storage element type of logits and dlogits.
         * @param Acc Accumulator type (float32_t for fp32/halfs, float64_t for fp64/integers/int4x2).
         * @param logits_ptr First logits element, offset applied.
         * @param dlogits_ptr First dlogits element, offset applied (written).
         * @param g_data Raw data pointer of the scalar dloss gradient tensor.
         */
        auto run_logits = [&]<typename T, typename Acc>(const T* logits_ptr, T* dlogits_ptr, const void* g_data) {
            const auto dloss = widen_dloss<T, Acc>(g_data);
            const auto dloss_scaled = dloss / static_cast<Acc>(rows);
            /**
             * Runs on a pool worker (or inline for small inputs) over the row range [b, e). Captures by reference; launch_rows copies it into each task and the caller drains before the captured locals die.
             * Dispatches on the targets dtype (run_targets) and writes the gradient for each row of the range.
             * @brief Row-range worker for softmax_ce_g_cpu.
             * @param b First row index of the range (inclusive).
             * @param e One past the last row index of the range.
             */
            drain(launch_rows(rows, [&](const size_t b, const size_t e) {
                /**
                 * Captures by reference: b, e, T, Acc, pointers and sizes of the enclosing worker. Generic lambda; I is deduced from t_ptr.
                 * For each row r in [b, e) it reads the bounds-checked target (target_index, throws IndexOutOfRange) and calls softmax_ce_g_row_wide for packed/int4x2 logits or softmax_ce_g_row otherwise.
                 * @brief Per-targets-dtype row loop for the softmax_ce backward.
                 * @param I Integer (or int4x2_t) element type of the targets.
                 * @param t_ptr First target element, offset applied.
                 */
                auto run_targets = [&]<typename I>(const I* t_ptr) {
                    for (auto r = b; r < e; ++r) {
                        const auto idx = target_index(t_ptr, r, cols_eff);
                        if constexpr (std::is_same_v<T, float16x2_t> || std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int4x2_t>) {
                            softmax_ce_g_row_wide<T, Acc>(logits_ptr + r * cols_store, dlogits_ptr + r * cols_store, cols_eff, idx, dloss_scaled);
                        } else {
                            softmax_ce_g_row<T, Acc>(logits_ptr + r * cols_store, dlogits_ptr + r * cols_store, cols_eff, idx, dloss_scaled);
                        }
                    }
                };
                switch (tgt_dtype) {
                    case DType::Int8: run_targets(static_cast<const int8_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt8: run_targets(static_cast<const uint8_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int16: run_targets(static_cast<const int16_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt16: run_targets(static_cast<const uint16_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int32: run_targets(static_cast<const int32_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt32: run_targets(static_cast<const uint32_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int64: run_targets(static_cast<const int64_t*>(tgt_data) + tgt_offset); break;
                    case DType::UInt64: run_targets(static_cast<const uint64_t*>(tgt_data) + tgt_offset); break;
                    case DType::Int4x2: run_targets(static_cast<const int4x2_t*>(tgt_data) + tgt_offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
            }, logits, targets, dlogits));
        };
        switch (logits.dtype) {
            case DType::Float32: run_logits.operator()<float32_t, float32_t>(static_cast<const float32_t*>(logits.data) + logits.offset, static_cast<float32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float64: run_logits.operator()<float64_t, float64_t>(static_cast<const float64_t*>(logits.data) + logits.offset, static_cast<float64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float16: run_logits.operator()<float16_t, float32_t>(static_cast<const float16_t*>(logits.data) + logits.offset, static_cast<float16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::BFloat16: run_logits.operator()<bfloat16_t, float32_t>(static_cast<const bfloat16_t*>(logits.data) + logits.offset, static_cast<bfloat16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float16x2: run_logits.operator()<float16x2_t, float32_t>(static_cast<const float16x2_t*>(logits.data) + logits.offset, static_cast<float16x2_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::BFloat16x2: run_logits.operator()<bfloat16x2_t, float32_t>(static_cast<const bfloat16x2_t*>(logits.data) + logits.offset, static_cast<bfloat16x2_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int8: run_logits.operator()<int8_t, float64_t>(static_cast<const int8_t*>(logits.data) + logits.offset, static_cast<int8_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt8: run_logits.operator()<uint8_t, float64_t>(static_cast<const uint8_t*>(logits.data) + logits.offset, static_cast<uint8_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int16: run_logits.operator()<int16_t, float64_t>(static_cast<const int16_t*>(logits.data) + logits.offset, static_cast<int16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt16: run_logits.operator()<uint16_t, float64_t>(static_cast<const uint16_t*>(logits.data) + logits.offset, static_cast<uint16_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int32: run_logits.operator()<int32_t, float64_t>(static_cast<const int32_t*>(logits.data) + logits.offset, static_cast<int32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt32: run_logits.operator()<uint32_t, float64_t>(static_cast<const uint32_t*>(logits.data) + logits.offset, static_cast<uint32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int64: run_logits.operator()<int64_t, float64_t>(static_cast<const int64_t*>(logits.data) + logits.offset, static_cast<int64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::UInt64: run_logits.operator()<uint64_t, float64_t>(static_cast<const uint64_t*>(logits.data) + logits.offset, static_cast<uint64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Int4x2: run_logits.operator()<int4x2_t, float64_t>(static_cast<const int4x2_t*>(logits.data) + logits.offset, static_cast<int4x2_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * Swaps the last two dimensions: out[..., j, i] = t[..., i, j]. Dtype-agnostic byte
     * copy of whole storage elements (a packed element moves as a unit, its lanes are
     * not swapped). out must already have the swapped shape and t's dtype. Work is
     * queued on the CPU pool and returns immediately; futures land on t and out.
     * @brief CPU transpose of the last two dimensions.
     * @param t Source tensor, rank >= 2.
     * @param out Destination tensor with the last two dimensions swapped (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if either tensor is not on the CPU.
     */
    void transpose_last2_cpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        // Dtype-agnostic: an elementwise byte copy. Transpose is memory-bound,
        // so it stays scalar; threading comes from the elementwise launcher.
        const auto eb = DataTypes::dtype_bytes(t.dtype);
        const auto rows = t.shape[t.ndim - 2];
        const auto cols = t.shape[t.ndim - 1];
        const auto slab = rows * cols;
        const auto numel = t.numel;
        const auto* in = static_cast<const uint8_t*>(t.data) + t.offset * eb;
        auto* outp = static_cast<uint8_t*>(out.data) + out.offset * eb;
        /**
         * Runs on a pool worker (or inline for small inputs) over the element range [b, e) of the source. Captures by value (the futures outlive this call): in, outp, slab, rows, cols, eb.
         * Each source element idx is decomposed into (batch, i, j) and its eb bytes are copied to the transposed position (batch * slab + j * rows + i).
         * @brief Worker for transpose_last2_cpu.
         * @param b First element index of the range (inclusive).
         * @param e One past the last element index of the range.
         */
        launch_elementwise([=](const size_t b, const size_t e) {
            for (auto idx = b; idx < e; ++idx) {
                const auto batch = idx / slab;
                const auto rem = idx - batch * slab;
                const auto i = rem / cols;
                const auto j = rem - i * cols;
                std::memcpy(outp + (batch * slab + j * rows + i) * eb, in + idx * eb, eb);
            }
        }, numel, t, t, out);
    }

    // --------------------------- node trampolines ---------------------------

    /**
     * Skips work when x does not require grad; otherwise computes dx on x's device
     * into a fresh tensor and hands it to x.accumulate_grad. Syncs output_grad and x first.
     * @brief Backward trampoline for relu.
     * @param output_grad Gradient of this node's output tensor.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 1 or the device is neither CPU nor GPU.
     */
    void relu_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs) {
        if (n_inputs != 1) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        if (!x.requires_grad) {
            return;
        }
        output_grad->sync();
        x.sync();
        auto dx = Tensor(std::vector<size_t>(x.shape, x.shape + x.ndim), x.dtype, x.device, false);
        if (x.device == Device::CPU) {
            relu_g_cpu(*output_grad, x, dx);
        } else if (x.device == Device::GPU) {
            relu_g_gpu(*output_grad, x, dx);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        x.accumulate_grad(std::move(dx));
    }

    /**
     * inputs are (x, bias). dx is a copy of output_grad (the forward add is bias-only per
     * element); dbias is the column-sum of output_grad. Each is computed only if its tensor
     * requires grad, then handed to accumulate_grad. Syncs the tensors it reads.
     * @brief Backward trampoline for bias_add.
     * @param output_grad Gradient of this node's output tensor.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void bias_add_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& x = *inputs[0];
        auto& bias = *inputs[1];
        output_grad->sync();
        if (x.requires_grad) {
            x.sync();
            // dx = dy, exactly (the forward add per element is bias-only).
            auto dx = output_grad->clone(true);
            x.accumulate_grad(std::move(dx));
        }
        if (bias.requires_grad) {
            bias.sync();
            auto dbias = Tensor(std::vector<size_t>(bias.shape, bias.shape + bias.ndim), bias.dtype, bias.device, false);
            if (bias.device == Device::CPU) {
                bias_add_g_cpu(*output_grad, dbias);
            } else if (bias.device == Device::GPU) {
                bias_add_g_gpu(*output_grad, dbias);
            } else [[unlikely]] {
                throw ErrorHandling::UnReachableCode();
            }
            bias.accumulate_grad(std::move(dbias));
        }
    }

    /**
     * inputs are (logits, targets). Only logits receives a gradient (targets are
     * class indices); skipped when logits does not require grad. Syncs the three tensors it reads.
     * @brief Backward trampoline for softmax_ce.
     * @param output_grad Gradient of this node's output tensor.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void softmax_ce_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& logits = *inputs[0];
        auto& targets = *inputs[1];
        if (!logits.requires_grad) {
            return;
        }
        output_grad->sync();
        logits.sync();
        targets.sync();
        auto dlogits = Tensor(std::vector<size_t>(logits.shape, logits.shape + logits.ndim), logits.dtype, logits.device, false);
        if (logits.device == Device::CPU) {
            softmax_ce_g_cpu(*output_grad, logits, targets, dlogits);
        } else if (logits.device == Device::GPU) {
            softmax_ce_g_gpu(*output_grad, logits, targets, dlogits);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        logits.accumulate_grad(std::move(dlogits));
    }

    /**
     * inputs are (a, b). dA = dOut * B^T and dB = A^T * dOut, each only if that input
     * requires grad. The transpose is materialized into a contiguous scratch tensor so the
     * existing matmul path (CPU kernels or GPU BLAS) is reused for every dtype.
     * @brief Backward trampoline for matmul (gemm).
     * @param output_grad Gradient of this node's output tensor.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void gemm_g_node(Tensor* output_grad, Tensor** inputs, const size_t n_inputs) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& a = *inputs[0];
        auto& b = *inputs[1];
        if (!a.requires_grad && !b.requires_grad) {
            return;
        }
        output_grad->sync();
        // dA = dOut·Bᵀ, dB = Aᵀ·dOut. The transpose is materialized into a
        // contiguous scratch tensor, so the existing validated matmul path
        // (CPU kernels or GPU BLAS) runs unchanged for every dtype.
        /**
         * Captures nothing. Allocates a new tensor with the last two dimensions of t swapped, same dtype and device, and fills it with transpose_last2_cpu or transpose_last2_gpu.
         * Returning the tensor moves it out, which drains the transpose workers.
         * Throws ErrorHandling::UnReachableCode if t is on neither CPU nor GPU.
         * @brief Materialize the transpose of the last two dimensions of t.
         * @param t Source tensor, rank >= 2, on CPU or GPU.
         * @return A new contiguous tensor holding the transposed data.
         */
        auto transpose = [](const Tensor& t) {
            auto t_shape = std::vector<size_t>(t.shape, t.shape + t.ndim);
            std::swap(t_shape[t.ndim - 1], t_shape[t.ndim - 2]);
            auto out = Tensor(t_shape, t.dtype, t.device, false);
            if (t.device == Device::CPU) {
                transpose_last2_cpu(t, out);
            } else if (t.device == Device::GPU) {
                transpose_last2_gpu(t, out);
            } else [[unlikely]] {
                throw ErrorHandling::UnReachableCode();
            }
            return out; // the move drains the transpose workers
        };
        if (a.requires_grad) {
            b.sync();
            auto bt = transpose(b);
            auto da = output_grad->matmul(bt);
            a.accumulate_grad(std::move(da));
        }
        if (b.requires_grad) {
            a.sync();
            auto at = transpose(a);
            auto db = at.matmul(*output_grad);
            b.accumulate_grad(std::move(db));
        }
    }
}

namespace BeanTensor::Tensors {

    /**
     * Allocates grad with this tensor's shape, dtype and device, zero-filled, and syncs
     * so the fill lands before any reader queues work. No-op if grad already exists.
     * Ownership: this tensor owns the heap-allocated grad.
     * @brief Lazily allocate a zeroed gradient tensor.
     */
    void Tensor::ensure_grad() {
        if (this->grad != nullptr) {
            return;
        }
        this->grad = new Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        this->grad->set_zero();
        this->grad->sync(); // the zero-fill must land before any reader queues work
    }

    /**
     * If no grad exists, adopts delta as the grad (the move drains delta's workers).
     * Otherwise syncs delta and the existing grad, then adds delta in place on this
     * tensor's device. delta must match this tensor's shape and dtype.
     * @brief Add a gradient contribution into this tensor's grad.
     * @param delta Gradient contribution; consumed (moved from or left drained).
     * @throws ErrorHandling::UnReachableCode if the device is neither CPU nor GPU.
     */
    void Tensor::accumulate_grad(Tensor&& delta) {
        if (this->grad == nullptr) {
            this->grad = new Tensor(std::move(delta)); // the move drains delta's workers
            return;
        }
        delta.sync();
        // grad may still have queued writers (backward()'s async set_zero on
        // GPU, or an earlier grad_acc from a fan-out sibling); accumulating
        // before they land would race them on the same buffer.
        this->grad->sync();
        if (this->device == Device::CPU) {
            detail::grad_acc_cpu(*this->grad, delta);
        } else if (this->device == Device::GPU) {
            detail::grad_acc_gpu(*this->grad, delta);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * No-op when neither input requires grad. Otherwise marks this tensor as a non-leaf
     * that requires grad, allocates a Node recording backward_fn and the input pointers
     * (non-owning: the inputs must outlive this tensor), and replaces any stale node.
     * @brief Attach an autograd node to this (output) tensor.
     * @param backward_fn Backward trampoline invoked as (output grad, inputs, input count).
     * @param a First input tensor.
     * @param b Optional second input tensor; nullptr for a unary op.
     */
    void Tensor::wire_node(void (*backward_fn)(Tensor*, Tensor**, size_t), const Tensor& a, const Tensor* b) {
        if (!a.requires_grad && (b == nullptr || !b->requires_grad)) {
            return;
        }
        this->requires_grad = true;
        this->is_leaf = false;
        auto* new_node = new Node{};
        new_node->_backward_fn = backward_fn;
        new_node->_inputs[0] = const_cast<Tensor*>(&a);
        new_node->_n_inputs = 1;
        this->parents[0] = new_node->_inputs[0];
        if (b != nullptr) {
            new_node->_inputs[1] = const_cast<Tensor*>(b);
            new_node->_n_inputs = 2;
            this->parents[1] = new_node->_inputs[1];
        }
        this->n_parents = new_node->_n_inputs;
        delete this->node; // re-wiring replaces any stale node
        this->node = new_node;
    }

    /**
     * Deletes grad (its destructor drains pending workers) and resets it to nullptr;
     * the next accumulation reallocates it.
     * @brief Drop this tensor's accumulated gradient.
     */
    void Tensor::zero_grad() {
        delete this->grad; // ~Tensor drains grad's own pending workers first
        this->grad = nullptr;
    }

    /**
     * Walks the graph from this tensor, zeroes the grads of node-owning (intermediate)
     * tensors so they hold only this call's delta (leaf grads accumulate across calls),
     * seeds this tensor's grad with ones, then fires each node's backward function in
     * topological order (Kahn's algorithm over consumer counts). Syncs every visited
     * tensor and its grad before returning.
     * @brief Reverse-mode backpropagation from this tensor.
     * @throws ErrorHandling::NoGradient if this tensor neither requires grad nor has a node.
     * @throws ErrorHandling::DimensionTooLarge if a tensor's consumer count exceeds 255.
     * @throws ErrorHandling::UnReachableCode if not every node fires (a cycle).
     */
    void Tensor::backward() {
        if (!this->requires_grad && this->node == nullptr) {
            throw ErrorHandling::NoGradient();
        }
        // Discover every tensor reachable through the graph.
        std::vector<Tensor*> visited;
        std::unordered_set<Tensor*> seen;
        std::vector<Tensor*> stack{this};
        while (!stack.empty()) {
            auto* t = stack.back();
            stack.pop_back();
            if (!seen.insert(t).second) {
                continue;
            }
            visited.push_back(t);
            if (t->node != nullptr) {
                for (auto i = size_t{0}; i < t->node->_n_inputs; ++i) {
                    stack.push_back(t->node->_inputs[i]);
                }
            }
        }
        // Intermediate (node-owning) grads hold only this call's delta; leaf
        // grads accumulate across calls.
        for (const auto t : visited) {
            if (t->node != nullptr) {
                t->ensure_grad();
                t->grad->set_zero();
            }
        }
        // Seed the root with ones.
        this->ensure_grad();
        auto ones = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        ones.set_one();
        this->accumulate_grad(std::move(ones));
        // Kahn's algorithm over consumer counts: a node fires once every
        // consumer of its tensor has delivered a gradient contribution.
        for (const auto t : visited) {
            if (t->node != nullptr) {
                t->node->_n_outputs_pending = 0;
            }
        }
        for (const auto t : visited) {
            if (t->node == nullptr) {
                continue;
            }
            for (auto i = size_t{0}; i < t->node->_n_inputs; ++i) {
                auto* const in = t->node->_inputs[i];
                if (in->node != nullptr) {
                    if (in->node->_n_outputs_pending == std::numeric_limits<uint8_t>::max()) {
                        throw ErrorHandling::DimensionTooLarge("backward graph fan-in exceeds 255");
                    }
                    ++in->node->_n_outputs_pending;
                }
            }
        }
        std::vector<Tensor*> ready;
        auto node_count = size_t{0};
        for (const auto t : visited) {
            if (t->node != nullptr) {
                ++node_count;
                if (t->node->_n_outputs_pending == 0) {
                    ready.push_back(t);
                }
            }
        }
        auto fired = size_t{0};
        while (!ready.empty()) {
            auto* const t = ready.back();
            ready.pop_back();
            auto* const n = t->node;
            n->_backward_fn(t->grad, n->_inputs, n->_n_inputs);
            ++fired;
            for (auto i = size_t{0}; i < n->_n_inputs; ++i) {
                auto* const in = n->_inputs[i];
                if (in->node != nullptr && --in->node->_n_outputs_pending == 0) {
                    ready.push_back(in);
                }
            }
        }
        if (fired != node_count) [[unlikely]] {
            // The graph is built exclusively by binary/unary ops on fresh
            // outputs, so a cycle means memory corruption, not user error.
            throw ErrorHandling::UnReachableCode();
        }
        for (const auto t : visited) {
            t->sync();
            if (t->grad != nullptr) {
                t->grad->sync();
            }
        }
    }
}
