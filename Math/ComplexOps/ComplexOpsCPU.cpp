/**
 * @file ComplexOpsCPU.cpp
 * @author GreenBean
 * @brief CPU backend for complex tensor ops: bias_add, relu, softmax_ce.
 * Hosts the Tensor method definitions (mirrors how Tensor::matmul lives in
 * MatmulCPU.cpp).
 * @version 0.1
 * @date 2026-10-18
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/ComplexOps/ComplexOps.h"
#include "Math/ComplexOps/ComplexOpsKernels.h"
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
        template<> struct packed_lane<float16x2_t> { using type = float16_t; };
        template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
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
         * Only the low 4 bits of each input survive, so out-of-range sums wrap.
         * @brief Pack two signed values into an int4 pair (modular, low 4 bits each).
         * @param lo Value for the low nibble.
         * @param hi Value for the high nibble.
         * @return Packed int4 pair.
         */
        [[nodiscard]] inline int4x2_t pack_int4(const int32_t lo, const int32_t hi) {
            return Intrinsics::detail::bit_cast_memcpy<int4x2_t>(
                static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4)));
        }

        // --------------------------- bias_add kernels ---------------------------

        /**
         * Single-rounding add, bit-identical to the AVX kernels.
         * @brief Scalar range bias_add (fp32/fp64).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void bias_add_range_scalar(T* out, const T* in, const T* bias, const size_t cols, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = in[i] + bias[i % cols];
            }
        }

        /**
         * Widen to fp32, add, narrow RNE (unaccel pathway).
         * Same conversion-pathway semantics as the other half kernels.
         * @brief Scalar range bias_add (fp16/bf16, fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename H>
        void bias_add_range_half(H* out, const H* in, const H* bias, const size_t cols, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = narrow_lane<H>(widen_lane(in[i]) + widen_lane(bias[i % cols]));
            }
        }

        /**
         * Per-lane widen/add/narrow, repack on store.
         * @brief Scalar range bias_add (fp16x2/bf16x2, per-lane fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename P>
        void bias_add_range_packed(P* out, const P* in, const P* bias, const size_t cols, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto in_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[i]);
                const auto b_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(bias[i % cols]);
                const auto lo = narrow_lane<H>(widen_lane(lane_lo<H>(in_bits)) + widen_lane(lane_lo<H>(b_bits)));
                const auto hi = narrow_lane<H>(widen_lane(lane_hi<H>(in_bits)) + widen_lane(lane_hi<H>(b_bits)));
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Modular (wrap) add through the unsigned twin. Same wrap contract as the integer sum kernels,
         * bit-identical on every backend and chunking.
         * @brief Scalar range bias_add (all integer dtypes, wrap add).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void bias_add_range_int(T* out, const T* in, const T* bias, const size_t cols, const size_t begin, const size_t end) {
            using U = std::make_unsigned_t<T>;
            for (auto i = begin; i < end; ++i) {
                out[i] = static_cast<T>(static_cast<U>(static_cast<U>(in[i]) + static_cast<U>(bias[i % cols])));
            }
        }

        /**
         * Per-nibble wrap add, repack on store.
         * @brief Scalar range bias_add (int4x2, per-nibble wrap).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param bias Bias vector, cols elements.
         * @param cols Last-dimension size; bias index is i % cols.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        void bias_add_range_int4x2(int4x2_t* out, const int4x2_t* in, const int4x2_t* bias, const size_t cols, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = pack_int4(int4_lo(in[i]) + int4_lo(bias[i % cols]),
                                   int4_hi(in[i]) + int4_hi(bias[i % cols]));
            }
        }

        // --------------------------- relu kernels ---------------------------

        /**
         * Computes out[i] = in[i] < 0 ? 0 : in[i]. NaN propagates (the comparison is false for NaN) and
         * -0.0 survives. Bit-identical to the AVX max(zero, x) kernels.
         * @brief Scalar range relu (fp32/fp64).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void relu_range_scalar(T* out, const T* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = in[i] < T{0} ? T{0} : in[i];
            }
        }

        /**
         * Widen to fp32, relu, narrow RNE. NaN propagates.
         * @brief Scalar range relu (fp16/bf16, fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename H>
        void relu_range_half(H* out, const H* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto wide = widen_lane(in[i]);
                out[i] = narrow_lane<H>(wide < 0.0f ? 0.0f : wide);
            }
        }

        /**
         * Per-lane widen/relu/narrow, repack on store.
         * @brief Scalar range relu (fp16x2/bf16x2, per-lane fp32 compute).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename P>
        void relu_range_packed(P* out, const P* in, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(in[i]);
                const auto lo_wide = widen_lane(lane_lo<H>(bits));
                const auto hi_wide = widen_lane(lane_hi<H>(bits));
                const auto lo = narrow_lane<H>(lo_wide < 0.0f ? 0.0f : lo_wide);
                const auto hi = narrow_lane<H>(hi_wide < 0.0f ? 0.0f : hi_wide);
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Signed dtypes clamp to 0; unsigned dtypes copy through.
         * @brief Scalar range relu (all integer dtypes).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        template<typename T>
        void relu_range_int(T* out, const T* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                if constexpr (std::is_signed_v<T>) {
                    out[i] = in[i] < T{0} ? T{0} : in[i];
                } else {
                    out[i] = in[i];
                }
            }
        }

        /**
         * Per-nibble clamp to 0, repack on store.
         * @brief Scalar range relu (int4x2, per-nibble).
         * @param out Output buffer.
         * @param in Input buffer.
         * @param begin First element computed.
         * @param end One past the last element computed.
         */
        void relu_range_int4x2(int4x2_t* out, const int4x2_t* in, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto lo = int4_lo(in[i]);
                const auto hi = int4_hi(in[i]);
                out[i] = pack_int4(lo < 0 ? 0 : lo, hi < 0 ? 0 : hi);
            }
        }

        // --------------------------- softmax_ce kernels ---------------------------

        /**
         * Covers fp32/fp16/bf16/packed x2 rows. Packed dtypes unpack each element's lanes as consecutive classes.
         * @brief Widen class j of a row to fp32.
         * @param row First element of the logits row.
         * @param j Effective class index within the row.
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
         * Covers fp64/integer/int4x2 rows. Int4x2 unpacks each element's nibbles as consecutive classes.
         * @brief Widen class j of a row to fp64.
         * @param row First element of the logits row.
         * @param j Effective class index within the row.
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
         * Numerically stable: loss = m + log(sum(exp(x - m))) - x_target, computed in the dtype's
         * accumulator width (fp32 for fp32/fp16/bf16/packed, fp64 for fp64/integers).
         * A NaN logit makes the row's loss NaN (max ignores NaN, exp(NaN - m) poisons the sum).
         * @brief Scalar row softmax cross-entropy.
         * @param row First element of the logits row.
         * @param cols Effective class count of the row.
         * @param target Class index of the row's target, already bounds-checked.
         * @return The row loss, widened to fp64.
         */
        template<typename T, typename Acc>
        [[nodiscard]] float64_t softmax_ce_row(const T* row, const size_t cols, const size_t target) {
            /**
             * Selects the fp32 or fp64 class widening to match the accumulator width.
             * @brief Widen class j of the captured row to Acc.
             * @param j Effective class index within the row.
             * @return Class j widened to Acc.
             * @note Lambda subcall. Self-execution prohibited.
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
            const auto loss = m + std::log(s) - widen(target);
            return static_cast<float64_t>(loss);
        }

        /**
         * Int4x2 targets use the low nibble (matches the packed->unpacked first-lane conversion contract).
         * Throws ErrorHandling::IndexOutOfRange for a negative index or one at or past cols.
         * @brief Read target class index r, bounds-checked against the effective class count.
         * @param targets Target class indices, one per row.
         * @param r Row index.
         * @param cols Effective class count of the row.
         * @return The class index, validated to [0, cols).
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
         * Chunks an elementwise range worker across the CPU thread pool. Futures land on every tensor the
         * workers touch so teardown drains them (they hold raw buffer pointers). Small inputs run inline.
         * @brief Threaded elementwise launcher.
         * @param worker Callable taking (begin, end); copied into each task.
         * @param numel Total element count to cover.
         * @param a First tensor the workers read.
         * @param b Second tensor the workers read.
         * @param out Tensor the workers write.
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
         * Chunks a row-indexed worker across the pool. Runs in the Tensor method, never inside a pool
         * worker, so the caller waiting on the returned futures is legal.
         * @brief Threaded row launcher. Returns the futures for the caller to drain.
         * @param rows Total row count to cover.
         * @param range_fn Callable taking (slot, begin, end); copied into each task.
         * @param t Logits tensor the workers read.
         * @param targets Targets tensor the workers read.
         * @return Futures of every submitted chunk; empty when the work ran inline.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_rows(const size_t rows, RangeFn&& range_fn, const Tensor& t, const Tensor& targets) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || rows <= t_count * 8) {
                range_fn(0, 0, rows);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (rows + t_count - 1) / t_count;
            auto slot = size_t{0};
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= rows) {
                    break;
                }
                // range_fn is captured by value; its own captures reference the
                // caller's partials, which stay alive until drain() returns.
                futures.push_back(threadMgr.submit([range_fn, slot, begin, end = std::min(begin + chunk, rows)] { range_fn(slot, begin, end); }, Threading::TaskPriority::Normal).share());
                ++slot;
            }
            t.give_futures(futures);
            targets.give_futures(futures);
            return futures;
        }

        /**
         * Rethrows the first task error through future::get.
         * @brief Drain futures on the caller thread.
         * @param futures Futures to wait on.
         */
        void drain(const std::vector<std::shared_future<void>>& futures) {
            for (const auto& f : futures) {
                f.get();
            }
        }

        /**
         * Row losses accumulate into partials[slot] in fp64. Not a friend of Tensor: the target
         * dtype/base pointer/offset arrive precomputed.
         * @brief Threaded softmax_ce partial over a row range.
         * @param logits Row-major logits, already offset to the first element.
         * @param tgt_dtype Dtype of the targets buffer; selects the index reader.
         * @param tgt_data Base pointer of the targets buffer.
         * @param tgt_offset Element offset into the targets buffer.
         * @param cols_store Stored element count per logits row.
         * @param cols_eff Effective class count per row (doubled for packed dtypes).
         * @param slot Index of this chunk's entry in partials.
         * @param begin First row computed.
         * @param end One past the last row computed.
         * @param partials Per-chunk fp64 loss sums; only partials[slot] is written.
         */
        template<typename T, typename Acc>
        void softmax_ce_rows(const T* logits, const DataTypes::DType tgt_dtype, const void* tgt_data, const size_t tgt_offset,
                             const size_t cols_store, const size_t cols_eff,
                             const size_t slot, const size_t begin, const size_t end, std::vector<float64_t>& partials) {
            auto acc = 0.0;
            /**
             * Reads each row's target index as I and accumulates that row's loss into acc.
             * @brief Accumulate the row losses for one target dtype.
             * @param t_ptr Targets buffer typed as I, already offset.
             * @note Lambda subcall. Self-execution prohibited.
             */
            auto run = [&]<typename I>(const I* t_ptr) {
                for (auto r = begin; r < end; ++r) {
                    const auto idx = target_index(t_ptr, r, cols_eff);
                    acc += softmax_ce_row<T, Acc>(logits + r * cols_store, cols_eff, idx);
                }
            };
            using DType = DataTypes::DType;
            switch (tgt_dtype) {
                case DType::Int8: run(static_cast<const int8_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt8: run(static_cast<const uint8_t*>(tgt_data) + tgt_offset); break;
                case DType::Int16: run(static_cast<const int16_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt16: run(static_cast<const uint16_t*>(tgt_data) + tgt_offset); break;
                case DType::Int32: run(static_cast<const int32_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt32: run(static_cast<const uint32_t*>(tgt_data) + tgt_offset); break;
                case DType::Int64: run(static_cast<const int64_t*>(tgt_data) + tgt_offset); break;
                case DType::UInt64: run(static_cast<const uint64_t*>(tgt_data) + tgt_offset); break;
                case DType::Int4x2: run(static_cast<const int4x2_t*>(tgt_data) + tgt_offset); break;
                [[unlikely]] default: throw ErrorHandling::UnReachableCode();
            }
            partials[slot] = acc;
        }

        /**
         * Floats narrow RNE (halfs via the unaccel pathway). Packed dtypes duplicate into both lanes
         * (matches the unpacked->packed conversion contract). Integers clamp to the dtype range and
         * truncate; a NaN loss stores 0 (there is no integer NaN to store).
         * Not a friend of Tensor: the output dtype and base pointer arrive precomputed.
         * @brief Store the narrowed softmax_ce mean loss.
         * @param out_dtype Dtype of the output element.
         * @param out_data Base pointer of the output element.
         * @param mean Mean loss in fp64.
         */
        void store_mean(const DataTypes::DType out_dtype, void* const out_data, const float64_t mean) {
            using DType = DataTypes::DType;
            switch (out_dtype) {
                case DType::Float32: *static_cast<float32_t*>(out_data) = static_cast<float32_t>(mean); break;
                case DType::Float64: *static_cast<float64_t*>(out_data) = mean; break;
                case DType::Float16: *static_cast<float16_t*>(out_data) = narrow_lane<float16_t>(static_cast<float32_t>(mean)); break;
                case DType::BFloat16: *static_cast<bfloat16_t*>(out_data) = narrow_lane<bfloat16_t>(static_cast<float32_t>(mean)); break;
                case DType::Float16x2: {
                    const auto lane = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<float16_t>(static_cast<float32_t>(mean)));
                    *static_cast<float16x2_t*>(out_data) = pack_lanes<float16x2_t>(lane, lane);
                    break;
                }
                case DType::BFloat16x2: {
                    const auto lane = Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<bfloat16_t>(static_cast<float32_t>(mean)));
                    *static_cast<bfloat16x2_t*>(out_data) = pack_lanes<bfloat16x2_t>(lane, lane);
                    break;
                }
                case DType::Int4x2: {
                    auto v = mean;
                    if (std::isnan(v)) {
                        v = 0.0;
                    }
                    if (v <= -8.0) {
                        v = -8.0;
                    } else if (v >= 7.0) {
                        v = 7.0;
                    }
                    const auto n = static_cast<int32_t>(v);
                    *static_cast<int4x2_t*>(out_data) = pack_int4(n, n);
                    break;
                }
                default: {
                    /**
                     * Maps NaN to 0, clamps the mean to I's range, truncates and stores.
                     * @brief Store the mean loss as integer type I.
                     * @param p Output element.
                     * @note Lambda subcall. Self-execution prohibited.
                     */
                    auto store_int = [&]<typename I>(I* p) {
                        auto v = mean;
                        if (std::isnan(v)) {
                            v = 0.0;
                        }
                        const auto lo = static_cast<float64_t>(std::numeric_limits<I>::lowest());
                        const auto hi = static_cast<float64_t>(std::numeric_limits<I>::max());
                        if (v <= lo) {
                            v = lo;
                        } else if (v >= hi) {
                            v = hi;
                        }
                        *p = static_cast<I>(v);
                    };
                    switch (out_dtype) {
                        case DType::Int8: store_int(static_cast<int8_t*>(out_data)); break;
                        case DType::UInt8: store_int(static_cast<uint8_t*>(out_data)); break;
                        case DType::Int16: store_int(static_cast<int16_t*>(out_data)); break;
                        case DType::UInt16: store_int(static_cast<uint16_t*>(out_data)); break;
                        case DType::Int32: store_int(static_cast<int32_t*>(out_data)); break;
                        case DType::UInt32: store_int(static_cast<uint32_t*>(out_data)); break;
                        case DType::Int64: store_int(static_cast<int64_t*>(out_data)); break;
                        case DType::UInt64: store_int(static_cast<uint64_t*>(out_data)); break;
                        [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                    }
                    break;
                }
            }
        }

        /**
         * @brief Function pointer type shared by every bias_add range kernel (scalar, AVX2, AVX512).
         */
        template<typename T>
        using BiasAddScalarFn = void(*)(T*, const T*, const T*, size_t, size_t, size_t);
        /**
         * @brief Function pointer type shared by every relu range kernel (scalar, AVX2, AVX512).
         */
        template<typename T>
        using ReluScalarFn = void(*)(T*, const T*, size_t, size_t);
    }

    /**
     * Float32/Float64 pick AVX512, then AVX2, then the scalar kernel. Every other dtype uses its scalar kernel.
     * Work is chunked across the CPU thread pool and the futures land on all three tensors.
     * @brief CPU bias_add: out[i] = t[i] + bias[i % cols].
     * @param t Input tensor, CPU resident, contiguous.
     * @param bias Rank-1 bias tensor matching the last dimension of t.
     * @param out Output tensor, same shape and dtype as t.
     */
    void bias_add_cpu(const Tensor& t, const Tensor& bias, Tensor& out) {
        if (t.device != Device::CPU || bias.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        const auto cols = t.shape[t.ndim - 1];
        switch (t.dtype) {
            case DType::Float32: {
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float32_t*>(t.data) + t.offset;
                const auto* b_ptr = static_cast<const float32_t*>(bias.data) + bias.offset;
                const BiasAddScalarFn<float32_t> kernel = Math::ComplexOps::complexops_avx512_available()
                    ? Math::ComplexOps::bias_add_f32_range_avx512
                    : (Math::ComplexOps::complexops_avx2_available() ? Math::ComplexOps::bias_add_f32_range_avx2 : bias_add_range_scalar<float32_t>);
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Float64: {
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float64_t*>(t.data) + t.offset;
                const auto* b_ptr = static_cast<const float64_t*>(bias.data) + bias.offset;
                const BiasAddScalarFn<float64_t> kernel = Math::ComplexOps::complexops_avx512_available()
                    ? Math::ComplexOps::bias_add_f64_range_avx512
                    : (Math::ComplexOps::complexops_avx2_available() ? Math::ComplexOps::bias_add_f64_range_avx2 : bias_add_range_scalar<float64_t>);
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Float16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for float16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16_t*>(out.data) + out.offset, in_ptr = static_cast<const float16_t*>(t.data) + t.offset, b_ptr = static_cast<const float16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_half<float16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for bfloat16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16_t*>(t.data) + t.offset, b_ptr = static_cast<const bfloat16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_half<bfloat16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for float16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const float16x2_t*>(t.data) + t.offset, b_ptr = static_cast<const float16x2_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_packed<float16x2_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for bfloat16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16x2_t*>(t.data) + t.offset, b_ptr = static_cast<const bfloat16x2_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_packed<bfloat16x2_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int8: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int8_t*>(out.data) + out.offset, in_ptr = static_cast<const int8_t*>(t.data) + t.offset, b_ptr = static_cast<const int8_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int8_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt8: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint8_t*>(out.data) + out.offset, in_ptr = static_cast<const uint8_t*>(t.data) + t.offset, b_ptr = static_cast<const uint8_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint8_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int16_t*>(out.data) + out.offset, in_ptr = static_cast<const int16_t*>(t.data) + t.offset, b_ptr = static_cast<const int16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt16: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint16_t*>(out.data) + out.offset, in_ptr = static_cast<const uint16_t*>(t.data) + t.offset, b_ptr = static_cast<const uint16_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint16_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int32: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int32_t*>(out.data) + out.offset, in_ptr = static_cast<const int32_t*>(t.data) + t.offset, b_ptr = static_cast<const int32_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int32_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt32: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint32_t*>(out.data) + out.offset, in_ptr = static_cast<const uint32_t*>(t.data) + t.offset, b_ptr = static_cast<const uint32_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint32_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int64: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int64_t*>(out.data) + out.offset, in_ptr = static_cast<const int64_t*>(t.data) + t.offset, b_ptr = static_cast<const int64_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<int64_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::UInt64: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for uint64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint64_t*>(out.data) + out.offset, in_ptr = static_cast<const uint64_t*>(t.data) + t.offset, b_ptr = static_cast<const uint64_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int<uint64_t>(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Forwards the captured bias_add pointers to the range kernel.
                 * @brief Elementwise bias_add range worker for int4x2.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int4x2_t*>(out.data) + out.offset, in_ptr = static_cast<const int4x2_t*>(t.data) + t.offset, b_ptr = static_cast<const int4x2_t*>(bias.data) + bias.offset, cols](const size_t b, const size_t e) { bias_add_range_int4x2(out_ptr, in_ptr, b_ptr, cols, b, e); }, numel, t, bias, out);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Float32/Float64 pick AVX512, then AVX2, then the scalar kernel. Every other dtype uses its scalar kernel.
     * Work is chunked across the CPU thread pool and the futures land on both tensors.
     * @brief CPU relu: out[i] = t[i] < 0 ? 0 : t[i].
     * @param t Input tensor, CPU resident, contiguous.
     * @param out Output tensor, same shape and dtype as t.
     */
    void relu_cpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        switch (t.dtype) {
            case DType::Float32: {
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float32_t*>(t.data) + t.offset;
                const ReluScalarFn<float32_t> kernel = Math::ComplexOps::complexops_avx512_available()
                    ? Math::ComplexOps::relu_f32_range_avx512
                    : (Math::ComplexOps::complexops_avx2_available() ? Math::ComplexOps::relu_f32_range_avx2 : relu_range_scalar<float32_t>);
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Float64: {
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                const auto* in_ptr = static_cast<const float64_t*>(t.data) + t.offset;
                const ReluScalarFn<float64_t> kernel = Math::ComplexOps::complexops_avx512_available()
                    ? Math::ComplexOps::relu_f64_range_avx512
                    : (Math::ComplexOps::complexops_avx2_available() ? Math::ComplexOps::relu_f64_range_avx2 : relu_range_scalar<float64_t>);
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker (fp32/fp64 SIMD or scalar).
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([=](const size_t b, const size_t e) { kernel(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Float16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for float16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16_t*>(out.data) + out.offset, in_ptr = static_cast<const float16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_half<float16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::BFloat16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for bfloat16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_half<bfloat16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Float16x2: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for float16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<float16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const float16x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_packed<float16x2_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::BFloat16x2: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for bfloat16x2_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset, in_ptr = static_cast<const bfloat16x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_packed<bfloat16x2_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int8: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int8_t*>(out.data) + out.offset, in_ptr = static_cast<const int8_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int8_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt8: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint8_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint8_t*>(out.data) + out.offset, in_ptr = static_cast<const uint8_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint8_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int16_t*>(out.data) + out.offset, in_ptr = static_cast<const int16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt16: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint16_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint16_t*>(out.data) + out.offset, in_ptr = static_cast<const uint16_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint16_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int32: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int32_t*>(out.data) + out.offset, in_ptr = static_cast<const int32_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int32_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt32: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint32_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint32_t*>(out.data) + out.offset, in_ptr = static_cast<const uint32_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint32_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int64: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int64_t*>(out.data) + out.offset, in_ptr = static_cast<const int64_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<int64_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::UInt64: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for uint64_t.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<uint64_t*>(out.data) + out.offset, in_ptr = static_cast<const uint64_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int<uint64_t>(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            case DType::Int4x2: {
                /**
                 * Forwards the captured relu pointers to the range kernel.
                 * @brief Elementwise relu range worker for int4x2.
                 * @param b First element computed.
                 * @param e One past the last element computed.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                launch_elementwise([out_ptr = static_cast<int4x2_t*>(out.data) + out.offset, in_ptr = static_cast<const int4x2_t*>(t.data) + t.offset](const size_t b, const size_t e) { relu_range_int4x2(out_ptr, in_ptr, b, e); }, numel, t, t, out);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Rows are chunked across the pool into fp64 partial sums, drained on the caller thread, then
     * averaged and narrowed into out. Throws ErrorHandling::IndexOutOfRange for a bad target.
     * @brief CPU softmax cross-entropy: mean over rows of logsumexp(row) - row[target].
     * @param t Rank-2 logits tensor (rows x classes), CPU resident, contiguous.
     * @param targets Rank-1 integer class indices, one per row.
     * @param out Single-element output tensor in the logits dtype.
     */
    void softmax_ce_cpu(const Tensor& t, const Tensor& targets, Tensor& out) {
        if (t.device != Device::CPU || targets.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto rows = t.shape[0];
        const auto cols_store = t.shape[1];
        const auto packed = t.dtype == DType::Float16x2 || t.dtype == DType::BFloat16x2 || t.dtype == DType::Int4x2;
        const auto cols_eff = cols_store * (packed ? size_t{2} : size_t{1});
        // softmax_ce_rows is not a Tensor friend: hand it the precomputed triple.
        const auto tgt_dtype = targets.dtype;
        const auto* const tgt_data = targets.data;
        const auto tgt_offset = targets.offset;
        std::vector<float64_t> partials(IncludedCPUFeatures().threads, 0.0);
        switch (t.dtype) {
            case DType::Float32: {
                const auto* logits = static_cast<const float32_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float32_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Float64: {
                const auto* logits = static_cast<const float64_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float64_t, float64_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Float16: {
                const auto* logits = static_cast<const float16_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float16_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::BFloat16: {
                const auto* logits = static_cast<const bfloat16_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<bfloat16_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Float16x2: {
                const auto* logits = static_cast<const float16x2_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<float16x2_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::BFloat16x2: {
                const auto* logits = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<bfloat16x2_t, float32_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64:
            case DType::Int4x2: {
                /**
                 * Launches the row workers for integer logits type T and drains them.
                 * @brief Run softmax_ce rows for one integer logits dtype.
                 * @param logits Logits buffer typed as T, already offset.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* logits) {
                    drain(launch_rows(rows, [&](const size_t slot, const size_t b, const size_t e) { softmax_ce_rows<T, float64_t>(logits, tgt_dtype, tgt_data, tgt_offset, cols_store, cols_eff, slot, b, e, partials); }, t, targets));
                };
                switch (t.dtype) {
                    case DType::Int8: launch_int(static_cast<const int8_t*>(t.data) + t.offset); break;
                    case DType::UInt8: launch_int(static_cast<const uint8_t*>(t.data) + t.offset); break;
                    case DType::Int16: launch_int(static_cast<const int16_t*>(t.data) + t.offset); break;
                    case DType::UInt16: launch_int(static_cast<const uint16_t*>(t.data) + t.offset); break;
                    case DType::Int32: launch_int(static_cast<const int32_t*>(t.data) + t.offset); break;
                    case DType::UInt32: launch_int(static_cast<const uint32_t*>(t.data) + t.offset); break;
                    case DType::Int64: launch_int(static_cast<const int64_t*>(t.data) + t.offset); break;
                    case DType::UInt64: launch_int(static_cast<const uint64_t*>(t.data) + t.offset); break;
                    case DType::Int4x2: launch_int(static_cast<const int4x2_t*>(t.data) + t.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
        auto total = 0.0;
        for (const auto p : partials) {
            total += p;
        }
        store_mean(out.dtype, out.data, total / static_cast<float64_t>(rows));
    }
}

namespace BeanTensor::Tensors {

    /**
     * Validates rank/dtype/device/contiguity, syncs both operands, then routes to the CPU or GPU backend
     * and wires the autograd node.
     * @brief Add a rank-1 bias along the last dimension.
     * @param bias Rank-1 tensor matching the last dimension, same dtype and device.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::bias_add(const Tensor& bias) const {
        if (bias.ndim != 1 || bias.shape[0] != this->shape[this->ndim - 1]) {
            throw ErrorHandling::MismatchedShapes("bias_add requires a rank-1 bias matching the last dimension");
        }
        if (this->dtype != bias.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != bias.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !bias.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        bias.sync();
        auto out = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::bias_add_cpu(*this, bias, out);
        } else if (this->device == Device::GPU) {
            detail::bias_add_gpu(*this, bias, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::bias_add_g_node, *this, &bias);
        return out;
    }

    /**
     * Validates contiguity, syncs this tensor, then routes to the CPU or GPU backend and wires the autograd node.
     * @brief Elementwise rectified linear unit.
     * @return New tensor shaped like this.
     */
    Tensor Tensor::relu() const {
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        auto out = Tensor(std::vector<size_t>(this->shape, this->shape + this->ndim), this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::relu_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::relu_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        out.wire_node(detail::relu_g_node, *this);
        return out;
    }

    /**
     * Validates shapes, devices and target dtype (integer kinds only), syncs both operands, then routes to the
     * backend. Only the logits join the autograd graph, as targets never receive a gradient.
     * @brief Mean softmax cross-entropy over the rows of rank-2 logits.
     * @param targets Rank-1 integer class indices, one per row, same device.
     * @return Single-element tensor in the logits dtype.
     */
    Tensor Tensor::softmax_ce(const Tensor& targets) const {
        if (this->ndim != 2) {
            throw ErrorHandling::MismatchedShapes("softmax_ce requires rank-2 logits (rows x classes)");
        }
        if (targets.ndim != 1 || targets.shape[0] != this->shape[0]) {
            throw ErrorHandling::MismatchedShapes("softmax_ce requires rank-1 targets with one class index per row");
        }
        if (this->shape[0] == 0 || this->shape[1] == 0) {
            throw ErrorHandling::MismatchedShapes("softmax_ce requires non-empty logits");
        }
        if (this->device != targets.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !targets.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        {
            using DType = DataTypes::DType;
            switch (targets.dtype) {
                case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
                case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64:
                case DType::Int4x2: {
                    break;
                }
                default: {
                    throw ErrorHandling::MismatchedDTypes();
                }
            }
        }
        sync();
        targets.sync();
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::softmax_ce_cpu(*this, targets, out);
        } else if (this->device == Device::GPU) {
            detail::softmax_ce_gpu(*this, targets, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        // Targets are class indices and never receive a gradient: only the
        // logits decide whether the output joins the graph.
        if (this->requires_grad) {
            out.wire_node(detail::softmax_ce_g_node, *this, &targets);
        }
        return out;
    }
}
