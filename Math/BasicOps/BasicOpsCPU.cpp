/**
 * @file BasicOpsCPU.cpp
 * @author GreenBean
 * @brief CPU backend for basic tensor ops: sgd_update, sum, argmax, reshape.
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
#include <utility>
#include <vector>

#include "Math/BasicOps/BasicOps.h"
#include "Math/BasicOps/BasicOpsKernels.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    namespace {
        /**
         * Routes through the unaccel conversion pathway; conversion flags are not tracked.
         * @brief Widen one fp16/bf16 lane to fp32.
         * @param v Half-precision lane.
         * @return The lane value as fp32.
         */
        template<typename H> requires (std::is_same_v<H, float16_t> || std::is_same_v<H, bfloat16_t>)
        [[nodiscard]] float32_t widen_lane(const H& v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp16_to_fp32_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_bf16_to_fp32(v);
            }
        }

        /**
         * Routes through the unaccel conversion pathway; untracked, PERMISSIVE clamp method.
         * @brief Narrow one fp32 accumulator to fp16/bf16 (RNE).
         * @param v FP32 accumulator.
         * @return The value rounded to the half-precision lane type.
         */
        template<typename H> requires (std::is_same_v<H, float16_t> || std::is_same_v<H, bfloat16_t>)
        [[nodiscard]] H narrow_lane(const float32_t v) {
            if constexpr (std::is_same_v<H, float16_t>) {
                return Intrinsics::detail::unaccel_fp32_to_fp16_untracked(v);
            } else {
                return Intrinsics::detail::unaccel_fp32_to_bf16_untracked(v);
            }
        }

        /**
         * Specialized per packed dtype; the primary template is declared only.
         * @brief Lane type of a packed dtype (float16x2_t -> float16_t, bfloat16x2_t -> bfloat16_t).
         */
        template<typename P> struct packed_lane;
        template<> struct packed_lane<float16x2_t> { using type = float16_t; };
        template<> struct packed_lane<bfloat16x2_t> { using type = bfloat16_t; };
        template<typename P> using packed_lane_t = typename packed_lane<P>::type;

        /**
         * Reads bits[15:0] of the packed 32-bit unit.
         * @brief Extract the low lane of a packed value, via its bit pattern.
         * @param bits Raw 32-bit packed unit.
         * @return The low lane as the half-precision lane type.
         */
        template<typename H>
        [[nodiscard]] H lane_lo(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits & 0xFFFFu));
        }

        /**
         * Reads bits[31:16] of the packed 32-bit unit.
         * @brief Extract the high lane of a packed value, via its bit pattern.
         * @param bits Raw 32-bit packed unit.
         * @return The high lane as the half-precision lane type.
         */
        template<typename H>
        [[nodiscard]] H lane_hi(const uint32_t bits) {
            return Intrinsics::detail::bit_cast_memcpy<H>(static_cast<uint16_t>(bits >> 16));
        }

        /**
         * Inverse of lane_lo/lane_hi.
         * @brief Pack two narrowed lanes back into a packed dtype value.
         * @param lo Raw bits of the low lane.
         * @param hi Raw bits of the high lane.
         * @return The packed value with lo in bits[15:0] and hi in bits[31:16].
         */
        template<typename P>
        [[nodiscard]] P pack_lanes(const uint16_t lo, const uint16_t hi) {
            return Intrinsics::detail::bit_cast_memcpy<P>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
        }

        /**
         * @brief Signed low lane of an int4x2 byte (bits[3:0], sign-extended).
         * @param v Packed int4x2 byte.
         * @return Lane value in [-8, 7].
         */
        [[nodiscard]] inline int8_t i4_lane_lo(const int4x2_t v) {
            const auto n = static_cast<int8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) & 0x0Fu);
            return n > 7 ? static_cast<int8_t>(n - 16) : n;
        }

        /**
         * @brief Signed high lane of an int4x2 byte (bits[7:4], sign-extended).
         * @param v Packed int4x2 byte.
         * @return Lane value in [-8, 7].
         */
        [[nodiscard]] inline int8_t i4_lane_hi(const int4x2_t v) {
            const auto n = static_cast<int8_t>(Intrinsics::detail::bit_cast_memcpy<uint8_t>(v) >> 4);
            return n > 7 ? static_cast<int8_t>(n - 16) : n;
        }

        /**
         * Each lane keeps only its low 4 bits, so out-of-range values wrap mod 16.
         * @brief Pack two signed lanes into an int4x2 byte.
         * @param lo Low lane value.
         * @param hi High lane value.
         * @return The packed int4x2 byte.
         */
        [[nodiscard]] inline int4x2_t i4_pack(const int8_t lo, const int8_t hi) {
            const auto lo_bits = static_cast<uint8_t>(lo) & 0x0Fu;
            const auto hi_bits = static_cast<uint8_t>(hi) & 0x0Fu;
            return Intrinsics::detail::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>(lo_bits | static_cast<uint8_t>(hi_bits << 4)));
        }

        // --------------------------- sgd_update kernels ---------------------------

        /**
         * Single rounding via fma matches the FMA-based AVX kernels bit-for-bit.
         * @note Computes w[i] = fma(neg_lr, g[i], w[i]).
         * @brief Scalar range SGD (fp32/fp64).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param neg_lr Negated learning rate.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void sgd_range_scalar(T* w, const T* g, const T neg_lr, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                if constexpr (std::is_same_v<T, float32_t>) {
                    w[i] = std::fmaf(neg_lr, g[i], w[i]);
                } else {
                    w[i] = std::fma(neg_lr, g[i], w[i]);
                }
            }
        }

        /**
         * Widen to fp32, fmaf, narrow RNE (unaccel pathway). Same conversion-pathway semantics as the GEMM half kernels.
         * @brief Scalar range SGD (fp16/bf16, fp32 compute).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param neg_lr Negated learning rate.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename H>
        void sgd_range_half(H* w, const H* g, const float32_t neg_lr, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                w[i] = narrow_lane<H>(std::fmaf(neg_lr, widen_lane(g[i]), widen_lane(w[i])));
            }
        }

        /**
         * Per-lane widen/fmaf/narrow, repacked on store.
         * @brief Scalar range SGD (fp16x2/bf16x2, per-lane fp32 compute).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param neg_lr Negated learning rate.
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename P>
        void sgd_range_packed(P* w, const P* g, const float32_t neg_lr, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto w_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(w[i]);
                const auto g_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(g[i]);
                const auto lo = narrow_lane<H>(std::fmaf(neg_lr, widen_lane(lane_lo<H>(g_bits)), widen_lane(lane_lo<H>(w_bits))));
                const auto hi = narrow_lane<H>(std::fmaf(neg_lr, widen_lane(lane_hi<H>(g_bits)), widen_lane(lane_hi<H>(w_bits))));
                w[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                     Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Computes w - lr*g in fp64 with a single fma rounding, clamps to the dtype range, narrows round-toward-zero.
         * At sub-1 magnitudes negative updates truncate up toward 0 while positive ones truncate down (up to 1 in the result).
         * A NaN update (NaN lr or grad lane) leaves the element unchanged, no integer NaN exists to store.
         * @brief Scalar range SGD (all integer dtypes, fp64 compute, clamp+truncate).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate (not negated).
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        template<typename T>
        void sgd_range_int(T* w, const T* g, const float64_t lr, const size_t begin, const size_t end) {
            const auto lo = static_cast<float64_t>(std::numeric_limits<T>::lowest());
            const auto hi = static_cast<float64_t>(std::numeric_limits<T>::max());
            for (auto i = begin; i < end; ++i) {
                const auto upd = std::fma(-lr, static_cast<float64_t>(g[i]), static_cast<float64_t>(w[i]));
                if (std::isnan(upd)) {
                    continue;
                }
                if (upd <= lo) {
                    w[i] = std::numeric_limits<T>::lowest();
                } else if (upd >= hi) {
                    w[i] = std::numeric_limits<T>::max();
                } else {
                    w[i] = static_cast<T>(upd);
                }
            }
        }

        /**
         * Per-lane fp64 compute, clamp to [-8, 7], narrow round-toward-zero (same contract as sgd_range_int).
         * A NaN update leaves that lane unchanged.
         * @brief Scalar range SGD (int4x2, per-lane fp64 compute).
         * @param w Weights, updated in place.
         * @param g Gradients, read only.
         * @param lr Learning rate (not negated).
         * @param begin First element processed.
         * @param end One past the last element processed.
         */
        inline void sgd_range_i4x2(int4x2_t* w, const int4x2_t* g, const float64_t lr, const size_t begin, const size_t end) {
            /**
             * Applies one clamped fp64 SGD step to a single signed int4 lane.
             * @brief Update one int4 lane.
             * @param w_lane Current weight lane, [-8, 7].
             * @param g_lane Gradient lane, [-8, 7].
             * @return Updated lane in [-8, 7]; w_lane itself when the update is NaN.
             * @note Lambda subcall. Self-execution prohibited.
             */
            const auto lane_step = [lr](const int8_t w_lane, const int8_t g_lane) {
                const auto upd = std::fma(-lr, static_cast<float64_t>(g_lane), static_cast<float64_t>(w_lane));
                if (std::isnan(upd)) {
                    return w_lane;
                }
                if (upd <= -8.0) {
                    return int8_t{-8};
                }
                if (upd >= 7.0) {
                    return int8_t{7};
                }
                return static_cast<int8_t>(upd);
            };
            for (auto i = begin; i < end; ++i) {
                w[i] = i4_pack(lane_step(i4_lane_lo(w[i]), i4_lane_lo(g[i])),
                               lane_step(i4_lane_hi(w[i]), i4_lane_hi(g[i])));
            }
        }

        // --------------------------- sum kernels ---------------------------

        /**
         * Sequential order; the AVX kernels use a tree order, so float sums are not bit-identical across dispatch tiers.
         * @brief Scalar range sum (fp32/fp64).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return Sum of the elements in [begin, end).
         */
        template<typename T>
        [[nodiscard]] T sum_range_scalar(const T* data, const size_t begin, const size_t end) {
            auto total = T{0};
            for (auto i = begin; i < end; ++i) {
                total += data[i];
            }
            return total;
        }

        /**
         * @brief Scalar range sum (fp16/bf16, fp32 accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return FP32 sum of the elements in [begin, end).
         */
        template<typename H>
        [[nodiscard]] float32_t sum_range_half(const H* data, const size_t begin, const size_t end) {
            auto total = 0.0f;
            for (auto i = begin; i < end; ++i) {
                total += widen_lane(data[i]);
            }
            return total;
        }

        /**
         * Lanes accumulate independently in fp32.
         * @brief Scalar range sum (fp16x2/bf16x2, per-lane fp32 accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return {sum_lo, sum_hi}.
         */
        template<typename P>
        [[nodiscard]] std::pair<float32_t, float32_t> sum_range_packed(const P* data, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            auto total_lo = 0.0f;
            auto total_hi = 0.0f;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(data[i]);
                total_lo += widen_lane(lane_lo<H>(bits));
                total_hi += widen_lane(lane_hi<H>(bits));
            }
            return {total_lo, total_hi};
        }

        /**
         * Each value's unsigned twin accumulates in a 64-bit modular accumulator. Modular addition is associative and
         * commutative, so every backend/chunking agrees bit-for-bit; narrowing truncates (wrap).
         * @brief Scalar range sum (all integer dtypes, uint64 wrap accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return Modular sum of the elements in [begin, end).
         */
        template<typename T>
        [[nodiscard]] uint64_t sum_range_int(const T* data, const size_t begin, const size_t end) {
            auto total = uint64_t{0};
            for (auto i = begin; i < end; ++i) {
                total += static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(data[i]));
            }
            return total;
        }

        /**
         * Per-lane uint64 modular accumulate of the raw nibbles. A nibble IS its two's-complement residue,
         * so mod-16 results match a signed wrap sum; narrowing keeps each lane mod 16.
         * @brief Scalar range sum (int4x2, per-lane modular accumulate).
         * @param data Input elements.
         * @param begin First element summed.
         * @param end One past the last element summed.
         * @return {sum_lo, sum_hi} raw nibble sums.
         */
        [[nodiscard]] inline std::pair<uint64_t, uint64_t> sum_range_i4x2(const int4x2_t* data, const size_t begin, const size_t end) {
            auto total_lo = uint64_t{0};
            auto total_hi = uint64_t{0};
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint8_t>(data[i]);
                total_lo += bits & 0x0Fu;
                total_hi += bits >> 4;
            }
            return {total_lo, total_hi};
        }

        // --------------------------- argmax helpers ---------------------------

        /**
         * PyTorch-compatible NaN rule: NaN beats every non-NaN value; the FIRST NaN wins against later NaNs;
         * otherwise strict greater wins, so ties keep the lower index.
         * @brief Is the candidate key better than the incumbent? (scalar keys)
         * @param cand Candidate key.
         * @param best Incumbent key.
         * @return T/F - the candidate should replace the incumbent.
         */
        [[nodiscard]] inline bool key_better(const float64_t cand, const float64_t best) {
            if (std::isnan(cand)) {
                return !std::isnan(best);
            }
            return !std::isnan(best) && cand > best;
        }

        /**
         * Keys are widened to fp64 and ranked by key_better.
         * @brief Scalar range argmax over fp32/fp64.
         * @param data Input elements.
         * @param begin First element considered.
         * @param end One past the last element considered.
         * @param best Best key so far; written when a better element is found.
         * @param best_idx Flat index of the best key.
         * @return T/F - the chunk held at least one element.
         */
        template<typename T>
        bool argmax_range_scalar(const T* data, const size_t begin, const size_t end, float64_t& best, uint64_t& best_idx) {
            auto has = false;
            for (auto i = begin; i < end; ++i) {
                const auto key = static_cast<float64_t>(data[i]);
                if (!has || key_better(key, best)) {
                    best = key;
                    best_idx = i;
                    has = true;
                }
            }
            return has;
        }

        /**
         * Keys are widened to fp64 and ranked by key_better.
         * @brief Scalar range argmax over fp16/bf16.
         * @param data Input elements.
         * @param begin First element considered.
         * @param end One past the last element considered.
         * @param best Best key so far; written when a better element is found.
         * @param best_idx Flat index of the best key.
         * @return T/F - the chunk held at least one element.
         */
        template<typename H>
        bool argmax_range_half(const H* data, const size_t begin, const size_t end, float64_t& best, uint64_t& best_idx) {
            auto has = false;
            for (auto i = begin; i < end; ++i) {
                const auto key = static_cast<float64_t>(widen_lane(data[i]));
                if (!has || key_better(key, best)) {
                    best = key;
                    best_idx = i;
                    has = true;
                }
            }
            return has;
        }

        /**
         * Lexicographic on the widened lanes, NaN-wins per lane.
         * @brief Is the candidate lane pair (clo, chi) better than (blo, bhi)?
         * @param clo Candidate low lane.
         * @param chi Candidate high lane.
         * @param blo Incumbent low lane.
         * @param bhi Incumbent high lane.
         * @return T/F - the candidate should replace the incumbent.
         */
        [[nodiscard]] inline bool pair_better(const float32_t clo, const float32_t chi, const float32_t blo, const float32_t bhi) {
            if (key_better(static_cast<float64_t>(clo), static_cast<float64_t>(blo))) {
                return true;
            }
            if (key_better(static_cast<float64_t>(blo), static_cast<float64_t>(clo))) {
                return false;
            }
            return key_better(static_cast<float64_t>(chi), static_cast<float64_t>(bhi));
        }

        /**
         * Per-element key is the widened (lo, hi) lane pair, lexicographic, NaN-wins per lane.
         * @brief Scalar range argmax over packed dtypes.
         * @param data Input elements.
         * @param begin First element considered.
         * @param end One past the last element considered.
         * @param best_lo Best low lane so far.
         * @param best_hi Best high lane so far.
         * @param best_idx Flat index of the best pair.
         * @return T/F - the chunk held at least one element.
         */
        template<typename P>
        bool argmax_range_packed(const P* data, const size_t begin, const size_t end,
                                 float32_t& best_lo, float32_t& best_hi, uint64_t& best_idx) {
            using H = packed_lane_t<P>;
            auto has = false;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(data[i]);
                const auto lo = widen_lane(lane_lo<H>(bits));
                const auto hi = widen_lane(lane_hi<H>(bits));
                if (!has || pair_better(lo, hi, best_lo, best_hi)) {
                    best_lo = lo;
                    best_hi = hi;
                    best_idx = i;
                    has = true;
                }
            }
            return has;
        }

        /**
         * Per-element key is the signed (lo, hi) lane pair widened to fp32 (every int4 value is exact), lexicographic.
         * No NaNs exist, so pair_better reduces to a plain ordered compare.
         * @brief Scalar range argmax over int4x2.
         * @param data Input elements.
         * @param begin First element considered.
         * @param end One past the last element considered.
         * @param best_lo Best low lane so far.
         * @param best_hi Best high lane so far.
         * @param best_idx Flat index of the best pair.
         * @return T/F - the chunk held at least one element.
         */
        inline bool argmax_range_i4x2(const int4x2_t* data, const size_t begin, const size_t end,
                                      float32_t& best_lo, float32_t& best_hi, uint64_t& best_idx) {
            auto has = false;
            for (auto i = begin; i < end; ++i) {
                const auto lo = static_cast<float32_t>(i4_lane_lo(data[i]));
                const auto hi = static_cast<float32_t>(i4_lane_hi(data[i]));
                if (!has || pair_better(lo, hi, best_lo, best_hi)) {
                    best_lo = lo;
                    best_hi = hi;
                    best_idx = i;
                    has = true;
                }
            }
            return has;
        }

        /**
         * Strict greater wins, so ties keep the lower index.
         * @brief Scalar range argmax over integer dtypes.
         * @param data Input elements.
         * @param begin First element considered.
         * @param end One past the last element considered.
         * @param best Best value so far; written when a larger element is found.
         * @param best_idx Flat index of the best value.
         * @return T/F - the chunk held at least one element.
         */
        template<typename T>
        bool argmax_range_int(const T* data, const size_t begin, const size_t end, T& best, uint64_t& best_idx) {
            auto has = false;
            for (auto i = begin; i < end; ++i) {
                if (!has || data[i] > best) {
                    best = data[i];
                    best_idx = i;
                    has = true;
                }
            }
            return has;
        }

        // --------------------------- launchers ---------------------------

        /**
         * Futures land on both tensors so teardown drains the workers (they hold raw buffer pointers).
         * Small inputs run inline. The worker is captured by value, the futures outlive this call.
         * @brief Threaded SGD launcher, chunks an elementwise two-operand range worker across the CPU thread pool.
         * @param worker Callable taking (begin, end).
         * @param numel Element count to cover.
         * @param w Weight tensor, receives the futures.
         * @param grad Gradient tensor, receives the futures.
         */
        template<typename Worker>
        void launch_sgd(Worker&& worker, const size_t numel, const Tensor& w, const Tensor& grad) {
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
            w.give_futures(futures);
            grad.give_futures(futures);
        }

        /**
         * Per-chunk partials are merged on the CALLER thread (runs in the Tensor method, never inside a pool worker,
         * so waiting is legal). Deterministic merge: partials combine in chunk order. Small inputs run inline.
         * @brief Threaded reduction launcher, chunks a reduction range worker across the pool.
         * @param numel Element count to cover.
         * @param range_fn Callable taking (slot, begin, end); slot indexes the caller's partials.
         * @param t Tensor receiving the futures.
         * @return Futures for the caller to drain.
         */
        template<typename RangeFn>
        std::vector<std::shared_future<void>> launch_reduce(const size_t numel, RangeFn&& range_fn, const Tensor& t) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                range_fn(0, 0, numel);
                return {};
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            auto slot = size_t{0};
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= numel) {
                    break;
                }
                // range_fn is captured by value; its own captures reference the
                // caller's partials, which stay alive until drain() returns.
                futures.push_back(threadMgr.submit([range_fn, slot, begin, end = std::min(begin + chunk, numel)] { range_fn(slot, begin, end); }, Threading::TaskPriority::Normal).share());
                ++slot;
            }
            t.give_futures(futures);
            return futures;
        }

        /**
         * @brief Drain futures on the caller thread.
         * @param futures Futures to wait on; rethrows the first stored worker error.
         */
        void drain(const std::vector<std::shared_future<void>>& futures) {
            for (const auto& f : futures) {
                f.get();
            }
        }

        /**
         * Threaded chunk partials merged on the caller thread, strict-greater so the first index wins ties.
         * Not a friend of Tensor: takes the base pointer and numel precomputed.
         * @brief Integer argmax over the whole tensor.
         * @param data Base pointer, offset already applied.
         * @param numel Element count.
         * @param t Tensor receiving the futures.
         * @return Flat index of the largest element.
         */
        template<typename T>
        [[nodiscard]] uint64_t argmax_int_typed(const T* data, const size_t numel, const Tensor& t) {
            const auto threads = static_cast<size_t>(IncludedCPUFeatures().threads);
            std::vector<std::pair<T, uint64_t>> partials(threads, {T{0}, 0});
            std::vector<char> has(threads, 0);
            drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                has[slot] = argmax_range_int<T>(data, b, e, partials[slot].first, partials[slot].second) ? 1 : 0;
            }, t));
            auto best = std::pair<T, uint64_t>{T{0}, 0};
            auto any = false;
            for (auto s = size_t{0}; s < partials.size(); ++s) {
                if (has[s] != 0 && (!any || partials[s].first > best.first)) {
                    best = partials[s];
                    any = true;
                }
            }
            return best.second;
        }

        /**
         * @brief Function pointer type shared by the scalar and SIMD SGD range kernels.
         */
        template<typename T>
        using SgdScalarFn = void(*)(T*, const T*, T, size_t, size_t);
    }

    /**
     * Dispatches on dtype: AVX512 > AVX2 > scalar for fp32/fp64, scalar paths for the rest.
     * Chunked across the CPU thread pool.
     * @brief CPU in-place SGD step, w -= lr * grad.
     * @param w Weights, updated in place.
     * @param grad Gradients, same shape and dtype as w.
     * @param lr Learning rate.
     */
    void sgd_update_cpu(Tensor& w, const Tensor& grad, const float64_t lr) {
        if (w.device != Device::CPU || grad.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(w.device));
        }
        using DType = DataTypes::DType;
        const auto numel = w.numel;
        switch (w.dtype) {
            case DType::Float32: {
                auto* w_ptr = static_cast<float32_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float32_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                const SgdScalarFn<float32_t> kernel = Math::BasicOps::basicops_avx512_available()
                    ? Math::BasicOps::sgd_f32_range_avx512
                    : (Math::BasicOps::basicops_avx2_available() ? Math::BasicOps::sgd_f32_range_avx2 : sgd_range_scalar<float32_t>);
                launch_sgd([=](const size_t b, const size_t e) { kernel(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Float64: {
                auto* w_ptr = static_cast<float64_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float64_t*>(grad.data) + grad.offset;
                const SgdScalarFn<float64_t> kernel = Math::BasicOps::basicops_avx512_available()
                    ? Math::BasicOps::sgd_f64_range_avx512
                    : (Math::BasicOps::basicops_avx2_available() ? Math::BasicOps::sgd_f64_range_avx2 : sgd_range_scalar<float64_t>);
                launch_sgd([=](const size_t b, const size_t e) { kernel(w_ptr, g_ptr, -lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Float16: {
                auto* w_ptr = static_cast<float16_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float16_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_half<float16_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::BFloat16: {
                auto* w_ptr = static_cast<bfloat16_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const bfloat16_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_half<bfloat16_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Float16x2: {
                auto* w_ptr = static_cast<float16x2_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const float16x2_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_packed<float16x2_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::BFloat16x2: {
                auto* w_ptr = static_cast<bfloat16x2_t*>(w.data) + w.offset;
                const auto* g_ptr = static_cast<const bfloat16x2_t*>(grad.data) + grad.offset;
                const auto neg_lr = static_cast<float32_t>(-lr);
                launch_sgd([=](const size_t b, const size_t e) { sgd_range_packed<bfloat16x2_t>(w_ptr, g_ptr, neg_lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int8: {
                launch_sgd([w_ptr = static_cast<int8_t*>(w.data) + w.offset, g_ptr = static_cast<const int8_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int8_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt8: {
                launch_sgd([w_ptr = static_cast<uint8_t*>(w.data) + w.offset, g_ptr = static_cast<const uint8_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint8_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int16: {
                launch_sgd([w_ptr = static_cast<int16_t*>(w.data) + w.offset, g_ptr = static_cast<const int16_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int16_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt16: {
                launch_sgd([w_ptr = static_cast<uint16_t*>(w.data) + w.offset, g_ptr = static_cast<const uint16_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint16_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int32: {
                launch_sgd([w_ptr = static_cast<int32_t*>(w.data) + w.offset, g_ptr = static_cast<const int32_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int32_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt32: {
                launch_sgd([w_ptr = static_cast<uint32_t*>(w.data) + w.offset, g_ptr = static_cast<const uint32_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint32_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int64: {
                launch_sgd([w_ptr = static_cast<int64_t*>(w.data) + w.offset, g_ptr = static_cast<const int64_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<int64_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::UInt64: {
                launch_sgd([w_ptr = static_cast<uint64_t*>(w.data) + w.offset, g_ptr = static_cast<const uint64_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_int<uint64_t>(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            case DType::Int4x2: {
                launch_sgd([w_ptr = static_cast<int4x2_t*>(w.data) + w.offset, g_ptr = static_cast<const int4x2_t*>(grad.data) + grad.offset, lr](const size_t b, const size_t e) { sgd_range_i4x2(w_ptr, g_ptr, lr, b, e); }, numel, w, grad);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Dispatches on dtype; per-chunk partials are merged in chunk order, then narrowed to the output dtype.
     * @brief CPU full reduction, sums every element into the scalar output tensor.
     * @param t Input tensor.
     * @param out Single element output tensor, same dtype as t.
     */
    void sum_cpu(const Tensor& t, Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        switch (t.dtype) {
            case DType::Float32: {
                const auto* data = static_cast<const float32_t*>(t.data) + t.offset;
                using SumFn = float32_t(*)(const float32_t*, size_t, size_t);
                const SumFn kernel = Math::BasicOps::basicops_avx512_available()
                    ? Math::BasicOps::sum_f32_range_avx512
                    : (Math::BasicOps::basicops_avx2_available() ? Math::BasicOps::sum_f32_range_avx2 : sum_range_scalar<float32_t>);
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = kernel(data, b, e); }, t));
                auto total = 0.0f;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<float32_t*>(out.data) = total;
                break;
            }
            case DType::Float64: {
                const auto* data = static_cast<const float64_t*>(t.data) + t.offset;
                using SumFn = float64_t(*)(const float64_t*, size_t, size_t);
                const SumFn kernel = Math::BasicOps::basicops_avx512_available()
                    ? Math::BasicOps::sum_f64_range_avx512
                    : (Math::BasicOps::basicops_avx2_available() ? Math::BasicOps::sum_f64_range_avx2 : sum_range_scalar<float64_t>);
                std::vector<float64_t> partials(IncludedCPUFeatures().threads, 0.0);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = kernel(data, b, e); }, t));
                auto total = 0.0;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<float64_t*>(out.data) = total;
                break;
            }
            case DType::Float16: {
                const auto* data = static_cast<const float16_t*>(t.data) + t.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = sum_range_half<float16_t>(data, b, e); }, t));
                auto total = 0.0f;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<float16_t*>(out.data) = narrow_lane<float16_t>(total);
                break;
            }
            case DType::BFloat16: {
                const auto* data = static_cast<const bfloat16_t*>(t.data) + t.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = sum_range_half<bfloat16_t>(data, b, e); }, t));
                auto total = 0.0f;
                for (const auto p : partials) {
                    total += p;
                }
                *static_cast<bfloat16_t*>(out.data) = narrow_lane<bfloat16_t>(total);
                break;
            }
            case DType::Float16x2: {
                const auto* data = static_cast<const float16x2_t*>(t.data) + t.offset;
                std::vector<std::pair<float32_t, float32_t>> partials(IncludedCPUFeatures().threads, {0.0f, 0.0f});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = sum_range_packed<float16x2_t>(data, b, e); }, t));
                auto total_lo = 0.0f;
                auto total_hi = 0.0f;
                for (const auto& p : partials) {
                    total_lo += p.first;
                    total_hi += p.second;
                }
                *static_cast<float16x2_t*>(out.data) = pack_lanes<float16x2_t>(
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<float16_t>(total_lo)),
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<float16_t>(total_hi)));
                break;
            }
            case DType::BFloat16x2: {
                const auto* data = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                std::vector<std::pair<float32_t, float32_t>> partials(IncludedCPUFeatures().threads, {0.0f, 0.0f});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = sum_range_packed<bfloat16x2_t>(data, b, e); }, t));
                auto total_lo = 0.0f;
                auto total_hi = 0.0f;
                for (const auto& p : partials) {
                    total_lo += p.first;
                    total_hi += p.second;
                }
                *static_cast<bfloat16x2_t*>(out.data) = pack_lanes<bfloat16x2_t>(
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<bfloat16_t>(total_lo)),
                    Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<bfloat16_t>(total_hi)));
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                std::vector<uint64_t> partials(IncludedCPUFeatures().threads, uint64_t{0});
                /**
                 * Runs sum_range_int<T> over every chunk and drains the futures.
                 * @brief Threaded integer sum for one concrete element type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* data) {
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = sum_range_int<T>(data, b, e); }, t));
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
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                auto total = uint64_t{0};
                for (const auto p : partials) {
                    total += p;
                }
                switch (t.dtype) {
                    case DType::Int8: *static_cast<int8_t*>(out.data) = static_cast<int8_t>(total); break;
                    case DType::UInt8: *static_cast<uint8_t*>(out.data) = static_cast<uint8_t>(total); break;
                    case DType::Int16: *static_cast<int16_t*>(out.data) = static_cast<int16_t>(total); break;
                    case DType::UInt16: *static_cast<uint16_t*>(out.data) = static_cast<uint16_t>(total); break;
                    case DType::Int32: *static_cast<int32_t*>(out.data) = static_cast<int32_t>(total); break;
                    case DType::UInt32: *static_cast<uint32_t*>(out.data) = static_cast<uint32_t>(total); break;
                    case DType::Int64: *static_cast<int64_t*>(out.data) = static_cast<int64_t>(total); break;
                    case DType::UInt64: *static_cast<uint64_t*>(out.data) = static_cast<uint64_t>(total); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            case DType::Int4x2: {
                const auto* data = static_cast<const int4x2_t*>(t.data) + t.offset;
                std::vector<std::pair<uint64_t, uint64_t>> partials(IncludedCPUFeatures().threads, {uint64_t{0}, uint64_t{0}});
                drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = sum_range_i4x2(data, b, e); }, t));
                auto total_lo = uint64_t{0};
                auto total_hi = uint64_t{0};
                for (const auto& p : partials) {
                    total_lo += p.first;
                    total_hi += p.second;
                }
                *static_cast<int4x2_t*>(out.data) = i4_pack(static_cast<int8_t>(total_lo & 0x0Fu), static_cast<int8_t>(total_hi & 0x0Fu));
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }

    /**
     * Dispatches on dtype; per-chunk bests are merged in chunk order with PyTorch-compatible NaN rules.
     * @brief CPU flat index of the largest element.
     * @param t Input tensor.
     * @param result Flat index of the winner.
     */
    void argmax_cpu(const Tensor& t, uint64_t& result) {
        if (t.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        struct FloatPart { float64_t key; uint64_t idx; bool has; };
        struct PackedPart { float32_t lo; float32_t hi; uint64_t idx; bool has; };
        switch (t.dtype) {
            case DType::Float32: case DType::Float64: case DType::Float16: case DType::BFloat16: {
                std::vector<FloatPart> partials(IncludedCPUFeatures().threads, {0.0, 0, false});
                /**
                 * Picks the scalar or half argmax range kernel by T and drains the futures.
                 * @brief Threaded float argmax for one concrete element type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_f = [&]<typename T>(const T* data) {
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                        if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                            partials[slot].has = argmax_range_scalar<T>(data, b, e, partials[slot].key, partials[slot].idx);
                        } else {
                            partials[slot].has = argmax_range_half<T>(data, b, e, partials[slot].key, partials[slot].idx);
                        }
                    }, t));
                };
                switch (t.dtype) {
                    case DType::Float32: launch_f(static_cast<const float32_t*>(t.data) + t.offset); break;
                    case DType::Float64: launch_f(static_cast<const float64_t*>(t.data) + t.offset); break;
                    case DType::Float16: launch_f(static_cast<const float16_t*>(t.data) + t.offset); break;
                    case DType::BFloat16: launch_f(static_cast<const bfloat16_t*>(t.data) + t.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                auto best = FloatPart{0.0, 0, false};
                for (const auto& p : partials) {
                    if (p.has && (!best.has || key_better(p.key, best.key))) {
                        best = p;
                    }
                }
                result = best.idx;
                break;
            }
            case DType::Float16x2: case DType::BFloat16x2: case DType::Int4x2: {
                std::vector<PackedPart> partials(IncludedCPUFeatures().threads, {0.0f, 0.0f, 0, false});
                /**
                 * Runs argmax_range_packed<P> over every chunk and drains the futures.
                 * @brief Threaded packed argmax for one concrete packed type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_p = [&]<typename P>(const P* data) {
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                        partials[slot].has = argmax_range_packed<P>(data, b, e, partials[slot].lo, partials[slot].hi, partials[slot].idx);
                    }, t));
                };
                if (t.dtype == DType::Float16x2) {
                    launch_p(static_cast<const float16x2_t*>(t.data) + t.offset);
                } else if (t.dtype == DType::BFloat16x2) {
                    launch_p(static_cast<const bfloat16x2_t*>(t.data) + t.offset);
                } else {
                    const auto* data = static_cast<const int4x2_t*>(t.data) + t.offset;
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                        partials[slot].has = argmax_range_i4x2(data, b, e, partials[slot].lo, partials[slot].hi, partials[slot].idx);
                    }, t));
                }
                auto best = PackedPart{0.0f, 0.0f, 0, false};
                for (const auto& p : partials) {
                    if (p.has && (!best.has || pair_better(p.lo, p.hi, best.lo, best.hi))) {
                        best = p;
                    }
                }
                result = best.idx;
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                switch (t.dtype) {
                    case DType::Int8: result = argmax_int_typed<int8_t>(static_cast<const int8_t*>(t.data) + t.offset, numel, t); break;
                    case DType::UInt8: result = argmax_int_typed<uint8_t>(static_cast<const uint8_t*>(t.data) + t.offset, numel, t); break;
                    case DType::Int16: result = argmax_int_typed<int16_t>(static_cast<const int16_t*>(t.data) + t.offset, numel, t); break;
                    case DType::UInt16: result = argmax_int_typed<uint16_t>(static_cast<const uint16_t*>(t.data) + t.offset, numel, t); break;
                    case DType::Int32: result = argmax_int_typed<int32_t>(static_cast<const int32_t*>(t.data) + t.offset, numel, t); break;
                    case DType::UInt32: result = argmax_int_typed<uint32_t>(static_cast<const uint32_t*>(t.data) + t.offset, numel, t); break;
                    case DType::Int64: result = argmax_int_typed<int64_t>(static_cast<const int64_t*>(t.data) + t.offset, numel, t); break;
                    case DType::UInt64: result = argmax_int_typed<uint64_t>(static_cast<const uint64_t*>(t.data) + t.offset, numel, t); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
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
     * Metadata-only: the result is a view over the same buffer, requires contiguous storage.
     * @brief Reshape into a view with a new shape of equal element count.
     * @param new_shape Target shape, 1..BT_TENSOR_MAX_DIMS dimensions.
     * @return View tensor sharing this tensor's buffer.
     */
    Tensor Tensor::reshape(const std::vector<size_t>& new_shape) {
        sync();
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        if (new_shape.empty() || new_shape.size() > BT_TENSOR_MAX_DIMS) {
            throw ErrorHandling::MismatchedShapes("reshape rank must be 1.." + std::to_string(BT_TENSOR_MAX_DIMS));
        }
        auto new_numel = size_t{1};
        for (const auto d : new_shape) {
            new_numel *= d;
        }
        if (new_numel != this->numel) {
            throw ErrorHandling::MismatchedShapes("reshape changes numel (" + std::to_string(this->numel) + " vs " + std::to_string(new_numel) + ")");
        }
        // Build a same-buffer view. The constructor allocates a fresh buffer;
        // release it while owns_data is still true, then re-point at ours.
        auto view = Tensor(new_shape, this->dtype, this->device, this->requires_grad);
        view.kill_cpu_data();
        view.kill_gpu_data();
        view.owns_data = false;
        view.data = this->data;
        this->children.emplace_back(&view);
        view.view_parents.emplace_back(this);
        return view;
    }

    /**
     * Both tensors are synced before dispatch; requires identical shape, dtype, device, contiguous.
     * @brief In-place SGD step, this -= lr * grad.
     * @param grad Gradient tensor.
     * @param lr Learning rate.
     */
    void Tensor::sgd_update(const Tensor& grad, const float64_t lr) {
        if (!this->is_same_shape(grad)) {
            throw ErrorHandling::MismatchedShapes("sgd_update requires identical shapes");
        }
        if (this->dtype != grad.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->device != grad.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (!this->is_contiguous || !grad.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        grad.sync();
        if (this->device == Device::CPU) {
            detail::sgd_update_cpu(*this, grad, lr);
        } else if (this->device == Device::GPU) {
            detail::sgd_update_gpu(*this, grad, lr);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * Uses the gradient attached to this tensor.
     * @brief In-place SGD step from this tensor's own gradient.
     * @param lr Learning rate.
     */
    void Tensor::sgd_step(const float64_t lr) {
        if (!this->requires_grad || this->grad == nullptr) {
            throw ErrorHandling::NoGradient();
        }
        sgd_update(*this->grad, lr);
    }

    /**
     * @brief Sum of every element.
     * @return Single element tensor of the same dtype and device.
     */
    Tensor Tensor::sum() const {
        sync();
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::sum_cpu(*this, out);
        } else if (this->device == Device::GPU) {
            detail::sum_gpu(*this, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    /**
     * Requires a non-empty contiguous tensor.
     * @brief Flat index of the largest element.
     * @return Flat index of the winner.
     */
    uint64_t Tensor::argmax() const {
        sync();
        if (this->numel == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        auto result = uint64_t{0};
        if (this->device == Device::CPU) {
            detail::argmax_cpu(*this, result);
        } else if (this->device == Device::GPU) {
            detail::argmax_gpu(*this, result);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return result;
    }
}
