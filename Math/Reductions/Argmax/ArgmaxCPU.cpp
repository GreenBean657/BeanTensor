/**
 * @file ArgmaxCPU.cpp
 * @author GreenBean
 * @brief CPU backend for argmax. Hosts the Tensor::argmax method definition
 * (mirrors how Tensor::matmul lives in MatmulCPU.cpp).
 * @version 0.6
 * @date 2026-10-26
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>
#include <vector>

#include "Math/Reductions/Argmax/Argmax.h"
#include "Math/Reductions/ReductionsShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace reductions_shared;

    namespace {
        /**
         * PyTorch-compatible NaN rule: NaN beats every non-NaN value; the FIRST NaN wins against later NaNs;
         * otherwise strict greater wins, so ties keep the lower index.
         * @brief Is the candidate key better than the incumbent? (scalar keys)
         * @param cand Candidate key.
         * @param best Incumbent key.
         * @return T/F - the candidate should replace the incumbent.
         */
        [[nodiscard]] bool key_better(const float64_t cand, const float64_t best) {
            if (__builtin_isnan(cand)) {
                return !__builtin_isnan(best);
            }
            return !__builtin_isnan(best) && cand > best;
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
                if (
                    const auto key = static_cast<float64_t>(data[i]);
                    !has || key_better(key, best)
                    ) {
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
                if (
                    const auto key = static_cast<float64_t>(widen_lane(data[i]));
                    !has || key_better(key, best)
                    ) {
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
        [[nodiscard]] bool pair_better(const float32_t clo, const float32_t chi, const float32_t blo, const float32_t bhi) {
            // Widen once, explicitly: the implicit float32 -> float64 promotion at the call sites trips -Werror=double-promotion.
            const auto clo64 = static_cast<float64_t>(clo);
            const auto chi64 = static_cast<float64_t>(chi);
            const auto blo64 = static_cast<float64_t>(blo);
            const auto bhi64 = static_cast<float64_t>(bhi);
            if (key_better(clo64, blo64)) {
                return true;
            }
            if (key_better(blo64, clo64)) {
                return false;
            }
            return key_better(chi64, bhi64);
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
        bool argmax_range_i4x2(const int4x2_t* data, const size_t begin, const size_t end,
                                      float32_t& best_lo, float32_t& best_hi, uint64_t& best_idx) {
            auto has = false;
            for (auto i = begin; i < end; ++i) {
                const auto lo = static_cast<float32_t>(i4_lane_lo(data[i]));
                if (
                    const auto hi = static_cast<float32_t>(i4_lane_hi(data[i]));
                    !has || pair_better(lo, hi, best_lo, best_hi)
                    ) {
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
            std::vector<uint8_t> has(threads, 0);
            drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                has[slot] = argmax_range_int<T>(data, b, e, partials[slot].first, partials[slot].second) ? 1 : 0;
            }, t, numel * sizeof(T)));
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

        // --------------------------- axis argmax kernels ---------------------------

        /**
         * Chunks an output-position range worker across the CPU thread pool. Futures land on the
         * input and output tensors so teardown drains them (workers hold raw buffer pointers).
         * Small outputs run inline. Mirrors linearalgebra_shared::launch_elementwise.
         * CCD-aware: axis argmax is a reduction — the output indices are tiny but every
         * worker reads the whole input span (each output position scans axis_len elements),
         * and cross-CCD reads are the cheap direction (~15-20% vs ~8x for writes), so the
         * gate is the INPUT span, not the output. Chunks are restricted to the submitting
         * thread's CCD group only when the input fits one CCD's LLC (that CCD typically
         * produced or last warmed the input); larger inputs stay on the whole pool for
         * bandwidth. The chunk count still assumes all t_count threads — group workers
         * simply claim multiple chunks from the queue.
         * @brief Threaded axis-argmax launcher with CCD-local routing by input span.
         * @param worker Callable taking (begin, end) over output positions; copied into each task.
         * @param out_numel Output element count to cover.
         * @param input_bytes Size in bytes of the input span the workers read.
         * @param t Input tensor the workers read.
         * @param out Output tensor the workers write.
         */
        template<typename Worker>
        void launch_axis(Worker&& worker, const size_t out_numel, const size_t input_bytes, const Tensor& t, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || out_numel <= t_count * 1024) {
                worker(0, out_numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (out_numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(input_bytes);
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            for (auto i = size_t{0}; i < t_count; ++i) {
                const auto begin = i * chunk;
                if (begin >= out_numel) {
                    break;
                }
                // worker is captured BY VALUE: these futures outlive this call
                // (they land on the tensors), so a reference would dangle.
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group([worker, begin, end = std::min(begin + chunk, out_numel)] { worker(begin, end); }, group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, out_numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
                }
            }
            t.give_futures(futures);
            out.give_futures(futures);
        }

        /**
         * Output position o decomposes as (outer, in_off) = (o / inner, o % inner) with the scan
         * base at outer * axis_len * inner + in_off. Keys rank by key_better, so the first
         * strict maximum wins ties and the first NaN beats every non-NaN value.
         * @brief Range axis argmax over fp32/fp64/fp16/bf16 (fp64 keys).
         * @param data Input elements.
         * @param out Output int64 indices.
         * @param axis_len Length of the reduced axis.
         * @param inner Stride of the reduced axis (product of the trailing dims).
         * @param begin First output position computed.
         * @param end One past the last output position computed.
         */
        template<typename T>
        void argmax_axis_range_float(const T* data, int64_t* out, const size_t axis_len, const size_t inner, const size_t begin, const size_t end) {
            for (auto o = begin; o < end; ++o) {
                const auto base = o / inner * (axis_len * inner) + o % inner;
                auto best_idx = size_t{0};
                auto best = float64_t{0.0};
                for (auto i = size_t{0}; i < axis_len; ++i) {
                    float64_t key;
                    if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                        key = static_cast<float64_t>(data[base + i * inner]);
                    } else {
                        key = static_cast<float64_t>(widen_lane(data[base + i * inner]));
                    }
                    if (i == 0 || key_better(key, best)) {
                        best = key;
                        best_idx = i;
                    }
                }
                out[o] = static_cast<int64_t>(best_idx);
            }
        }

        /**
         * Per-element key is the widened (lo, hi) lane pair, lexicographic, NaN-wins per lane —
         * the same ranking as the flat packed argmax.
         * @brief Range axis argmax over fp16x2/bf16x2.
         * @param data Input elements.
         * @param out Output int64 indices.
         * @param axis_len Length of the reduced axis.
         * @param inner Stride of the reduced axis (product of the trailing dims).
         * @param begin First output position computed.
         * @param end One past the last output position computed.
         */
        template<typename P>
        void argmax_axis_range_packed(const P* data, int64_t* out, const size_t axis_len, const size_t inner, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto o = begin; o < end; ++o) {
                const auto base = o / inner * (axis_len * inner) + o % inner;
                auto best_idx = size_t{0};
                auto best_lo = float32_t{0.0f};
                auto best_hi = float32_t{0.0f};
                for (auto i = size_t{0}; i < axis_len; ++i) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(data[base + i * inner]);
                    const auto lo = widen_lane(lane_lo<H>(bits));
                    const auto hi = widen_lane(lane_hi<H>(bits));
                    if (i == 0 || pair_better(lo, hi, best_lo, best_hi)) {
                        best_lo = lo;
                        best_hi = hi;
                        best_idx = i;
                    }
                }
                out[o] = static_cast<int64_t>(best_idx);
            }
        }

        /**
         * Per-element key is the signed (lo, hi) nibble pair widened to fp32 (exact), lexicographic.
         * @brief Range axis argmax over int4x2.
         * @param data Input elements.
         * @param out Output int64 indices.
         * @param axis_len Length of the reduced axis.
         * @param inner Stride of the reduced axis (product of the trailing dims).
         * @param begin First output position computed.
         * @param end One past the last output position computed.
         */
        void argmax_axis_range_i4x2(const int4x2_t* data, int64_t* out, const size_t axis_len, const size_t inner, const size_t begin, const size_t end) {
            for (auto o = begin; o < end; ++o) {
                const auto base = o / inner * (axis_len * inner) + o % inner;
                auto best_idx = size_t{0};
                auto best_lo = float32_t{0.0f};
                auto best_hi = float32_t{0.0f};
                for (auto i = size_t{0}; i < axis_len; ++i) {
                    const auto lo = static_cast<float32_t>(i4_lane_lo(data[base + i * inner]));
                    if (
                        const auto hi = static_cast<float32_t>(i4_lane_hi(data[base + i * inner]));
                        i == 0 || pair_better(lo, hi, best_lo, best_hi)
                        ) {
                        best_lo = lo;
                        best_hi = hi;
                        best_idx = i;
                    }
                }
                out[o] = static_cast<int64_t>(best_idx);
            }
        }

        /**
         * Strict greater wins, so ties keep the lower axis index.
         * @brief Range axis argmax over integer dtypes.
         * @param data Input elements.
         * @param out Output int64 indices.
         * @param axis_len Length of the reduced axis.
         * @param inner Stride of the reduced axis (product of the trailing dims).
         * @param begin First output position computed.
         * @param end One past the last output position computed.
         */
        template<typename T>
        void argmax_axis_range_int(const T* data, int64_t* out, const size_t axis_len, const size_t inner, const size_t begin, const size_t end) {
            for (auto o = begin; o < end; ++o) {
                const auto base = o / inner * (axis_len * inner) + o % inner;
                auto best_idx = size_t{0};
                auto best = data[base];
                for (auto i = size_t{1}; i < axis_len; ++i) {
                    if (data[base + i * inner] > best) {
                        best = data[base + i * inner];
                        best_idx = i;
                    }
                }
                out[o] = static_cast<int64_t>(best_idx);
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
                std::vector<FloatPart> partials(IncludedCPUFeatures().threads, {.key = 0.0, .idx = 0, .has = false});
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
                    }, t, numel * sizeof(T)));
                };
                switch (t.dtype) {
                    case DType::Float32: launch_f(static_cast<const float32_t*>(t.data) + t.offset); break;
                    case DType::Float64: launch_f(static_cast<const float64_t*>(t.data) + t.offset); break;
                    case DType::Float16: launch_f(static_cast<const float16_t*>(t.data) + t.offset); break;
                    case DType::BFloat16: launch_f(static_cast<const bfloat16_t*>(t.data) + t.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                auto best = FloatPart{.key = 0.0, .idx = 0, .has = false};
                for (const auto& p : partials) {
                    if (p.has && (!best.has || key_better(p.key, best.key))) {
                        best = p;
                    }
                }
                result = best.idx;
                break;
            }
            case DType::Float16x2: case DType::BFloat16x2: case DType::Int4x2: {
                std::vector<PackedPart> partials(IncludedCPUFeatures().threads, {.lo = 0.0f, .hi = 0.0f, .idx = 0, .has = false});
                /**
                 * Runs argmax_range_packed<P> over every chunk and drains the futures.
                 * @brief Threaded packed argmax for one concrete packed type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_p = [&]<typename P>(const P* data) {
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                        partials[slot].has = argmax_range_packed<P>(data, b, e, partials[slot].lo, partials[slot].hi, partials[slot].idx);
                    }, t, numel * sizeof(P)));
                };
                if (t.dtype == DType::Float16x2) {
                    launch_p(static_cast<const float16x2_t*>(t.data) + t.offset);
                } else if (t.dtype == DType::BFloat16x2) {
                    launch_p(static_cast<const bfloat16x2_t*>(t.data) + t.offset);
                } else {
                    const auto* data = static_cast<const int4x2_t*>(t.data) + t.offset;
                    drain(launch_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) {
                        partials[slot].has = argmax_range_i4x2(data, b, e, partials[slot].lo, partials[slot].hi, partials[slot].idx);
                    }, t, numel * sizeof(int4x2_t)));
                }
                auto best = PackedPart{.lo = 0.0f, .hi = 0.0f, .idx = 0, .has = false};
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

    /**
     * Dispatches on dtype; every output position scans the reduced axis independently, so chunk
     * partials need no merge. axis_len/inner come from the contiguous strides.
     * @brief CPU argmax along one axis, written as int64 indices into the output tensor.
     * @param t Input tensor.
     * @param out Output tensor (input shape minus the axis), Int64 dtype.
     * @param axis Axis to reduce.
     */
    void argmax_axis_cpu(const Tensor& t, Tensor& out, const size_t axis) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto axis_len = t.shape[axis];
        const auto inner = t.dims[axis];
        const auto out_numel = out.numel;
        auto* out_data = static_cast<int64_t*>(out.data);
        switch (t.dtype) {
            case DType::Float32: case DType::Float64: case DType::Float16: case DType::BFloat16: {
                /**
                 * Launches argmax_axis_range_float<T> over the output positions.
                 * @brief Threaded float axis argmax for one concrete element type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_f = [&]<typename T>(const T* data) {
                    launch_axis([data, out_data, axis_len, inner](const size_t b, const size_t e) { argmax_axis_range_float<T>(data, out_data, axis_len, inner, b, e); }, out_numel, t.numel * sizeof(T), t, out);
                };
                switch (t.dtype) {
                    case DType::Float32: launch_f(static_cast<const float32_t*>(t.data) + t.offset); break;
                    case DType::Float64: launch_f(static_cast<const float64_t*>(t.data) + t.offset); break;
                    case DType::Float16: launch_f(static_cast<const float16_t*>(t.data) + t.offset); break;
                    case DType::BFloat16: launch_f(static_cast<const bfloat16_t*>(t.data) + t.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                break;
            }
            case DType::Float16x2: case DType::BFloat16x2: case DType::Int4x2: {
                /**
                 * Launches the packed/i4x2 axis range kernel over the output positions.
                 * @brief Threaded packed axis argmax for one concrete packed type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_p = [&]<typename P>(const P* data) {
                    launch_axis([data, out_data, axis_len, inner](const size_t b, const size_t e) { argmax_axis_range_packed<P>(data, out_data, axis_len, inner, b, e); }, out_numel, t.numel * sizeof(P), t, out);
                };
                if (t.dtype == DType::Float16x2) {
                    launch_p(static_cast<const float16x2_t*>(t.data) + t.offset);
                } else if (t.dtype == DType::BFloat16x2) {
                    launch_p(static_cast<const bfloat16x2_t*>(t.data) + t.offset);
                } else {
                    const auto* data = static_cast<const int4x2_t*>(t.data) + t.offset;
                    launch_axis([data, out_data, axis_len, inner](const size_t b, const size_t e) { argmax_axis_range_i4x2(data, out_data, axis_len, inner, b, e); }, out_numel, t.numel * sizeof(int4x2_t), t, out);
                }
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                /**
                 * Launches argmax_axis_range_int<T> over the output positions.
                 * @brief Threaded integer axis argmax for one concrete element type.
                 * @param data Base pointer, offset already applied.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_i = [&]<typename T>(const T* data) {
                    launch_axis([data, out_data, axis_len, inner](const size_t b, const size_t e) { argmax_axis_range_int<T>(data, out_data, axis_len, inner, b, e); }, out_numel, t.numel * sizeof(T), t, out);
                };
                switch (t.dtype) {
                    case DType::Int8: launch_i(static_cast<const int8_t*>(t.data) + t.offset); break;
                    case DType::UInt8: launch_i(static_cast<const uint8_t*>(t.data) + t.offset); break;
                    case DType::Int16: launch_i(static_cast<const int16_t*>(t.data) + t.offset); break;
                    case DType::UInt16: launch_i(static_cast<const uint16_t*>(t.data) + t.offset); break;
                    case DType::Int32: launch_i(static_cast<const int32_t*>(t.data) + t.offset); break;
                    case DType::UInt32: launch_i(static_cast<const uint32_t*>(t.data) + t.offset); break;
                    case DType::Int64: launch_i(static_cast<const int64_t*>(t.data) + t.offset); break;
                    case DType::UInt64: launch_i(static_cast<const uint64_t*>(t.data) + t.offset); break;
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

    /**
     * Requires a non-empty contiguous tensor and axis < ndim. The output holds one int64 index
     * per slice orthogonal to the axis, in the input shape with the axis removed (a rank-1 input
     * reduces to shape {1}). The compute is queued asynchronously, but the returned tensor is
     * moved out, and the move drains pending work, so argmax(axis) effectively blocks before
     * returning (same as matmul()); a drained failure is re-queued on the result as a poisoned
     * future, surfacing at its next sync().
     * @brief Index of the largest element along one axis.
     * @param axis Axis to reduce.
     * @return Int64 tensor of per-slice argmax indices.
     */
    Tensor Tensor::argmax(const size_t axis) const {
        sync();
        if (axis >= this->ndim) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (this->numel == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        if (!this->is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        std::vector<size_t> out_shape;
        out_shape.reserve(this->ndim);
        for (auto i = size_t{0}; i < this->ndim; ++i) {
            if (i != axis) {
                out_shape.push_back(this->shape[i]);
            }
        }
        if (out_shape.empty()) {
            // A rank-1 input reduces to a scalar index, held as shape {1}.
            out_shape.push_back(1);
        }
        auto out = Tensor(out_shape, DataTypes::DType::Int64, this->device, false);
        if (this->device == Device::CPU) {
            detail::argmax_axis_cpu(*this, out, axis);
        } else if (this->device == Device::GPU) {
            detail::argmax_axis_gpu(*this, out, axis);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return out;
    }

    /**
     * Lane-aware argmax over packed x2 logits. Element j holds two classes: its
     * low lane is class 2j and its high lane class 2j+1, the layout softmax_ce
     * trains. Per row, the largest lane wins; the first strict maximum wins
     * ties, and NaN never wins. CPU-only, rank-2 {rows, classes}, axis 1.
     * @brief Index of the largest class lane per row of packed x2 logits.
     * @param axis Must be 1.
     * @return Int64 tensor of shape {rows} holding class indices in [0, 2*classes).
     * @throws ErrorHandling::MismatchedDTypes The tensor is not Float16x2 or BFloat16x2.
     * @throws ErrorHandling::NotImplemented Not rank-2, axis is not 1, or the tensor is on the GPU.
     * @throws ErrorHandling::IndexOutOfRange The tensor is empty.
     */
    Tensor Tensor::argmax_lanes(const size_t axis) {
        namespace shared = detail::reductions_shared;
        sync();
        const auto is_half = this->dtype == DataTypes::DType::Float16x2;
        if (!is_half && this->dtype != DataTypes::DType::BFloat16x2) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (this->ndim != 2 || axis != 1 || this->device != Device::CPU) {
            throw ErrorHandling::NotImplemented();
        }
        if (this->numel == 0) {
            throw ErrorHandling::IndexOutOfRange();
        }
        const auto rows = this->shape[0];
        const auto cols = this->shape[1];
        auto out = Tensor(std::vector<size_t>{rows}, DataTypes::DType::Int64, this->device, false);
        for (auto r = size_t{0}; r < rows; ++r) {
            auto has = false;
            auto best = float32_t{0};
            auto best_class = size_t{0};
            for (auto c = size_t{0}; c < cols; ++c) {
                auto bits = uint32_t{0};
                this->read_element(this->checked_element_offset({r, c}), &bits, sizeof(bits));
                for (auto lane = size_t{0}; lane < 2; ++lane) {
                    const auto value = is_half
                        ? (lane == 0 ? shared::widen_lane(shared::lane_lo<float16_t>(bits)) : shared::widen_lane(shared::lane_hi<float16_t>(bits)))
                        : (lane == 0 ? shared::widen_lane(shared::lane_lo<bfloat16_t>(bits)) : shared::widen_lane(shared::lane_hi<bfloat16_t>(bits)));
                    if (std::isnan(value)) {
                        continue;
                    }
                    const auto class_index = 2 * c + lane;
                    if (!has || value > best) {
                        has = true;
                        best = value;
                        best_class = class_index;
                    }
                }
            }
            out.set<int64_t>({r}, static_cast<int64_t>(best_class));
        }
        return out;
    }
}
