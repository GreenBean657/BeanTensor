/**
 * @file ArgmaxCPU.cpp
 * @author GreenBean
 * @brief CPU backend for argmax. Hosts the Tensor::argmax method definition
 * (mirrors how Tensor::matmul lives in MatmulCPU.cpp).
 * @version 0.1
 * @date 2026-10-20
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>
#include <vector>

#include "Math/BasicOps/Argmax/Argmax.h"
#include "Math/BasicOps/BasicOpsShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace basicops_shared;

    namespace {
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
