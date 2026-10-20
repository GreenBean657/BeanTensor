/**
 * @file SumCPU.cpp
 * @author GreenBean
 * @brief CPU backend for sum. Hosts the Tensor::sum method definition
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

#include "Math/BasicOps/Sum/Sum.h"
#include "Math/BasicOps/BasicOpsShared.h"
#include "Math/BasicOps/Sum/SumKernels.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace basicops_shared;

    namespace {
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
                const SumFn kernel = Math::BasicOps::sum_avx512_available()
                    ? Math::BasicOps::sum_f32_range_avx512
                    : (Math::BasicOps::sum_avx2_available() ? Math::BasicOps::sum_f32_range_avx2 : sum_range_scalar<float32_t>);
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
                const SumFn kernel = Math::BasicOps::sum_avx512_available()
                    ? Math::BasicOps::sum_f64_range_avx512
                    : (Math::BasicOps::sum_avx2_available() ? Math::BasicOps::sum_f64_range_avx2 : sum_range_scalar<float64_t>);
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
}

namespace BeanTensor::Tensors {

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
}
