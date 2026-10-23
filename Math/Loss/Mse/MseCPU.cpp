/**
 * @file MseCPU.cpp
 * @author GreenBean
 * @brief CPU backend for mse. Hosts the Tensor::mse method definition
 * (mirrors how Tensor::softmax_ce lives in SoftmaxCECPU.cpp).
 * @version 0.1
 * @date 2026-10-23
 */
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

#include "Math/Loss/LossShared.h"
#include "Math/Loss/MseG/MseG.h"
#include "Math/Loss/Mse/Mse.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_shared;

    namespace {
        // --------------------------- mse kernels ---------------------------

        /**
         * fp32 accumulates in fp32, fp64 in fp64 — the same accumulator-width
         * taxonomy as sum()/mean(); the per-chunk partials merge in fp64 below.
         * @brief Scalar range MSE partial (fp32/fp64).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return Sum of (p[i] - t[i])^2 over [begin, end), in Acc.
         */
        template<typename T, typename Acc>
        [[nodiscard]] Acc mse_range_scalar(const T* p, const T* t, const size_t begin, const size_t end) {
            auto total = Acc{0};
            for (auto i = begin; i < end; ++i) {
                const auto d = static_cast<Acc>(p[i]) - static_cast<Acc>(t[i]);
                total += d * d;
            }
            return total;
        }

        /**
         * @brief Scalar range MSE partial (fp16/bf16, widened fp32 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return FP32 sum of (p[i] - t[i])^2 over [begin, end).
         */
        template<typename H>
        [[nodiscard]] float32_t mse_range_half(const H* p, const H* t, const size_t begin, const size_t end) {
            auto total = 0.0f;
            for (auto i = begin; i < end; ++i) {
                const auto d = widen_lane(p[i]) - widen_lane(t[i]);
                total += d * d;
            }
            return total;
        }

        /**
         * Both lanes of every element feed ONE scalar accumulator: the loss is
         * a single mean over all logical values, so packed dtypes divide by
         * 2 * numel on store (see Tensor::mse's contract).
         * @brief Scalar range MSE partial (fp16x2/bf16x2, per-lane fp32 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return FP32 sum of the squared lane differences over [begin, end).
         */
        template<typename P>
        [[nodiscard]] float32_t mse_range_packed(const P* p, const P* t, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            auto total = 0.0f;
            for (auto i = begin; i < end; ++i) {
                const auto p_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(p[i]);
                const auto t_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(t[i]);
                const auto d_lo = widen_lane(lane_lo<H>(p_bits)) - widen_lane(lane_lo<H>(t_bits));
                const auto d_hi = widen_lane(lane_hi<H>(p_bits)) - widen_lane(lane_hi<H>(t_bits));
                total += d_lo * d_lo + d_hi * d_hi;
            }
            return total;
        }

        /**
         * Unlike sum()'s modular uint64 contract, the loss accumulates in fp64
         * (a wrapped mean would be meaningless); narrowing clamps and
         * truncates toward zero, a NaN mean stores 0 (store_loss_mean).
         * @brief Scalar range MSE partial (all plain integer dtypes, fp64 accumulate).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return FP64 sum of (p[i] - t[i])^2 over [begin, end).
         */
        template<typename T>
        [[nodiscard]] float64_t mse_range_int(const T* p, const T* t, const size_t begin, const size_t end) {
            auto total = 0.0;
            for (auto i = begin; i < end; ++i) {
                const auto d = static_cast<float64_t>(p[i]) - static_cast<float64_t>(t[i]);
                total += d * d;
            }
            return total;
        }

        /**
         * Both nibbles feed one fp64 accumulator (2 * numel divides on store).
         * @brief Scalar range MSE partial (int4x2, per-nibble fp64 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return FP64 sum of the squared nibble differences over [begin, end).
         */
        [[nodiscard]] float64_t mse_range_i4x2(const int4x2_t* p, const int4x2_t* t, const size_t begin, const size_t end) {
            auto total = 0.0;
            for (auto i = begin; i < end; ++i) {
                const auto d_lo = static_cast<float64_t>(i4_lane_lo(p[i])) - static_cast<float64_t>(i4_lane_lo(t[i]));
                const auto d_hi = static_cast<float64_t>(i4_lane_hi(p[i])) - static_cast<float64_t>(i4_lane_hi(t[i]));
                total += d_lo * d_lo + d_hi * d_hi;
            }
            return total;
        }
    }

    /**
     * Chunks the squared-difference reduction across the pool into per-chunk
     * partials (accumulator width per dtype: fp32 for fp32, fp32-widened for
     * halfs/packed, fp64 for fp64/integers), merges the partials in fp64,
     * divides by the logical element count (2 * numel for packed/int4x2, whose
     * lanes/nibbles are logical elements) and narrows into out. Reduction
     * order is backend-defined, so float losses are not guaranteed
     * bit-identical between CPU/SIMD/GPU tiers (same documented contract as
     * sum()/mean()/softmax_ce).
     * @brief CPU mean squared error: mean of (pred - target)^2 over all elements.
     * @param t Predictions tensor, CPU resident, contiguous.
     * @param target Targets tensor, same shape/dtype/device, contiguous.
     * @param out Single-element output tensor in the operand dtype.
     */
    void mse_cpu(const Tensor& t, const Tensor& target, Tensor& out) {
        if (t.device != Device::CPU || target.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        const auto packed_kind = t.dtype == DType::Float16x2 || t.dtype == DType::BFloat16x2;
        const auto lane_factor = (packed_kind || t.dtype == DType::Int4x2) ? size_t{2} : size_t{1};
        const auto n_logical = static_cast<float64_t>(numel) * static_cast<float64_t>(lane_factor);
        switch (t.dtype) {
            case DType::Float32: {
                const auto* p = static_cast<const float32_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float32_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_scalar<float32_t, float32_t>(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Float64: {
                const auto* p = static_cast<const float64_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float64_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_scalar<float64_t, float64_t>(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += part;
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Float16: {
                const auto* p = static_cast<const float16_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float16_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_half<float16_t>(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::BFloat16: {
                const auto* p = static_cast<const bfloat16_t*>(t.data) + t.offset;
                const auto* g = static_cast<const bfloat16_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_half<bfloat16_t>(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Float16x2: {
                const auto* p = static_cast<const float16x2_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float16x2_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_packed<float16x2_t>(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::BFloat16x2: {
                const auto* p = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                const auto* g = static_cast<const bfloat16x2_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_packed<bfloat16x2_t>(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Int8: case DType::UInt8: case DType::Int16: case DType::UInt16:
            case DType::Int32: case DType::UInt32: case DType::Int64: case DType::UInt64: {
                std::vector partials(IncludedCPUFeatures().threads, 0.0);
                /**
                 * Runs mse_range_int<T> over every chunk and drains the futures.
                 * @brief Threaded integer MSE partial for one concrete element type.
                 * @param p Predictions buffer typed as T, already offset.
                 * @param g Targets buffer typed as T, already offset.
                 * @note Lambda subcall. Self-execution prohibited.
                 */
                auto launch_int = [&]<typename T>(const T* p, const T* g) {
                    drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_int<T>(p, g, b, e); }, t, target));
                };
                switch (t.dtype) {
                    case DType::Int8: launch_int(static_cast<const int8_t*>(t.data) + t.offset, static_cast<const int8_t*>(target.data) + target.offset); break;
                    case DType::UInt8: launch_int(static_cast<const uint8_t*>(t.data) + t.offset, static_cast<const uint8_t*>(target.data) + target.offset); break;
                    case DType::Int16: launch_int(static_cast<const int16_t*>(t.data) + t.offset, static_cast<const int16_t*>(target.data) + target.offset); break;
                    case DType::UInt16: launch_int(static_cast<const uint16_t*>(t.data) + t.offset, static_cast<const uint16_t*>(target.data) + target.offset); break;
                    case DType::Int32: launch_int(static_cast<const int32_t*>(t.data) + t.offset, static_cast<const int32_t*>(target.data) + target.offset); break;
                    case DType::UInt32: launch_int(static_cast<const uint32_t*>(t.data) + t.offset, static_cast<const uint32_t*>(target.data) + target.offset); break;
                    case DType::Int64: launch_int(static_cast<const int64_t*>(t.data) + t.offset, static_cast<const int64_t*>(target.data) + target.offset); break;
                    case DType::UInt64: launch_int(static_cast<const uint64_t*>(t.data) + t.offset, static_cast<const uint64_t*>(target.data) + target.offset); break;
                    [[unlikely]] default: throw ErrorHandling::UnReachableCode();
                }
                auto total = 0.0;
                for (const auto part : partials) {
                    total += part;
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Int4x2: {
                const auto* p = static_cast<const int4x2_t*>(t.data) + t.offset;
                const auto* g = static_cast<const int4x2_t*>(target.data) + target.offset;
                std::vector partials(IncludedCPUFeatures().threads, 0.0);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = mse_range_i4x2(p, g, b, e); }, t, target));
                auto total = 0.0;
                for (const auto part : partials) {
                    total += part;
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
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
     * Validates shapes, dtype, devices and contiguity, syncs both operands,
     * then routes to the backend. Only pred decides whether the output joins
     * the autograd graph, as targets never receive a gradient.
     * @brief Mean squared error: mean of (this - target)^2 over all elements.
     * @param target Targets, identical shape/dtype/device, contiguous.
     * @return Shape-{1} tensor in this tensor's dtype and device.
     * @throws ErrorHandling::MismatchedShapes Shapes differ, or a tensor is empty.
     * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
     * @throws ErrorHandling::MismatchedDevices Operand devices differ.
     * @throws ErrorHandling::NotImplemented Non-contiguous operand.
     */
    Tensor Tensor::mse(const Tensor& target) const {
        if (!this->is_same_shape(target)) {
            throw ErrorHandling::MismatchedShapes("mse requires identical shapes");
        }
        if (this->numel == 0) {
            throw ErrorHandling::MismatchedShapes("mse requires non-empty tensors");
        }
        if (this->device != target.device) {
            throw ErrorHandling::MismatchedDevices();
        }
        if (this->dtype != target.dtype) {
            throw ErrorHandling::MismatchedDTypes();
        }
        if (!this->is_contiguous || !target.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        target.sync();
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::mse_cpu(*this, target, out);
        } else if (this->device == Device::GPU) {
            detail::mse_gpu(*this, target, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        // Targets never receive a gradient: only pred decides whether the
        // output joins the graph.
        if (this->requires_grad) {
            out.wire_node(detail::mse_g_bridge, *this, &target);
        }
        return out;
    }
}
