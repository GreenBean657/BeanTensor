/**
 * @file BceCPU.cpp
 * @author GreenBean
 * @brief CPU backend for bce (probability form). Hosts the Tensor::bce method
 * definition, mirroring how Tensor::mse lives in MseCPU.cpp.
 * @version 0.1
 * @date 2026-11-10
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#include <vector>

#include "Math/Loss/Bce/Bce.h"
#include "Math/Loss/BceG/BceG.h"
#include "Math/Loss/LossShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_shared;

    namespace {
        // --------------------------- target validation ---------------------------

        /**
         * BCE targets must be finite probabilities in [0, 1]. The check runs on
         * the host: GPU targets are deep-copied to CPU once and then inspected,
         * avoiding a per-element D2H round trip.
         * @brief Verify every target element is a finite value in [0, 1].
         * @param target Target tensor, contiguous, CPU resident or GPU resident.
         * @throws ErrorHandling::InvalidProbability if any target is non-finite or outside [0, 1].
         */
        void validate_bce_targets_cpu(const Tensor& target) {
            if (target.get_device() == Device::GPU) {
                auto host = const_cast<Tensor&>(target).clone(true);
                host.to(Device::CPU);
                host.sync();
                validate_bce_targets_cpu(host);
                return;
            }
            using DType = DataTypes::DType;
            const auto numel = target.element_count();
            switch (target.get_dtype()) {
                case DType::Float32: {
                    for (auto i = size_t{0}; i < numel; ++i) {
                        auto v = float32_t{0.0f};
                        target.read_element(i, &v, sizeof(v));
                        if (!std::isfinite(v) || v < float32_t{0.0f} || v > float32_t{1.0f}) {
                            throw ErrorHandling::InvalidProbability("BCE targets must be finite and in [0, 1]");
                        }
                    }
                    break;
                }
                case DType::Float64: {
                    for (auto i = size_t{0}; i < numel; ++i) {
                        auto v = float64_t{0.0};
                        target.read_element(i, &v, sizeof(v));
                        if (!std::isfinite(v) || v < float64_t{0.0} || v > float64_t{1.0}) {
                            throw ErrorHandling::InvalidProbability("BCE targets must be finite and in [0, 1]");
                        }
                    }
                    break;
                }
                case DType::Float16: {
                    for (auto i = size_t{0}; i < numel; ++i) {
                        auto v = float16_t{};
                        target.read_element(i, &v, sizeof(v));
                        const auto w = widen_lane(v);
                        if (!std::isfinite(w) || w < float32_t{0.0f} || w > float32_t{1.0f}) {
                            throw ErrorHandling::InvalidProbability("BCE targets must be finite and in [0, 1]");
                        }
                    }
                    break;
                }
                case DType::BFloat16: {
                    for (auto i = size_t{0}; i < numel; ++i) {
                        auto v = bfloat16_t{};
                        target.read_element(i, &v, sizeof(v));
                        const auto w = widen_lane(v);
                        if (!std::isfinite(w) || w < float32_t{0.0f} || w > float32_t{1.0f}) {
                            throw ErrorHandling::InvalidProbability("BCE targets must be finite and in [0, 1]");
                        }
                    }
                    break;
                }
                case DType::Float16x2: {
                    for (auto i = size_t{0}; i < numel; ++i) {
                        auto v = float16x2_t{};
                        target.read_element(i, &v, sizeof(v));
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(v);
                        for (const auto hi : {false, true}) {
                            const auto lane = widen_lane(hi ? lane_hi<float16_t>(bits) : lane_lo<float16_t>(bits));
                            if (!std::isfinite(lane) || lane < float32_t{0.0f} || lane > float32_t{1.0f}) {
                                throw ErrorHandling::InvalidProbability("BCE targets must be finite and in [0, 1]");
                            }
                        }
                    }
                    break;
                }
                case DType::BFloat16x2: {
                    for (auto i = size_t{0}; i < numel; ++i) {
                        auto v = bfloat16x2_t{};
                        target.read_element(i, &v, sizeof(v));
                        const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(v);
                        for (const auto hi : {false, true}) {
                            const auto lane = widen_lane(hi ? lane_hi<bfloat16_t>(bits) : lane_lo<bfloat16_t>(bits));
                            if (!std::isfinite(lane) || lane < float32_t{0.0f} || lane > float32_t{1.0f}) {
                                throw ErrorHandling::InvalidProbability("BCE targets must be finite and in [0, 1]");
                            }
                        }
                    }
                    break;
                }
                [[unlikely]] default: {
                    throw ErrorHandling::NotImplemented();
                }
            }
        }

        // --------------------------- bce kernels ---------------------------

        /**
         * @brief Clamp a probability to [eps, 1 - eps] in the compute width.
         * @param p Raw probability value.
         * @param eps Lower/upper epsilon.
         * @return Clamped probability.
         */
        template<typename Acc>
        [[nodiscard]] Acc bce_clamp_p(const Acc p, const Acc eps) {
            if (p < eps) {
                return eps;
            }
            if (p > Acc{1} - eps) {
                return Acc{1} - eps;
            }
            return p;
        }

        /**
         * @brief Scalar BCE contribution for one lane: -y*log(p) - (1-y)*log(1-p).
         * @param p_clamped Clamped probability.
         * @param y Target probability.
         * @return BCE loss contribution.
         */
        template<typename Acc>
        [[nodiscard]] Acc bce_lane_loss(const Acc p_clamped, const Acc y) {
            return -(y * std::log(p_clamped)) - ((Acc{1} - y) * std::log(Acc{1} - p_clamped));
        }

        /**
         * fp32 accumulates in fp32, fp64 in fp64. Predictions are clamped to
         * [eps, 1 - eps] before the log; the same clamped value is used in
         * backward by construction.
         * @brief Scalar range BCE partial (fp32/fp64).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @param eps Clamping epsilon in Acc.
         * @return Sum of BCE(p[i], t[i]) over [begin, end), in Acc.
         */
        template<typename T, typename Acc>
        [[nodiscard]] Acc bce_range_scalar(const T* p, const T* t, const size_t begin, const size_t end, const Acc eps) {
            auto total = Acc{0};
            for (auto i = begin; i < end; ++i) {
                const auto pc = bce_clamp_p(static_cast<Acc>(p[i]), eps);
                const auto y = static_cast<Acc>(t[i]);
                total += bce_lane_loss(pc, y);
            }
            return total;
        }

        /**
         * @brief Scalar range BCE partial (fp16/bf16, widened fp32 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return FP32 sum of BCE over [begin, end).
         */
        template<typename H>
        [[nodiscard]] float32_t bce_range_half(const H* p, const H* t, const size_t begin, const size_t end) {
            constexpr auto eps = float32_t{1e-7f};
            auto total = float32_t{0.0f};
            for (auto i = begin; i < end; ++i) {
                const auto pc = bce_clamp_p(widen_lane(p[i]), eps);
                const auto y = widen_lane(t[i]);
                total += bce_lane_loss(pc, y);
            }
            return total;
        }

        /**
         * Both lanes of every element feed ONE scalar accumulator: the loss is
         * a single mean over all logical values, so packed dtypes divide by
         * 2 * numel on store.
         * @brief Scalar range BCE partial (fp16x2/bf16x2, per-lane fp32 compute).
         * @param p Predictions, read only.
         * @param t Targets, read only.
         * @param begin First element processed.
         * @param end One past the last element processed.
         * @return FP32 sum of the lane BCE losses over [begin, end).
         */
        template<typename P>
        [[nodiscard]] float32_t bce_range_packed(const P* p, const P* t, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            constexpr auto eps = float32_t{1e-7f};
            auto total = float32_t{0.0f};
            for (auto i = begin; i < end; ++i) {
                const auto p_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(p[i]);
                const auto t_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(t[i]);
                const auto pc_lo = bce_clamp_p(widen_lane(lane_lo<H>(p_bits)), eps);
                const auto pc_hi = bce_clamp_p(widen_lane(lane_hi<H>(p_bits)), eps);
                const auto y_lo = widen_lane(lane_lo<H>(t_bits));
                const auto y_hi = widen_lane(lane_hi<H>(t_bits));
                total += bce_lane_loss(pc_lo, y_lo);
                total += bce_lane_loss(pc_hi, y_hi);
            }
            return total;
        }
    }

    /**
     * Chunks the BCE reduction across the pool into per-chunk partials (fp32
     * for fp32/halfs/packed, fp64 for fp64), merges the partials in fp64,
     * divides by the logical element count (2 * numel for packed) and narrows
     * into out. Float reduction order is backend-defined, so float losses are
     * not guaranteed bit-identical between CPU/SIMD/GPU tiers.
     * @brief CPU binary cross-entropy (probability form): mean of -y*log(p) - (1-y)*log(1-p) over all elements.
     * @param t Predictions tensor, CPU resident, contiguous.
     * @param target Targets tensor, same shape/dtype/device, contiguous.
     * @param out Single-element output tensor in the operand dtype.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if any tensor is not on the CPU.
     * @throws ErrorHandling::NotImplemented for an unsupported operand dtype.
     */
    void bce_cpu(const Tensor& t, const Tensor& target, Tensor& out) {
        if (t.device != Device::CPU || target.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        using DType = DataTypes::DType;
        const auto numel = t.numel;
        const auto packed_kind = t.dtype == DType::Float16x2 || t.dtype == DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto n_logical = static_cast<float64_t>(numel) * static_cast<float64_t>(lane_factor);
        switch (t.dtype) {
            case DType::Float32: {
                constexpr auto eps = float32_t{1e-7f};
                const auto* p = static_cast<const float32_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float32_t*>(target.data) + target.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = bce_range_scalar<float32_t, float32_t>(p, g, b, e, eps); }, t, target));
                auto total = float64_t{0.0};
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Float64: {
                constexpr auto eps = float64_t{1e-15};
                const auto* p = static_cast<const float64_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float64_t*>(target.data) + target.offset;
                std::vector<float64_t> partials(IncludedCPUFeatures().threads, 0.0);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = bce_range_scalar<float64_t, float64_t>(p, g, b, e, eps); }, t, target));
                auto total = float64_t{0.0};
                for (const auto part : partials) {
                    total += part;
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Float16: {
                const auto* p = static_cast<const float16_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float16_t*>(target.data) + target.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = bce_range_half<float16_t>(p, g, b, e); }, t, target));
                auto total = float64_t{0.0};
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::BFloat16: {
                const auto* p = static_cast<const bfloat16_t*>(t.data) + t.offset;
                const auto* g = static_cast<const bfloat16_t*>(target.data) + target.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = bce_range_half<bfloat16_t>(p, g, b, e); }, t, target));
                auto total = float64_t{0.0};
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::Float16x2: {
                const auto* p = static_cast<const float16x2_t*>(t.data) + t.offset;
                const auto* g = static_cast<const float16x2_t*>(target.data) + target.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = bce_range_packed<float16x2_t>(p, g, b, e); }, t, target));
                auto total = float64_t{0.0};
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
                }
                store_loss_mean(out.dtype, out.data, total / n_logical);
                break;
            }
            case DType::BFloat16x2: {
                const auto* p = static_cast<const bfloat16x2_t*>(t.data) + t.offset;
                const auto* g = static_cast<const bfloat16x2_t*>(target.data) + target.offset;
                std::vector<float32_t> partials(IncludedCPUFeatures().threads, 0.0f);
                drain(launch_loss_reduce(numel, [&](const size_t slot, const size_t b, const size_t e) { partials[slot] = bce_range_packed<bfloat16x2_t>(p, g, b, e); }, t, target));
                auto total = float64_t{0.0};
                for (const auto part : partials) {
                    total += static_cast<float64_t>(part);
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
     * Validates shapes, dtype, devices and contiguity, verifies that targets
     * are finite probabilities in [0, 1], syncs both operands, then routes to
     * the backend. Only pred decides whether the output joins the autograd
     * graph, as targets never receive a gradient.
     * @brief Binary cross-entropy (probability form): mean of -y*log(p) - (1-y)*log(1-p) over all elements.
     * @param target Targets, identical shape/dtype/device, contiguous.
     * @return Shape-{1} tensor in this tensor's dtype and device.
     * @throws ErrorHandling::MismatchedShapes Shapes differ, or a tensor is empty.
     * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
     * @throws ErrorHandling::MismatchedDevices Operand devices differ.
     * @throws ErrorHandling::NotImplemented Non-contiguous operand, integer storage, or int4x2.
     * @throws ErrorHandling::InvalidProbability Target is non-finite or outside [0, 1].
     */
    Tensor Tensor::bce(const Tensor& target) const & {
        if (!this->is_same_shape(target)) {
            throw ErrorHandling::MismatchedShapes("bce requires identical shapes");
        }
        if (this->numel == 0) {
            throw ErrorHandling::MismatchedShapes("bce requires non-empty tensors");
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
        using DType = DataTypes::DType;
        if (
            const auto dt = this->dtype;
            dt != DType::Float32 && dt != DType::Float64 && dt != DType::Float16 && dt != DType::BFloat16 && dt != DType::Float16x2 && dt != DType::BFloat16x2
        ) {
            throw ErrorHandling::NotImplemented();
        }
        sync();
        target.sync();
        detail::validate_bce_targets_cpu(target);
        auto out = Tensor(std::vector<size_t>{1}, this->dtype, this->device, false);
        if (this->device == Device::CPU) {
            detail::bce_cpu(*this, target, out);
        } else if (this->device == Device::GPU) {
            detail::bce_gpu(*this, target, out);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        if (this->requires_grad) {
            out.wire_node(detail::bce_g_bridge, *this, &target);
        }
        return out;
    }
}
