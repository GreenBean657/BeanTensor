/**
 * @file BceGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for bce backward (probability form), plus the bce_g
 * backward trampoline.
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

#include "Math/Autograd/GradDType.h"
#include "Math/Loss/BceG/BceG.h"
#include "Math/Loss/LossShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_shared;

    namespace {
        // --------------------------- bce_g kernels ---------------------------

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
            } else {
                return Acc{0};
            }
        }

        template<typename T, typename Acc>
        [[nodiscard]] T narrow_grad(const Acc v_raw) {
            if constexpr (std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t>) {
                return static_cast<T>(v_raw);
            } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                return narrow_lane<T>(static_cast<float32_t>(v_raw));
            } else {
                return T{};
            }
        }

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

        template<typename Acc>
        [[nodiscard]] Acc bce_g_lane(const Acc p, const Acc y) {
            const auto eps = [] {
                if constexpr (std::is_same_v<Acc, float64_t>) {
                    return float64_t{1e-15};
                } else {
                    return Acc{1e-7f};
                }
            }();
            const auto pc = bce_clamp_p(p, eps);
            return ((Acc{1} - y) / (Acc{1} - pc)) - (y / pc);
        }

        template<typename T, typename Acc>
        void bce_g_range_scalar(const T* p, const T* t, T* out, const Acc scale, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = narrow_grad<T, Acc>(scale * bce_g_lane(static_cast<Acc>(p[i]), static_cast<Acc>(t[i])));
            }
        }

        template<typename H>
        void bce_g_range_half(const H* p, const H* t, H* out, const float32_t scale, const size_t begin, const size_t end) {
            constexpr auto eps = float32_t{1e-7f};
            for (auto i = begin; i < end; ++i) {
                const auto pc = bce_clamp_p(widen_lane(p[i]), eps);
                const auto y = widen_lane(t[i]);
                const auto g = scale * (((float32_t{1} - y) / (float32_t{1} - pc)) - (y / pc));
                out[i] = narrow_lane<H>(g);
            }
        }

        template<typename P>
        void bce_g_range_packed(const P* p, const P* t, P* out, const float32_t scale, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            constexpr auto eps = float32_t{1e-7f};
            for (auto i = begin; i < end; ++i) {
                const auto p_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(p[i]);
                const auto t_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(t[i]);
                const auto pc_lo = bce_clamp_p(widen_lane(lane_lo<H>(p_bits)), eps);
                const auto pc_hi = bce_clamp_p(widen_lane(lane_hi<H>(p_bits)), eps);
                const auto y_lo = widen_lane(lane_lo<H>(t_bits));
                const auto y_hi = widen_lane(lane_hi<H>(t_bits));
                const auto g_lo = scale * (((float32_t{1} - y_lo) / (float32_t{1} - pc_lo)) - (y_lo / pc_lo));
                const auto g_hi = scale * (((float32_t{1} - y_hi) / (float32_t{1} - pc_hi)) - (y_hi / pc_hi));
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(g_lo)),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(g_hi)));
            }
        }
    }

    void bce_g_cpu(const Tensor& dloss_grad, const Tensor& pred, const Tensor& targets, Tensor& dpred) {
        if (dloss_grad.device != Device::CPU || pred.device != Device::CPU || targets.device != Device::CPU || dpred.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(pred.device));
        }
        using DType = DataTypes::DType;
        const auto numel = pred.numel;
        const auto packed_kind = pred.dtype == DType::Float16x2 || pred.dtype == DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto n_logical = numel * lane_factor;
        auto run_pred = [&]<typename T, typename Acc>(const T* p_ptr, const T* t_ptr, T* o_ptr, const void* g_data) {
            const auto dloss = widen_dloss<T, Acc>(g_data);
            const auto scale = dloss / static_cast<Acc>(n_logical);
            drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_g_range_scalar<T, Acc>(p_ptr, t_ptr, o_ptr, scale, b, e); }, pred, targets, dpred));
        };
        switch (pred.dtype) {
            case DType::Float32: run_pred.operator()<float32_t, float32_t>(static_cast<const float32_t*>(pred.data) + pred.offset, static_cast<const float32_t*>(targets.data) + targets.offset, static_cast<float32_t*>(dpred.data) + dpred.offset, dloss_grad.data); break;
            case DType::Float64: run_pred.operator()<float64_t, float64_t>(static_cast<const float64_t*>(pred.data) + pred.offset, static_cast<const float64_t*>(targets.data) + targets.offset, static_cast<float64_t*>(dpred.data) + dpred.offset, dloss_grad.data); break;
            case DType::Float16: {
                const auto dloss = widen_dloss<float16_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_g_range_half<float16_t>(static_cast<const float16_t*>(pred.data) + pred.offset, static_cast<const float16_t*>(targets.data) + targets.offset, static_cast<float16_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::BFloat16: {
                const auto dloss = widen_dloss<bfloat16_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_g_range_half<bfloat16_t>(static_cast<const bfloat16_t*>(pred.data) + pred.offset, static_cast<const bfloat16_t*>(targets.data) + targets.offset, static_cast<bfloat16_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::Float16x2: {
                const auto dloss = widen_dloss<float16x2_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_g_range_packed<float16x2_t>(static_cast<const float16x2_t*>(pred.data) + pred.offset, static_cast<const float16x2_t*>(targets.data) + targets.offset, static_cast<float16x2_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            case DType::BFloat16x2: {
                const auto dloss = widen_dloss<bfloat16x2_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_g_range_packed<bfloat16x2_t>(static_cast<const bfloat16x2_t*>(pred.data) + pred.offset, static_cast<const bfloat16x2_t*>(targets.data) + targets.offset, static_cast<bfloat16x2_t*>(dpred.data) + dpred.offset, scale, b, e); }, pred, targets, dpred));
                break;
            }
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    void bce_g_node(Node*, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& pred = *inputs[0];
        const auto& target = *inputs[1];
        if (!pred.requires_grad) {
            return;
        }
        output_grad->sync();
        pred.sync();
        target.sync();
        const auto dloss = grad_converted_copy(*output_grad, pred.dtype);
        auto dpred = Tensor(std::vector(pred.shape, pred.shape + pred.ndim), pred.dtype, pred.device, false);
        if (pred.device == Device::CPU) {
            bce_g_cpu(dloss, pred, target, dpred);
        } else if (pred.device == Device::GPU) {
            bce_g_gpu(dloss, pred, target, dpred);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        grad_convert_inplace(dpred, grad_dtype);
        if (pred.node != nullptr) {
            grad_probe_hit(dpred.dtype);
        }
        pred.accumulate_grad(std::move(dpred));
    }

    void bce_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        bce_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
