/**
 * @file BceWithLogitsGCPU.cpp
 * @author GreenBean
 * @brief CPU backend for bce_with_logits backward (stable sigmoid form), plus
 * the backward trampoline.
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
#include "Math/Loss/BceWithLogitsG/BceWithLogitsG.h"
#include "Math/Loss/LossShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Tensors::detail {
    using namespace loss_shared;

    namespace {
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
        [[nodiscard]] Acc stable_sigmoid(const Acc x) {
            if (x >= Acc{0}) {
                const auto z = std::exp(-x);
                return Acc{1} / (Acc{1} + z);
            }
            const auto z = std::exp(x);
            return z / (Acc{1} + z);
        }

        template<typename Acc>
        [[nodiscard]] Acc bce_logits_g_lane(const Acc x, const Acc y) {
            return stable_sigmoid(x) - y;
        }

        template<typename T, typename Acc>
        void bce_logits_g_range_scalar(const T* x, const T* t, T* out, const Acc scale, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = narrow_grad<T, Acc>(scale * bce_logits_g_lane(static_cast<Acc>(x[i]), static_cast<Acc>(t[i])));
            }
        }

        template<typename H>
        void bce_logits_g_range_half(const H* x, const H* t, H* out, const float32_t scale, const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                const auto logit = widen_lane(x[i]);
                const auto y = widen_lane(t[i]);
                const auto g = scale * bce_logits_g_lane(logit, y);
                out[i] = narrow_lane<H>(g);
            }
        }

        template<typename P>
        void bce_logits_g_range_packed(const P* x, const P* t, P* out, const float32_t scale, const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto x_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(x[i]);
                const auto t_bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(t[i]);
                const auto x_lo = widen_lane(lane_lo<H>(x_bits));
                const auto x_hi = widen_lane(lane_hi<H>(x_bits));
                const auto y_lo = widen_lane(lane_lo<H>(t_bits));
                const auto y_hi = widen_lane(lane_hi<H>(t_bits));
                const auto g_lo = scale * bce_logits_g_lane(x_lo, y_lo);
                const auto g_hi = scale * bce_logits_g_lane(x_hi, y_hi);
                out[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(g_lo)),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(narrow_lane<H>(g_hi)));
            }
        }
    }

    void bce_with_logits_g_cpu(const Tensor& dloss_grad, const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        if (dloss_grad.device != Device::CPU || logits.device != Device::CPU || targets.device != Device::CPU || dlogits.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(logits.device));
        }
        using DType = DataTypes::DType;
        const auto numel = logits.numel;
        const auto packed_kind = logits.dtype == DType::Float16x2 || logits.dtype == DType::BFloat16x2;
        const auto lane_factor = packed_kind ? size_t{2} : size_t{1};
        const auto n_logical = numel * lane_factor;
        auto run_logits = [&]<typename T, typename Acc>(const T* x_ptr, const T* t_ptr, T* o_ptr, const void* g_data) {
            const auto dloss = widen_dloss<T, Acc>(g_data);
            const auto scale = dloss / static_cast<Acc>(n_logical);
            drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_logits_g_range_scalar<T, Acc>(x_ptr, t_ptr, o_ptr, scale, b, e); }, logits, targets, dlogits));
        };
        switch (logits.dtype) {
            case DType::Float32: run_logits.operator()<float32_t, float32_t>(static_cast<const float32_t*>(logits.data) + logits.offset, static_cast<const float32_t*>(targets.data) + targets.offset, static_cast<float32_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float64: run_logits.operator()<float64_t, float64_t>(static_cast<const float64_t*>(logits.data) + logits.offset, static_cast<const float64_t*>(targets.data) + targets.offset, static_cast<float64_t*>(dlogits.data) + dlogits.offset, dloss_grad.data); break;
            case DType::Float16: {
                const auto dloss = widen_dloss<float16_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_logits_g_range_half<float16_t>(static_cast<const float16_t*>(logits.data) + logits.offset, static_cast<const float16_t*>(targets.data) + targets.offset, static_cast<float16_t*>(dlogits.data) + dlogits.offset, scale, b, e); }, logits, targets, dlogits));
                break;
            }
            case DType::BFloat16: {
                const auto dloss = widen_dloss<bfloat16_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_logits_g_range_half<bfloat16_t>(static_cast<const bfloat16_t*>(logits.data) + logits.offset, static_cast<const bfloat16_t*>(targets.data) + targets.offset, static_cast<bfloat16_t*>(dlogits.data) + dlogits.offset, scale, b, e); }, logits, targets, dlogits));
                break;
            }
            case DType::Float16x2: {
                const auto dloss = widen_dloss<float16x2_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_logits_g_range_packed<float16x2_t>(static_cast<const float16x2_t*>(logits.data) + logits.offset, static_cast<const float16x2_t*>(targets.data) + targets.offset, static_cast<float16x2_t*>(dlogits.data) + dlogits.offset, scale, b, e); }, logits, targets, dlogits));
                break;
            }
            case DType::BFloat16x2: {
                const auto dloss = widen_dloss<bfloat16x2_t, float32_t>(dloss_grad.data);
                const auto scale = dloss / static_cast<float32_t>(n_logical);
                drain(launch_loss_rows(numel, [&](const size_t b, const size_t e) { bce_logits_g_range_packed<bfloat16x2_t>(static_cast<const bfloat16x2_t*>(logits.data) + logits.offset, static_cast<const bfloat16x2_t*>(targets.data) + targets.offset, static_cast<bfloat16x2_t*>(dlogits.data) + dlogits.offset, scale, b, e); }, logits, targets, dlogits));
                break;
            }
            [[unlikely]] default: throw ErrorHandling::NotImplemented();
        }
    }

    void bce_with_logits_g_node(Node*, const Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& logits = *inputs[0];
        const auto& target = *inputs[1];
        if (!logits.requires_grad) {
            return;
        }
        output_grad->sync();
        logits.sync();
        target.sync();
        const auto dloss = grad_converted_copy(*output_grad, logits.dtype);
        auto dlogits = Tensor(std::vector(logits.shape, logits.shape + logits.ndim), logits.dtype, logits.device, false);
        if (logits.device == Device::CPU) {
            bce_with_logits_g_cpu(dloss, logits, target, dlogits);
        } else if (logits.device == Device::GPU) {
            bce_with_logits_g_gpu(dloss, logits, target, dlogits);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        grad_convert_inplace(dlogits, grad_dtype);
        if (logits.node != nullptr) {
            grad_probe_hit(dlogits.dtype);
        }
        logits.accumulate_grad(std::move(dlogits));
    }

    void bce_with_logits_g_bridge(Node* node, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        bce_with_logits_g_node(node, output_grad, inputs, n_inputs, grad_dtype);
    }
}
