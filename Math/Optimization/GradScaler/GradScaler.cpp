/**
 * @file GradScaler.cpp
 * @author GreenBean
 * @brief CPU backend and main GradScaler class implementation: in-place scalar
 * division of float-kind gradients, finite check via sub/equal/min, and
 * dynamic scale management. GPU backend is in GradScalerCUDA.cu /
 * GradScalerHIP.cpp.
 * @version 0.1
 * @date 2026-11-14
 */
#include "Math/Optimization/GradScaler/GradScaler.h"

#include "Module/Module.h"

#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <vector>

#include "Helpers/CPUThreading.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"
#include "Math/Optimization/OptimizationShared.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Optimization::detail {
    /**
     * Defined in this file for CPU builds; the CUDA/HIP backends provide their
     * own definitions when USE_CUDA/USE_HIP are set.
     * @brief Forward declaration of the GPU in-place divide backend.
     */
    void gradscaler_div_scalar_inplace_gpu(Tensors::Tensor&, float64_t);
}

#if !defined(USE_HIP) && !defined(USE_CUDA)
namespace BeanTensor::Optimization::detail {
    /**
     * @brief CPU-only build stub for the GPU in-place divide. Always throws.
     */
    inline void gradscaler_div_scalar_inplace_gpu(Tensors::Tensor&, float64_t) {
        throw ErrorHandling::GPUTaskOnCPUBuild();
    }
}
#endif

namespace BeanTensor::Optimization {
    using namespace Tensors;
    using namespace Modules;
    using namespace Tensors::detail::optimization_shared;

    namespace {
        /**
         * fp32 in-place division. The inverse scale is precomputed in fp32 so
         * division is a single fp32 multiply.
         * @brief Divide a float32 buffer by scale.
         */
        void div_scalar_range_fp32(float32_t* ptr, const float32_t inv_scale,
                                   const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                ptr[i] *= inv_scale;
            }
        }

        /**
         * fp64 in-place division in full fp64 precision.
         * @brief Divide a float64 buffer by scale.
         */
        void div_scalar_range_fp64(float64_t* ptr, const float64_t inv_scale,
                                   const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                ptr[i] *= inv_scale;
            }
        }

        /**
         * fp16/bf16 in-place division: widen to fp32, multiply by the fp32
         * inverse scale, narrow RNE back to storage. This keeps 1/scale in
         * fp32 even when it would underflow if narrowed to fp16 first.
         * @brief Divide a half-precision buffer by scale.
         */
        template<typename H>
        void div_scalar_range_half(H* ptr, const float32_t inv_scale,
                                   const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                ptr[i] = narrow_lane<H>(widen_lane(ptr[i]) * inv_scale);
            }
        }

        /**
         * Packed x2 in-place division: per-lane widen/fp32-multiply/narrow.
         * @brief Divide a packed half-precision buffer by scale.
         */
        template<typename P>
        void div_scalar_range_packed(P* ptr, const float32_t inv_scale,
                                     const size_t begin, const size_t end) {
            using H = packed_lane_t<P>;
            for (auto i = begin; i < end; ++i) {
                const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(ptr[i]);
                const auto lo = narrow_lane<H>(widen_lane(lane_lo<H>(bits)) * inv_scale);
                const auto hi = narrow_lane<H>(widen_lane(lane_hi<H>(bits)) * inv_scale);
                ptr[i] = pack_lanes<P>(Intrinsics::detail::bit_cast_memcpy<uint16_t>(lo),
                                       Intrinsics::detail::bit_cast_memcpy<uint16_t>(hi));
            }
        }

        /**
         * Single-operand thread-pool launcher. Small inputs run inline; large
         * inputs are chunked across the pool. CCD-aware because the write is
         * in place.
         * @brief Chunk a unary elementwise worker across the CPU thread pool.
         * @param worker Callable taking (begin, end).
         * @param numel Element count.
         * @param out_bytes Bytes written by the worker.
         * @param t Tensor receiving the futures.
         */
        template<typename Worker>
        void launch_unary(Worker&& worker, const size_t numel, const size_t out_bytes, const Tensor& t) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(out_bytes);
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            for (auto tid = size_t{0}; tid < t_count; ++tid) {
                const auto begin = tid * chunk;
                if (begin >= numel) {
                    break;
                }
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group(
                        [worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); },
                        group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit(
                        [worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); },
                        Threading::TaskPriority::Normal).share());
                }
            }
            t.give_futures(futures);
        }
    }
}

namespace BeanTensor::Optimization::detail {
    using namespace Tensors::detail::optimization_shared;

    /**
     * CPU in-place division of a contiguous float-kind tensor by a scalar.
     * Computes 1/scale in the matching precision (fp64 for fp64 storage,
     * otherwise fp32) and multiplies in place; fp16/bf16/packed lanes widen to
     * fp32 for the multiply and narrow RNE on store.
     * @brief CPU backend for GradScaler::unscale_tensor_.
     * @param grad Gradient tensor (updated in place).
     * @param scale Current scale factor.
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource On a GPU tensor.
     * @throws ErrorHandling::NotImplemented Non-contiguous or non-float dtype.
     */
    void gradscaler_div_scalar_inplace_cpu(Tensors::Tensor& grad, const float64_t scale) {
        if (grad.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(grad.device));
        }
        if (!grad.is_contiguous) {
            throw ErrorHandling::NotImplemented();
        }
        using DType = DataTypes::DType;
        const auto numel = grad.numel;
        switch (grad.dtype) {
            case DType::Float32: {
                auto* ptr = static_cast<float32_t*>(grad.data) + grad.offset;
                const auto inv = static_cast<float32_t>(float64_t{1.0} / scale);
                launch_unary([=](const size_t b, const size_t e) { div_scalar_range_fp32(ptr, inv, b, e); },
                             numel, numel * sizeof(float32_t), grad);
                break;
            }
            case DType::Float64: {
                auto* ptr = static_cast<float64_t*>(grad.data) + grad.offset;
                const auto inv = float64_t{1.0} / scale;
                launch_unary([=](const size_t b, const size_t e) { div_scalar_range_fp64(ptr, inv, b, e); },
                             numel, numel * sizeof(float64_t), grad);
                break;
            }
            case DType::Float16: {
                auto* ptr = static_cast<float16_t*>(grad.data) + grad.offset;
                const auto inv = static_cast<float32_t>(float64_t{1.0} / scale);
                launch_unary([=](const size_t b, const size_t e) { div_scalar_range_half<float16_t>(ptr, inv, b, e); },
                             numel, numel * sizeof(float16_t), grad);
                break;
            }
            case DType::BFloat16: {
                auto* ptr = static_cast<bfloat16_t*>(grad.data) + grad.offset;
                const auto inv = static_cast<float32_t>(float64_t{1.0} / scale);
                launch_unary([=](const size_t b, const size_t e) { div_scalar_range_half<bfloat16_t>(ptr, inv, b, e); },
                             numel, numel * sizeof(bfloat16_t), grad);
                break;
            }
            case DType::Float16x2: {
                auto* ptr = static_cast<float16x2_t*>(grad.data) + grad.offset;
                const auto inv = static_cast<float32_t>(float64_t{1.0} / scale);
                launch_unary([=](const size_t b, const size_t e) { div_scalar_range_packed<float16x2_t>(ptr, inv, b, e); },
                             numel, numel * sizeof(float16x2_t), grad);
                break;
            }
            case DType::BFloat16x2: {
                auto* ptr = static_cast<bfloat16x2_t*>(grad.data) + grad.offset;
                const auto inv = static_cast<float32_t>(float64_t{1.0} / scale);
                launch_unary([=](const size_t b, const size_t e) { div_scalar_range_packed<bfloat16x2_t>(ptr, inv, b, e); },
                             numel, numel * sizeof(bfloat16x2_t), grad);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
    }
}

namespace BeanTensor::Optimization {

    GradScaler::GradScaler(const float64_t init_scale, const float64_t growth_factor,
                           const float64_t backoff_factor, const int32_t growth_interval,
                           const bool enabled, const bool dynamic)
        : scale_(init_scale),
          growth_factor_(growth_factor),
          backoff_factor_(backoff_factor),
          growth_interval_(growth_interval),
          growth_tracker_(0),
          enabled_(enabled),
          dynamic_(dynamic) {
    }

    Tensor GradScaler::scale(const Tensor& loss) const {
        const auto factor = this->enabled_ ? this->scale_ : float64_t{1.0};
        return loss.mul(factor);
    }

    bool GradScaler::unscale_and_check(Module& module) {
        bool all_finite = true;
        const auto params = module.named_parameters();
        for (const auto&[name, param] : params) {
            if (!param->has_grad()) {
                continue;
            }
            auto& grad = param->gradient();
            if (!grad.is_contiguous) {
                throw ErrorHandling::NotImplemented();
            }
            if (this->enabled_) {
                this->unscale_tensor_(grad);
            }
            if (!this->check_finite_(grad)) {
                all_finite = false;
            }
        }
        return all_finite;
    }

    void GradScaler::update(const bool finite) {
        if (!this->dynamic_ || !this->enabled_) {
            return;
        }
        if (!finite) {
            this->scale_ *= this->backoff_factor_;
            this->growth_tracker_ = 0;
            return;
        }
        this->growth_tracker_ += 1;
        if (this->growth_tracker_ >= this->growth_interval_) {
            this->scale_ *= this->growth_factor_;
            this->growth_tracker_ = 0;
        }
    }

    bool GradScaler::step(Module& module, const float64_t lr, const float64_t beta1,
                          const float64_t beta2, const float64_t eps) {
        const auto finite = this->unscale_and_check(module);
        this->update(finite);
        if (finite) {
            module.adam_step(lr, beta1, beta2, eps);
        }
        return finite;
    }

    bool GradScaler::step(Module& module, const float64_t lr) {
        const auto finite = this->unscale_and_check(module);
        this->update(finite);
        if (finite) {
            module.sgd_step(lr);
        }
        return finite;
    }

    void GradScaler::zero_grad(Module& module) {
        module.zero_grad();
    }

    float64_t GradScaler::get_scale() const {
        return this->scale_;
    }

    int32_t GradScaler::get_growth_tracker() const {
        return this->growth_tracker_;
    }

    bool GradScaler::is_enabled() const {
        return this->enabled_;
    }

    bool GradScaler::is_dynamic() const {
        return this->dynamic_;
    }

    void GradScaler::set_enabled(const bool enabled) {
        this->enabled_ = enabled;
    }

    void GradScaler::set_dynamic(const bool dynamic) {
        this->dynamic_ = dynamic;
    }

    void GradScaler::unscale_tensor_(Tensor& grad) {
        grad.sync();
        if (grad.device == Device::CPU) {
            detail::gradscaler_div_scalar_inplace_cpu(grad, this->scale_);
        } else if (grad.device == Device::GPU) {
            detail::gradscaler_div_scalar_inplace_gpu(grad, this->scale_);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    bool GradScaler::check_finite_(const Tensor& grad) const {
        // NaN and Inf are exactly the values for which x - x is NaN.
        // equal() writes a UInt8 0 at NaN positions and 1 elsewhere, so
        // min() over the mask is 1 iff every element is finite.
        auto diff = grad.sub(grad);
        auto mask = diff.equal(diff);
        auto all = mask.min();
        all.sync();
        return all.at<uint8_t>({0}) != uint8_t{0};
    }

}
