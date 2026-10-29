/**
 * @file DropoutCPU.cpp
 * @brief CPU backend for inverted dropout mask construction, plus the public
 * train_mask dispatcher (routes on device to train_mask_cpu / train_mask_gpu).
 * Builds a same-shape scratch tensor, fills it with Tensor::random in [0, 1),
 * then applies the threshold-scale kernel `out = (rand >= p) ? 1/(1-p) : 0`
 * composed from existing compare/select/multiply lane primitives.
 * @version 0.2
 * @date 2026-10-29
 */
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "Math/Dropout/Dropout.h"
#include "Math/Dropout/DropoutShared.h"
#include "Tensor/Tensor.h"

#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Native/DTypeConversions.h"

namespace BeanTensor::Math::Dropout {

    namespace {
        using namespace BeanTensor::Tensors::detail::dropout_shared;
        using namespace BeanTensor::Tensors::detail::activations_shared;

        /**
         * @brief Scalar threshold-scale range kernel for native float kinds.
         */
        template<typename T>
        void mask_range_native(T* out, const T* rand, const float64_t p, const float64_t scale,
                               const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = mask_scalar_native<T>(rand[i], p, scale);
            }
        }

        /**
         * @brief Scalar threshold-scale range kernel for half kinds.
         */
        template<typename H>
        void mask_range_half(H* out, const H* rand, const float64_t p, const float64_t scale,
                             const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = mask_scalar_half<H>(rand[i], p, scale);
            }
        }

        /**
         * @brief Scalar threshold-scale range kernel for packed half kinds.
         */
        template<typename P>
        void mask_range_packed(P* out, const P* rand, const float64_t p, const float64_t scale,
                               const size_t begin, const size_t end) {
            for (auto i = begin; i < end; ++i) {
                out[i] = mask_scalar_packed<P>(rand[i], p, scale);
            }
        }
    }

    Tensors::Tensor train_mask_cpu(const Tensors::Tensor& x, const float64_t p, const int64_t seed) {
        using DType = DataTypes::DType;
        if (!is_dropout_float_kind(x.dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        if (x.device != Tensors::Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(x.device));
        }
        if (p < float64_t{0.0} || p >= float64_t{1.0}) {
            throw ErrorHandling::InvalidProbability("p must satisfy 0 <= p < 1");
        }
        auto rand = Tensors::Tensor(std::vector(x.shape, x.shape + x.ndim), x.dtype, x.device, false);
        rand.random(0.0, 1.0, seed);
        rand.sync();
        auto out = Tensors::Tensor(std::vector(x.shape, x.shape + x.ndim), x.dtype, x.device, false);
        const auto scale = float64_t{1.0} / (float64_t{1.0} - p);
        const auto numel = x.numel;
        switch (x.dtype) {
            case DType::Float32: {
                const auto* in_ptr = static_cast<const float32_t*>(rand.data) + rand.offset;
                auto* out_ptr = static_cast<float32_t*>(out.data) + out.offset;
                launch_elementwise([=](const size_t b, const size_t e) { mask_range_native<float32_t>(out_ptr, in_ptr, p, scale, b, e); },
                                   numel, numel * sizeof(float32_t), rand, rand, out);
                break;
            }
            case DType::Float64: {
                const auto* in_ptr = static_cast<const float64_t*>(rand.data) + rand.offset;
                auto* out_ptr = static_cast<float64_t*>(out.data) + out.offset;
                launch_elementwise([=](const size_t b, const size_t e) { mask_range_native<float64_t>(out_ptr, in_ptr, p, scale, b, e); },
                                   numel, numel * sizeof(float64_t), rand, rand, out);
                break;
            }
            case DType::Float16: {
                const auto* in_ptr = static_cast<const float16_t*>(rand.data) + rand.offset;
                auto* out_ptr = static_cast<float16_t*>(out.data) + out.offset;
                launch_elementwise([=](const size_t b, const size_t e) { mask_range_half<float16_t>(out_ptr, in_ptr, p, scale, b, e); },
                                   numel, numel * sizeof(float16_t), rand, rand, out);
                break;
            }
            case DType::BFloat16: {
                const auto* in_ptr = static_cast<const bfloat16_t*>(rand.data) + rand.offset;
                auto* out_ptr = static_cast<bfloat16_t*>(out.data) + out.offset;
                launch_elementwise([=](const size_t b, const size_t e) { mask_range_half<bfloat16_t>(out_ptr, in_ptr, p, scale, b, e); },
                                   numel, numel * sizeof(bfloat16_t), rand, rand, out);
                break;
            }
            case DType::Float16x2: {
                const auto* in_ptr = static_cast<const float16x2_t*>(rand.data) + rand.offset;
                auto* out_ptr = static_cast<float16x2_t*>(out.data) + out.offset;
                launch_elementwise([=](const size_t b, const size_t e) { mask_range_packed<float16x2_t>(out_ptr, in_ptr, p, scale, b, e); },
                                   numel, numel * sizeof(float16x2_t), rand, rand, out);
                break;
            }
            case DType::BFloat16x2: {
                const auto* in_ptr = static_cast<const bfloat16x2_t*>(rand.data) + rand.offset;
                auto* out_ptr = static_cast<bfloat16x2_t*>(out.data) + out.offset;
                launch_elementwise([=](const size_t b, const size_t e) { mask_range_packed<bfloat16x2_t>(out_ptr, in_ptr, p, scale, b, e); },
                                   numel, numel * sizeof(bfloat16x2_t), rand, rand, out);
                break;
            }
            [[unlikely]] default: {
                throw ErrorHandling::NotImplemented();
            }
        }
        return out;
    }

    Tensors::Tensor train_mask(const Tensors::Tensor& x, const float64_t p, const int64_t seed) {
        if (!is_dropout_float_kind(x.dtype)) {
            throw ErrorHandling::NotImplemented();
        }
        if (p < float64_t{0.0} || p >= float64_t{1.0}) {
            throw ErrorHandling::InvalidProbability("p must satisfy 0 <= p < 1");
        }
        if (x.device == Tensors::Device::CPU) {
            return train_mask_cpu(x, p, seed);
        } else if (x.device == Tensors::Device::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP)
            return train_mask_gpu(x, p, seed);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

}
