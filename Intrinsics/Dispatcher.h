#pragma once
#include <future>
#include <vector>
#include <Tensor/Tensor.h>
namespace BeanTensor::Intrinsics::detail {

    void launch_basic() {

    }

    [[nodiscard]] inline std::pmr::vector<std::vector<void>> launch_conv_cpu(Tensors::Tensor& tensor, BeanTensor::Casting::DType dtype, ConversionClampMethod method) {
        // Ending

        //All standardized paths have been discovered.

        using DType = Casting::DType;

        static const std::unordered_map<DType, std::numeric_limits<void>> test = {
            {},
        };



        /*
         *    void Tensor::convert_dtype(const Casting::DType& new_dtype, const bool use_nan_conversions) {
        if (this->dtype == new_dtype) return;
        sync();
        const size_t new_nbytes = this->numel * Casting::dtype_size(new_dtype);
        if (this->device == Device::CPU) {

            // SCRATCH BUFFER: grow convert_buf if needed, never shrink, never free between calls.
            // Replaces aligned_alloc per call — after warmup this is zero allocations.
            if (new_nbytes > this->convert_buf_size) {
                std::free(this->convert_buf);
                this->convert_buf = std::aligned_alloc(64, new_nbytes);
                if (this->convert_buf == nullptr) throw ErrorHandling::OutOfMemory();
                this->convert_buf_size = new_nbytes;
            }
            void* dst = this->convert_buf;

            bool converted = false;

            if (Casting::dtype_is_float(this->dtype)) {
                if (this->dtype == Casting::DType::Float32 && new_dtype == Casting::DType::BFloat16) {
                    if (Hardware::CPU().avx512bf16) {
                        this->give_futures(Intrinsics::detail::avx512_fp32_to_bf16(static_cast<Casting::float32_t*>(this->data), static_cast<Casting::bfloat16_t*>(dst), this->numel));
                        converted = true;
                    } else {
                        throw ErrorHandling::NotImplemented();
                    }
                } else if (this->dtype == Casting::DType::BFloat16 && new_dtype == Casting::DType::Float32) {
                    if (Hardware::CPU().avx512bf16) {
                        this->give_futures(Intrinsics::detail::avx512_bf16_to_fp32(static_cast<Casting::bfloat16_t*>(this->data), static_cast<Casting::float32_t*>(dst), this->numel));
                    } else {
                        throw ErrorHandling::NotImplemented();
                        //this->give_futures(Intrinsics::detail::unaccelerated_bf16_to_fp32(static_cast<Casting::bfloat16_t*>(this->data), static_cast<Casting::float32_t*>(dst), this->numel));
                    }
                    converted = true;
                }
            }

            if (!converted) {
                throw ErrorHandling::NotImplemented();
            }
            this->convert_buf = this->data;
            this->convert_buf_size = this->nbytes;
            this->data = dst;
            this->dtype = new_dtype;
            this->nbytes = new_nbytes;
            this->owns_data = true;
        } else if (this->device == Device::GPU) {
            detail::convert_gpu(*this, new_dtype, use_nan_conversions);
        } else {
            __builtin_unreachable();
        }
    }


        return std::pmr::vector<std::vector<void>>();
        */
    }
}