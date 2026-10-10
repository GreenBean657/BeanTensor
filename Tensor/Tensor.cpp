#include "Tensor.h"

#include <sstream>

namespace BeanTensor::Tensors::detail {
    template<typename T>
    void print_recursive(
        std::ostringstream& os,
        const void* data,
        const size_t* shape,
        const size_t ndim,
        size_t& idx,
        const size_t depth)
    {
        const T* ptr = static_cast<const T*>(data);
        os << "[";
        if (ndim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    os << "TODO: IMPLI_BF16";
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    os << "TODO: IMPLI_FP16";
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    os << "TODO: IMPLI_FP16x2";
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    os << "TODO: IMPLI_BF16x2";
                } else {
                    os << ptr[idx++];
                }
                if (i < shape[0] - 1) os << ", ";
            }
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                print_recursive<T>(os, ptr, shape + 1, ndim - 1, idx, depth + 1);
                if (i < shape[0] - 1) os << ",\n" << std::string(depth + 1, ' ');
            }
        }
        os << "]";
    }
}

namespace BeanTensor::Tensors {
    Tensor Tensor::clone(const bool hard_copy) {

        std::vector<size_t> new_shape;
        new_shape.reserve(this->ndim);
        for (size_t i = 0; i < this->ndim; i++) {
            new_shape.push_back(this->shape[i]);
        }

        auto clone = Tensor(new_shape, this->dtype, this->device, this->requires_grad);
        clone.sync();

        if (device == Device::CPU) {
            if (hard_copy) {
                clone.owns_data = true;
                memcpy(clone.data, this->data, this->nbytes);
            } else {
                clone.owns_data = false;
                clone.data = this->data;
                this->children.emplace_back(&clone);
            }
        } else if (device == Device::GPU) {
            if (hard_copy) {
                clone.owns_data = true;
            } else {
                clone.owns_data = false;
                clone.data = this->data;
                this->children.emplace_back(&clone);
            }
        } else {
            __builtin_unreachable();
        }
        return clone;
    }

    void Tensor::compute_strides() {
        sync();
        this->dims[this->ndim - 1] = 1;
        for (int32_t i = static_cast<int32_t>(this->ndim) - 2; i >= 0; --i)
            this->dims[i] = this->dims[i + 1] * this->shape[i + 1];
    }

    std::string Tensor::contents_to_string() const {
        sync();
        std::ostringstream os;

        using PrintFn = void(*)(std::ostringstream&, const void*, const size_t*, const size_t, size_t&, const size_t);
        using DType = DataTypes::DType;
        static const std::unordered_map<DType, PrintFn> print_fns = {
            {DType::Int8,     detail::print_recursive<int8_t>},
            {DType::Int16,    detail::print_recursive<int16_t>},
            {DType::Int32,    detail::print_recursive<int32_t>},
            {DType::Int64,    detail::print_recursive<int64_t>},
            {DType::UInt8,    detail::print_recursive<uint8_t>},
            {DType::UInt16,   detail::print_recursive<uint16_t>},
            {DType::UInt32,   detail::print_recursive<uint32_t>},
            {DType::UInt64,   detail::print_recursive<uint64_t>},
            {DType::Float16,  detail::print_recursive<float16_t>},
            {DType::Float16x2,  detail::print_recursive<float16x2_t>},
            {DType::BFloat16, detail::print_recursive<bfloat16_t>},
            {DType::BFloat16x2, detail::print_recursive<bfloat16x2_t>},
            {DType::Float32,  detail::print_recursive<float32_t>},
            {DType::Float64,  detail::print_recursive<float64_t>},
        };

        const auto it = print_fns.find(this->dtype);
        if (it == print_fns.end()) throw ErrorHandling::NotImplemented();
        size_t idx = 0;

        if (this->device == Device::CPU) {
            it->second(os, this->data, this->shape, this->ndim, idx, 0);
        } else if (this->device == Device::GPU) {
#if defined(USE_CUDA)
            CUDA_CHECK_ERROR(cudaStreamSynchronize(this->stream));
            void* host_buf = std::malloc(this->nbytes);
            CUDA_CHECK_ERROR(cudaMemcpy(host_buf, this->data, this->nbytes, cudaMemcpyDeviceToHost));
            it->second(os, host_buf, this->shape, this->ndim, idx, 0);
            std::free(host_buf);
#elif defined(USE_HIP)
            HIP_CHECK_ERROR(hipStreamSynchronize(this->stream));
            void* host_buf = std::aligned_alloc(64, this->nbytes);
            HIP_CHECK_ERROR(hipMemcpy(host_buf, this->data, this->nbytes, hipMemcpyDeviceToHost));
            it->second(os, host_buf, this->shape, this->ndim, idx, 0);
            std::free(host_buf);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else {
            assert(false && "Invalid Device"); std::unreachable();
        }
        return os.str();
    }

    void Tensor::kill_buffer() {
        if (this->device == Device::CPU) {
            if (this->convert_buffer != nullptr && this->convert_buffer_size != 0) {
                std::free(convert_buffer);
                this->convert_buffer = nullptr;
                this->convert_buffer_size = 0;
            }
        } else if (this->device == Device::GPU) {
            if (this->convert_buffer != nullptr && this->convert_buffer_size != 0) {
#if defined(USE_HIP)
                HIP_CHECK_ERROR(hipFree(this->convert_buffer));
                this->convert_buffer = nullptr;
                this->convert_buffer_size = 0;
#elif defined(USE_CUDA)
                CUDA_CHECK_ERROR(cudaFree(this->convert_buf));
                this->convert_buf = nullptr;
                this->convert_buf_size = 0;
#else
                throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
            }
        }
    }
    void Tensor::kill_cpu_data() {
        if (this->owns_data) {
            if (this->data != nullptr && this->device == Device::CPU) {
                std::free(this->data);
                this->data = nullptr;
            }
        }
    }

    void Tensor::kill_gpu_data() {
        if (this->device != Device::GPU) return;
        if (this->owns_data && this->data != nullptr) {
#if defined(USE_HIP)
            HIP_CHECK_ERROR(hipFree(this->data));
#elif defined(USE_CUDA)
            CUDA_CHECK_ERROR(cudaFree(this->data));
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
            this->data = nullptr;
        }
    }

    void Tensor::notify_child(const void* tensor_data) {
        if (this->device == Device::CPU) {
            this->kill_cpu_data();
            this->kill_buffer();
            this->sync();
            this->data = std::aligned_alloc(64, this->nbytes);
            memcpy(this->data, tensor_data, this->nbytes);
        } else if (this->device == Device::GPU) {
            this->kill_gpu_data();
            this->kill_buffer();
            this->sync();
#if defined(USE_HIP)
            throw ErrorHandling::NotImplemented();
#elif defined(USE_CUDA)
            void* new_data = nullptr;
            CUDA_CHECK_ERROR(cudaSetDevice(this->gpu_id));
            CUDA_CHECK_ERROR(cudaMalloc(&new_data, this->nbytes));
            CUDA_CHECK_ERROR(cudaMemcpy(new_data, data, this->nbytes, cudaMemcpyDeviceToDevice));
            this->data = new_data;
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        }
    }

    void Tensor::random(const float64_t &min, const float64_t &max, int64_t seed) {
        return;
    }

    void Tensor::set_one() {
        return;
    }

    void Tensor::set_tensor(const std::vector<size_t> &new_shape, const DataTypes::DType new_dtype, const Device new_device, const bool new_grad_status) {
        sync();
        const size_t new_ndim = new_shape.size();
        if (new_ndim == 0 || new_ndim > BT_TENSOR_MAX_DIMS) {
            throw std::invalid_argument("Tensor dimension exceeds supported dimensions");
        }
        this->ndim = new_ndim;
        this->dtype = new_dtype;
        this->device = new_device;
        this->requires_grad = new_grad_status;
        this->owns_data = true;

        this->numel = 1;
        for (uint32_t i = 0; i < new_ndim; ++i) {
            this->shape[i] = new_shape[i];
            this->numel *= new_shape[i];
        }
        this->nbytes = this->numel * (DataTypes::dtype_bytes(dtype));
        compute_strides();

        if (device == BeanTensor::Tensors::Device::CPU) {
            // DEPLOYED PATCH: aligned_alloc(64) instead of malloc — guarantees 64-byte alignment
            // enabling NT stores in conversion functions without runtime branching
            this->data = std::aligned_alloc(64, this->nbytes);
            assert(this->data != nullptr);
        } else if (device == BeanTensor::Tensors::Device::GPU) {
#if defined(USE_CUDA)
            cudaMalloc(&this->data, this->nbytes);
#elif defined(USE_HIP)
            HIP_CHECK_ERROR(hipMalloc(&this->data, this->nbytes));
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else {
            throw ErrorHandling::NotImplemented();
        }
    }

    void Tensor::set_zero() {
        return;
    }

    std::string Tensor::shape_to_string() const {
        std::ostringstream os;
        os << "{";
        for (uint32_t i = 0; i < this->ndim; ++i) {
            os << this->shape[i];
            if (i < this->ndim - 1) os << ", ";
        }
        os << "}";
        return os.str();
    }
}
