#include "Tensor.h"
#include <immintrin.h>
#include <vector>
#include <sstream>
#include "Helper/Casting/casting_detail.h"
#include "Intrinsics/AVX512/DTypeConversions.h"
#include "Intrinsics/Native/DTypeConversions.h"
using namespace BeanTensor;

namespace BeanTensor::Tensors::detail {


    /*
     * Recursive & Print
     */
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
                if constexpr (std::is_same_v<T, Casting::bfloat16_t> || std::is_same_v<T, Casting::float16_t>) {
                    os << Casting::to_float(ptr[idx++]);
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

    template<typename T>
    void fill_fixed_impli(void* data, const size_t numel, const Tensors::Tensor& exec) {
        auto& threadMgr = Threading::get_cpu_thread_pool();
        static size_t t_count = Hardware::CPU().threads;

        auto worker = [data](const size_t t_start, const size_t t_end) {
            T* ptr = static_cast<T*>(data);
            for (size_t i = t_start; i < t_end; ++i) {
                if constexpr (std::is_same_v<T, Casting::float16_t>) {
                    ptr[i] = Casting::detail::f32_to_f16(1.0f);
                } else if constexpr (std::is_same_v<T, Casting::bfloat16_t>) {
                    ptr[i] = Casting::detail::f32_to_bf16(1.0f);
                } else {
                    ptr[i] = static_cast<T>(1);
                }
            }
        };

        if (numel <= t_count * 1024) {
            worker(0, numel);
            return;
        }

        std::vector<std::shared_future<void>> futures;
        const size_t chunk_size = (numel + t_count - 1) / t_count;
        for (size_t t = 0; t < t_count; ++t) {
            const size_t t_start = t * chunk_size;
            const size_t t_end = std::min(t_start + chunk_size, numel);
            if (t_start >= numel) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(t_start, t_end); }
            ).share());
        }
        exec.give_futures(futures);
    }

    template<typename T>
    void fill_random_impl(void* data, const size_t numel, const size_t seed, const double min, const double max, const Tensors::Tensor& exec) {
        auto& threadMgr = Threading::get_cpu_thread_pool();
        static size_t t_count = Hardware::CPU().threads;
        auto* ptr = static_cast<T*>(data);
        const static std::string err = "Random number generation range is out of supported range";
        if constexpr (std::is_same_v<T, Casting::bfloat16_t> || std::is_same_v<T, Casting::float16_t>) {
            if constexpr (std::is_same_v<T, Casting::bfloat16_t>) {
                if (min < Casting::detail::BF16_MIN || max > Casting::detail::BF16_MAX) {
                    throw std::invalid_argument(err);
                }
            } else {
                if (min < Casting::detail::F16_MIN || max > Casting::detail::F16_MAX) {
                    throw std::invalid_argument(err);
                }
            }
        } else {
            auto max_supported = std::numeric_limits<T>::max();
            auto min_supported = std::numeric_limits<T>::lowest();
            if (min < min_supported || max > max_supported) {
                throw std::invalid_argument(err);
            }
        }

        auto worker = [ptr](const size_t t_start, const size_t t_end, const size_t l_seed, const double t_min, const double t_max) {
            thread_local std::mt19937 rng;
            rng.seed(l_seed);
            if constexpr (std::is_integral_v<T>) {
                if constexpr(std::is_unsigned_v<T>) {
                    std::uniform_int_distribution<uint64_t> dist(
                        static_cast<uint64_t>(t_min),
                        static_cast<uint64_t>(t_max));
                    for (size_t i = t_start; i < t_end; ++i)
                        ptr[i] = static_cast<T>(dist(rng));
                } else {
                    std::uniform_int_distribution<int64_t> dist(
                        static_cast<int64_t>(t_min),
                        static_cast<int64_t>(t_max));
                    for (size_t i = t_start; i < t_end; ++i)
                        ptr[i] = static_cast<T>(dist(rng));
                }
            } else {
                std::uniform_real_distribution<double> dist(t_min, t_max);
                for (size_t i = t_start; i < t_end; ++i)
                    ptr[i] = static_cast<T>(dist(rng));
            }
        };

        if (numel <= t_count * 1024) {
            worker(0, numel, seed, min, max);
            return;
        }
        std::vector<std::shared_future<void>> futures;
        const size_t chunk_size = (numel + t_count - 1) / t_count;
        for (size_t t = 0; t < t_count; ++t) {
            const size_t t_start = t * chunk_size;
            const size_t t_end = std::min(t_start + chunk_size, numel);
            if (t_start >= numel) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(t_start, t_end, seed + t, min, max); }
            ).share());
        }
        exec.give_futures(futures);
    }
}

namespace BeanTensor::Tensors {

    void BeanTensor::Tensors::Tensor::compute_strides() {
        sync();
        strides[ndim - 1] = 1;
        for (int i = static_cast<int>(ndim) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }

    void BeanTensor::Tensors::Tensor::set_tensor(const std::vector<size_t> &new_shape, const Casting::DType new_dtype, const BeanTensor::Tensors::Device new_device, const bool new_grad_status) {
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
        this->nbytes = this->numel * (Casting::dtype_size(dtype));
        compute_strides();

        if (device == BeanTensor::Tensors::Device::CPU) {
            // FIX: aligned_alloc(64) instead of malloc — guarantees 64-byte alignment
            // enabling NT stores in conversion functions without runtime branching
            this->data = std::aligned_alloc(64, this->nbytes);
            assert(this->data != nullptr);
        } else if (device == BeanTensor::Tensors::Device::GPU) {
#if defined(USE_CUDA)
            int curr;
            cudaGetDevice(&curr);
            this->gpu_id = curr;
            cudaMalloc(&this->data, this->nbytes);
#elif defined(USE_HIP)
            int curr;
            hipGetDevice(&curr);
            this->gpu_id = curr;
            hipMalloc(&this->data, this->nbytes);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else {
            throw ErrorHandling::NotImplemented();
        }
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

    std::string Tensor::contents_to_string() const {
        sync();
        std::ostringstream os;

        using PrintFn = void(*)(std::ostringstream&, const void*, const size_t*, const size_t, size_t&, const size_t);
        using DType = Casting::DType;
        static const std::unordered_map<DType, PrintFn> print_fns = {
            {DType::Int8,     detail::print_recursive<int8_t>},
            {DType::Int16,    detail::print_recursive<int16_t>},
            {DType::Int32,    detail::print_recursive<int32_t>},
            {DType::Int64,    detail::print_recursive<int64_t>},
            {DType::UInt8,    detail::print_recursive<uint8_t>},
            {DType::UInt16,   detail::print_recursive<uint16_t>},
            {DType::UInt32,   detail::print_recursive<uint32_t>},
            {DType::UInt64,   detail::print_recursive<uint64_t>},
            {DType::Float16,  detail::print_recursive<Casting::float16_t>},
            {DType::BFloat16, detail::print_recursive<Casting::bfloat16_t>},
            {DType::Float32,  detail::print_recursive<Casting::float32_t>},
            {DType::Float64,  detail::print_recursive<Casting::float64_t>},
        };

        const auto it = print_fns.find(this->dtype);
        if (it == print_fns.end()) throw ErrorHandling::NotImplemented();
        size_t idx = 0;

        if (this->device == Device::CPU) {
            it->second(os, this->data, this->shape, this->ndim, idx, 0);
        } else if (this->device == Device::GPU) {
#if defined(USE_CUDA)
            cudaStreamSynchronize(stream);
            void* host_buf = std::malloc(this->nbytes);
            cudaMemcpy(host_buf, this->data, this->nbytes, cudaMemcpyDeviceToHost);
            it->second(os, host_buf, this->shape, this->ndim, idx, 0);
            std::free(host_buf);
#elif defined(USE_HIP)
            hipStreamSynchronize(stream);
            void* host_buf = std::aligned_alloc(64, this->nbytes);
            hipMemcpy(host_buf, this->data, this->nbytes, hipMemcpyDeviceToHost);
            it->second(os, host_buf, this->shape, this->ndim, idx, 0);
            std::free(host_buf);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else {
            __builtin_unreachable();
        }
        return os.str();
    }

    void Tensor::random(const double& min, const double& max, const int64_t seed) {
        sync();
        if (this->device == Device::CPU) {
            using FillFn = void(*)(void*, const size_t, const size_t, const double, const double, const Tensor&);
            using DType = Casting::DType;

            static const std::unordered_map<DType, FillFn> fill_fns = {
                {DType::Int8,     detail::fill_random_impl<int8_t>},
                {DType::Int16,    detail::fill_random_impl<int16_t>},
                {DType::Int32,    detail::fill_random_impl<int32_t>},
                {DType::Int64,    detail::fill_random_impl<int64_t>},
                {DType::UInt8,    detail::fill_random_impl<uint8_t>},
                {DType::UInt16,   detail::fill_random_impl<uint16_t>},
                {DType::UInt32,   detail::fill_random_impl<uint32_t>},
                {DType::UInt64,   detail::fill_random_impl<uint64_t>},
                {DType::Float16,  detail::fill_random_impl<Casting::float16_t>},
                {DType::BFloat16, detail::fill_random_impl<Casting::bfloat16_t>},
                {DType::Float32,  detail::fill_random_impl<Casting::float32_t>},
                {DType::Float64,  detail::fill_random_impl<Casting::float64_t>}
            };
            const auto it = fill_fns.find(this->dtype);
            if (it == fill_fns.end()) {
                throw ErrorHandling::NotImplemented();
            }
            it->second(this->data, this->numel, seed, min, max, *this);
        } else if (this->device == Device::GPU) {
            detail::set_random_gpu(*this, min, max, seed);
        } else {
            __builtin_unreachable();
        }
    }

    void Tensor::set_zero() {
        sync();
        if (this->device == Device::CPU) {
            std::memset(this->data, 0, this->nbytes);
        } else {
            detail::set_amount_gpu(*this, 0);
        }
    }

    void Tensor::set_one() {
        sync();
        using FillFn = void(*)(void*, const size_t, const Tensor&);
        using DType = Casting::DType;

        static const std::unordered_map<DType, FillFn> fill_fns = {
            {DType::Int8,     detail::fill_fixed_impli<int8_t>},
            {DType::Int16,    detail::fill_fixed_impli<int16_t>},
            {DType::Int32,    detail::fill_fixed_impli<int32_t>},
            {DType::Int64,    detail::fill_fixed_impli<int64_t>},
            {DType::UInt8,    detail::fill_fixed_impli<uint8_t>},
            {DType::UInt16,   detail::fill_fixed_impli<uint16_t>},
            {DType::UInt32,   detail::fill_fixed_impli<uint32_t>},
            {DType::UInt64,   detail::fill_fixed_impli<uint64_t>},
            {DType::Float16,  detail::fill_fixed_impli<Casting::float16_t>},
            {DType::BFloat16, detail::fill_fixed_impli<Casting::bfloat16_t>},
            {DType::Float32,  detail::fill_fixed_impli<Casting::float32_t>},
            {DType::Float64,  detail::fill_fixed_impli<Casting::float64_t>}
        };
        if (this->device == Device::CPU) {
            const auto it = fill_fns.find(this->dtype);
            if (it == fill_fns.end()) {
                throw ErrorHandling::NotImplemented();
            }
            it->second(this->data, this->numel, *this);
        } else if (this->device == Device::GPU) {
            detail::set_amount_gpu(*this, 1);
        } else {
            __builtin_unreachable();
        }
    }

    void Tensor::convert_dtype(const Casting::DType& new_dtype, const bool use_nan_conversions) {
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
                        this->give_futures(Intrinsics::detail::unaccelerated_bf16_to_fp32(static_cast<Casting::bfloat16_t*>(this->data), static_cast<Casting::float32_t*>(dst), this->numel));
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

    void Tensor::to(const Device& new_device, const uint64_t id) {
        if (this->device == Device::CPU && new_device == Device::CPU) {
            return;
        }
        if (this->device == Device::GPU && new_device == Device::GPU) {
            if (this->gpu_id == id) return;
            // GPU > GPU
            sync();
            detail::move_gpu_to_gpu(*this, id);
            return;
        }
        if (this->device == Device::CPU && new_device == Device::GPU) {
            // CPU > GPU
            sync();
            detail::move_to_gpu(*this, id);
            return;
        }
        if (this->device == Device::GPU && new_device == Device::CPU) {
            // GPU > CPU
            sync();
            detail::move_to_cpu(*this);
            return;
        }
        throw ErrorHandling::NotImplemented();
    }

    std::vector<size_t> Tensor::contents_to_flat_vector() const {
        sync();
        std::vector<size_t> result(this->numel);
        if (this->device == Device::CPU) {
            std::memcpy(result.data(), this->data, this->nbytes);
        } else if (this->device == Device::GPU) {
            throw ErrorHandling::NotImplemented();
        } else {
            __builtin_unreachable();
        }
        return result;
    }
}
