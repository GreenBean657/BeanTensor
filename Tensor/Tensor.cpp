/**
 * @file Tensor.cpp
 * @author GreenBean
 * @brief Tensor definitions: allocation, fills (incl. pool-chunked set_zero), dtype guards, conversion, transfer, printing and cloning.
 * @version 0.5
 * @date 2026-10-25
 */
#include "Tensor.h"

#include <cstring>
#include <random>
#include <sstream>
#include <unordered_map>

#include "Helpers/Alloc.h"
#include "Helpers/CPUThreading.h"
#include "Helpers/Errors.h"
#include "Helpers/Hardware.h"
#include "Intrinsics/Dispatcher.h"
#include "Intrinsics/Native/DTypeConversions.h"

#define BT_RECURSIVE_MAX_DEPTH 128
namespace BeanTensor::Tensors::detail {
    /**
     * Recursively prints the elements of a tensor. Stride-aware: views with a
     * nonzero offset or non-contiguous strides print their own elements, not
     * the root buffer's prefix.
     * @tparam T The data type of the tensor.
     * @param os The output stream to print to.
     * @param data The data pointer (buffer base the offsets index into).
     * @param shape The shape of the tensor.
     * @param strides Element strides of the tensor.
     * @param ndim The number of dimensions.
     * @param base Element offset of the current sub-array (view offset included).
     * @param depth The current depth.
     * @throws std::runtime_error stack overflow per BT_RECURSIVE_MAX_DEPTH.
     */
    template<typename T>
    static void print_recursive (
        std::ostringstream& os,
        const void* data,
        const size_t* shape,
        const size_t* strides,
        const size_t ndim,
        const size_t base,
        const size_t depth)
    {
        // Check for excessive depth. (Depth increments by 1 for each recursive call - prediction isn't reliable)
        if (depth > BT_RECURSIVE_MAX_DEPTH) [[unlikely]]
            throw std::runtime_error("Tensor print depth exceeded maximum limit.");{
        }
        const T* ptr = static_cast<const T*>(data);
        os << "[";
        if (ndim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                const auto elem = ptr[base + i * strides[0]];
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    os << Intrinsics::detail::unaccel_bf16_to_fp32(elem);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    os << Intrinsics::detail::unaccel_fp16_to_fp32_untracked(elem);
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(elem);
                    os << "(" << Intrinsics::detail::unaccel_fp16_to_fp32_untracked(Intrinsics::detail::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(bits & 0xFFFF)))
                       << ", " << Intrinsics::detail::unaccel_fp16_to_fp32_untracked(Intrinsics::detail::bit_cast_memcpy<float16_t>(static_cast<uint16_t>(bits >> 16))) << ")";
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint32_t>(elem);
                    os << "(" << Intrinsics::detail::unaccel_bf16_to_fp32(Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(bits & 0xFFFF)))
                       << ", " << Intrinsics::detail::unaccel_bf16_to_fp32(Intrinsics::detail::bit_cast_memcpy<bfloat16_t>(static_cast<uint16_t>(bits >> 16))) << ")";
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    const auto bits = Intrinsics::detail::bit_cast_memcpy<uint8_t>(elem);
                    const auto lo_nib = static_cast<int8_t>(bits & 0x0Fu);
                    const auto hi_nib = static_cast<int8_t>(bits >> 4);
                    const auto lo = lo_nib > 7 ? static_cast<int16_t>(lo_nib - 16) : static_cast<int16_t>(lo_nib);
                    const auto hi = hi_nib > 7 ? static_cast<int16_t>(hi_nib - 16) : static_cast<int16_t>(hi_nib);
                    os << "(" << lo << ", " << hi << ")";
                } else {
                    os << elem;
                }
                if (i < shape[0] - 1) os << ", ";
            }
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                print_recursive<T>(os, ptr, shape + 1, strides + 1, ndim - 1, base + i * strides[0], depth + 1);
                if (i < shape[0] - 1) os << ",\n" << std::string(depth + 1, ' ');
            }
        }
        os << "]";
    }

    /**
     * TODO - Write documentation.
     * @tparam T
     * @param data
     * @param numel
     * @param exec
     */
    template<typename T>
    static void fill_fixed_impli(void* data, const size_t numel, const Tensor& exec) {
        auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t t_count = IncludedCPUFeatures().threads;

        auto worker = [data](const size_t t_start, const size_t t_end) {
            T* ptr = static_cast<T*>(data);
            for (size_t i = t_start; i < t_end; ++i) {
                if constexpr (std::is_same_v<T, float16_t>) {
                    ptr[i] = Intrinsics::detail::unaccel_fp32_to_fp16_untracked(1.0f);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    ptr[i] = Intrinsics::detail::unaccel_fp32_to_bf16_untracked(1.0f);
                } else if constexpr (std::is_same_v<T, float16x2_t>) {
                    const auto one = Intrinsics::detail::bit_cast_memcpy<uint16_t>(Intrinsics::detail::unaccel_fp32_to_fp16_untracked(1.0f));
                    ptr[i] = Intrinsics::detail::bit_cast_memcpy<float16x2_t>(static_cast<uint32_t>(one) | (static_cast<uint32_t>(one) << 16));
                } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                    const auto one = Intrinsics::detail::bit_cast_memcpy<uint16_t>(Intrinsics::detail::unaccel_fp32_to_bf16_untracked(1.0f));
                    ptr[i] = Intrinsics::detail::bit_cast_memcpy<bfloat16x2_t>(static_cast<uint32_t>(one) | (static_cast<uint32_t>(one) << 16));
                } else if constexpr (std::is_same_v<T, int4x2_t>) {
                    // Both signed nibbles = 1 (matches the GPU set_amount int4x2 fill).
                    ptr[i] = Intrinsics::detail::bit_cast_memcpy<int4x2_t>(uint8_t{0x11});
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
        // The fill writes the whole destination buffer; when it fits one
        // CCD's LLC the chunks go to the calling thread's CCD group so the
        // cache-resident buffer is written locally (cross-CCD writes ~8x).
        const bool keep_local = KeepWorkLocal(numel * sizeof(T));
        const uint32_t group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
        for (size_t t = 0; t < t_count; ++t) {
            const size_t t_start = t * chunk_size;
            const size_t t_end = std::min(t_start + chunk_size, numel);
            if (t_start >= numel) break;
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { worker(t_start, t_end); }, group
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { worker(t_start, t_end); }
                ).share());
            }
        }
        exec.give_futures(futures);
    }

    /**
     * TODO - Write documentation.
     * @tparam T
     * @param data
     * @param numel
     * @param seed
     * @param min
     * @param max
     * @param exec
     */
    template<typename T>
    static void fill_random_impl(void* data, const size_t numel, const size_t seed, const float64_t min, const float64_t max, const Tensor& exec) {
        auto& threadMgr = Threading::GetCPUThreadPool();
        const size_t t_count = IncludedCPUFeatures().threads;
        auto* ptr = static_cast<T*>(data);
        const static std::string err = "Random number generation range is out of supported range";
        if constexpr (std::is_same_v<T, int4x2_t>) {
            // Signed 4-bit lanes: [-8, 7].
            if (min < -8.0 || max > 7.0) {
                throw std::invalid_argument(err);
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, bfloat16x2_t>) {
            if (min < static_cast<float64_t>(DataTypes::detail::BF16_MIN) || max > static_cast<float64_t>(DataTypes::detail::BF16_MAX)) {
                throw std::invalid_argument(err);
            }
        } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, float16x2_t>) {
            if (min < static_cast<float64_t>(DataTypes::detail::FP16_MIN) || max > static_cast<float64_t>(DataTypes::detail::FP16_MAX)) {
                throw std::invalid_argument(err);
            }
        } else {
            auto max_supported = std::numeric_limits<T>::max();
            auto min_supported = std::numeric_limits<T>::lowest();
            // Explicit widening — identical semantics to the implicit promotion.
            if (min < static_cast<float64_t>(min_supported) || max > static_cast<float64_t>(max_supported)) {
                throw std::invalid_argument(err);
            }
        }

        auto worker = [ptr](const size_t t_start, const size_t t_end, const size_t l_seed, const float64_t t_min, const float64_t t_max) {
            // Threadlocal initializer won't reseed rng per call... no lint and redefine after threadlocal.
            // NOLINTNEXTLINE(cert-msc51-cpp)
            thread_local std::mt19937 rng;
            rng.seed(l_seed);
            if constexpr (std::is_same_v<T, float16_t>) {
                std::uniform_real_distribution dist(t_min, t_max);
                for (size_t i = t_start; i < t_end; ++i)
                    ptr[i] = Intrinsics::detail::unaccel_fp32_to_fp16_untracked(static_cast<float32_t>(dist(rng)));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                std::uniform_real_distribution dist(t_min, t_max);
                for (size_t i = t_start; i < t_end; ++i)
                    ptr[i] = Intrinsics::detail::unaccel_fp32_to_bf16_untracked(static_cast<float32_t>(dist(rng)));
            } else if constexpr (std::is_same_v<T, float16x2_t>) {
                std::uniform_real_distribution dist(t_min, t_max);
                for (size_t i = t_start; i < t_end; ++i) {
                    const auto lo = Intrinsics::detail::bit_cast_memcpy<uint16_t>(Intrinsics::detail::unaccel_fp32_to_fp16_untracked(static_cast<float32_t>(dist(rng))));
                    const auto hi = Intrinsics::detail::bit_cast_memcpy<uint16_t>(Intrinsics::detail::unaccel_fp32_to_fp16_untracked(static_cast<float32_t>(dist(rng))));
                    ptr[i] = Intrinsics::detail::bit_cast_memcpy<float16x2_t>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
                }
            } else if constexpr (std::is_same_v<T, bfloat16x2_t>) {
                std::uniform_real_distribution dist(t_min, t_max);
                for (size_t i = t_start; i < t_end; ++i) {
                    const auto lo = Intrinsics::detail::bit_cast_memcpy<uint16_t>(Intrinsics::detail::unaccel_fp32_to_bf16_untracked(static_cast<float32_t>(dist(rng))));
                    const auto hi = Intrinsics::detail::bit_cast_memcpy<uint16_t>(Intrinsics::detail::unaccel_fp32_to_bf16_untracked(static_cast<float32_t>(dist(rng))));
                    ptr[i] = Intrinsics::detail::bit_cast_memcpy<bfloat16x2_t>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
                }
            } else if constexpr (std::is_same_v<T, int4x2_t>) {
                std::uniform_int_distribution dist(static_cast<int64_t>(t_min), static_cast<int64_t>(t_max));
                for (size_t i = t_start; i < t_end; ++i) {
                    const auto lo = static_cast<uint8_t>(static_cast<uint8_t>(static_cast<int8_t>(dist(rng))) & 0x0Fu);
                    const auto hi = static_cast<uint8_t>(static_cast<uint8_t>(static_cast<int8_t>(dist(rng))) & 0x0Fu);
                    ptr[i] = Intrinsics::detail::bit_cast_memcpy<int4x2_t>(static_cast<uint8_t>(lo | (hi << 4)));
                }
            } else if constexpr (std::is_integral_v<T>) {
                if constexpr(std::is_unsigned_v<T>) {
                    std::uniform_int_distribution dist(
                        static_cast<uint64_t>(t_min),
                        static_cast<uint64_t>(t_max));
                    for (size_t i = t_start; i < t_end; ++i)
                        ptr[i] = static_cast<T>(dist(rng));
                } else {
                    std::uniform_int_distribution dist(
                        static_cast<int64_t>(t_min),
                        static_cast<int64_t>(t_max));
                    for (size_t i = t_start; i < t_end; ++i)
                        ptr[i] = static_cast<T>(dist(rng));
                }
            } else {
                std::uniform_real_distribution dist(t_min, t_max);
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
        // The fill writes the whole destination buffer; when it fits one
        // CCD's LLC the chunks go to the calling thread's CCD group so the
        // cache-resident buffer is written locally (cross-CCD writes ~8x).
        const bool keep_local = KeepWorkLocal(numel * sizeof(T));
        const uint32_t group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
        for (size_t t = 0; t < t_count; ++t) {
            const size_t t_start = t * chunk_size;
            const size_t t_end = std::min(t_start + chunk_size, numel);
            if (t_start >= numel) break;
            if (keep_local) {
                futures.push_back(threadMgr.submit_to_group(
                    [=] { worker(t_start, t_end, seed + t, min, max); }, group
                ).share());
            } else {
                futures.push_back(threadMgr.submit(
                    [=] { worker(t_start, t_end, seed + t, min, max); }
                ).share());
            }
        }
        exec.give_futures(futures);
    }
}

namespace BeanTensor::Tensors {
    /**
     * Clone this tensor. Autograd state is NOT copied: the clone keeps
     * requires_grad but gets no grad, node, or parents (it is a fresh leaf).
     * @param hard_copy
     * @return
     */
    Tensor Tensor::clone(const bool hard_copy) {

        std::vector<size_t> new_shape;
        new_shape.reserve(this->ndim);
        for (size_t i = 0; i < this->ndim; i++) {
            new_shape.push_back(this->shape[i]);
        }

        auto clone = Tensor(new_shape, this->dtype, this->device, this->requires_grad);
        sync(); // the source must settle before its data is read/shared

        if (device == Device::CPU) {
            if (hard_copy) {
                clone.owns_data = true;
                memcpy(clone.data, this->data, this->nbytes);
            } else {
                clone.owns_data = false;
                clone.data = this->data;
                this->children.emplace_back(&clone);
                clone.view_parents.emplace_back(this);
            }
        } else if (device == Device::GPU) {
            if (hard_copy) {
                clone.owns_data = true;
#if defined(USE_HIP)
                HIP_CHECK_ERROR(hipMemcpy(clone.data, this->data, this->nbytes, hipMemcpyDeviceToDevice));
#elif defined(USE_CUDA)
                CUDA_CHECK_ERROR(cudaMemcpy(clone.data, this->data, this->nbytes, cudaMemcpyDeviceToDevice));
#else
                throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
            } else {
                clone.owns_data = false;
                clone.data = this->data;
                this->children.emplace_back(&clone);
                clone.view_parents.emplace_back(this);
            }
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return clone;
    }

    /**
     * Move this tensor (and its accumulated gradient, if any) to new_device.
     * Blocking: returns only once the copy has landed and the device flag
     * has flipped. The move workers rewrite data/device and capture this
     * tensor by reference, so an op validating devices (or a move of this
     * tensor) right after an async to() would race them.
     * @param new_device Destination device.
     * @throws ErrorHandling::NotImplemented Unsupported device pair.
     */
    void Tensor::to(const Device& new_device) {
        if (this->device == new_device) return;
        sync();
        if (this->device == Device::CPU && new_device == Device::GPU) {
            detail::move_to_gpu(*this);
        } else if (this->device == Device::GPU && new_device == Device::CPU) {
            detail::move_to_cpu(*this);
        } else [[unlikely]] {
            throw ErrorHandling::NotImplemented();
        }
        sync(); // surfaces move errors here, not at an unrelated later sync()
        // The accumulated gradient rides along — a grad stranded on the old
        // device would break the next backward()'s device checks.
        if (this->grad != nullptr) {
            this->grad->to(new_device);
        }
    }

    /**
     * TODO - Write documentation.
     * @param new_dtype
     * @param method
     */
    void Tensor::convert_dtype(const DataTypes::DType& new_dtype, const ConversionClampMethod method) {
        // sync() BEFORE the dtype check: a previously queued conversion's
        // finalize flips dtype on a pool worker — reading dtype first races it
        // (TSAN-verified). Same-dtype calls pay one drained-queue sync.
        sync();
        if (this->dtype == new_dtype) return;
        if (this->device == Device::CPU) {
            Intrinsics::detail::launch_conv_cpu(*this, new_dtype, method);
        } else if (this->device == Device::GPU) {
            // GPU kernels mirror the CPU dispatcher's clamp/flag semantics per
            // method; non-PERMISSIVE failures surface as std::range_error at sync().
            detail::convert_gpu(*this, new_dtype, method);
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * TODO - Write documentation.
     */
    void Tensor::compute_strides() {
        sync();
        this->dims[this->ndim - 1] = 1;
        for (int32_t i = static_cast<int32_t>(this->ndim) - 2; i >= 0; --i) {
            this->dims[i] = this->dims[i + 1] * this->shape[i + 1];
        }
    }

    /**
     * TODO - Write documentation.
     * @return
     */
    std::string Tensor::contents_to_string() const {
        sync();
        std::ostringstream os;

        using PrintFn = void(*)(std::ostringstream&, const void*, const size_t*, const size_t*, size_t, size_t, size_t);
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
            {DType::Int4x2,   detail::print_recursive<int4x2_t>},
        };

        const auto it = print_fns.find(this->dtype);
        if (it == print_fns.end()) [[unlikely]] {
            throw ErrorHandling::NotImplemented();
        }

        if (this->device == Device::CPU) {
            it->second(os, this->data, this->shape, this->dims, this->ndim, this->offset, 0);
        } else if (this->device == Device::GPU) {
#if defined(USE_CUDA)
            CUDA_CHECK_ERROR(cudaStreamSynchronize(this->stream));
            // A view can reach past nbytes from the buffer base (offset +
            // strides); stage the full reachable span so printing stays in
            // bounds. Computed per-backend: CPU-only builds would see these
            // as unused variables.
            auto span_elems = size_t{0};
            if (this->numel != 0) {
                span_elems = this->offset + 1;
                for (auto d = size_t{0}; d < this->ndim; ++d) {
                    span_elems += (this->shape[d] - 1) * this->dims[d];
                }
            }
            const auto span_bytes = span_elems * DataTypes::dtype_bytes(this->dtype);
            void* host_buf = std::malloc(span_bytes == 0 ? 1 : span_bytes);
            if (span_bytes != 0) {
                CUDA_CHECK_ERROR(cudaMemcpy(host_buf, this->data, span_bytes, cudaMemcpyDeviceToHost));
            }
            it->second(os, host_buf, this->shape, this->dims, this->ndim, this->offset, 0);
            std::free(host_buf);
#elif defined(USE_HIP)
            HIP_CHECK_ERROR(hipStreamSynchronize(this->stream));
            auto span_elems = size_t{0};
            if (this->numel != 0) {
                span_elems = this->offset + 1;
                for (auto d = size_t{0}; d < this->ndim; ++d) {
                    span_elems += (this->shape[d] - 1) * this->dims[d];
                }
            }
            const auto span_bytes = span_elems * DataTypes::dtype_bytes(this->dtype);
            void* host_buf = bt_aligned_alloc(span_bytes);
            if (span_bytes != 0) {
                HIP_CHECK_ERROR(hipMemcpy(host_buf, this->data, span_bytes, hipMemcpyDeviceToHost));
            }
            it->second(os, host_buf, this->shape, this->dims, this->ndim, this->offset, 0);
            std::free(host_buf);
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        return os.str();
    }

    /**
     * TODO - Write documentation.
     */
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
                CUDA_CHECK_ERROR(cudaFree(this->convert_buffer));
                this->convert_buffer = nullptr;
                this->convert_buffer_size = 0;
#else
                throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
            }
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * TODO - Write documentation.
     */
    void Tensor::kill_cpu_data() {
        if (this->device != Device::CPU) return;
        if (this->owns_data) {
            if (this->data != nullptr) {
                std::free(this->data);
                this->data = nullptr;
            }
        }
    }

    /**
     * TODO - Write documentation.
     */
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

    /**
     * Teardown path for the destructor and the noexcept move assignment: same
     * frees as kill_gpu_data plus the convert buffer (mirroring kill_buffer),
     * but the GPU frees are unchecked. At static teardown the driver may
     * already be unloading (cudaErrorCudartUnloading, "driver shutting down")
     * and no GPU call can succeed; those allocations are reclaimed with the
     * process, so the failure is unactionable and must not escape these
     * noexcept callers. Operational callers keep the checked variants.
     */
    void Tensor::kill_gpu_teardown_noexcept() noexcept {
        if (
            this->device == Device::GPU && this->owns_data && this->data != nullptr
        ) {
#if defined(USE_HIP)
            (void)hipFree(this->data);
#elif defined(USE_CUDA)
            (void)cudaFree(this->data);
#endif
            this->data = nullptr;
        }
        if (this->convert_buffer != nullptr && this->convert_buffer_size != 0) {
            if (this->device == Device::CPU) {
                std::free(this->convert_buffer);
            } else if (this->device == Device::GPU) {
#if defined(USE_HIP)
                (void)hipFree(this->convert_buffer);
#elif defined(USE_CUDA)
                (void)cudaFree(this->convert_buffer);
#endif
            }
            this->convert_buffer = nullptr;
            this->convert_buffer_size = 0;
        }
    }

    /**
     * TODO - Write documentation.
     * @param parent
     */
    void Tensor::notify_child(Tensor* parent) {
        if (this->device == Device::CPU) {
            this->sync();
            this->kill_cpu_data();
            this->kill_buffer();
            this->data = bt_aligned_alloc(this->nbytes);
            memcpy(this->data, parent->data, this->nbytes);
        } else if (this->device == Device::GPU) {
            this->sync();
            this->kill_gpu_data();
            this->kill_buffer();
#if defined(USE_HIP)
            void* new_data = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&new_data, this->nbytes));
            HIP_CHECK_ERROR(hipMemcpy(new_data, parent->data, this->nbytes, hipMemcpyDeviceToDevice));
            this->data = new_data;
#elif defined(USE_CUDA)
            void* new_data = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&new_data, this->nbytes));
            CUDA_CHECK_ERROR(cudaMemcpy(new_data, parent->data, this->nbytes, cudaMemcpyDeviceToDevice));
            this->data = new_data;
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        }
        // The fresh buffer is ours now — without this the destructor never
        // frees it and every parent death leaks a full clone.
        this->owns_data = true;
        std::erase(this->view_parents, parent);
        // This view's own children aliased the dying parent's buffer too;
        // re-point them onto the fresh copy so the chain stays valid.
        for (const auto& i : this->children) {
            i->data = this->data;
        }
    }

    /**
     * TODO - Write documentation.
     * @param idx
     * @return
     */
    size_t Tensor::checked_element_offset(const std::initializer_list<size_t> idx) const {
        sync();
        if (idx.size() != ndim) {
            throw ErrorHandling::IndexOutOfRange();
        }
        auto elem_offset = this->offset;
        auto dim = size_t{0};
        for (const size_t i : idx) {
            if (i >= shape[dim]) {
                throw ErrorHandling::IndexOutOfRange();
            }
            elem_offset += i * dims[dim];
            ++dim;
        }
        return elem_offset;
    }

    /**
     * Guards Tensor::set and Tensor::at: the element type they were called
     * with must be the dtype this tensor stores, or the byte stride is wrong.
     * Callers sync first (through checked_element_offset), so a pending
     * convert_dtype has already flipped dtype when this runs.
     * @param expected DType of the element type being accessed.
     * @throws ErrorHandling::MismatchedDTypes If expected differs from this tensor's dtype.
     */
    void Tensor::require_dtype(const DataTypes::DType expected) const {
        if (expected != this->dtype) [[unlikely]] {
            throw ErrorHandling::MismatchedDTypes();
        }
    }

    /**
     * TODO - Write documentation.
     * @param elem_offset
     * @param dst
     * @param bytes
     */
    void Tensor::read_element(const size_t elem_offset, void* dst, const size_t bytes) {
        if (this->device == Device::CPU) {
            memcpy(dst, static_cast<uint8_t*>(this->data) + elem_offset * bytes, bytes);
        } else if (this->device == Device::GPU) {
#if defined(USE_HIP)
            HIP_CHECK_ERROR(hipMemcpy(dst, static_cast<uint8_t*>(this->data) + elem_offset * bytes, bytes, hipMemcpyDeviceToHost));
#elif defined(USE_CUDA)
            CUDA_CHECK_ERROR(cudaMemcpy(dst, static_cast<uint8_t*>(this->data) + elem_offset * bytes, bytes, cudaMemcpyDeviceToHost));
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
          throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * TODO - Write documentation.
     * @param elem_offset
     * @param src
     * @param bytes
     */
    void Tensor::write_element(const size_t elem_offset, const void* src, const size_t bytes) {
        if (this->device == Device::CPU) {
            memcpy(static_cast<uint8_t*>(this->data) + elem_offset * bytes, src, bytes);
        } else if (this->device == Device::GPU) {
#if defined(USE_HIP)
            HIP_CHECK_ERROR(hipMemcpy(static_cast<uint8_t*>(this->data) + elem_offset * bytes, src, bytes, hipMemcpyHostToDevice));
#elif defined(USE_CUDA)
            CUDA_CHECK_ERROR(cudaMemcpy(static_cast<uint8_t*>(this->data) + elem_offset * bytes, src, bytes, cudaMemcpyHostToDevice));
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * TODO - Write documentation.
     * @param min
     * @param max
     * @param seed
     */
    void Tensor::random(const float64_t &min, const float64_t &max, const int64_t seed) {
        sync();
        if (this->device == Device::CPU) {
            using FillFn = void(*)(void*, size_t, size_t, float64_t, float64_t, const Tensor&);
            using DType = DataTypes::DType;

            static const std::unordered_map<DType, FillFn> fill_fns = {
                {DType::Int8,     detail::fill_random_impl<int8_t>},
                {DType::Int16,    detail::fill_random_impl<int16_t>},
                {DType::Int32,    detail::fill_random_impl<int32_t>},
                {DType::Int64,    detail::fill_random_impl<int64_t>},
                {DType::UInt8,    detail::fill_random_impl<uint8_t>},
                {DType::UInt16,   detail::fill_random_impl<uint16_t>},
                {DType::UInt32,   detail::fill_random_impl<uint32_t>},
                {DType::UInt64,   detail::fill_random_impl<uint64_t>},
                {DType::Float16,   detail::fill_random_impl<float16_t>},
                {DType::Float16x2, detail::fill_random_impl<float16x2_t>},
                {DType::BFloat16,   detail::fill_random_impl<bfloat16_t>},
                {DType::BFloat16x2, detail::fill_random_impl<bfloat16x2_t>},
                {DType::Float32,  detail::fill_random_impl<float32_t>},
                {DType::Float64,  detail::fill_random_impl<float64_t>},
                {DType::Int4x2,   detail::fill_random_impl<int4x2_t>}
            };
            const auto it = fill_fns.find(this->dtype);
            if (it == fill_fns.end()) [[unlikely]] {
                throw ErrorHandling::NotImplemented();
            }
            it->second(this->data, this->numel, static_cast<size_t>(seed), min, max, *this);
        } else if (this->device == Device::GPU) {
            detail::set_random_gpu(*this, min, max, seed);
        } else [[unlikely]] {
            throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * TODO - Write documentation.
     */
    void Tensor::set_one() {
        sync();
        if (this->device == Device::CPU) {
            using FillFn = void(*)(void*, size_t, const Tensor&);
            using DType = DataTypes::DType;

            static const std::unordered_map<DType, FillFn> fill_fns = {
                {DType::Int8,     detail::fill_fixed_impli<int8_t>},
                {DType::Int16,    detail::fill_fixed_impli<int16_t>},
                {DType::Int32,    detail::fill_fixed_impli<int32_t>},
                {DType::Int64,    detail::fill_fixed_impli<int64_t>},
                {DType::UInt8,    detail::fill_fixed_impli<uint8_t>},
                {DType::UInt16,   detail::fill_fixed_impli<uint16_t>},
                {DType::UInt32,   detail::fill_fixed_impli<uint32_t>},
                {DType::UInt64,   detail::fill_fixed_impli<uint64_t>},
                {DType::Float16,   detail::fill_fixed_impli<float16_t>},
                {DType::Float16x2, detail::fill_fixed_impli<float16x2_t>},
                {DType::BFloat16,   detail::fill_fixed_impli<bfloat16_t>},
                {DType::BFloat16x2, detail::fill_fixed_impli<bfloat16x2_t>},
                {DType::Float32,  detail::fill_fixed_impli<float32_t>},
                {DType::Float64,  detail::fill_fixed_impli<float64_t>},
                {DType::Int4x2,   detail::fill_fixed_impli<int4x2_t>}
            };
            const auto it = fill_fns.find(this->dtype);
            if (it == fill_fns.end()) [[unlikely]] {
                throw ErrorHandling::NotImplemented();
            }
            it->second(this->data, this->numel, *this);
        } else if (this->device == Device::GPU) {
            detail::set_amount_gpu(*this, int64_t{1});
        } else [[unlikely]] {
            throw ErrorHandling::NotImplemented();
        }
    }

    /**
     * TODO - Write documentation.
     * @param new_shape
     * @param new_dtype
     * @param new_device
     * @param new_grad_status
     */
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
        this->nbytes = this->numel * DataTypes::dtype_bytes(dtype);
        compute_strides();

        if (device == Device::CPU) {
            // bt_aligned_alloc(64) instead of malloc — guarantees 64-byte alignment
            // enabling NT stores in conversion functions without runtime branching
            this->data = bt_aligned_alloc(this->nbytes);
            if (this->data == nullptr) {
                throw ErrorHandling::OutOfMemory();
            }
        } else if (device == Device::GPU) {
#if defined(USE_CUDA)
            CUDA_CHECK_ERROR(cudaMalloc(&this->data, this->nbytes));
#elif defined(USE_HIP)
            HIP_CHECK_ERROR(hipMalloc(&this->data, this->nbytes));
#else
            throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
        } else [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
    }

    /**
     * Zero the tensor's storage. Synchronous: any pending tasks are drained
     * first, and the buffer reads all-zero on return. CPU path: inline memset
     * from inside a pool worker (never submit from a worker) or for small
     * buffers; larger buffers are chunked across the pool, CCD-routed exactly
     * like fill_fixed_impli, with every chunk future drained before returning.
     * GPU path: unchanged (routed fill on the pool).
     * @brief Set every element of the tensor to zero.
     */
    void Tensor::set_zero() {
        sync();
        if (this->device == Device::CPU) {
            constexpr size_t inline_threshold = size_t{256} * 1024u;
            if (Threading::IsInPoolWorker() || this->nbytes <= inline_threshold) {
                std::memset(this->data, 0, this->nbytes);
                return;
            }
            // Chunked path mirroring fill_fixed_impli: byte ranges are
            // element-aligned (chunking by whole elements), routed to the
            // calling thread's CCD group when the span fits one CCD's LLC.
            auto& threadMgr = Threading::GetCPUThreadPool();
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            const auto elem_bytes = DataTypes::dtype_bytes(this->dtype);
            const auto chunk_elems = (this->numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(this->nbytes);
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            auto* const base = static_cast<char*>(this->data);
            auto futures = std::vector<std::shared_future<void>>{};
            for (auto t = size_t{0}; t < t_count; t++) {
                const auto begin_elem = t * chunk_elems;
                if (begin_elem >= this->numel) {
                    break;
                }
                const auto end_elem = std::min(begin_elem + chunk_elems, this->numel);
                const auto begin = begin_elem * elem_bytes;
                const auto end = end_elem * elem_bytes;
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group([base, begin, end] {
                        std::memset(base + begin, 0, end - begin);
                    }, group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit([base, begin, end] {
                        std::memset(base + begin, 0, end - begin);
                    }, Threading::TaskPriority::Normal).share());
                }
            }
            for (auto& f : futures) {
                f.get();
            }
            return;
        }
        if (this->device == Device::GPU) {
            detail::set_amount_gpu(*this, int64_t{0});
            return;
        }
        throw ErrorHandling::UnReachableCode();
    }

    /**
     * TODO - Write documentation.
     * @return
     */
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
