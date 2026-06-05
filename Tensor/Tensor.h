#pragma once
#define BT_TENSOR_MAX_DIMS 8 // PUT BEFORE INCLUDES
#define BT_TENSOR_MAX_PARENTS 3 // THIS HAS TO BE AT THE TOP

#include <cassert>
#include <future>
#include <random>
#include <unordered_map>

#include "Dtypes.h"
#include "Errors/logic.h"
#include "Helper/Threads/cpu_threading.h"

#include "TensorOps.h"
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace BeanTensor::Tensors { class Tensor; }
namespace BeanTensor::Tensors::detail {
    template<typename DType, typename TensorType> requires std::is_arithmetic_v<DType> || std::is_same_v<DType, Casting::float16_t> || std::is_same_v<DType, Casting::bfloat16_t>
    void flat_vector_impli(std::vector<DType>& vec, const Tensors::Tensor& exec);
}

namespace BeanTensor::Tensors {
    enum class Device {
        CPU,
        GPU
    };
    struct Node;
    class Tensor {
    public:

        Tensor() = delete;

        ~Tensor() {
            sync();
            if (!this->children.empty() && this->owns_data) {
                for (const auto & i : this->children) {
                    i->notify_child(this->data);
                }
            }
            kill_cpu_data();
            kill_gpu_data();
            kill_buffer();
        }
        explicit Tensor(const std::vector<size_t>& shape,
                        const Casting::DType dtype = Casting::DType::Float32,
                        const Device device = Device::CPU,
                        const bool requires_grad = false) {
            set_tensor(shape, dtype, device, requires_grad);
        }


        /**
         *
         * @param min Lowest possible value for this tensor.
         * @param max Highest possible value for this tensor.
         * @param seed Seed to use. Default is randomized.
         */
        void random(const double& min = 0.0, const double& max = 1.0, int64_t seed = std::random_device()());
        void set_one();
        void set_zero();

        /**
         * Manually force a resync of this Tensor, blocking the calling thread until all pending operations are complete. Used for multithreaded syncs.
         * @note Safe to call multiple times.
         */
        void sync() const {
            for (auto& f : pending) f.get();
            pending.clear();
        }

        /**
         * Give multiple, vectorized futures. Used for multithreaded syncs.
         * @note Do not use this for single futures, use give_future().
         * @param futures Vectorized futures.
         */
        void give_futures(const std::vector<std::shared_future<void>>& futures) const {
            sync();
            pending = futures;
        }

        /**
         * Give a singular, non-vectorized future. Used for multithreaded syncs.
         * @note Do not use for multiple futures, use give_futures().
         * @param future Future to give.
         */
        void give_future(const std::shared_future<void>& future) const {
            sync();
            pending.push_back(future);
        }

        void convert_dtype(const Casting::DType& new_dtype, bool use_nan_conversions = false);

        /**
         * Get the shape of a tensor.
         * @return std::string representing the shape.
         */
        [[nodiscard]] std::string shape_to_string() const;

        /**
         *
         * @return String showing this tensor's contents.
         */
        [[nodiscard]] std::string contents_to_string() const;

        /**
         * Get a flat, 1D vector made up of this tensor.
         * @tparam DType DType of the resulting vector.
         * @return Vector to output.
         */
        template<typename DType> requires std::is_arithmetic_v<DType>
                                          or std::is_same_v<DType, Casting::float16_t>
                                          or std::is_same_v<DType, Casting::bfloat16_t>
        [[nodiscard]] std::vector<DType> contents_to_flat_vector() const {
            std::vector<DType> vec;
            vec.resize(this->numel);
            if (this->device == Device::CPU) {
                using FlatFn = void(*)(std::vector<DType>&, const Tensor&);
                static const std::unordered_map<Casting::DType, FlatFn> flat_map = {
                    {Casting::DType::Int8,     detail::flat_vector_impli<DType, int8_t>},
                    {Casting::DType::Int16,    detail::flat_vector_impli<DType, int16_t>},
                    {Casting::DType::Int32,    detail::flat_vector_impli<DType, int32_t>},
                    {Casting::DType::Int64,    detail::flat_vector_impli<DType, int64_t>},
                    {Casting::DType::UInt8,    detail::flat_vector_impli<DType, uint8_t>},
                    {Casting::DType::UInt16,   detail::flat_vector_impli<DType, uint16_t>},
                    {Casting::DType::UInt32,   detail::flat_vector_impli<DType, uint32_t>},
                    {Casting::DType::UInt64,   detail::flat_vector_impli<DType, uint64_t>},
                    {Casting::DType::Float16,  detail::flat_vector_impli<DType, Casting::float16_t>},
                    {Casting::DType::BFloat16, detail::flat_vector_impli<DType, Casting::bfloat16_t>},
                    {Casting::DType::Float32,  detail::flat_vector_impli<DType, float>},
                    {Casting::DType::Float64,  detail::flat_vector_impli<DType, double>},
                };
                const auto it = flat_map.find(this->dtype);
                if (it == flat_map.end()) throw ErrorHandling::NotImplemented();
                it->second(vec, *this);
            } else if (device == Device::GPU) {
                throw ErrorHandling::NotImplemented();
            } else {
                __builtin_unreachable();
            }
            return vec;
        }

        /**
         * Make a copy of this tensor.
         * @param hard_copy Should the clone share the same pointer?
         * @return New Tensor.
         */
        [[nodiscard]] Tensor clone(bool hard_copy = true);

        template<typename T>
        [[nodiscard]] T* at(const std::initializer_list<size_t> idx) {
            throw ErrorHandling::NotImplemented();
        }

        [[nodiscard]] bool is_same_shape(const Tensor& t2) {
            return (ndim == t2.ndim) && std::equal(shape, shape + ndim, t2.shape);
        }

        void to(const Device &new_device, uint64_t id = 0);

    private:
        size_t shape[BT_TENSOR_MAX_DIMS]{}; // Multidimensional Tensor indexes
        size_t strides[BT_TENSOR_MAX_DIMS]{}; // Multidimensional Tensor strides
        size_t ndim = 0; // length of shape (Number of indexes)
        size_t numel = 0; // Flat number of elements
        size_t nbytes = 0; // Number of bytes

        mutable std::vector<std::shared_future<void>> pending; // Promise for if this tensor is ready to be touched.

        void* convert_buf = nullptr; // Conversion and operation buffers
        size_t convert_buf_size = 0; // SIze of the current conversion buffer

#ifdef USE_HIP
        hipStream_t stream = nullptr; // hipStream_t for GPUs
#elif defined(USE_CUDA)
        cudaStream_t stream = nullptr; // cudaStream_t for GPUS
#else
        void* stream = nullptr; // unused on CPU references
#endif
        uint32_t gpu_id = 0; // GPU owner ID

        bool owns_data = true; // Is this tensor the sole owner of its data?
        size_t offset = 0; // How far in the data is this tensor's beginning? (In elements)
        bool is_contiguous = true; // Is this tensor contiguous in memory?

        Casting::DType dtype = Casting::DType::Float32; // What datatype is this tensor?

        Device device = Device::CPU; // What device owns this tensors' data?

        void* data{}; // CPU or GPU pointer to data

        bool requires_grad = false; // Does this tensor require gradient tracking?
        Tensor* grad = nullptr; // Accumulated gradient of same shape as this tensor
        Node* grad_fn = nullptr; // Parent grad function for autograd
        Tensor* parents[BT_TENSOR_MAX_PARENTS] = {}; // Parent tensors for autograd
        uint8_t n_parents = 0; // Number of parent tensors (for 2-4)
        bool is_leaf = true; // Is this tensor a leaf in the autograd graph?

        std::vector<Tensor*>children{}; // Should this tensor notify anyone when it's killed?
        /**
         * Compute the strides of this tensor based on its shape. Should be called after setting shape and ndim.
         */
        void compute_strides();

        /**
         * Set the properties of this tensor and allocate memory for it.
         * @param new_shape Vector representing this tensor shape.
         * @param new_dtype New DType of this tensor.
         * @param new_device Device to launch this tensor on.
         * @param new_grad_status Should this tensor be gradient monitored?
         * @throws ErrorHandling::GPUTaskOnCPUBuild Tensor is set to a GPU device on CPU build.
         */
        void set_tensor(const std::vector<size_t>& new_shape,
                        Casting::DType new_dtype = Casting::DType::Float32,
                        Device new_device = Device::CPU,
                        bool new_grad_status = false
        );

        /**
         * Deconstruct CPU data, besides buffer.
         * @note DOES NOT DECONSTRUCT BUFFER.
         */
        void kill_cpu_data() {
            if (this->owns_data) {
                if (this->data != nullptr && this->device == Device::CPU) {
                    std::free(this->data);
                    this->data = nullptr;
                }
            }
        }

        /**
         * Deconstruct the active Scratch Buffer.
         * @note DOES NOT DECONSTRUCT CPU/GPU DATA.
         */
        void kill_buffer() {
            if (this->device == Device::CPU) {
                if (this->convert_buf != nullptr && this->convert_buf_size != 0) {
                    std::free(convert_buf);
                    this->convert_buf = nullptr;
                    this->convert_buf_size = 0;
                }
            } else if (this->device == Device::GPU) {
                if (this->convert_buf != nullptr && this->convert_buf_size != 0) {
#if defined(USE_HIP)
                    HIP_CHECK_ERROR(hipFree(this->convert_buf));
                    this->convert_buf = nullptr;
                    this->convert_buf_size = 0;
#elif defined(USE_CUDA)
                    CUDA_CHECK_ERROR(cudaFree(this->convert_buf)); // HERE
                    this->convert_buf = nullptr;
                    this->convert_buf_size = 0;
#else
                    throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
                }
            }
        }

        /**
         * Deconstruct GPU data, besides buffer.
         * @note DOES NOT DECONSTRUCT BUFFER.
         */
        void kill_gpu_data() {
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
         * Notify a child tensor it needs to clone the data to stay alive.
         */
        void notify_child(const void* data);

        /*
         * GPU Definitions
         */
        friend void detail::set_amount_gpu(Tensor& t, int32_t amount);
        friend void detail::set_random_gpu(Tensor& t, double min, double max, int64_t seed);
        friend void detail::move_to_gpu(Tensor& t, uint32_t device);
        friend void detail::move_to_cpu(Tensor& t);
        friend void detail::move_gpu_to_gpu(Tensor& t, uint32_t new_device);
        friend void detail::convert_gpu(Tensor& t, const BeanTensor::Casting::DType& new_dtype, bool use_inf_conversions);

        template<typename DType, typename TensorType> requires std::is_arithmetic_v<DType> || std::is_same_v<DType, Casting::float16_t> || std::is_same_v<DType, Casting::bfloat16_t>
        friend void detail::flat_vector_impli(std::vector<DType>& vec, const Tensors::Tensor& exec);

    };
    struct Node {
        void (*_backward_fn)(Tensor* output_grad, Tensor** inputs, size_t n_inputs);
        Tensor* _inputs[BT_TENSOR_MAX_PARENTS];
        size_t  _n_inputs;
        Tensor* _saved[BT_TENSOR_MAX_PARENTS];
        size_t  _n_saved;
        size_t  _n_outputs_pending;
    };
}

namespace BeanTensor::Tensors::detail {
    template<typename DType, typename TensorType> requires std::is_arithmetic_v<DType> || std::is_same_v<DType, Casting::float16_t> || std::is_same_v<DType, Casting::bfloat16_t>
    void flat_vector_impli(std::vector<DType>& vec, const Tensors::Tensor& exec) {
        if (exec.numel == 0) return;
        auto& threadMgr = Threading::get_cpu_thread_pool();
        assert(exec.device == Device::CPU);
        assert(vec.size() > 0);
        auto worker = [&vec, &exec](const size_t t_start, const size_t t_end) {
            DType* ptr = vec.data();
            const auto* src = static_cast<const TensorType*>(exec.data);
            for (size_t i = t_start; i < t_end; ++i) {
                if constexpr (std::is_same_v<TensorType, Casting::float16_t> || std::is_same_v<TensorType, Casting::bfloat16_t>) {
                    ptr[i] = static_cast<DType>(Casting::to_float(src[i]));
                } else {
                    ptr[i] = static_cast<DType>(src[i]);
                }
            }
        };
        if (vec.size() <= 1024) {
            worker(0, vec.size());
        } else {
            std::vector<std::shared_future<void>> futures;
            size_t threads = Hardware::CPU().threads;
            const size_t chunk_size = (vec.size() + threads - 1) / threads;
            for (size_t i = 0; i < threads; i++) {
                const size_t t_start = i * chunk_size;
                const size_t t_end = std::min(t_start + chunk_size, vec.size());
                futures.push_back(threadMgr.submit(
                    [=] { worker(t_start, t_end); },
                    Threading::ThreadPriority::High
                ).share());
            }
            for (auto& future : futures) future.wait();
        }
    }
}