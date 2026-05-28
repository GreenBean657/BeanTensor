#pragma once
#define BT_TENSOR_MAX_DIMS 8 // PUT BEFORE INCLUDES
#define BT_TENSOR_MAX_PARENTS 3 // THIS HAS TO BE AT THE TOP

#include <future>
#include <random>

#include "Dtypes.h"
#include "Errors/logic.h"
#include "Helper/Threads/cpu_threading.h"

#include "TensorOps.h"
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#endif

namespace BeanTensor::Tensors {
    enum class Device {
        CPU,
        GPU
    };
    struct Node;
    class Tensor {
    private:

    public:

        Tensor() = delete;

        ~Tensor() {
            sync();

            kill_cpu_data();
            kill_gpu_data();
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

        [[nodiscard]] std::vector<size_t> contents_to_flat_vector() const;

        template<typename T>
        [[nodiscard]] T* at(const std::initializer_list<size_t> idx) {
            throw ErrorHandling::NotImplemented();
        }

        [[nodiscard]] bool is_same_shape(const Tensor& t2) {
            return (ndim == t2.ndim) && std::equal(shape, shape + ndim, t2.shape);
        }

        [[nodiscard]] Tensor make_slice(const size_t dim, const size_t index) {
            if (this->ndim <= 1) {
                throw std::invalid_argument("Cannot slice a 1D tensor into a 0D scalar.");
            }
            if (dim > this->ndim) {
                throw std::invalid_argument("Dimension to slice along is out of bounds.");
            }
            if (index > this->shape[dim]) {
                throw std::invalid_argument("Index out of bounds.");
            }
            /*
            assert(parent._ndim > 1); // slicing a 1D tensor would produce a 0-dim scalar;
            assert(dim < parent._ndim);
            assert(index < parent._shape[dim]);
            */
            std::vector<size_t> new_shape{};
            for (size_t i = 0; i < this->ndim; i++) {
                new_shape.push_back(shape[i]);
            }
            auto child = Tensor(new_shape, this->dtype, this->device, this->requires_grad);
            child.owns_data = false;

            child.offset = this->offset + index * this->strides[dim];

            for (size_t i = dim; i < this->ndim - 1; ++i) {
                this->shape[i]   = this->shape[i + 1];
                this->strides[i] = this->strides[i + 1];
            }
            child.ndim  -= 1;
            child.numel  = 1;
            for (size_t i = 0; i < child.ndim; ++i)
                child.numel *= child.shape[i];
            child.nbytes = child.numel * Casting::dtype_size(child.dtype);
            return child;
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
         * Deconstruct all CPU data.
         */
        void kill_cpu_data() {
            if (this->owns_data) {
                if (this->data != nullptr && this->device == Device::CPU) {
                    std::free(this->data);
                    this->data = nullptr;
                }
            }
            if (this->device == Device::CPU) {
                std::free(convert_buf);
                this->convert_buf = nullptr;
                this->convert_buf_size = 0;
            }
        }

        /**
         * Deconstruct all GPU data.
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
            if (this->convert_buf != nullptr) {
#if defined(USE_HIP)
                HIP_CHECK_ERROR(hipFree(this->convert_buf));
#elif defined(USE_CUDA)
                CUDA_CHECK_ERROR(cudaFree(this->convert_buf));
#else
                throw ErrorHandling::GPUTaskOnCPUBuild();
#endif
                this->convert_buf = nullptr;
                this->convert_buf_size = 0;
            }
        }

        /*
         * GPU Definitions
         */
        friend void detail::set_amount_gpu(Tensor& t, int32_t amount);
        friend void detail::set_random_gpu(Tensor& t, double min, double max, int64_t seed);
        friend void detail::move_to_gpu(Tensor& t, uint32_t device);
        friend void detail::move_to_cpu(Tensor& t);
        friend void detail::move_gpu_to_gpu(Tensor& t, uint32_t new_device);
        friend void detail::convert_gpu(Tensor& t, const BeanTensor::Casting::DType& new_dtype, bool use_inf_conversions);

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