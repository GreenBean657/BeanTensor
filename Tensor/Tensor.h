#pragma once
#define BT_TENSOR_MAX_DIMS 2
#define BT_TENSOR_MAX_PARENTS 3

#include <cstddef>
#include <future>
#include <random>
#include <vector>

#include "DTypes.h"
#include "Helpers/Errors.h"
#include "Intrinsics/ConversionMethods.h"
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

#include "Ops/TensorOps.h"
namespace BeanTensor::Tensors { class Tensor; }
namespace BeanTensor::Intrinsics::detail {
    void launch_conv_cpu(Tensors::Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);
}
namespace BeanTensor::Tensors {
    enum class Device {
        CPU,
        GPU
    };
    struct Node;
    class Tensor {
        public:
        ~Tensor() {
            // Drain pending pool work before freeing buffers — tasks hold raw
            // pointers into data/convert_buffer. Errors are swallowed here:
            // a destructor cannot throw, and the buffers must be freed
            // regardless (the error already surfaced at sync() if anyone
            // was listening).
            for (auto& f : this->pending_tasks) {
                try { f.get(); } catch (...) {}
            }
            pending_tasks.clear();
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
                const DataTypes::DType dtype = DataTypes::DType::Float32,
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
        void random(const float64_t& min = 0.0, const float64_t& max = 1.0, int64_t seed = std::random_device()());
        void set_one();
        void set_zero();


        /**
         * @brief Sync Threaded Tensor tasks.
         * Manually force a resync of this Tensor, blocking the calling thread until all pending operations are complete. Used for multithreaded syncs.
         * @note Safe to call multiple times. An error from a pending task is
         * thrown once; the task list is drained regardless, so a later sync()
         * does not rethrow it.
         */
        void sync() const {
            const auto tasks = std::move(pending_tasks);
            pending_tasks.clear();
            for (auto& f : tasks) f.get();
        }

        /**
         * Give multiple, vectorized futures. Used for multithreaded syncs.
         * @note Do not use this for single futures, use give_future().
         * @param futures Vectorized futures.
         */
        void give_futures(const std::vector<std::shared_future<void>>& futures) const {
            for (const auto& f : futures) pending_tasks.push_back(f);
        }

        /**
         * Give a singular, non-vectorized future. Used for multithreaded syncs.
         * @note Do not use for multiple futures, use give_futures().
         * @param future Future to give.
         */
        void give_future(const std::shared_future<void>& future) const {
            pending_tasks.push_back(future);
        }

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
         * Make a copy of this tensor.
         * @param hard_copy Should the clone share the same pointer?
         * @return New Tensor.
         */
        [[nodiscard]] Tensor clone(bool hard_copy = true);

        /**
         * Move this tensor between devices (single-GPU; CPU <-> GPU only).
         * @param new_device Device to move to.
         */
        void to(const Device& new_device);

        /**
         * Convert this tensor's data to a new dtype in place. Asynchronous on
         * both devices: the call queues the conversion and returns; the dtype
         * flip and any clamp error materialize at sync(). Back-to-back
         * conversions of different tensors overlap on the thread pool.
         * Packed dtypes (Float16x2/BFloat16x2) preserve numel/shape: unpacked->packed
         * duplicates the value into both lanes, packed->unpacked converts the first lane.
         * @param new_dtype New DType of this tensor.
         * @param method Clamp tolerance for unsafe conversions.
         * @throws std::range_error At sync(): values clamped/invalid under a non-PERMISSIVE method.
         */
        void convert_dtype(const DataTypes::DType& new_dtype, ConversionClampMethod method = ConversionClampMethod::PERMISSIVE);

        template<typename T>
        [[nodiscard]] T* at(const std::initializer_list<size_t> idx) {
            throw ErrorHandling::NotImplemented();
        }

        [[nodiscard]] bool is_same_shape(const Tensor& t2) {
            return (ndim == t2.ndim) && std::equal(shape, shape + ndim, t2.shape);
        }


    private:
        //Multi-dimensional array data
        size_t shape[BT_TENSOR_MAX_DIMS]{}; // Indexes
        size_t dims[BT_TENSOR_MAX_DIMS]{}; // Strides
        size_t ndim = 0; // Length of shape (number of indexes)
        size_t numel = 0; // Flat number of elements, as if it was 1D.
        size_t nbytes = 0; // Number of bytes taken by the tensor data, NOT the tensor itself.

        mutable std::vector<std::shared_future<void>> pending_tasks;

        void* convert_buffer = nullptr;
        size_t convert_buffer_size = 0;

#ifdef USE_HIP
        hipStream_t stream = nullptr; // hipStream_t for GPUs
#elif defined(USE_CUDA)
        cudaStream_t stream = nullptr; // cudaStream_t for GPUS
#else
        void* stream = nullptr; // unused on CPU references
#endif

        bool owns_data = true; // Does this tensor own its data? If not, it is a view of another tensor.
        size_t offset = 0; // How far in the data is this tensor's beginning? Used for views of other tensors.
        bool is_contiguous = true; // Is this tensor contiguous in memory? If not, it is a view of another tensor.

        DataTypes::DType dtype = DataTypes::DType::Float32; // Data type of the tensor

        Device device = Device::CPU; // What device owns the tensor?

        void* data{}; //CPU/GPU data pointer. Actual tensor data.

        bool requires_grad = false; // Does this tensor require gradients?
        Tensor* grad = nullptr; // Accumulated gradient tensor. If this tensor requires gradients, this is where they are stored.
        Node* node = nullptr; // The node that represents this tensor in the computation graph.
        Tensor* parents[BT_TENSOR_MAX_PARENTS]{}; // The parent tensors of this tensor. Used for backpropagation.
        uint8_t n_parents = 0; // Number of parents of this tensor.
        bool is_leaf = true; // Is this tensor a leaf in the computation graph? If so, it is a leaf!

        std::vector<Tensor*> children{}; // Should anyone be informed this tensor was killed?

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
                        DataTypes::DType new_dtype = DataTypes::DType::Float32,
                        Device new_device = Device::CPU,
                        bool new_grad_status = false
        );

        /**
          * Deconstruct CPU data, besides buffer.
          * @note DOES NOT DECONSTRUCT BUFFER.
          */
        void kill_cpu_data();

        /**
 * Deconstruct the active Scratch Buffer.
 * @note DOES NOT DECONSTRUCT CPU/GPU DATA.
 */
        void kill_buffer();

        /**
         * Deconstruct GPU data, besides buffer.
         * @note DOES NOT DECONSTRUCT BUFFER.
         */
        void kill_gpu_data();

        /**
         * Notify a child tensor it needs to clone the data to stay alive.
         */
        void notify_child(const void* tensor_data);

        friend void detail::set_amount_gpu(Tensor& t, int32_t amount);
        friend void detail::set_random_gpu(Tensor& t, double min, double max, int64_t seed);
        friend void detail::move_to_gpu(Tensor& t, uint32_t device);
        friend void detail::move_to_cpu(Tensor& t);
        friend void detail::convert_gpu(Tensor& t, const BeanTensor::DataTypes::DType& new_dtype, ConversionClampMethod method);
        friend void Intrinsics::detail::launch_conv_cpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);

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