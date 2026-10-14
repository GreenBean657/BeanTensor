#pragma once
#define BT_TENSOR_MAX_DIMS 2
#define BT_TENSOR_MAX_PARENTS 3

#include <algorithm>
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
namespace BeanTensor::Tensors {
    class Tensor;
}
namespace BeanTensor::Intrinsics::detail {
    void launch_conv_cpu(Tensors::Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);
}
namespace BeanTensor::Tensors {

    enum class Device {
        CPU,
        GPU
    };

    /**
     * Convert device enum to text - when a device is unexplored, default to Unknown.
     * @brief Convert Enum Device to a string representation.
     * @param device Device source to check.
     * @return String representation of the device source.
     */
    [[nodiscard]] __inline__ std::string stringify_device_source(const Device& device) {
        switch (device) {
            case Device::CPU: return "CPU";
            case Device::GPU: return "GPU";
            default: return "Unknown";
        }
    };

    struct Node;
    class alignas(8) Tensor {
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
            hand_off_children();
            detach_from_parents();
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

        // Copying is disabled: a memberwise copy would double-free data and
        // dangling-link the view graph. clone(true) is the explicit deep copy.
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        /**
         * Move constructor. The source's pending tasks are drained first
         * (workers capture the source's address), then its buffers and view
         * links are stolen and re-pointed to this object.
         * @param o Source tensor, left empty and unlinked.
         */
        Tensor(Tensor&& o) {
            steal_from(o);
        }

        /**
         * Move assignment. Releases this tensor's current view links and
         * buffers (child views are handed their own copies, as in the
         * destructor), then steals from the source.
         * @param o Source tensor, left empty and unlinked.
         */
        Tensor& operator=(Tensor&& o) {
            if (this == &o) {
                return *this;
            }
            sync();
            hand_off_children();
            detach_from_parents();
            kill_cpu_data();
            kill_gpu_data();
            kill_buffer();
            steal_from(o);
            return *this;
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

        /**
         * Bounds-checked strided element access.
         * @brief Read an element by multidimensional index.
         * @tparam T Element type, matching the tensor's dtype.
         * @param idx One index per dimension; must match ndim exactly.
         * @return Copy of the element's value. A raw pointer is never exposed:
         * it would only be valid for CPU tensors, never GPU ones.
         * @throws ErrorHandling::IndexOutOfRange If the rank or any index is out of bounds.
         */
        template<typename T>
        [[nodiscard]] T at(const std::initializer_list<size_t> idx) {
            T value;
            read_element(checked_element_offset(idx), &value, sizeof(T));
            return value;
        }

        /**
         * Bounds-checked strided element write.
         * @brief Assign an element by multidimensional index.
         * @tparam T Element type, matching the tensor's dtype.
         * @param idx One index per dimension; must match ndim exactly.
         * @param value Value to write into the element.
         * @throws ErrorHandling::IndexOutOfRange If the rank or any index is out of bounds.
         */
        template<typename T>
        void set(const std::initializer_list<size_t> idx, const T& value) {
            write_element(checked_element_offset(idx), &value, sizeof(T));
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
        size_t offset = 0; // How far in the data is this tensor's beginning? Used for views of other tensors.

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

        void* data{}; //CPU/GPU data pointer. Actual tensor data.

        Tensor* grad = nullptr; // Accumulated gradient tensor. If this tensor requires gradients, this is where they are stored.
        Node* node = nullptr; // The node that represents this tensor in the computation graph.
        Tensor* parents[BT_TENSOR_MAX_PARENTS]{}; // The parent tensors of this tensor. Used for backpropagation.

        std::vector<Tensor*> children{}; // Should anyone be informed this tensor was killed?
        std::vector<Tensor*> view_parents{}; // Tensors whose data this tensor views. Deregistered on death/move.

        DataTypes::DType dtype = DataTypes::DType::Float32; // Data type of the tensor

        Device device = Device::CPU; // What device owns the tensor?

        // Alignment > 1 byte with 1 byte decreases size verus scattered pools.
        bool owns_data = true; // Does this tensor own its data? If not, it is a view of another tensor.
        bool is_contiguous = true; // Is this tensor contiguous in memory? If not, it is a view of another tensor.
        bool requires_grad = false; // Does this tensor require gradients?
        uint8_t n_parents = 0; // Number of parents of this tensor.
        bool is_leaf = true; // Is this tensor a leaf in the computation graph? If so, it is a leaf!

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
         * The child takes ownership of its fresh buffer and unregisters from
         * the dying parent.
         * @param parent The tensor being destroyed (source of the clone).
         */
        void notify_child(Tensor* parent);

        /**
         * Remove this view from every parent it aliases. Shared by the
         * destructor and move assignment.
         */
        void detach_from_parents() {
            for (const auto p : this->view_parents) {
                std::erase(p->children, this);
            }
            view_parents.clear();
        }

        /**
         * Hand every child view off before this tensor dies or is overwritten.
         * Owners clone the data into each child (notify_child). A view owns no
         * data to clone, so its children are re-homed to this view's own
         * parents instead — they alias the same buffer, and the link chain
         * stays valid after this view is gone.
         */
        void hand_off_children() {
            if (this->children.empty()) {
                return;
            }
            if (this->owns_data) {
                for (const auto& i : this->children) {
                    i->notify_child(this);
                }
            } else {
                for (const auto& i : this->children) {
                    std::erase(i->view_parents, this);
                    for (const auto p : this->view_parents) {
                        i->view_parents.push_back(p);
                        p->children.push_back(i);
                    }
                }
            }
            children.clear();
        }

        /**
         * Move workhorse: drain the source, steal every member, and re-point
         * the view links (parents' children lists and children's view_parents
         * lists) from the source address to this object.
         * @param o Source tensor, left empty and unlinked.
         */
        void steal_from(Tensor& o) {
            o.sync(); // pending workers capture &o — must drain before the move
            std::copy(o.shape, o.shape + BT_TENSOR_MAX_DIMS, this->shape);
            std::copy(o.dims, o.dims + BT_TENSOR_MAX_DIMS, this->dims);
            this->ndim = o.ndim;
            this->numel = o.numel;
            this->nbytes = o.nbytes;
            this->offset = o.offset;
            this->pending_tasks = std::move(o.pending_tasks);
            this->convert_buffer = o.convert_buffer;
            this->convert_buffer_size = o.convert_buffer_size;
            this->stream = o.stream;
            this->data = o.data;
            this->grad = o.grad;
            this->node = o.node;
            std::copy(o.parents, o.parents + BT_TENSOR_MAX_PARENTS, this->parents);
            this->children = std::move(o.children);
            this->view_parents = std::move(o.view_parents);
            this->dtype = o.dtype;
            this->device = o.device;
            this->owns_data = o.owns_data;
            this->is_contiguous = o.is_contiguous;
            this->requires_grad = o.requires_grad;
            this->n_parents = o.n_parents;
            this->is_leaf = o.is_leaf;

            o.convert_buffer = nullptr;
            o.convert_buffer_size = 0;
            o.data = nullptr;
            o.children.clear();
            o.view_parents.clear();

            for (const auto p : this->view_parents) {
                std::replace(p->children.begin(), p->children.end(), &o, this);
            }
            for (const auto c : this->children) {
                std::replace(c->view_parents.begin(), c->view_parents.end(), &o, this);
            }
        }

        /**
         * Compute the flat element offset (view offset included) for a
         * multidimensional index. Syncs first, then bounds-checks the rank
         * and every index.
         * @param idx One index per dimension; must match ndim exactly.
         * @return Element offset into the data buffer.
         * @throws ErrorHandling::IndexOutOfRange If the rank or any index is out of bounds.
         */
        size_t checked_element_offset(std::initializer_list<size_t> idx) const;

        /**
         * Read one element (bytes at elem_offset * bytes) into dst.
         * Blocks on GPU tensors via a synchronous device-to-host copy.
         */
        void read_element(size_t elem_offset, void* dst, size_t bytes);

        /**
         * Write one element (bytes at elem_offset * bytes) from src.
         * Blocks on GPU tensors via a synchronous host-to-device copy.
         */
        void write_element(size_t elem_offset, const void* src, size_t bytes);

        friend void detail::set_amount_gpu(Tensor& t, int64_t amount);
        friend void detail::set_amount_gpu(Tensor& t, uint64_t amount);
        friend void detail::set_amount_gpu(Tensor& t, float64_t amount);
        friend void detail::set_random_gpu(Tensor& t, float64_t min, float64_t max, int64_t seed);
        friend void detail::move_to_gpu(Tensor& t);
        friend void detail::move_to_cpu(Tensor& t);
        friend void detail::convert_gpu(Tensor& t, const BeanTensor::DataTypes::DType& new_dtype, ConversionClampMethod method);
        friend void Intrinsics::detail::launch_conv_cpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);

    };
    struct alignas(8) Node {
        void (*_backward_fn)(Tensor* output_grad, Tensor** inputs, size_t n_inputs);
        Tensor* _inputs[BT_TENSOR_MAX_PARENTS];
        Tensor* _saved[BT_TENSOR_MAX_PARENTS];
        uint8_t _n_inputs;
        uint8_t _n_saved;
        uint8_t _n_outputs_pending;
    };
}