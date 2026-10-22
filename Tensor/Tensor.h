/**
 * @file Tensor.h
 * @author GreenBean
 * @brief Tensor class declaration: dtype, shape, device and buffer ownership.
 * @version 0.2
 * @date 2026-10-18
 */
#pragma once
#define BT_TENSOR_MAX_DIMS 3
#define BT_TENSOR_MAX_PARENTS 3

#include <algorithm>
#include <cstddef>
#include <future>
#include <random>
#include <utility>
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
#include "Math/LinearAlgebra/LinearAlgebra.h" // IWYU pragma: keep
#include "Math/Reductions/Reductions.h" // IWYU pragma: keep
#include "Math/Activations/Activations.h" // IWYU pragma: keep
#include "Math/Optimization/Optimization.h" // IWYU pragma: keep
#include "Math/Autograd/Autograd.h" // IWYU pragma: keep
#include "Math/Shape/Shape.h" // IWYU pragma: keep
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
            [[unlikely]] default: return "Unknown";
        }
    };

    // Defined ahead of Tensor: Tensor's inline destructor and move assignment
    // delete their owned Node, which requires the complete type.
    struct alignas(8) Node {
        void (*_backward_fn)(Tensor*, Tensor**, size_t, DataTypes::DType);
        Tensor* _inputs[BT_TENSOR_MAX_PARENTS];
        Tensor* _saved[BT_TENSOR_MAX_PARENTS];
        uint8_t _n_inputs;
        uint8_t _n_saved;
        uint8_t _n_outputs_pending;
    };
    class alignas(8) Tensor {
        public:
        ~Tensor() {
            // Drain pending pool work before freeing buffers — tasks hold raw
            // pointers into data/convert_buffer. Errors are swallowed here:
            // a destructor cannot throw, and the buffers must be freed
            // regardless (the error already surfaced at sync() if anyone
            // was listening).
            (void)drain_tasks(); // OK to discard, a destructor doesn't need to care about errors - the Tensor is gone anyway.
            hand_off_children();
            detach_from_parents();
            kill_cpu_data();
            kill_gpu_data();
            kill_buffer();
            // Autograd state is owned: grad is a heap tensor, node a heap Node.
            delete this->node;
            delete this->grad;
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
         * (workers capture the source's address); a drained task error is
         * re-queued on this tensor as a poisoned future, surfacing at this
         * tensor's next sync() instead of being discarded. Buffers and view
         * links are then stolen and re-pointed to this object.
         * @param o Source tensor, left empty and unlinked.
         */
        Tensor(Tensor&& o) noexcept {
            steal_from(o);
        }


        /**
         * Move assignment. Releases this tensor's current view links and
         * buffers (child views are handed their own copies, as in the
         * destructor), then steals from the source. Drained task errors —
         * this tensor's own and the source's — are re-queued as poisoned
         * futures, surfacing at the next sync() instead of being discarded.
         * @param o Source tensor, left empty and unlinked.
         */
        Tensor& operator=(Tensor&& o) noexcept {
            if (this == &o) {
                return *this;
            }
            // Drain this tensor's own pending work first; its error is
            // re-queued after the steal so it survives the overwrite.
            const auto own_error = drain_tasks();
            hand_off_children();
            detach_from_parents();
            kill_cpu_data();
            kill_gpu_data();
            kill_buffer();
            delete this->node;
            delete this->grad;
            this->node = nullptr;
            this->grad = nullptr;
            steal_from(o);
            if (own_error.valid()) {
                this->pending_tasks.insert(this->pending_tasks.begin(), own_error);
            }
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
         * Matrix multiply. Rank-2: this(m×k) · other(k×n) → out(m×n). Rank-3 is
         * batched matmul: this(B×m×k) · other(B×k×n) → out(B×m×n), an
         * independent GEMM per batch index (GPU: strided-batched BLAS). Both
         * operands must share rank, dtype, and device, and be contiguous.
         * Packed dtypes (Float16x2/BFloat16x2) multiply per lane: the result's
         * lo/hi lanes are the products of the operands' lo/hi lanes. Half
         * dtypes accumulate in FP32 on every backend (CPU via the unaccel
         * conversion pathway, GPU via gemm_ex/GemmStridedBatchedEx FP32
         * compute). Integer dtypes multiply modularly (wrap accumulation in
         * the unsigned twin, truncate on store — the same contract as the integer
         * sum/bias_add kernels). When either operand has requires_grad=true the
         * output joins the computation graph (GEMM_G backward: dA = dOut·Bᵀ,
         * dB = Aᵀ·dOut, accumulated into the operands' Tensor::grad by
         * backward()).
         * The compute is queued asynchronously — failures materialize at
         * out.sync() — but the returned tensor is moved out, and the move
         * drains pending work, so matmul effectively blocks before returning
         * (same as clone()). A failure drained by that move is re-queued on
         * the result as a poisoned future and still surfaces at its next
         * sync(), per the async error contract.
         * @brief Matrix multiplication (rank-2) / batched matrix multiplication (rank-3).
         * @param other Right operand (k×n, or B×k×n with matching batch extent).
         * @return New tensor (m×n, or B×m×n) holding the product.
         * @throws ErrorHandling::MismatchedShapes Rank or inner/batch dimension mismatch.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous input.
         */
        [[nodiscard]] Tensor matmul(const Tensor& other) const;

        /**
         * Reshape this tensor into a zero-copy view with a new shape. The view
         * shares this tensor's buffer (view semantics: writes through either are
         * visible in both; if this tensor dies first, the view takes its own
         * copy via notify_child). Works identically on CPU and GPU — no data
         * moves. The compute is metadata-only, so the call is synchronous.
         * @brief Reshape into a view with the same element count.
         * @param new_shape New shape; numel must be unchanged, rank within
         * 1..BT_TENSOR_MAX_DIMS.
         * @return New tensor viewing this tensor's data with new_shape.
         * @throws ErrorHandling::MismatchedShapes numel changes or bad rank.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] Tensor reshape(const std::vector<size_t>& new_shape);

        /**
         * View into a per-dimension [start, end) box of this tensor. The
         * result shares this tensor's buffer (view semantics: bumped offset,
         * inherited strides, owns_data=false; writes through either are
         * visible in both; if this tensor dies first, the view takes its own
         * strided copy via notify_child). Slicing a leading dimension prefix
         * (or any box whose inherited strides match the contiguous strides of
         * its own shape) yields a contiguous view; slicing an inner dimension
         * yields a non-contiguous one, which ops requiring contiguous operands
         * reject with NotImplemented. Works identically on CPU and GPU — no
         * data moves, so the call is synchronous.
         * @brief Strided view over per-dimension [start, end) ranges.
         * @param ranges One [start, end) pair per dimension; rank must match
         * exactly. Empty ranges (start == end) yield a zero-extent dimension.
         * @return View tensor sharing this tensor's buffer.
         * @throws ErrorHandling::MismatchedShapes Range count differs from the rank.
         * @throws ErrorHandling::IndexOutOfRange start > end or end > shape[d].
         */
        [[nodiscard]] Tensor slice(const std::vector<std::pair<size_t, size_t>>&);

        /**
         * Split into equal view chunks along an axis — the lazy inverse of
         * cat: cat(split(axis, n), axis) reproduces this tensor. Each chunk
         * is a slice view (leading-axis chunks are contiguous; inner-axis
         * chunks are not). Synchronous and metadata-only, like slice.
         * @brief Split into equal slice views along an axis.
         * @param axis Axis to split along.
         * @param sections Number of chunks; must be nonzero and divide shape[axis].
         * @return One view per chunk, in axis order.
         * @throws ErrorHandling::IndexOutOfRange axis out of bounds.
         * @throws ErrorHandling::MismatchedShapes sections is 0 or does not divide shape[axis].
         */
        [[nodiscard]] std::vector<Tensor> split(size_t, size_t);

        /**
         * Select slabs along an axis by index: out[..., j, ...] =
         * this[..., indices[j], ...]. indices must be rank-1, of a plain
         * integer dtype (Int8..UInt64), contiguous, and on the same device;
         * this tensor must be contiguous. The output is a fresh contiguous
         * tensor with shape[axis] replaced by indices.numel (duplicate and
         * out-of-order indices are allowed; the output never joins the
         * autograd graph). CPU: the copy runs on the caller thread and index
         * bounds errors throw directly. GPU: the copy queues onto the thread
         * pool; the returned tensor is moved out, and the move drains pending
         * work, so index_select effectively blocks before returning, with
         * drained failures re-queued on the result and surfacing at its next
         * sync() (same contract as matmul).
         * @brief Select slabs along an axis by an index tensor.
         * @param axis Axis to select along.
         * @param indices Rank-1 integer index tensor.
         * @return New tensor with the selected slabs.
         * @throws ErrorHandling::IndexOutOfRange axis out of bounds, or an index
         * outside [0, shape[axis]) (GPU: at sync()).
         * @throws ErrorHandling::MismatchedShapes indices is not rank-1.
         * @throws ErrorHandling::MismatchedDTypes indices dtype is not an integer kind.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor index_select(size_t, const Tensor&) const;

        /**
         * Gather elements along an axis: out[m] = this[m with m[axis] replaced
         * by indices[m]]. indices must share this tensor's rank, be of a plain
         * integer dtype, contiguous, on the same device, and its extents must
         * not exceed this tensor's on the non-axis dimensions (PyTorch rule);
         * the output takes the indices' shape and this tensor's dtype. The
         * output never joins the autograd graph. Blocking contract matches
         * index_select (CPU: direct throw; GPU: at the result's sync()).
         * @brief Gather elements along an axis by an index tensor.
         * @param axis Axis to gather along.
         * @param indices Integer index tensor, same rank as this tensor.
         * @return New tensor with the indices' shape.
         * @throws ErrorHandling::IndexOutOfRange axis out of bounds, or an index
         * outside [0, shape[axis]) (GPU: at sync()).
         * @throws ErrorHandling::MismatchedShapes Rank mismatch, or an off-axis
         * index extent exceeds the input's.
         * @throws ErrorHandling::MismatchedDTypes indices dtype is not an integer kind.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor gather(size_t, const Tensor&) const;

        /**
         * Concatenate a list of tensors along an axis into a fresh contiguous
         * tensor (the materializing inverse of split). Every input must share
         * rank, dtype and device, have equal shapes except along the axis, and
         * be contiguous; the output extent along the axis is the sum of the
         * inputs'. The output never joins the autograd graph. Blocking
         * contract matches index_select (CPU: caller thread; GPU: queued, then
         * drained by the move out, failures surfacing at the result's sync()).
         * @brief Concatenate tensors along an axis.
         * @param tensors Non-empty list of inputs.
         * @param axis Axis to concatenate along.
         * @return New contiguous tensor holding all inputs in list order.
         * @throws ErrorHandling::MismatchedShapes Empty list, rank mismatch, or
         * off-axis shape mismatch.
         * @throws ErrorHandling::IndexOutOfRange axis out of bounds.
         * @throws ErrorHandling::MismatchedDTypes Dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous input.
         */
        [[nodiscard]] static Tensor cat(const std::vector<Tensor>&, size_t);

        /**
         * Plain SGD update, in place: w = w - lr * grad, elementwise. grad must
         * share this tensor's shape and device, and both must be contiguous.
         * When grad's dtype differs from this tensor's (backward(DType) makes
         * the grad dtype a per-call knob), a private copy of grad is converted
         * into the weight dtype via the single grad-conversion rule
         * (Math/Autograd/GradDType.h) and the update proceeds on the copy —
         * no MismatchedDTypes throw. Async on both devices: the update queues
         * onto the thread pool and errors materialize at sync().
         * Compute contract: fp32/fp64 use a single fma rounding (AVX2/AVX512
         * kernels when the CPU supports them); fp16/bf16 widen to fp32, fmaf,
         * narrow RNE; packed x2 dtypes compute per lane; integer dtypes compute
         * in fp64, clamp to the dtype range, and narrow round-toward-zero — at
         * sub-1 magnitudes negative updates truncate up toward 0 while positive
         * ones truncate down (asymmetric by up to 1), and a NaN update leaves
         * the integer element unchanged.
         * @brief In-place SGD: w -= lr * grad.
         * @param grad Gradient tensor (same shape/device as this; any dtype —
         * a mismatch converts a copy into this tensor's dtype).
         * @param lr Learning rate.
         * @throws ErrorHandling::MismatchedShapes Shapes differ.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        void sgd_update(const Tensor& grad, float64_t lr);

        /**
         * SGD step using this tensor's accumulated gradient: sgd_update(*grad, lr).
         * @brief In-place SGD step from Tensor::grad.
         * @param lr Learning rate.
         * @throws ErrorHandling::NoGradient requires_grad is false or grad is null.
         */
        void sgd_step(float64_t lr);

        /**
         * Backpropagate from this tensor through the recorded computation
         * graph. matmul/bias_add/relu/softmax_ce attach a Node to their output
         * whenever an input has requires_grad=true; backward() seeds this
         * tensor's gradient with ones IN grad_dtype and fires every reachable
         * node in reverse-topological order (Kahn's algorithm over
         * Node::_n_outputs_pending), each node's _backward_fn accumulating the
         * local gradients into its inputs' Tensor::grad. grad_dtype is a free
         * per-call parameter (any supported dtype, integer kinds included) with
         * no engine default: the seed and every intermediate node-to-node delta
         * ARE grad_dtype — backward nodes boundary-cast operands/deltas into it
         * via the single rule in Math/Autograd/GradDType.h. Leaf gradients
         * (requires_grad tensors with no node) accumulate across calls: the
         * first call's buffer pins the leaf's grad dtype and later calls with a
         * different grad_dtype convert their deltas into the pinned buffer
         * dtype (same single rule). Intermediate (node-owning) tensors' grads
         * are per-call: they are rebuilt in grad_dtype whenever it changed and
         * zeroed at the start of every call, so they always hold the current
         * call's delta in the current call's dtype. Parents are raw pointers:
         * every tensor in the graph must outlive backward(). Blocks before
         * returning (syncs every visited tensor and gradient).
         * @brief Run backpropagation from this tensor.
         * @throws ErrorHandling::NoGradient requires_grad is false and this
         * tensor has no graph node.
         */
        void backward(DataTypes::DType);

        /**
         * Drop this tensor's accumulated gradient (frees Tensor::grad).
         * @brief Clear the accumulated gradient.
         */
        void zero_grad();

        /**
         * @brief Does this tensor hold an accumulated gradient?
         * @return True when Tensor::grad is allocated.
         */
        [[nodiscard]] bool has_grad() const {
            return this->grad != nullptr;
        }

        /**
         * @brief Read this tensor's dtype (e.g. to check a gradient buffer's dtype in tests).
         * @return The tensor's DataTypes::DType.
         */
        [[nodiscard]] DataTypes::DType get_dtype() const {
            return this->dtype;
        }

        /**
         * @brief Access the accumulated gradient tensor.
         * @return Reference to Tensor::grad.
         * @throws ErrorHandling::NoGradient No gradient accumulated yet.
         */
        [[nodiscard]] Tensor& gradient() const {
            if (this->grad == nullptr) {
                throw ErrorHandling::NoGradient();
            }
            return *this->grad;
        }

        /**
         * Sum of every element, reduced to a shape-{1} tensor of the same dtype
         * and device. Accumulation: fp32 in fp32, fp64 in fp64, fp16/bf16 in
         * widened fp32 (RNE narrow on store), packed x2 per-lane fp32, integers
         * in a uint64 modular accumulator (wrap; bit-identical across backends
         * and chunkings). Float reduction order is backend-defined, so float
         * sums are not guaranteed bit-identical between CPU/SIMD/GPU tiers.
         * Blocks before returning (the merge happens on the caller thread).
         * @brief Full-tensor sum reduction.
         * @return Shape-{1} tensor holding the sum.
         */
        [[nodiscard]] Tensor sum() const;

        /**
         * Flat index of the maximum element. PyTorch-compatible NaN rule: NaN
         * beats every non-NaN value, so the FIRST NaN's index is returned when
         * one is present; otherwise the first strict maximum wins ties.
         * Packed dtypes compare per element on the widened (lo, hi) lane pair,
         * lexicographic, NaN-wins per lane. Blocks before returning.
         * @brief Full-tensor argmax.
         * @return Flat index of the maximum element.
         * @throws ErrorHandling::IndexOutOfRange Tensor is empty.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] uint64_t argmax() const;

        /**
         * Add a rank-1 bias vector to every row of this tensor, out of place:
         * out[..., c] = this[..., c] + bias[c]. bias must have shape
         * {shape[ndim-1]}, share this tensor's dtype and device, and both must
         * be contiguous. Async on both devices: the add queues onto the thread
         * pool and errors materialize at out.sync() — but the returned tensor
         * is moved out, and the move drains pending work, so bias_add
         * effectively blocks before returning (same as matmul()). A failure
         * drained by that move is re-queued on the result as a poisoned
         * future and still surfaces at its next sync().
         * Compute contract: fp32/fp64 add in their storage width (AVX2/AVX512
         * kernels when the CPU supports them, bit-identical to scalar);
         * fp16/bf16 widen to fp32, add, narrow RNE; packed x2 dtypes compute
         * per lane; integer dtypes add modularly (wrap, same contract as the
         * integer sum kernels); int4x2 wraps per nibble. When either operand
         * has requires_grad=true the output joins the computation graph
         * (bias_add_G backward: dx = dy, dbias = column-sums of dy).
         * @brief Out-of-place per-row bias add along the last dimension.
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::MismatchedShapes Bias is not rank-1 or its length differs from the last dimension.
         * @throws ErrorHandling::MismatchedDTypes Dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor bias_add(const Tensor&) const;

        /**
         * Rectified linear unit, out of place: out[i] = this[i] < 0 ? 0 :
         * this[i]. NaN propagates (the comparison is false for NaN) and -0.0
         * survives, on every backend and dispatch tier (the AVX kernels use
         * max(zero, x), whose unordered result is exactly x). Unsigned integer
         * dtypes copy through; int4x2 clamps per nibble. Async on both devices;
         * effectively blocks before returning (same move-drain as bias_add,
         * with drained failures re-queued on the result, surfacing at its
         * next sync()).
         * When this tensor has requires_grad=true the output joins the
         * computation graph (relu_G backward: dx = x > 0 ? dy : 0 — a select,
         * so a NaN input zeroes its gradient lane).
         * @brief Out-of-place ReLU.
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::NotImplemented Non-contiguous tensor.
         */
        [[nodiscard]] Tensor relu() const;

        /**
         * Numerically-stable softmax cross-entropy over the class dimension:
         * this must be rank-2 (rows x classes) logits; targets must be rank-1
         * with one integer class index per row (any integer dtype; int4x2 uses
         * the low nibble). Returns the mean loss in a shape-{1} tensor of this
         * tensor's dtype and device: per row, loss = m + log(sum(exp(x - m))) -
         * x_target with m = row max; a NaN logit makes that row's loss NaN.
         * Packed logits dtypes (Float16x2/BFloat16x2) and Int4x2 unpack each
         * element's two lanes as consecutive classes (effective class count is
         * 2 x shape[1]). Accumulation: fp32 for fp32/fp16/bf16/packed logits,
         * fp64 for fp64 and integer logits; rows merge in fp64. The stored mean
         * narrows to the logits dtype (RNE for floats, both lanes for packed,
         * clamp+truncate for integers — a NaN mean stores 0, as there is no
         * integer NaN). A target index outside [0, classes) throws
         * IndexOutOfRange. Float reduction order is backend-defined, so the
         * mean is not guaranteed bit-identical between CPU/SIMD/GPU tiers.
         * Blocks before returning (the merge happens on the caller thread).
         * When the logits have requires_grad=true the output joins the
         * computation graph (softmax_ce_G backward: dlogits = (softmax -
         * onehot(target)) * dloss / rows; targets never require gradients).
         * @brief Mean softmax cross-entropy loss over rank-2 logits.
         * @return Shape-{1} tensor holding the mean loss.
         * @throws ErrorHandling::MismatchedShapes Logits not rank-2, empty, or targets rank/length mismatch.
         * @throws ErrorHandling::MismatchedDTypes Targets dtype is not an integer kind.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         * @throws ErrorHandling::IndexOutOfRange A target index is out of bounds (at sync()).
         */
        [[nodiscard]] Tensor softmax_ce(const Tensor&) const;

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

        [[nodiscard]] bool is_same_shape(const Tensor& t2) const {
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
         * Drain pending pool work, awaiting every task, without throwing. The
         * first task error is captured into a poisoned shared_future that
         * rethrows at get(): the move operations re-queue it on the surviving
         * tensor so it surfaces at that tensor's next sync(); the destructor
         * discards it (unsynced errors are dropped on destroy). noexcept: a
         * poison allocation failure terminates — the standard noexcept bargain.
         * @return Poisoned future holding the first drained error, or invalid.
         */
        [[nodiscard]] std::shared_future<void> drain_tasks() const noexcept {
            const auto tasks = std::move(pending_tasks);
            pending_tasks.clear();
            std::shared_future<void> deferred_error;
            for (const auto& f : tasks) {
                try {
                    f.get();
                } catch (...) {
                    if (!deferred_error.valid()) {
                        std::promise<void> poison;
                        poison.set_exception(std::current_exception());
                        deferred_error = poison.get_future().share();
                    }
                }
            }
            return deferred_error;
        }

        /**
         * Move workhorse: drain the source, steal every member, and re-point
         * the view links (parents' children lists and children's view_parents
         * lists) from the source address to this object. A drained task error
         * is not discarded: it is prepended to this tensor's pending_tasks as
         * a poisoned future, surfacing at this tensor's next sync() per the
         * async error contract.
         * @param o Source tensor, left empty and unlinked.
         */
        void steal_from(Tensor& o) noexcept {
            const auto deferred_error = o.drain_tasks(); // pending workers capture &o — must drain before the steal
            std::copy_n(o.shape, BT_TENSOR_MAX_DIMS, this->shape);
            std::copy_n(o.dims, BT_TENSOR_MAX_DIMS, this->dims);
            this->ndim = o.ndim;
            this->numel = o.numel;
            this->nbytes = o.nbytes;
            this->offset = o.offset;
            this->pending_tasks = std::move(o.pending_tasks);
            if (deferred_error.valid()) {
                this->pending_tasks.insert(this->pending_tasks.begin(), deferred_error);
            }
            this->convert_buffer = o.convert_buffer;
            this->convert_buffer_size = o.convert_buffer_size;
            this->stream = o.stream;
            this->data = o.data;
            this->grad = o.grad;
            this->node = o.node;
            std::copy_n(o.parents, BT_TENSOR_MAX_PARENTS, this->parents);
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
            o.node = nullptr; // stolen above — the source must not delete it
            o.grad = nullptr; // stolen above — the source must not delete it
            o.children.clear();
            o.view_parents.clear();

            for (const auto p : this->view_parents) {
                std::ranges::replace(p->children, &o, this);
            }
            for (const auto c : this->children) {
                std::ranges::replace(c->view_parents, &o, this);
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

        /**
         * Lazily allocate this tensor's gradient buffer (same shape and device,
         * grad_dtype element type, zero-filled). No-op when grad already exists.
         * @brief Lazily allocate a zeroed gradient tensor in grad_dtype.
         */
        void ensure_grad(DataTypes::DType);

        /**
         * Accumulate a fresh gradient contribution into Tensor::grad: the
         * first contribution's buffer is adopted (moved); later ones are
         * added elementwise per dtype contract (floats single-rounding add,
         * halfs widen-add-narrow RNE, packed per lane, integers wrap). When
         * the existing grad buffer's dtype differs from delta's (a leaf buffer
         * pinned by an earlier backward() call), delta is first converted into
         * the buffer's dtype via the single grad-conversion rule
         * (Math/Autograd/GradDType.h).
         * @param delta Fresh gradient contribution, same shape/device.
         */
        void accumulate_grad(Tensor&& delta);

        /**
         * Attach a computation-graph node to this tensor (an op output).
         * No-op when no input requires gradients. Sets requires_grad,
         * is_leaf=false, parents/n_parents, and allocates the Node; _inputs
         * receive ALL op inputs (backward needs non-grad operands too, e.g.
         * GEMM's dA needs B), gated on requires_grad inside the node fn.
         * Parents are borrowed raw pointers — they must outlive backward().
         * @param backward_fn Node::_backward_fn implementation for the op.
         * @param a First op input.
         * @param b Second op input, when the op is binary.
         */
        void wire_node(void (*backward_fn)(Tensor*, Tensor**, size_t, DataTypes::DType), const Tensor& a, const Tensor* b = nullptr);

        friend void detail::set_amount_gpu(Tensor& t, int64_t amount);
        friend void detail::set_amount_gpu(Tensor& t, uint64_t amount);
        friend void detail::set_amount_gpu(Tensor& t, float64_t amount);
        friend void detail::set_random_gpu(Tensor& t, float64_t min, float64_t max, int64_t seed);
        friend void detail::move_to_gpu(Tensor& t);
        friend void detail::move_to_cpu(Tensor& t);
        friend void detail::convert_gpu(Tensor& t, const BeanTensor::DataTypes::DType& new_dtype, ConversionClampMethod method);
        friend void detail::matmul_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::matmul_gpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::sgd_update_cpu(Tensor&, const Tensor&, float64_t);
        friend void detail::sgd_update_gpu(Tensor&, const Tensor&, float64_t);
        friend void detail::sum_cpu(const Tensor&, Tensor&);
        friend void detail::sum_gpu(const Tensor&, Tensor&);
        friend void detail::argmax_cpu(const Tensor&, uint64_t&);
        friend void detail::argmax_gpu(const Tensor&, uint64_t&);
        friend void detail::bias_add_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::bias_add_gpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::relu_cpu(const Tensor&, const Tensor&);
        friend void detail::relu_gpu(const Tensor&, const Tensor&);
        friend void detail::softmax_ce_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::softmax_ce_gpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::gemm_g_node(Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::bias_add_g_node(Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::relu_g_node(Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::softmax_ce_g_node(Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::grad_acc_cpu(Tensor&, const Tensor&);
        friend void detail::grad_acc_gpu(Tensor&, const Tensor&);
        friend void detail::transpose_last2_cpu(const Tensor&, Tensor&);
        friend void detail::transpose_last2_gpu(const Tensor&, Tensor&);
        friend void detail::relu_g_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::relu_g_gpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::bias_add_g_cpu(const Tensor&, Tensor&);
        friend void detail::bias_add_g_gpu(const Tensor&, Tensor&);
        friend void detail::softmax_ce_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        friend void detail::softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        friend void detail::index_select_cpu(const Tensor&, const Tensor&, Tensor&, size_t);
        friend void detail::index_select_gpu(const Tensor&, const Tensor&, Tensor&, size_t);
        friend void detail::gather_cpu(const Tensor&, const Tensor&, Tensor&, size_t);
        friend void detail::gather_gpu(const Tensor&, const Tensor&, Tensor&, size_t);
        friend void detail::cat_cpu(const std::vector<Tensor>&, Tensor&, size_t);
        friend void detail::cat_gpu(const std::vector<Tensor>&, Tensor&, size_t);
        friend void Intrinsics::detail::launch_conv_cpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);

    };
}