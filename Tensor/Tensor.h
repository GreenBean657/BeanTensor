/**
 * @file Tensor.h
 * @author GreenBean
 * @brief Tensor class declaration: dtype, shape, device and buffer ownership.
 * @version 0.4.10
 * @date 2026-10-29
 */
#pragma once
#define BT_TENSOR_MAX_DIMS 4
#define BT_TENSOR_MAX_PARENTS 3

#include <algorithm>
#include <cstddef>
#include <future>
#include <optional>
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
#include "Math/Elementwise/Elementwise.h" // IWYU pragma: keep
#include "Math/LinearAlgebra/LinearAlgebra.h" // IWYU pragma: keep
#include "Math/Reductions/Reductions.h" // IWYU pragma: keep
#include "Math/Reductions/Mean/Mean.h" // IWYU pragma: keep
#include "Math/Compare/Compare.h" // IWYU pragma: keep
#include "Math/Activations/Activations.h" // IWYU pragma: keep
#include "Math/Optimization/Optimization.h" // IWYU pragma: keep
#include "Math/Loss/Loss.h" // IWYU pragma: keep
#include "Math/Autograd/Autograd.h" // IWYU pragma: keep
#include "Math/Shape/Shape.h" // IWYU pragma: keep
#include "Math/Convolution/Convolution.h" // IWYU pragma: keep
#include "Math/Spatial/Spatial.h" // IWYU pragma: keep
namespace BeanTensor::Tensors {
    class Tensor;
    struct Node;
}
namespace BeanTensor::Intrinsics::detail {
    void launch_conv_cpu(Tensors::Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);
}
namespace BeanTensor::Math::Dropout {
    Tensors::Tensor train_mask(const Tensors::Tensor& x, float64_t p, int64_t seed);
    Tensors::Tensor train_mask_cpu(const Tensors::Tensor& x, float64_t p, int64_t seed);
    Tensors::Tensor train_mask_gpu(const Tensors::Tensor& x, float64_t p, int64_t seed);
}
namespace BeanTensor::Tensors::detail {
    struct BcastReducePlan;
    void conv2d_g_node_gpu(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
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
    }

    // Defined ahead of Tensor: Tensor's inline destructor and move assignment
    // delete their owned Node, which requires the complete type.
    struct alignas(8) Node {
        // Trampolines receive the owning Node so they can read per-op state
        // (_scalar, _saved) without a bit-cast channel through _inputs.
        void (*_backward_fn)(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        float64_t _scalar; // Tensor-scalar operand of the op; zero when the op has none.
        Tensor* _inputs[BT_TENSOR_MAX_PARENTS];
        Tensor* _saved[BT_TENSOR_MAX_PARENTS];
        uint8_t _n_inputs;
        uint8_t _n_saved;
        uint8_t _n_outputs_pending;
    };
    class alignas(8) Tensor final {
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
            kill_gpu_teardown_noexcept();
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
            kill_gpu_teardown_noexcept();
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
         * @brief Fill this tensor with uniformly distributed random values in [min, max].
         * @param min Lowest possible value for this tensor.
         * @param max Highest possible value for this tensor.
         * @param seed Explicit seed, or omitted/nullopt to draw one seed from the
         * process-wide default generator (see Random/Generator.h). The draw happens
         * on the calling thread before dispatch.
         */
        void random(const float64_t& min = 0.0, const float64_t& max = 1.0, std::optional<int64_t> seed = std::nullopt);
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
         * Mask semantics: the UInt8 0/1 mask produced by equal() casts to float
         * through the generic integer->float pathway, so false -> 0.0 and
         * true -> 1.0 exactly (0 and 1 are representable in every float
         * dtype, so no clamp error can occur); accuracy =
         * equal(pred, labels).convert_dtype(Float32).mean() is therefore exact.
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
         * 2-D convolution over NCHW input using im2col + the existing GEMM
         * helpers. Input must be rank-4 {N,C,H,W}; weight must be rank-4
         * {C,R,S,K} (input channels, kernel height, kernel width, output
         * channels). Output shape is {N,K,Hout,Wout} with Hout/Wout floored.
         * Both operands must share dtype and device and be contiguous. The
         * output joins the computation graph when any input requires grad
         * (Conv2dG backward: recomputes im2col(x), transposes dy into {M,K}
         * scratch, then dW = colᵀ·dyT, db = rowsum(dyT), dx = col2im(dyT·Wᵀ)).
         * The compute is queued asynchronously; failures materialize at sync().
         * @brief 2-D convolution (NCHW) computed as im2col + GEMM.
         * @param weight Convolution kernel {C,R,S,K}.
         * @param bias Per-output-channel bias {K}.
         * @param stride Stride along H and W.
         * @param padding Zero-padding on all sides.
         * @return Convolved output {N,K,Hout,Wout}.
         * @throws ErrorHandling::MismatchedShapes rank mismatch, channel mismatch,
         *         kernel larger than padded input, or stride == 0.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous input.
         */
        [[nodiscard]] Tensor conv2d(const Tensor& weight, const Tensor& bias, uint32_t stride, uint32_t padding) const;
        /**
         * Bias-less overload of conv2d. Same contract as the three-argument
         * form, but wires a 2-parent graph node (no bias gradient).
         * @brief 2-D convolution without bias.
         * @param weight Convolution kernel {C,R,S,K}.
         * @param stride Stride along H and W.
         * @param padding Zero-padding on all sides.
         * @return Convolved output {N,K,Hout,Wout}.
         * @throws ErrorHandling::MismatchedShapes rank mismatch, channel mismatch,
         *         kernel larger than padded input, or stride == 0.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous input.
         */
        [[nodiscard]] Tensor conv2d(const Tensor& weight, uint32_t stride, uint32_t padding) const;

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
         * Flatten a contiguous tensor to rank 2: out shape is
         * {shape[0], numel / shape[0]}. This is a metadata-only view over the
         * same buffer; the output joins the computation graph when the input
         * requires grad (FlattenG backward: dx = reshape(dy, x.shape())).
         * @brief Flatten to a rank-2 view.
         * @return Rank-2 tensor viewing this tensor's data as {d0, numel/d0}.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] Tensor flatten() const;

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
         * ranges: One [start, end) pair per dimension; rank must match
         * exactly. Empty ranges (start == end) yield a zero-extent dimension.
         * @return View tensor sharing this tensor's buffer.
         * @throws ErrorHandling::MismatchedShapes Range count differs from the rank.
         * @throws ErrorHandling::IndexOutOfRange start > end or end > shape[d].
         */
        [[nodiscard]] Tensor slice(const std::vector<std::pair<size_t, size_t>>&);

        /**
         * Split into equal view chunks along an axis. Each chunk is a slice
         * view (leading-axis chunks are contiguous; inner-axis chunks are
         * not, so they are rejected by cat and every other op requiring
         * contiguous operands — only a leading-axis split feeds straight
         * back into cat). Synchronous and metadata-only, like slice.
         * @brief Split into equal slice views along an axis.
         * axis: Axis to split along.
         * sections: Number of chunks; must be nonzero and divide shape[axis].
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
         * axis: Axis to select along.
         * indices: Rank-1 integer index tensor.
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
         * axis: Axis to gather along.
         * indices: Integer index tensor, same rank as this tensor.
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
         * tensors: Non-empty list of inputs.
         * axis: Axis to concatenate along.
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
        void sgd_update(const Tensor&, float64_t);

        /**
         * SGD step using this tensor's accumulated gradient: sgd_update(*grad, lr).
         * @brief In-place SGD step from Tensor::grad.
         * @param lr Learning rate.
         * @throws ErrorHandling::NoGradient requires_grad is false or grad is null.
         */
        void sgd_step(float64_t lr);

        /**
         * Adam update from an explicit gradient, in place:
         * m = beta1*m + (1-beta1)*grad, v = beta2*v + (1-beta2)*grad^2,
         * w = w - lr * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + epsilon),
         * elementwise. The first/second moments m/v and the step counter t
         * are optimizer-side state kept in an external map keyed by this
         * tensor's address (Math/Optimization/Adam/AdamCPU.cpp) — Tensor's
         * verified sizeof layout does not grow. The moments persist across
         * calls; the map entry dies with the process, and this tensor must
         * outlive the optimizer's use of it (the same raw-pointer discipline
         * as backward()'s graph parents). Float kinds only (fp32/fp64 in
         * their storage width, fp16/bf16 and the lanes of packed x2 widened
         * to fp32 with RNE narrowing, exactly like sgd_update's compute
         * contract); integer kinds throw ErrorHandling::NotImplemented, as
         * Adam's moment math is float-only. When grad's dtype differs from
         * this tensor's, a private copy converts into the weight dtype via
         * the single grad-conversion rule (Math/Autograd/GradDType.h) — no
         * MismatchedDTypes throw. Requires 0 <= beta1, beta2 < 1 and
         * epsilon > 0 (documented contract). Async on both devices: the
         * update queues onto the thread pool and errors materialize at sync().
         * @brief In-place Adam moment step, w -= lr * m_hat / (sqrt(v_hat) + eps).
         * @param grad Gradient tensor (same shape/device as this; any dtype —
         * a mismatch converts a copy into this tensor's dtype).
         * @param lr Learning rate.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Denominator epsilon.
         * @throws ErrorHandling::MismatchedShapes Shapes differ.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand, or a non-float dtype.
         */
        void adam_update(const Tensor&, float64_t, float64_t, float64_t, float64_t);

        /**
         * Adam step using this tensor's accumulated gradient: adam_update(*grad, lr, beta1, beta2, epsilon).
         * @brief In-place Adam step from Tensor::grad.
         * @param lr Learning rate.
         * @param beta1 First-moment decay.
         * @param beta2 Second-moment decay.
         * @param epsilon Denominator epsilon.
         * @throws ErrorHandling::NoGradient requires_grad is false or grad is null.
         */
        void adam_step(float64_t, float64_t, float64_t, float64_t);

        /**
         * Backpropagate from this tensor through the recorded computation
         * graph. matmul/bias_add/relu/softmax_ce and the elementwise
         * add/sub/mul/div/pow (tensor-tensor and tensor-scalar) and exp/log
         * ops attach a Node to their output
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
         * Pins this leaf's gradient buffer to dtype dt, so later backward()
         * calls accumulate into that dtype. With no gradient held yet, a zeroed
         * one is allocated in dt. With one held, the buffer is converted in place
         * under the single grad-conversion rule (Math/Autograd/GradDType.h).
         * A change that narrows the buffer (see DataTypes::is_narrowing_conversion)
         * is refused unless narrowing is Allow; widening is always allowed.
         * @brief Set the dtype of this leaf's gradient buffer.
         * @param dt Dtype the gradient buffer must have.
         * @param narrowing Reject (default) refuses a narrowing change; Allow converts it.
         * @throws ErrorHandling::NoGradient requires_grad is false.
         * @throws ErrorHandling::NonLeafGradDType this tensor owns a graph node.
         * @throws ErrorHandling::NarrowingGradConversion narrowing refused under Reject.
         */
        void set_grad_dtype(DataTypes::DType dt, DataTypes::GradNarrowing narrowing = DataTypes::GradNarrowing::Reject);

        /**
         * Checks every precondition of set_grad_dtype(dt, narrowing) without
         * changing anything, so a caller applying it to several tensors can
         * validate all of them before changing any.
         * @brief Throw if set_grad_dtype(dt, narrowing) would fail.
         * @param dt Dtype the gradient buffer would have.
         * @param narrowing Narrowing policy the change would use.
         * @throws ErrorHandling::NoGradient requires_grad is false.
         * @throws ErrorHandling::NonLeafGradDType this tensor owns a graph node.
         * @throws ErrorHandling::NarrowingGradConversion narrowing refused under Reject.
         */
        void validate_grad_dtype(DataTypes::DType dt, DataTypes::GradNarrowing narrowing = DataTypes::GradNarrowing::Reject) const;

        /**
         * @brief Does this tensor hold an accumulated gradient?
         * @return True when Tensor::grad is allocated.
         */
        [[nodiscard]] bool has_grad() const {
            return this->grad != nullptr;
        }

        /**
         * Dtype of the held gradient buffer, or nullopt when no gradient is held.
         * @brief Gradient dtype of this tensor.
         * @return The gradient buffer's dtype, or std::nullopt.
         */
        [[nodiscard]] std::optional<DataTypes::DType> grad_dtype() const {
            if (this->grad == nullptr) {
                return std::nullopt;
            }
            return this->grad->dtype;
        }

        /**
         * @brief Read this tensor's dtype (e.g. to check a gradient buffer's dtype in tests).
         * @return The tensor's DataTypes::DType.
         */
        [[nodiscard]] DataTypes::DType get_dtype() const {
            return this->dtype;
        }

        /**
         * Extent of one axis of this tensor's shape.
         * @brief Dimension extent at the requested axis.
         * @param axis Axis index, must be < ndim.
         * @return The size of the requested dimension.
         * @throws ErrorHandling::IndexOutOfRange axis is out of bounds.
         */
        [[nodiscard]] size_t dim(size_t axis) const {
            if (axis >= this->ndim) {
                throw ErrorHandling::IndexOutOfRange();
            }
            return this->shape[axis];
        }

        /**
         * @brief Number of dimensions of this tensor.
         * @return The tensor's rank.
         */
        [[nodiscard]] size_t rank() const {
            return this->ndim;
        }

        /**
         * @brief Device this tensor lives on.
         * @return The tensor's Device.
         */
        [[nodiscard]] Device get_device() const {
            return this->device;
        }

        /**
         * Number of elements in this tensor (the product of its shape). A view
         * reports its own element count.
         * @brief Element count.
         * @return Number of elements.
         */
        [[nodiscard]] size_t element_count() const {
            return this->numel;
        }

        /**
         * Bytes of element data this tensor addresses: element_count() times the
         * dtype's element size. Gradients and optimizer state are not included.
         * @brief Byte size of the tensor data.
         * @return Number of bytes.
         */
        [[nodiscard]] size_t byte_count() const {
            return this->nbytes;
        }

        /**
         * Bytes of the accumulated gradient (Tensor::grad), counted at the
         * gradient's own dtype. Zero when no gradient is held.
         * @brief Byte size of this tensor's gradient.
         * @return Number of bytes.
         */
        [[nodiscard]] size_t grad_byte_count() const {
            return this->grad == nullptr ? size_t{0} : this->grad->byte_count();
        }

        /**
         * Bytes of the Adam moment state the optimizer holds for this tensor:
         * first and second moments, each the same shape and dtype as this
         * tensor. Zero until Adam has stepped this tensor. The state lives in the
         * optimizer's map, not in this object. A tensor owned by a Module loses
         * its state when that Module is destroyed; a bare tensor keeps it until
         * forget_adam_state() or until its address is reused.
         * @brief Byte size of this tensor's Adam moment state.
         * @return Number of bytes.
         */
        [[nodiscard]] size_t adam_state_byte_count() const;

        /**
         * Drops this tensor's Adam moment state from the optimizer's map.
         * Module's destructor calls this for every parameter it owns.
         * @brief Forget this tensor's Adam moment state.
         */
        void forget_adam_state() const;

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
         * When requires_grad=true the output joins the computation graph
         * (SumG backward: dx = the scalar seed broadcast to this tensor's
         * shape).
         * @brief Full-tensor sum reduction.
         * @return Shape-{1} tensor holding the sum.
         */
        [[nodiscard]] Tensor sum() const;

        /**
         * Arithmetic mean of every element, reduced to a shape-{1} tensor of
         * this tensor's dtype and device. Accumulation: fp32 in fp32, fp64 in
         * fp64, fp16/bf16 widened to fp32 (RNE narrow on store), packed x2
         * dtypes per-lane fp32, integers in fp64 and narrowed by
         * truncate-toward-zero with clamping to the dtype range (a NaN mean
         * stores 0, as there is no integer NaN; values past 2^53 in magnitude
         * can lose low bits). Float reduction order is backend-defined, so
         * float means are not guaranteed bit-identical between CPU/SIMD/GPU
         * tiers. Blocks before returning (the merge happens on the caller
         * thread). Mean consumes the UInt8 0/1 masks equal() produces
         * directly, e.g. accuracy = equal(pred, labels) cast to Float32,
         * then mean().
         * When requires_grad=true the output joins the computation graph
         * (MeanG backward: dx = the scalar seed divided by numel, broadcast
         * to this tensor's shape).
         * @brief Full-tensor mean reduction.
         * @return Shape-{1} tensor holding the mean.
         * @throws ErrorHandling::IndexOutOfRange Tensor is empty.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] Tensor mean() const;

        /**
         * Smallest of every element, reduced to a shape-{1} tensor of this
         * tensor's dtype and device. Comparison reduction: the non-NaN result
         * is always bit-for-bit one of the input elements, and the combine is
         * associative and commutative, so min results ARE bit-identical
         * across CPU/GPU tiers and across chunk counts (unlike sum/mean,
         * whose float accumulation order is backend-defined). PyTorch-
         * compatible NaN rule: if ANY element is NaN the result is the
         * canonical quiet NaN of the dtype (payloads are not preserved).
         * Packed x2 dtypes (Float16x2/BFloat16x2) reduce per lane, so the two
         * lanes' winners may come from different elements; a +0.0/-0.0 tie
         * canonicalizes to -0.0. Integer and Int4x2 dtypes compare plainly
         * (Int4x2 per lane). Blocks before returning (same move-drain
         * contract as sum()). The output never joins the autograd graph.
         * @brief Full-tensor min reduction.
         * @return Shape-{1} tensor holding the minimum element.
         * @throws ErrorHandling::IndexOutOfRange Tensor is empty.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] Tensor min() const;

        /**
         * Largest of every element, reduced to a shape-{1} tensor of this
         * tensor's dtype and device. Comparison reduction: the non-NaN result
         * is always bit-for-bit one of the input elements, and the combine is
         * associative and commutative, so max results ARE bit-identical
         * across CPU/GPU tiers and across chunk counts (unlike sum/mean,
         * whose float accumulation order is backend-defined). PyTorch-
         * compatible NaN rule: if ANY element is NaN the result is the
         * canonical quiet NaN of the dtype (payloads are not preserved).
         * Packed x2 dtypes (Float16x2/BFloat16x2) reduce per lane, so the two
         * lanes' winners may come from different elements; a +0.0/-0.0 tie
         * canonicalizes to +0.0. Integer and Int4x2 dtypes compare plainly
         * (Int4x2 per lane). Blocks before returning (same move-drain
         * contract as sum()). The output never joins the autograd graph.
         * @brief Full-tensor max reduction.
         * @return Shape-{1} tensor holding the maximum element.
         * @throws ErrorHandling::IndexOutOfRange Tensor is empty.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] Tensor max() const;

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
         * Index of the maximum element along one axis, written as int64 indices into an output tensor.
         * The output holds one index per slice orthogonal to `axis`, in this tensor's shape with the
         * reduced dimension removed (a rank-1 input reduces to shape {1}). PyTorch-compatible NaN rule:
         * NaN beats every non-NaN value, so the first NaN along the axis wins when present; otherwise the
         * first strict maximum wins ties. Packed dtypes compare per element on the widened (lo, hi) lane
         * pair, lexicographic, NaN-wins per lane. Accumulation order is backend-defined, but the indices
         * themselves are exact and bit-identical across backends. Blocks before returning (same move-drain
         * contract as argmax()/matmul()). The output never joins the autograd graph.
         * @brief Index of the largest element along one axis.
         * @param axis Axis to reduce; must be < ndim.
         * @return Int64 tensor of per-slice argmax indices (input shape minus `axis`).
         * @throws ErrorHandling::IndexOutOfRange axis out of bounds or tensor is empty.
         * @throws ErrorHandling::NotImplemented Tensor is non-contiguous.
         */
        [[nodiscard]] Tensor argmax(size_t axis) const;

        /**
         * Lane-aware argmax over packed x2 logits (Float16x2 or BFloat16x2). Each
         * element's low lane is class 2j and its high lane is class 2j+1, the
         * layout softmax_ce trains, so the result is a class index in
         * [0, 2*classes). argmax() compares whole elements and cannot recover
         * the class. CPU-only, rank-2 input, axis 1.
         * @brief Class index of the largest lane per row of packed x2 logits.
         * @param axis Must be 1.
         * @return Int64 tensor of shape {rows}.
         * @throws ErrorHandling::MismatchedDTypes Not a packed x2 dtype.
         * @throws ErrorHandling::NotImplemented Not rank-2, axis is not 1, or on the GPU.
         * @throws ErrorHandling::IndexOutOfRange The tensor is empty.
         */
        [[nodiscard]] Tensor argmax_lanes(size_t axis);

        /**
         * Elementwise equality mask, out of place: out[i] = this[i] == other[i]
         * ? 1 : 0, as a UInt8 tensor of this tensor's shape and device (DType
         * has no Bool; UInt8 is the smallest dtype with full
         * conversion/reduction support, so the mask feeds convert_dtype() and
         * mean() directly). Float comparison follows IEEE == on widened fp64
         * keys: NaN never equals anything (not even NaN), and -0.0 equals
         * 0.0. Packed dtypes (Float16x2/BFloat16x2) widen both lanes and are
         * equal only when both lanes are; Int4x2 compares raw bytes.
         * Both operands must share shape, dtype and device and be contiguous.
         * The compute is queued asynchronously — failures materialize at
         * out.sync() — but the returned tensor is moved out, and the move
         * drains pending work, so equal effectively blocks before returning
         * (same as matmul()); a drained failure is re-queued on the result as
         * a poisoned future and still surfaces at its next sync().
         * @brief Elementwise equality mask (UInt8 0/1).
         * @param other Right operand, same shape/dtype/device.
         * @return UInt8 tensor, same shape/device as this.
         * @throws ErrorHandling::MismatchedShapes Shapes differ.
         * @throws ErrorHandling::MismatchedDTypes Dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor equal(const Tensor& other) const;

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
         * 2D max pooling over the spatial dimensions of a rank-4 NCHW tensor.
         * Output shape is {N, C, Hout, Wout} with Hout = floor((H - kernel) / stride) + 1
         * and Wout = floor((W - kernel) / stride) + 1; no padding. The first
         * occurrence of a repeated window maximum wins in both forward and backward.
         * @brief 2D max pooling (square kernel, no padding).
         * @param kernel Window height and width.
         * @param stride Step between adjacent windows.
         * @return New rank-4 tensor of the same dtype/device as this.
         * @throws ErrorHandling::MismatchedShapes Input is not rank-4, kernel > H/W,
         *         kernel == 0, or stride == 0.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor max_pool2d(uint32_t kernel, uint32_t stride) const;

        /**
         * Numerically-stable last-axis softmax over the class dimension: this
         * must be rank-2 (rows x classes). Returns a tensor of the same
         * shape/dtype/device where each row sums to 1. Computed as
         * exp(x - row_max) / sum(exp(x - row_max)); a NaN logit poisons its
         * whole row. Packed logits dtypes unpack each element's two lanes as
         * consecutive classes. Accumulation: fp32 for fp32/fp16/bf16/packed
         * logits, fp64 for fp64 and integer logits. Integer outputs
         * clamp/truncate and map NaN to 0. Blocks before returning.
         * When this tensor has requires_grad=true the output joins the
         * computation graph (softmax_G backward: dx = y * (dy - rowsum(dy*y)),
         * with y recomputed from the saved input).
         * @brief Standalone rank-2 softmax.
         * @return New tensor (rows x classes) holding softmax probabilities.
         * @throws ErrorHandling::MismatchedShapes Input is not rank-2 or is empty.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor softmax() const;

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
         * Mean squared error loss, rank-agnostic: out = mean of (this -
         * target)^2 over ALL elements, any rank 1..BT_TENSOR_MAX_DIMS, reduced
         * to a shape-{1} tensor of this tensor's dtype and device. this and
         * target must have identical shapes and dtypes, share a device, and be
         * contiguous (a loss does not broadcast). Accumulation follows the
         * mean()/softmax_ce taxonomy: fp32 in fp32, fp64 in fp64, fp16/bf16
         * widened to fp32 (RNE narrow on store), the lanes of packed x2 dtypes
         * and the nibbles of int4x2 are logical elements of the same single
         * scalar mean (dividing by 2 * numel, stored duplicated into both
         * lanes like softmax_ce's mean), integer dtypes accumulate in fp64
         * and narrow by clamp + truncate (a NaN mean stores 0). Float
         * reduction order is backend-defined, so the loss is not guaranteed
         * bit-identical between CPU/GPU tiers. Blocks before returning (the
         * merge happens on the caller thread / a stream-syncing worker).
         * When this tensor has requires_grad=true the output joins the
         * computation graph (mse_g backward: d/dthis = 2*(this-target)/numel
         * * dloss; targets never receive a gradient). A callable free-function
         * form lives at BeanTensor::Loss::mse (Math/Loss/Loss.h) with an
         * identical contract, for callers that do not go through a Tensor
         * method (e.g. the Module layer).
         * @brief Mean squared error: mean of (this - target)^2 over all elements.
         * @param target Targets, identical shape/dtype/device, contiguous.
         * @return Shape-{1} tensor holding the mean loss.
         * @throws ErrorHandling::MismatchedShapes Shapes differ, or a tensor is empty.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor mse(const Tensor&) const;

        /**
         * Broadcasting semantics (identical for add/sub/mul/div/pow): the two
         * operand shapes are right-aligned and compared per dimension under
         * NumPy rules — each dim pair must be equal or one of them must be 1;
         * the output dim is the max. This covers the scalar case (a rank-1
         * shape-{1} operand, or the float64_t scalar overloads), the
         * same-shape case, and the per-row/bias cases ([1,n] against [m,n],
         * [m,1] against [m,n], [1,1] against anything). A dim pair that is
         * unequal with neither side 1 is not broadcastable and throws
         * ErrorHandling::MismatchedShapes. A size-1 dim is materialized by
         * re-reading that operand's single element (stride 0), never by a
         * physical copy.
         *
         * Dtype/conversion rules: there is NO implicit dtype promotion —
         * tensor-tensor ops require both operands to share this tensor's
         * dtype (ErrorHandling::MismatchedDTypes otherwise) and produce an
         * output of that same dtype. The scalar overloads take a float64_t
         * scalar that is narrowed ONCE into this tensor's dtype under the
         * PERMISSIVE convert contract (Math/Autograd/GradDType.h): floats
         * round RNE (halfs through an fp32 intermediate), floats-to-ints
         * truncate toward zero and clamp to the dtype range with NaN -> 0,
         * packed dtypes duplicate the narrowed value into both lanes, and
         * int4x2 clamps to [-8, 7] per nibble (both nibbles).
         *
         * Compute contract: fp32/fp64 compute in their storage width with a
         * single rounding; fp16/bf16 (and the lanes of Float16x2/BFloat16x2)
         * widen to fp32, compute, and narrow RNE through the same unaccel
         * pathway as the conversion dispatcher (BF16 flush-to-zero on fp32
         * subnormal inputs, fp16 overflow clamps to +/-65504, NaN/Inf per the
         * ConversionClampMethod flag semantics); integer add/sub/mul wrap
         * modularly through the unsigned twin (int4x2 wraps per nibble);
         * integer div truncates toward zero with division by zero yielding 0
         * and min/-1 yielding min (wrap) — never UB; float div is IEEE
         * (x/0 -> +/-inf, 0/0 -> NaN). pow requires a float-kind dtype
         * (integer pow throws ErrorHandling::NotImplemented) and is the
         * backend libm's pow (std::pow on CPU, powf/pow on GPU).
         *
         * Async contract: the compute queues onto the thread pool and errors
         * materialize at out.sync() — but the returned tensor is moved out,
         * and the move drains pending work, so these ops effectively block
         * before returning (same as bias_add()). A failure drained by that
         * move is re-queued on the result as a poisoned future and still
         * surfaces at its next sync().
         *
         * Autograd: when either operand has requires_grad=true the output
         * joins the computation graph (elementwise backward: the analytic
         * local derivatives composed onto the upstream delta in the assigned
         * grad dtype, sum-reduced over each broadcast operand's broadcast
         * dims — Math/Autograd/ElementwiseG). The scalar overloads
         * differentiate with respect to the tensor only; the scalar is
         * persisted as a bit-cast payload in Node::_inputs[1] (never
         * dereferenced, never walked). exp/log wire the matching unary
         * backward (d exp = dy*exp(x), d log = dy/x).
         * @brief Elementwise add with NumPy-style broadcasting.
         * o: Right operand, same dtype and device, contiguous.
         * @return New tensor, the broadcast shape, this tensor's dtype.
         * @throws ErrorHandling::MismatchedShapes Operands are not broadcast-compatible.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor add(const Tensor&) const;
        /**
         * Tensor-scalar add: out[i] = this[i] + scalar, scalar narrowed into
         * this tensor's dtype. Broadcasting/compute/async contracts: see the
         * tensor-tensor add doc block above.
         * @brief Elementwise add of a scalar.
         * scalar: Scalar operand (float64_t).
         * @return New tensor, same shape/dtype/device as this.
         */
        [[nodiscard]] Tensor add(const float64_t&) const;
        /**
         * @brief Elementwise subtract with NumPy-style broadcasting.
         * o: Right operand (the subtrahend), same dtype and device, contiguous.
         * @return New tensor, the broadcast shape, this tensor's dtype.
         * @throws ErrorHandling::MismatchedShapes Operands are not broadcast-compatible.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor sub(const Tensor&) const;
        /**
         * Tensor-scalar subtract: out[i] = this[i] - scalar, scalar narrowed
         * into this tensor's dtype. Contracts: see the tensor-tensor add doc
         * block above.
         * @brief Elementwise subtract of a scalar.
         * scalar: Scalar operand (float64_t).
         * @return New tensor, same shape/dtype/device as this.
         */
        [[nodiscard]] Tensor sub(const float64_t&) const;
        /**
         * @brief Elementwise multiply with NumPy-style broadcasting.
         * o: Right operand, same dtype and device, contiguous.
         * @return New tensor, the broadcast shape, this tensor's dtype.
         * @throws ErrorHandling::MismatchedShapes Operands are not broadcast-compatible.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor mul(const Tensor&) const;
        /**
         * Tensor-scalar multiply: out[i] = this[i] * scalar, scalar narrowed
         * into this tensor's dtype. Contracts: see the tensor-tensor add doc
         * block above.
         * @brief Elementwise multiply by a scalar.
         * scalar: Scalar operand (float64_t).
         * @return New tensor, same shape/dtype/device as this.
         */
        [[nodiscard]] Tensor mul(const float64_t&) const;
        /**
         * Division edge semantics: float div is IEEE (x/0 -> +/-inf, 0/0 ->
         * NaN); integer div truncates toward zero, division by zero yields 0,
         * and min/-1 yields min (wrap) — never UB.
         * @brief Elementwise divide with NumPy-style broadcasting.
         * o: Right operand (the divisor), same dtype and device, contiguous.
         * @return New tensor, the broadcast shape, this tensor's dtype.
         * @throws ErrorHandling::MismatchedShapes Operands are not broadcast-compatible.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand.
         */
        [[nodiscard]] Tensor div(const Tensor&) const;
        /**
         * Tensor-scalar divide: out[i] = this[i] / scalar, scalar narrowed
         * into this tensor's dtype. Division edge semantics: see the
         * tensor-tensor div doc block above.
         * @brief Elementwise divide by a scalar.
         * scalar: Scalar operand (float64_t).
         * @return New tensor, same shape/dtype/device as this.
         */
        [[nodiscard]] Tensor div(const float64_t&) const;
        /**
         * pow requires a float-kind dtype (Float16/BFloat16/Float16x2/
         * BFloat16x2/Float32/Float64); integer pow throws
         * ErrorHandling::NotImplemented. Float-kind bases/exponents compute
         * per the widening contract: halfs widen to fp32, std::pow, narrow
         * RNE. IEEE edge semantics come from the backend libm (e.g. negative
         * base with non-integer exponent -> NaN, 0^0 -> 1).
         * @brief Elementwise power with NumPy-style broadcasting.
         * o: Right operand (the exponent), same dtype and device, contiguous.
         * @return New tensor, the broadcast shape, this tensor's dtype.
         * @throws ErrorHandling::MismatchedShapes Operands are not broadcast-compatible.
         * @throws ErrorHandling::MismatchedDTypes Operand dtypes differ.
         * @throws ErrorHandling::MismatchedDevices Operand devices differ.
         * @throws ErrorHandling::NotImplemented Non-contiguous operand, or integer dtype.
         */
        [[nodiscard]] Tensor pow(const Tensor&) const;
        /**
         * Tensor-scalar power: out[i] = this[i]^scalar, scalar narrowed into
         * this tensor's dtype; integer dtypes are rejected (same contract as
         * the tensor-tensor pow).
         * @brief Elementwise power with a scalar exponent.
         * scalar: Scalar exponent (float64_t).
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::NotImplemented Integer dtype.
         */
        [[nodiscard]] Tensor pow(const float64_t&) const;
        /**
         * Float kinds only — integer exp throws ErrorHandling::NotImplemented.
         * fp16/bf16 (and packed lanes) widen to fp32, std::exp, narrow RNE;
         * fp32/fp64 exp in their storage width. Overflow yields +inf.
         * @brief Elementwise exponential.
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::NotImplemented Non-contiguous tensor, or integer dtype.
         */
        [[nodiscard]] Tensor exp() const;
        /**
         * Float kinds only — integer log throws ErrorHandling::NotImplemented.
         * fp16/bf16 (and packed lanes) widen to fp32, std::log, narrow RNE;
         * fp32/fp64 log in their storage width. IEEE edges come from the
         * library: log(0) is -inf, log of a negative is NaN.
         * @brief Elementwise natural logarithm.
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::NotImplemented Non-contiguous tensor, or integer dtype.
         */
        [[nodiscard]] Tensor log() const;
        /**
         * Float kinds only — integer sigmoid throws ErrorHandling::NotImplemented.
         * fp16/bf16 (and packed lanes) widen to fp32, 1/(1+exp(-x)), narrow RNE;
         * fp32/fp64 sigmoid in their storage width. Large positive/negative inputs
         * saturate to 1/0 without overflow.
         * @brief Elementwise sigmoid.
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::NotImplemented Non-contiguous tensor, or integer dtype.
         */
        [[nodiscard]] Tensor sigmoid() const;
        /**
         * Float kinds only — integer tanh throws ErrorHandling::NotImplemented.
         * fp16/bf16 (and packed lanes) widen to fp32, 1 - 2/(exp(2x)+1), narrow RNE;
         * fp32/fp64 tanh in their storage width. Large positive/negative inputs
         * saturate to +/-1 without overflow.
         * @brief Elementwise hyperbolic tangent.
         * @return New tensor, same shape/dtype/device as this.
         * @throws ErrorHandling::NotImplemented Non-contiguous tensor, or integer dtype.
         */
        [[nodiscard]] Tensor tanh() const;

        /**
         * Bounds-checked strided element access. T must be a type with a
         * DataTypes::DTypeOf mapping, and it must be the dtype this tensor
         * stores; the stride is sizeof(T), so a mismatch would read the wrong bytes.
         * @brief Read an element by multidimensional index.
         * @tparam T Element type, matching the tensor's dtype.
         * @param idx One index per dimension; must match ndim exactly.
         * @return Copy of the element's value. A raw pointer is never exposed:
         * it would only be valid for CPU tensors, never GPU ones.
         * @throws ErrorHandling::MismatchedDTypes T does not map to this tensor's dtype.
         * @throws ErrorHandling::IndexOutOfRange If the rank or any index is out of bounds.
         */
        template<typename T> requires DataTypes::DTypeMapped<T>
        [[nodiscard]] T at(const std::initializer_list<size_t> idx) {
            // Offset first: it syncs, and a pending convert_dtype only flips dtype at sync().
            const auto elem_offset = checked_element_offset(idx);
            require_dtype(DataTypes::dtype_of_v<T>);
            T value;
            read_element(elem_offset, &value, sizeof(T));
            return value;
        }

        /**
         * Bounds-checked strided element write. T must be a type with a
         * DataTypes::DTypeOf mapping, and it must be the dtype this tensor
         * stores; the stride is sizeof(T), so a mismatch would write the wrong bytes.
         * On a dtype mismatch nothing is written.
         * @brief Assign an element by multidimensional index.
         * @tparam T Element type, matching the tensor's dtype.
         * @param idx One index per dimension; must match ndim exactly.
         * @param value Value to write into the element.
         * @throws ErrorHandling::MismatchedDTypes T does not map to this tensor's dtype.
         * @throws ErrorHandling::IndexOutOfRange If the rank or any index is out of bounds.
         */
        template<typename T> requires DataTypes::DTypeMapped<T>
        void set(const std::initializer_list<size_t> idx, const T& value) {
            // Offset first: it syncs, and a pending convert_dtype only flips dtype at sync().
            const auto elem_offset = checked_element_offset(idx);
            require_dtype(DataTypes::dtype_of_v<T>);
            write_element(elem_offset, &value, sizeof(T));
        }

        [[nodiscard]] bool is_same_shape(const Tensor& t2) const {
            return ndim == t2.ndim && std::equal(shape, shape + ndim, t2.shape);
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
         * this is the out-shaped delta of a broadcast elementwise op; op is
         * the operand the delta is being reduced into. See
         * Math/Autograd/ElementwiseG/ElementwiseGShared.h for how the plan's
         * free (size-1 or left-padded missing) operand dims map to full
         * output extents.
         * @brief Build the broadcast-reduction plan for (this, op).
         * @param op The operand the delta will be reduced into.
         * @return The broadcast-reduction execution plan.
         */
        [[nodiscard]] detail::BcastReducePlan build_bcast_reduce_plan(const Tensor& op) const;

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
         * Teardown variant of kill_gpu_data/kill_buffer for the destructor and
         * the noexcept move assignment: frees GPU data and the convert buffer
         * without throwing. At static teardown the CUDA/HIP driver may already
         * be unloading (cudaErrorCudartUnloading, "driver shutting down"); the
         * allocation is reclaimed with the process, so a free error there is
         * unactionable — and these callers cannot throw anyway.
         * @note Swallows GPU free errors; operational callers must use the checked variants.
         */
        void kill_gpu_teardown_noexcept() noexcept;

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
         * Throw unless the element type a caller is accessing matches this
         * tensor's dtype. Reads the dtype as-is, so callers must sync first
         * (checked_element_offset does): a pending convert_dtype flips dtype at sync().
         * @throws ErrorHandling::MismatchedDTypes If expected differs from this tensor's dtype.
         */
        void require_dtype(DataTypes::DType) const;

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
         * Shared implementation of the tensor-tensor arithmetic ops
         * (add/sub/mul/div/pow); the public methods are thin wrappers. See
         * Tensor::add for the full broadcasting/dtype/async contract.
         * @brief Tensor-tensor elementwise op with broadcasting.
         * o: Right operand, same dtype and device.
         * op: Operation selector.
         * @return New tensor with the broadcast shape and the input dtype.
         */
        [[nodiscard]] Tensor elementwise_binary(const Tensor&, ElementwiseOp) const;
        /**
         * Shared implementation of the tensor-scalar arithmetic ops; the
         * public methods are thin wrappers. See Tensor::add for the full
         * scalar-narrowing/async contract.
         * @brief Tensor-scalar elementwise op.
         * scalar: Scalar operand (float64_t, narrowed into this tensor's dtype).
         * op: Operation selector.
         * @return New tensor shaped like this.
         */
        [[nodiscard]] Tensor elementwise_scalar(const float64_t&, ElementwiseOp) const;
        /**
         * Shared implementation of the unary elementwise ops (exp/log); the
         * public methods are thin wrappers. See Tensor::exp for the contract.
         * @brief Unary elementwise op (exp/log).
         * op: Operation selector.
         * @return New tensor shaped like this.
         */
        [[nodiscard]] Tensor elementwise_unary(ElementwiseUnaryOp) const;

        /**
         * Attach a computation-graph node to this tensor (an op output).
         * No-op when no input requires gradients. Sets requires_grad,
         * is_leaf=false, parents/n_parents, and allocates the Node; _inputs
         * receive ALL op inputs (backward needs non-grad operands too, e.g.
         * GEMM's dA needs B), gated on requires_grad inside the node fn.
         * Parents are borrowed raw pointers — they must outlive backward().
         * @param backward_fn Node::_backward_fn implementation for the op. The
         * new node is passed to it on every fire, so the op can store per-node
         * state in Node::_scalar before wiring (or after, via Tensor::node).
         * @param a First op input.
         * @param b Second op input, when the op is binary.
         */
        void wire_node(void (*backward_fn)(Node*, Tensor*, Tensor**, size_t, DataTypes::DType), const Tensor& a, const Tensor* b = nullptr);
        /**
         * Three-input variant of wire_node for ops like Conv2dG that need
         * parents (x, weight, bias). Follows the same contract as the unary/
         * binary overload: no-op when no input requires gradients, otherwise
         * records all three borrowed parent pointers and sets _n_inputs = 3.
         * @brief Attach a 3-input autograd node to this tensor.
         * @param backward_fn Backward trampoline invoked as (owning node, output grad, inputs, input count, grad dtype).
         * @param a First input tensor.
         * @param b Second input tensor.
         * @param c Third input tensor.
         */
        void wire_node(void (*backward_fn)(Node*, Tensor*, Tensor**, size_t, DataTypes::DType), const Tensor& a, const Tensor& b, const Tensor& c);

        friend void detail::set_amount_gpu(Tensor& t, int64_t amount);
        friend void detail::set_amount_gpu(Tensor& t, uint64_t amount);
        friend void detail::set_amount_gpu(Tensor& t, float64_t amount);
        friend void detail::set_random_gpu(Tensor& t, float64_t min, float64_t max, int64_t seed);
        friend void detail::move_to_gpu(Tensor& t);
        friend void detail::move_to_cpu(Tensor& t);
        friend void detail::convert_gpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);
        friend void detail::matmul_cpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::matmul_gpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::im2col_cpu(const Tensor&, const Tensor&, uint32_t, uint32_t, size_t, size_t);
        friend void detail::im2col_gpu(const Tensor&, const Tensor&, uint32_t, uint32_t, size_t, size_t);
        friend void detail::col2im_cpu(const Tensor&, const Tensor&, uint32_t, uint32_t, size_t, size_t);
        friend void detail::col2im_gpu(const Tensor&, const Tensor&, uint32_t, uint32_t, size_t, size_t);
        friend void detail::conv2d_cpu(const Tensor&, const Tensor&, const Tensor*, const Tensor&, uint32_t, uint32_t);
        friend void detail::conv2d_gpu(const Tensor&, const Tensor&, const Tensor*, const Tensor&, uint32_t, uint32_t);
        friend void detail::conv2d_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::conv2d_g_node_gpu(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::conv_zero_tensor(const Tensor&);
        friend void detail::col2im_dispatch(const Tensor&, const Tensor&, const detail::Conv2dShape&, uint32_t, uint32_t);
        friend void detail::rowsum_mk_dispatch(const Tensor&, const Tensor&);
        friend void detail::bias_add_nchw_dispatch(const Tensor&, const Tensor&);
        friend void detail::sgd_update_cpu(const Tensor&, const Tensor&, float64_t);
        friend void detail::sgd_update_gpu(const Tensor&, const Tensor&, float64_t);
        friend void detail::adam_update_cpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float64_t, float64_t, float64_t, float64_t, float64_t, float64_t);
        friend void detail::adam_update_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float64_t, float64_t, float64_t, float64_t, float64_t, float64_t);
        friend void detail::sum_cpu(const Tensor&, const Tensor&);
        friend void detail::sum_gpu(const Tensor&, const Tensor&);
        friend void detail::argmax_cpu(const Tensor&, uint64_t&);
        friend void detail::argmax_gpu(const Tensor&, uint64_t&);
        friend void detail::argmax_axis_cpu(const Tensor&, Tensor&, size_t);
        friend void detail::argmax_axis_gpu(const Tensor&, const Tensor&, size_t);
        friend void detail::mean_cpu(const Tensor&, const Tensor&);
        friend void detail::mean_gpu(const Tensor&, const Tensor&);
        friend void detail::min_cpu(const Tensor&, const Tensor&);
        friend void detail::min_gpu(const Tensor&, const Tensor&);
        friend void detail::max_cpu(const Tensor&, const Tensor&);
        friend void detail::max_gpu(const Tensor&, const Tensor&);
        friend void detail::equal_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::equal_gpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::bias_add_cpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::bias_add_gpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::relu_cpu(const Tensor&, const Tensor&);
        friend void detail::relu_gpu(const Tensor&, const Tensor&);
        friend void detail::max_pool2d_cpu(const Tensor&, Tensor&, uint32_t, uint32_t);
        friend void detail::max_pool2d_gpu(const Tensor&, Tensor&, uint32_t, uint32_t);
        friend void detail::max_pool2d_g_cpu(const Tensor&, const Tensor&, Tensor&, uint32_t, uint32_t);
        friend void detail::max_pool2d_g_gpu(const Tensor&, const Tensor&, Tensor&, uint32_t, uint32_t);
        friend void detail::max_pool2d_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::max_pool2d_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::softmax_cpu(const Tensor&, Tensor&);
        friend void detail::softmax_gpu(const Tensor&, const Tensor&);
        friend void detail::softmax_g_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::softmax_g_gpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::softmax_ce_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::softmax_ce_gpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::mse_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::mse_gpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::gemm_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::bias_add_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::sum_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::mean_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::relu_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::sigmoid_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::tanh_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::softmax_g_node(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::softmax_g_bridge(Node*, Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::flatten_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::softmax_ce_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::mse_g_node(Node*, const Tensor*, Tensor**, size_t, DataTypes::DType);
        friend void detail::ew_binary_g_node_impl(Node*, Tensor*, Tensor**, size_t, DataTypes::DType, ElementwiseOp);
        friend void detail::ew_scalar_g_node_impl(Node*, Tensor*, Tensor**, size_t, DataTypes::DType, ElementwiseOp);
        friend void detail::ew_unary_g_node_impl(Node*, Tensor*, Tensor**, size_t, DataTypes::DType, ElementwiseUnaryOp);
        friend void detail::ew_deliver_delta(Tensor&&, Tensor&, const DataTypes::DType&, const DataTypes::DType&);
        friend Tensor detail::ew_operand_copy(const Tensor&, const DataTypes::DType&);
        friend void detail::bcast_sum_to_cpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::bcast_sum_to_gpu(const Tensor&, const Tensor&, Tensor&);
        friend void detail::grad_acc_cpu(const Tensor&, const Tensor&);
        friend void detail::grad_acc_gpu(const Tensor&, const Tensor&);
        friend void detail::transpose_last2_cpu(const Tensor&, const Tensor&);
        friend void detail::transpose_last2_gpu(const Tensor&, const Tensor&);
        friend void detail::relu_g_cpu(const Tensor&, const Tensor&, const Tensor&);
        friend void detail::relu_g_gpu(const Tensor&, const Tensor&, const Tensor&);

        friend void detail::bias_add_g_cpu(const Tensor&, Tensor&);
        friend void detail::bias_add_g_gpu(const Tensor&, const Tensor&);
        friend void detail::softmax_ce_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        friend void detail::softmax_ce_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
        friend void detail::mse_g_cpu(const Tensor&, const Tensor&, const Tensor&, Tensor&);
        friend void detail::mse_g_gpu(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
        friend void detail::index_select_cpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        friend void detail::index_select_gpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        friend void detail::gather_cpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        friend void detail::gather_gpu(const Tensor&, const Tensor&, const Tensor&, size_t);
        friend void detail::cat_cpu(const std::vector<Tensor>&, const Tensor&, size_t);
        friend void detail::cat_gpu(const std::vector<Tensor>&, const Tensor&, size_t);
        friend void detail::elementwise_cpu(const Tensor&, const Tensor&, const Tensor&, ElementwiseOp, const detail::ElementwiseBcast&);
        friend void detail::elementwise_gpu(const Tensor&, const Tensor&, const Tensor&, ElementwiseOp, const detail::ElementwiseBcast&);
        friend void detail::elementwise_scalar_cpu(const Tensor&, const float64_t&, const Tensor&, ElementwiseOp);
        friend void detail::elementwise_scalar_gpu(const Tensor&, const float64_t&, const Tensor&, ElementwiseOp);
        friend void detail::elementwise_unary_cpu(const Tensor&, const Tensor&, ElementwiseUnaryOp);
        friend void detail::elementwise_unary_gpu(const Tensor&, const Tensor&, ElementwiseUnaryOp);
        friend void Intrinsics::detail::launch_conv_cpu(Tensor& t, const DataTypes::DType& new_dtype, ConversionClampMethod method);
        friend Tensor BeanTensor::Math::Dropout::train_mask(const Tensor&, float64_t, int64_t);
        friend Tensor BeanTensor::Math::Dropout::train_mask_cpu(const Tensor&, float64_t, int64_t);
        friend Tensor BeanTensor::Math::Dropout::train_mask_gpu(const Tensor&, float64_t, int64_t);

    };
}
