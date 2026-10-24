/**
 * @file GemmGCPU.cpp
 * @author GreenBean
 * @brief Backward trampoline for matmul (gemm_g_node). No dedicated backend:
 * dA = dOut * B^T and dB = A^T * dOut reuse the existing matmul path, with
 * the transposes materialized by transpose_last2_cpu/gpu.
 * @version 0.2
 * @date 2026-10-23
 */
#include <algorithm>
#include <vector>

#include "Math/Autograd/GemmG/GemmG.h"
#include "Math/Autograd/GradDType.h"
#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {

    /**
     * inputs are (a, b). dA = dOut * B^T and dB = A^T * dOut, each only if that input
     * requires grad. output_grad already carries grad_dtype (the seed and every
     * node-to-node delta do); matmul requires matching dtypes, so each transposed
     * operand is boundary-cast into grad_dtype (GradDType.h, the single conversion
     * rule) before the multiply — dA/dB are then computed in and carry grad_dtype.
     * The transpose is materialized into a contiguous scratch tensor so the
     * existing matmul path (CPU kernels or GPU BLAS) is reused for every dtype.
     * @brief Backward trampoline for matmul (gemm).
     * @param output_grad Gradient of this node's output tensor, in grad_dtype.
     * @param inputs Input tensors of the forward op.
     * @param n_inputs Number of entries in inputs.
     * @param grad_dtype The calling backward()'s assigned gradient dtype.
     * @throws ErrorHandling::UnReachableCode if n_inputs is not 2 or the device is neither CPU nor GPU.
     */
    void gemm_g_node(Node*, Tensor* output_grad, Tensor** inputs, const size_t n_inputs, const DataTypes::DType grad_dtype) {
        if (n_inputs != 2) [[unlikely]] {
            throw ErrorHandling::UnReachableCode();
        }
        auto& a = *inputs[0];
        auto& b = *inputs[1];
        if (!a.requires_grad && !b.requires_grad) {
            return;
        }
        output_grad->sync();
        // dA = dOut·Bᵀ, dB = Aᵀ·dOut. The transpose is materialized into a
        // contiguous scratch tensor, so the existing validated matmul path
        // (CPU kernels or GPU BLAS) runs unchanged for every dtype.
        /**
         * Captures nothing. Allocates a new tensor with the last two dimensions of t swapped, same dtype and device, and fills it with transpose_last2_cpu or transpose_last2_gpu.
         * Returning the tensor moves it out, which drains the transpose workers.
         * Throws ErrorHandling::UnReachableCode if t is on neither CPU nor GPU.
         * @brief Materialize the transpose of the last two dimensions of t.
         * @param t Source tensor, rank >= 2, on CPU or GPU.
         * @return A new contiguous tensor holding the transposed data.
         */
        auto transpose = [](const Tensor& t) {
            auto t_shape = std::vector(t.shape, t.shape + t.ndim);
            std::swap(t_shape[t.ndim - 1], t_shape[t.ndim - 2]);
            auto out = Tensor(t_shape, t.dtype, t.device, false);
            if (t.device == Device::CPU) {
                transpose_last2_cpu(t, out);
            } else if (t.device == Device::GPU) {
                transpose_last2_gpu(t, out);
            } else [[unlikely]] {
                throw ErrorHandling::UnReachableCode();
            }
            return out; // the move drains the transpose workers
        };
        if (a.requires_grad) {
            b.sync();
            auto bt = transpose(b);
            grad_convert_inplace(bt, grad_dtype); // no-op when b already carries grad_dtype
            auto da = output_grad->matmul(bt);
            if (a.node != nullptr) {
                grad_probe_hit(da.dtype); // test hook: an intermediate received a delta
            }
            a.accumulate_grad(std::move(da));
        }
        if (b.requires_grad) {
            a.sync();
            auto at = transpose(a);
            grad_convert_inplace(at, grad_dtype);
            auto db = at.matmul(*output_grad);
            if (b.node != nullptr) {
                grad_probe_hit(db.dtype);
            }
            b.accumulate_grad(std::move(db));
        }
    }
}
