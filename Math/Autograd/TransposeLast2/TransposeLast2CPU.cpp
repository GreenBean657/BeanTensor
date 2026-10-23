/**
 * @file TransposeLast2CPU.cpp
 * @author GreenBean
 * @brief CPU backend for the last-two-dims transpose. Dtype-agnostic byte copy;
 * transpose is memory-bound, so it stays scalar (threading comes from the
 * elementwise launcher).
 * @version 0.1
 * @date 2026-10-20
 */
#include <cstdint>
#include <cstring>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    /**
     * Swaps the last two dimensions: out[..., j, i] = t[..., i, j]. Dtype-agnostic byte
     * copy of whole storage elements (a packed element moves as a unit, its lanes are
     * not swapped). out must already have the swapped shape and t's dtype. Work is
     * queued on the CPU pool and returns immediately; futures land on t and out.
     * @brief CPU transpose of the last two dimensions.
     * @param t Source tensor, rank >= 2.
     * @param out Destination tensor with the last two dimensions swapped (written).
     * @throws ErrorHandling::DirectFunctionCallOnWrongDataSource if either tensor is not on the CPU.
     */
    void transpose_last2_cpu(const Tensor& t, const Tensor& out) {
        if (t.device != Device::CPU || out.device != Device::CPU) {
            throw ErrorHandling::DirectFunctionCallOnWrongDataSource("CPU", stringify_device_source(t.device));
        }
        // Dtype-agnostic: an elementwise byte copy. Transpose is memory-bound,
        // so it stays scalar; threading comes from the elementwise launcher.
        const auto eb = DataTypes::dtype_bytes(t.dtype);
        const auto rows = t.shape[t.ndim - 2];
        const auto cols = t.shape[t.ndim - 1];
        const auto slab = rows * cols;
        const auto numel = t.numel;
        const auto* in = static_cast<const uint8_t*>(t.data) + t.offset * eb;
        auto* outp = static_cast<uint8_t*>(out.data) + out.offset * eb;
        /**
         * Runs on a pool worker (or inline for small inputs) over the element range [b, e) of the source. Captures by value (the futures outlive this call): in, outp, slab, rows, cols, eb.
         * Each source element idx is decomposed into (batch, i, j) and its eb bytes are copied to the transposed position (batch * slab + j * rows + i).
         * @brief Worker for transpose_last2_cpu.
         * @param b First element index of the range (inclusive).
         * @param e One past the last element index of the range.
         */
        launch_elementwise([=](const size_t b, const size_t e) {
            for (auto idx = b; idx < e; ++idx) {
                const auto batch = idx / slab;
                const auto rem = idx - batch * slab;
                const auto i = rem / cols;
                const auto j = rem - i * cols;
                std::memcpy(outp + (batch * slab + j * rows + i) * eb, in + idx * eb, eb);
            }
        }, numel, t, t, out);
    }
}
