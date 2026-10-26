/**
 * @file TransposeLast2CPU.cpp
 * @author GreenBean
 * @brief CPU backend for the last-two-dims transpose. Dtype-agnostic byte copy;
 * transpose is memory-bound, so it stays scalar (threading comes from the
 * elementwise launcher).
 * @version 0.2
 * @date 2026-10-25
 */
#include <cstdint>
#include <cstring>

#include "Math/Autograd/AutogradShared.h"
#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Tensor/Tensor.h"

namespace BeanTensor::Tensors::detail {
    using namespace autograd_shared;

    namespace {
        /**
         * Local twin of autograd_shared::launch_elementwise with CCD-aware
         * scheduling: the transpose writes the destination buffer, so the gate
         * is the bytes written (numel * element size). When that fits one
         * CCD's LLC the chunks are submitted to the calling thread's CCD group
         * (submit_to_group) so the cache-resident destination is written
         * locally; otherwise they go to the whole pool exactly as before. The
         * chunk count still assumes all t_count threads — group workers simply
         * claim multiple chunks from the queue, so restricting the submit never
         * starves the decomposition.
         * @brief Threaded transpose launcher with CCD-local routing.
         * @param worker Callable taking (begin, end); copied into each task.
         * @param numel Total element count to cover.
         * @param out_bytes Bytes of the destination buffer actually written by the workers.
         * @param a First tensor the workers touch (receives the futures).
         * @param b Second tensor the workers touch (receives the futures).
         * @param out Output tensor (receives the futures).
         */
        template<typename Worker>
        void launch_elementwise_ccd(Worker&& worker, const size_t numel, const size_t out_bytes,
                                    const Tensor& a, const Tensor& b, const Tensor& out) {
            const auto t_count = static_cast<size_t>(IncludedCPUFeatures().threads);
            if (t_count <= 1 || numel <= t_count * 1024) {
                worker(0, numel);
                return;
            }
            auto& threadMgr = Threading::GetCPUThreadPool();
            std::vector<std::shared_future<void>> futures;
            const auto chunk = (numel + t_count - 1) / t_count;
            const bool keep_local = KeepWorkLocal(out_bytes);
            const auto group = keep_local ? CcdGroupOfCurrentCpu() : uint32_t{0};
            for (auto t = size_t{0}; t < t_count; ++t) {
                const auto begin = t * chunk;
                if (begin >= numel) {
                    break;
                }
                // worker is captured BY VALUE: these futures outlive this call
                // (they land on the tensors), so a reference would dangle.
                if (keep_local) {
                    futures.push_back(threadMgr.submit_to_group([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, group, Threading::TaskPriority::Normal).share());
                } else {
                    futures.push_back(threadMgr.submit([worker, begin, end = std::min(begin + chunk, numel)] { worker(begin, end); }, Threading::TaskPriority::Normal).share());
                }
            }
            a.give_futures(futures);
            b.give_futures(futures);
            out.give_futures(futures);
        }
    }

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
        launch_elementwise_ccd([=](const size_t b, const size_t e) {
            for (auto idx = b; idx < e; ++idx) {
                const auto batch = idx / slab;
                const auto rem = idx - batch * slab;
                const auto i = rem / cols;
                const auto j = rem - i * cols;
                std::memcpy(outp + (batch * slab + j * rows + i) * eb, in + idx * eb, eb);
            }
        }, numel, numel * static_cast<size_t>(eb), t, t, out);
    }
}
