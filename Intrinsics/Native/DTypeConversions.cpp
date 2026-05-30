//
// Created by greenbean on 2026-05-30.
//

#include "DTypeConversions.h"
namespace BeanTensor::Intrinsics::detail {
    std::vector<std::shared_future<void>> unaccelerated_bf16_to_fp32(const Casting::bfloat16_t* src, float* dst, const size_t end) {
        const size_t t = Hardware::CPU().threads;

        auto compute = [src, dst](const size_t start, const size_t end_l) {
            for (size_t pos = start; pos < end_l; ++pos) {
                uint16_t raw;
                std::memcpy(&raw, &src[pos], sizeof(raw));
                uint32_t bits = static_cast<uint32_t>(raw) << 16;
                dst[pos] = std::bit_cast<Casting::float32_t>(bits);
            }
        };

        std::vector<std::shared_future<void>> futures;
        if (end <= t * 1024) {
            compute(0, end);
            return futures;
        }

        static auto& threadMgr = Threading::get_cpu_thread_pool();
        const size_t chunk_size = ((end / t + 15) / 16) * 16;

        size_t actual_threads = 0;
        for (size_t tid = 0; tid < t; ++tid) {
            if (tid * chunk_size >= end) break;
            ++actual_threads;
        }

        auto worker = [compute](const size_t start, const size_t end_l) {
            compute(start, end_l);
        };
        for (size_t thread_id = 0; thread_id < t; ++thread_id) {
            const size_t start = thread_id * chunk_size;
            const size_t end_l = std::min(start + chunk_size, end);
            if (start >= end) break;
            futures.push_back(threadMgr.submit(
                [=] { worker(start, end_l); },
                Threading::ThreadPriority::Normal
            ).share());
        }
        return futures;
    }
}