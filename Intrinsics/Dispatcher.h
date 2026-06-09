#pragma once
#include <future>
#include <vector>
#include <Tensor/Tensor.h>
#include <iostream>
#include "Helper/Threads/CPUThreading.h"
namespace BeanTensor::Intrinsics::detail {

    inline void launch_conv_cpu(Tensors::Tensor& tensor, BeanTensor::Casting::DType dtype, ConversionClampMethod method) {

        const size_t new_nbytes = tensor.numel * Casting::dtype_size(dtype);

        // SCRATCH BUFFER: grow convert_buf if needed, never shrink, never free between calls.
        // Replaces aligned_alloc per call — after warmup this is zero allocations.
        if (new_nbytes > tensor.convert_buf_size) {
            std::free(tensor.convert_buf);
            tensor.convert_buf = std::aligned_alloc(64, new_nbytes);
            if (tensor.convert_buf == nullptr) throw ErrorHandling::OutOfMemory();
            tensor.convert_buf_size = new_nbytes;
        }
        void* dst = tensor.convert_buf;
        // Ending

        //All standardized paths have been discovered.

        void* src_data = tensor.data;
        const size_t numel = tensor.numel;
        const size_t old_nbytes = tensor.nbytes;

        BT_DISPATCH_DTYPE_AS(tensor.dtype, src_t, [&] {
            BT_DISPATCH_DTYPE_AS(dtype, dst_t, [&] {
                if constexpr (            (
            (std::is_floating_point_v<src_t> &&
            std::is_floating_point_v<dst_t>)
            ) || (
            (std::is_integral_v<src_t> &&
            std::is_integral_v<dst_t>)
            ) || (
            std::is_integral_v<src_t> &&
            std::is_floating_point_v<dst_t>
            ) || (
                std::is_floating_point_v<src_t> &&
                std::is_integral_v<dst_t>
            )) {
                    auto global_standard_manager = [src_data, numel, method, dst]() {
                        const size_t threads = Hardware::CPU().threads;
                        auto& threadMgr = Threading::get_cpu_thread_pool();
                        auto stan2stand_fun = [src_data, method](const size_t start, const size_t end, void* dst_ptr) {
                            bool flag = false;
                            auto* ptr = static_cast<dst_t*>(dst_ptr);
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = standard2standard<src_t, dst_t>(static_cast<src_t*>(src_data)[i], flag, method);
                            }
                            if (flag == true) throw ErrorHandling::ConversionError();
                        };
                        if (numel <= 1024) {
                            stan2stand_fun(0, numel, dst);
                            return;
                        } else {
                            std::vector<std::shared_future<void>> futures;
                            size_t chunk = (numel + threads - 1) / threads;
                            for (size_t i = 0; i < threads; i++) {
                                const size_t t_start = i * chunk;
                                const size_t end = std::min(t_start + chunk, numel);
                                futures.push_back(threadMgr.submit(
                                    [=] {stan2stand_fun(t_start, end, dst);},
                                    Threading::ThreadPriority::High
                                    ).share());
                            }
                            for (size_t i = 0; i < futures.size(); i++) {
                                futures[i].get();
                            }
                            return;
                        }
                    };
                    const size_t threads = Hardware::CPU().threads;
                    auto& threadMgr = Threading::get_cpu_thread_pool();
                    std::shared_future l_future = threadMgr.submit(
                        [=] {
                            global_standard_manager();
                        }, Threading::ThreadPriority::High
                    ).share();

                    tensor.give_future(l_future);

                    tensor.convert_buf = tensor.data;
                    tensor.convert_buf_size = old_nbytes;
                    tensor.data = dst;
                    tensor.dtype = dtype;
                    tensor.nbytes = new_nbytes;
                } else {
                    assert(false);
                }
            });
        });
    }
}