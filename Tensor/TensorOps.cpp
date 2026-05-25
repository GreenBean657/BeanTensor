/*
 * ROCM
 */
#if defined(USE_HIP)
#include "Tensor/TensorOps.h"
#include <hip/hip_runtime.h>

#include "Tensor.h"

/**
 * [Detail] Generate GPU values based on specified parameter.
 * @tparam Ptr Dtype pointer
 * @param pointer Pointer to replace.
 * @param end Number of elements in this tensor.
 * @param value Value to set.
 */
template<typename Ptr>
__global__ void implementation_gpu_set_val(Ptr* __restrict__ pointer , const int64_t end, int32_t value) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < end; i += stride) {
        pointer[i] = value;
    }
}

namespace BeanTensor::Tensors::detail {

    __host__ void set_amount_gpu(Tensor& t, int32_t amount) {
        if (t.device != Device::GPU) return;
        t.sync();
        static auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto worker = [&t, amount] {

            constexpr size_t threads = 256;
            const size_t blocks = (t.numel + threads - 1) / threads;
            using Args = void(*)(Tensor* t, size_t blocks, size_t threads, int32_t quantity);
            using DType = Casting::DType;
            static const std::unordered_map<DType, Args> kernel_fns = {
                {
                    DType::Int8, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int8_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int8_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
            {
                    DType::Int16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int16_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::Int32, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int32_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int32_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::Int64, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int64_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int64_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::UInt8, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint8_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint8_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::UInt16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint16_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::UInt32, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint32_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint32_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::UInt64, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint64_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint64_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::BFloat16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::bfloat16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::bfloat16_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::Float16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::float16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::float16_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::Float32, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::float32_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::float32_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
                {
                    DType::Float64, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::float64_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::float64_t*>(t2->data),
                            t2->numel,
                            quantity
                        );
                    },
                },
            };
            const auto it = kernel_fns.find(t.dtype);
            if (it == kernel_fns.end()) {
                throw ErrorHandling::NotImplemented();
            }
            it->second(&t, blocks, threads, amount);
        };
        const auto future = threadMgr.submit(
            worker,
            Threading::ThreadPriority::High
        ).share();
        t.give_future(future);
    }

    __host__ void set_random_gpu(Tensor& t, double min, double max, int64_t seed) {
        if (t.device != Device::GPU) return;
        t.sync();
        auto worker = [&t, min, max, seed] {
            return nullptr;
            /*
            Tensor temp({t.numel}, t.dtype, Device::GPU);
            temp.random(min, max, seed);
            temp.to(t.device);
            // Copy temp to t
            hipError_t err = hipMemcpyAsync(
                t.data, temp.data, t.nbytes,
                hipMemcpyDeviceToDevice,
                t.stream
            );
            */
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(
            worker,
            Threading::ThreadPriority::High
        ).share();
        t.give_future(future);
    }

    __host__ void move_to_gpu(Tensor& t, uint32_t device) {
        if (t.device != Device::CPU) return;
        t.sync();
        const auto worker = [&t, device] {
            int count = 0;
            HIP_CHECK_ERROR(hipGetDeviceCount(&count));
            if (count < device) {
                throw std::invalid_argument("Invalid GPU device ID");
            }
            HIP_CHECK_ERROR(hipSetDevice(static_cast<int32_t>(device)));
            void* buffer = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&buffer,(t.numel*Casting::dtype_size(t.dtype))));
            HIP_CHECK_ERROR(hipMemcpy(buffer, t.data, t.nbytes, hipMemcpyHostToDevice));
            t.kill_cpu_data();
            t.data = buffer;
            t.device = Device::GPU;
            t.gpu_id = device;
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(worker,Threading::ThreadPriority::High).share();
        t.give_future(future);
    };

    __host__ void move_to_cpu(Tensor& t) {
        if (t.device != Device::GPU) return;
        const auto worker = [&t] {
            hipDeviceProp_t prop;
            HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, t.gpu_id));
            if (prop.warpSize == 0) throw std::invalid_argument("Invalid GPU id");
            void* buffer = std::aligned_alloc(64, t.nbytes);
            HIP_CHECK_ERROR(hipMemcpy(buffer, t.data, t.nbytes, hipMemcpyDeviceToHost));
            HIP_CHECK_ERROR(hipFree(t.data));
            t.data = buffer;
            t.device = Device::CPU;
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(worker,Threading::ThreadPriority::High).share();
        t.give_future(future);
    }

    void move_gpu_to_gpu(Tensor& t, uint32_t new_device) {
        throw ErrorHandling::NotImplemented();
    }
}

#endif
