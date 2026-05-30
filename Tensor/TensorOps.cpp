/*
 * ROCM
 */
#if defined(USE_HIP)
#include "Tensor/TensorOps.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "Tensor.h"

template<typename Ptr>
__global__ void implementation_gpu_set_val(Ptr* __restrict__ pointer, const size_t end, int32_t value) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < end; i += stride) {
        pointer[i] = value;
    }
}

template<typename Ptr> requires std::is_floating_point_v<Ptr>
__global__ void implementation_bf16_to_fp32_fp64(Casting::bfloat16_t* __restrict__ src, Ptr* __restrict__ dst, const size_t end, const bool allow_unsafe_conversions, int32_t* __restrict__ fail_check) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t end2 = end / 2;

    for (size_t i = tid; i < end2; i += stride) {
        const __hip_bfloat162 v = reinterpret_cast<const __hip_bfloat162*>(src)[i];
        dst[i * 2]     = static_cast<Ptr>(__bfloat162float(v.x));
        dst[i * 2 + 1] = static_cast<Ptr>(__bfloat162float(v.y));
    }

    if (end % 2 != 0 && tid == 0) {
        dst[end - 1] = static_cast<Ptr>(__bfloat162float(src[end - 1]));
    }
    if (tid == 0) {
        fail_check[0] = false;
    }
}

template<typename Ptr> requires std::is_floating_point_v<Ptr>
__device__ int helper_fp32_fp64_to_bf16(Ptr* val1) {
    constexpr Ptr BF16_MAX = static_cast<Ptr>(3.38953138925153547590470800371487866880e+38);
    if constexpr (std::is_same_v<Ptr, float>) {
        if (__isnanf(*val1) || __isinff(*val1) || *val1 > BF16_MAX || *val1 < -BF16_MAX) return 1;
    } else {
        if (__isnan(*val1) || __isinf(*val1) || *val1 > BF16_MAX || *val1 < -BF16_MAX) return 1;
    }
    return 0;
}

template<typename Ptr> requires std::is_floating_point_v<Ptr>
__global__ void implementation_fp32_fp64_to_bf16(Ptr* __restrict__ src, Casting::bfloat16_t* __restrict__ dst, const size_t end, const bool allow_unsafe_conversions, int32_t* __restrict__ fail_check) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t end2 = end / 2;

    for (size_t i = tid; i < end2; i += stride) {
        if (!allow_unsafe_conversions) {
            if (helper_fp32_fp64_to_bf16(&src[i * 2]) || helper_fp32_fp64_to_bf16(&src[i * 2 + 1])) {
                atomicOr(fail_check, 1);
                return;
            }
        }
        const __hip_bfloat162 packed = {
            __float2bfloat16(static_cast<float>(src[i * 2])),
            __float2bfloat16(static_cast<float>(src[i * 2 + 1]))
        };
        reinterpret_cast<__hip_bfloat162*>(dst)[i] = packed;
    }

    if (end % 2 != 0 && tid == 0) {
        if (!allow_unsafe_conversions && helper_fp32_fp64_to_bf16(&src[end - 1])) {
            atomicAdd(fail_check, 1);
            return;
        }
        dst[end - 1] = __float2bfloat16(static_cast<float>(src[end - 1]));
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
                            static_cast<int8_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::Int16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int16_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::Int32, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int32_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int32_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::Int64, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<int64_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<int64_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::UInt8, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint8_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint8_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::UInt16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint16_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::UInt32, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint32_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint32_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::UInt64, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<uint64_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<uint64_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::BFloat16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::bfloat16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::bfloat16_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::Float16, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::float16_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::float16_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::Float32, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::float32_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::float32_t*>(t2->data), t2->numel, quantity
                        );
                    },
                },
                {
                    DType::Float64, [](Tensor* t2, size_t blocks2, size_t threads2, int32_t quantity) {
                        hipLaunchKernelGGL(
                            implementation_gpu_set_val<Casting::float64_t>,
                            dim3(blocks2), dim3(threads2), 0, 0,
                            static_cast<Casting::float64_t*>(t2->data), t2->numel, quantity
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
            auto temp = Tensor({t.numel}, t.dtype, Device::CPU, false);
            temp.random(min, max, seed);
            temp.sync();
            move_to_gpu(temp, t.gpu_id);
            temp.sync();
            t.kill_cpu_data();
            t.data = temp.data;
            temp.data = nullptr;
            // NEVER EXECUTE .SYNC WITHIN A TENSORS OWN WORKER
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
        t.kill_buffer();
        const auto worker = [&t, device] {
            {
                int count = 0;
                HIP_CHECK_ERROR(hipGetDeviceCount(&count));
                if (count < static_cast<int>(device)) {
                    throw std::invalid_argument("Invalid GPU device ID");
                }
                size_t free, total = 0;
                HIP_CHECK_ERROR(hipMemGetInfo(&free, &total));
                if (free <= t.nbytes) throw ErrorHandling::OutOfMemory();
            }
            HIP_CHECK_ERROR(hipSetDevice(static_cast<int32_t>(device)));
            void* buffer = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&buffer, t.numel * Casting::dtype_size(t.dtype)));
            HIP_CHECK_ERROR(hipMemcpy(buffer, t.data, t.nbytes, hipMemcpyHostToDevice));
            t.kill_cpu_data();
            t.data = buffer;
            t.device = Device::GPU;
            t.gpu_id = device;
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(worker, Threading::ThreadPriority::High).share();
        t.give_future(future);
    }

    __host__ void move_to_cpu(Tensor& t) {
        if (t.device != Device::GPU) return;
        t.sync();
        t.kill_buffer();
        const auto worker = [&t] {
            hipDeviceProp_t prop{};
            HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, static_cast<int>(t.gpu_id)));
            if (prop.warpSize == 0) throw std::invalid_argument("Invalid GPU id");
            void* buffer = std::aligned_alloc(64, t.nbytes);
            HIP_CHECK_ERROR(hipMemcpy(buffer, t.data, t.nbytes, hipMemcpyDeviceToHost));
            HIP_CHECK_ERROR(hipFree(t.data));
            t.data = buffer;
            t.device = Device::CPU;
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(worker, Threading::ThreadPriority::High).share();
        t.give_future(future);
    }

    __host__ void move_gpu_to_gpu(Tensor& t, uint32_t new_device) {
        if (t.device != Device::GPU) return;
        t.sync();
        t.kill_buffer();
        const auto worker = [&t, new_device] {
            {
                hipDeviceProp_t prop{};
                hipDeviceProp_t new_prop{};
                HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, static_cast<int>(t.gpu_id)));
                HIP_CHECK_ERROR(hipGetDeviceProperties(&new_prop, static_cast<int>(new_device)));
                if (prop.warpSize == 0 || new_prop.warpSize == 0) throw std::invalid_argument("Invalid GPU id");
                size_t free, total = 0;
                HIP_CHECK_ERROR(hipMemGetInfo(&free, &total));
                if (free <= t.nbytes) throw ErrorHandling::OutOfMemory();
            }
            void* buffer = nullptr;
            // hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device
            HIP_CHECK_ERROR(hipSetDevice(static_cast<int32_t>(new_device)));
            HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(static_cast<int32_t>(t.gpu_id), 0));
            HIP_CHECK_ERROR(hipDeviceEnablePeerAccess(static_cast<int32_t>(new_device), 0));
            HIP_CHECK_ERROR(hipSetDevice(static_cast<int32_t>(new_device)));
            HIP_CHECK_ERROR(hipMalloc(&buffer, t.nbytes));
            HIP_CHECK_ERROR(hipMemcpyPeer(buffer, new_device, t.data, t.gpu_id, t.nbytes));
            HIP_CHECK_ERROR(hipFree(t.data));
            t.data = buffer;
            t.device = Device::GPU;
            t.gpu_id = new_device;
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(worker, Threading::ThreadPriority::High).share();
        t.give_future(future);
    }

    __host__ void convert_gpu(Tensor& t, const BeanTensor::Casting::DType& new_dtype, const bool use_inf_conversions) {
        if (t.device != Device::GPU) return;
        if (t.dtype == new_dtype) return;
        t.sync();
        t.kill_buffer();
        // NOTE: new_dtype/use_inf_conversions must be captured BY VALUE — the worker runs
        // async on the thread pool after convert_gpu returns, by which point new_dtype (a
        // reference bound to a temporary in the caller) and the local bool have been destroyed.
        const auto worker = [&t, new_dtype, use_inf_conversions] {
            const size_t target_buffer_size = t.numel * Casting::dtype_size(new_dtype);
            if (t.convert_buf == nullptr || t.convert_buf_size < target_buffer_size) {
                HIP_CHECK_ERROR(hipFree(t.convert_buf));
                HIP_CHECK_ERROR(hipMalloc(&t.convert_buf, target_buffer_size));
                t.convert_buf_size = target_buffer_size;
            }
            constexpr size_t threads = 256;
            const size_t blocks = (t.numel + threads - 1) / threads;
            int32_t* check_err = nullptr;
            HIP_CHECK_ERROR(hipMalloc(&check_err, sizeof(int32_t)));
            constexpr int32_t zero = 0;
            HIP_CHECK_ERROR(hipMemcpy(check_err, &zero, sizeof(int32_t), hipMemcpyHostToDevice));

            if (Casting::dtype_is_float(new_dtype)) {
                if (t.dtype == Casting::DType::Float32 && new_dtype == Casting::DType::BFloat16) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(implementation_fp32_fp64_to_bf16<Casting::float32_t>),
                        dim3(blocks), dim3(threads), 0, nullptr,
                        static_cast<Casting::float32_t*>(t.data),
                        static_cast<Casting::bfloat16_t*>(t.convert_buf),
                        t.numel, use_inf_conversions, check_err
                    );
                } else if (t.dtype == Casting::DType::BFloat16 && new_dtype == Casting::DType::Float32) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(implementation_bf16_to_fp32_fp64<Casting::float32_t>),
                        dim3(blocks), dim3(threads), 0, nullptr,
                        static_cast<Casting::bfloat16_t*>(t.data),
                        static_cast<Casting::float32_t*>(t.convert_buf),
                        t.numel, use_inf_conversions, check_err
                    );
                }
            }

            int32_t ret = 0;
            HIP_CHECK_ERROR(hipMemcpy(&ret, check_err, sizeof(int32_t), hipMemcpyDeviceToHost));
            assert(check_err != nullptr && ret == 0);
            HIP_CHECK_ERROR(hipFree(check_err));

            void* old_data = t.data;
            t.data = t.convert_buf;
            t.convert_buf = old_data;
            t.convert_buf_size = t.nbytes;
            t.dtype = new_dtype;
            t.nbytes = target_buffer_size;
            t.owns_data = true;
        };
        auto& threadMgr = Threading::get_cpu_thread_pool();
        const auto future = threadMgr.submit(worker, Threading::ThreadPriority::High).share();
        t.give_future(future);
    }
}

#endif