#include <cstdint>

#include "Tensor/Tensor.h"
#include "Tensor/DTypes.h"
#include "TensorOps.h"

#include "Helpers/CPUThreading.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template<typename Ptr> requires (!BeanTensor::DataTypes::requires_unpack_v<Ptr>)
__global__ void impli_gpu_set_static_val_unpacked(Ptr* __restrict__ ptr, const size_t end, const int32_t val) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    if (tid >= end) return;
    for (size_t i = tid; i < end; i += stride) {
        ptr[i] = val;
    }
}

/**
 * Force static assertion failure without compilation.
 * @brief Trigger static assertion failure.
 */
template <typename>
inline constexpr bool dependent_false_v = false;

/**
 * Implementation - set a static value into a packed type, on the GPU.
 * @brief GPU kernel, set static value into packed data.
 * @tparam Ptr Target pointer, must be a packed type.
 * @tparam Val Target value for packing, must be a packable (but unpacked) type.
 * @param ptr Target pointer.
 * @param end End index of the range to set.
 * @param value Target value to set.
 */
template<typename Ptr, typename Val>
requires (BeanTensor::DataTypes::requires_unpack_v<Ptr> && BeanTensor::DataTypes::can_be_packed_v<Val>)
__global__ void impli_gpu_set_static_val_packed(Ptr* __restrict__ ptr, const size_t end, const Val value) {
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;

    __shared__ Ptr shared_value;

    if (threadIdx.x == 0) {
        if constexpr (std::is_same_v<Ptr, BeanTensor::float16x2_t>) {
            shared_value = __halves2half2(value, value);
        } else if constexpr (std::is_same_v<Ptr, BeanTensor::bfloat16x2_t>) {
            shared_value = __halves2bfloat162(value, value);
        } else {
            static_assert(dependent_false_v<Ptr>, "Unsupported packed type");
        }
    }
    __syncthreads();

    for (size_t i = tid; i < end; i += stride) {
        ptr[i] = shared_value;
    }
}


    namespace BeanTensor::Tensors::detail {
        void set_amount_gpu(Tensor& t, int32_t amount) {
            if (t.device != Device::GPU) {
                assert(false && "Call of GPU function on CPU tensor"); std::unreachable();
            }
            t.sync();
            static auto& threadMgr = Threading::CPUThreadPool;
            const auto worker = [&t, amount] {
                constexpr size_t threads = 256;
                const size_t blocks = (t.numel + threads - 1) / threads;
                using Args = void(*)(Tensor* t, size_t blocks, size_t threads, int32_t amount);

            };
        }

        void set_random_gpu(Tensor& t, double min, double max, std::int64_t seed) {}


        void move_to_gpu(Tensor& t, uint32_t device) {

        }

        void move_to_cpu(Tensor& t) {

        }

        void convert_gpu(Tensor& t, const DataTypes::DType& new_dtype, bool use_inf_conversions) {

        }
    }


#include "TensorOps.hpp"