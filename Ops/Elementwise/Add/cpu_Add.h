#pragma once
#include "Tensor/tensor.h"

/*
*    enum class DType {
BFloat16,
Float16,
Float32,
Float64,
Int64,
Int32,
Int16,
Int8,
UInt64,
UInt32,
UInt16,
UInt8
};
*/

namespace BeanTensor::Ops::detail::Add {
    void add_cpu_int8(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_int16(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_int32(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_int64(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_uint8(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_uint16(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_uint32(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_uint64(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_fp16(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_bf16(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_fp32(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
    void add_cpu_fp64(const Tensors::Tensor& t1, const Tensors::Tensor& t2, Tensors::Tensor& returnedTensor);
}
