#pragma once

#include <vector>
#include <memory>
#include <functional>
#include "Dtypes.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#elif  USE_CUDA
#include <cuda_runtime.h>
#else

#endif

#define BT_TENSOR_MAX_DIMS 8
namespace BeanTensor::Tensors {


    struct Tensor { // 256

        size_t shape[BT_TENSOR_MAX_DIMS]{}; // Multinational tensor indexing
        size_t strides[BT_TENSOR_MAX_DIMS]{}; // Multi-dimensional tensor indexing
        uint32_t ndim = 0; // lengthof(shape)
        size_t numel = 0; // Number of elements, flat
        size_t nbytes = 0; // Number of bytes

        void* stream = nullptr; // hipStream_t / cudaStream_t (GPU Only)
        uint32_t gpu_id = 0; //Owner GPU

        bool owns_data = false; // Is this tensor the owner of its data?
        size_t offset = 0; // How far in the data is this tensor?
        bool is_contiguous = true; // Is this tensor contiguous in memory?

        Casting::DType dtype = Casting::DType::Float32; // What datatype is this tensor?

        enum class Device {
            CPU,
            GPU
        } device = Device::CPU; // What device type owns this tensors' data?

        void* data{};   // CPU or GPU pointer

        bool requires_grad = false; // Does this tensor require grad?
        struct Node* grad_fn{}; // Parent grad functions?
        Tensor* parents[4] = {}; //Parent tensors?
        uint8_t n_parents = 0; //Parent count (for 2-4)
        bool is_leaf = true; // Autograd property
    };

    inline void _compute_strides(const size_t* shape, size_t* strides, const uint32_t ndim) {
        strides[ndim - 1] = 1;
        for (int i = static_cast<int>(ndim) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }

    /* Get a value at a tensor based on idx (tensor[1][4] = {1, 4})*/
    template<typename T>
    T* at_location(Tensor* tensor, const std::initializer_list<int> idx) {
        assert(static_cast<int>(idx.size()) == tensor->ndim);
        auto* base = static_cast<T*>(tensor->data) + tensor->offset;
        for (unsigned int i = 0; i < tensor->ndim; ++i) {
            assert(idx.begin()[i] >= 0 && idx.begin()[i] < tensor->shape[i]);
            base += idx.begin()[i] * tensor->strides[i];
        }
        return base;
    }

    /* Make a new tensor */
    inline Tensor make_tensor(const size_t* shape,
        const uint32_t ndim,
        const Casting::DType dtype = Casting::DType::Float32,
        const Tensor::Device device = Tensor::Device::CPU,
        const bool requires_grad = false
        ) {
        if (ndim == 0 || ndim > BT_TENSOR_MAX_DIMS) {
            throw std::invalid_argument("Tensor dimension exceeds maximum supported dimensions");
        }
        Tensor new_tensor{};
        new_tensor.ndim = ndim;
        new_tensor.dtype = dtype;
        new_tensor.device = device;
        new_tensor.requires_grad = requires_grad;
        new_tensor.owns_data = true;

        new_tensor.numel = 1;
        for (uint32_t i = 0; i < ndim; ++i) {
            new_tensor.shape[i] = shape[i];
            new_tensor.numel *= shape[i];
        }
        new_tensor.nbytes = new_tensor.numel * (BeanTensor::Casting::dtype_size(dtype));

        _compute_strides(new_tensor.shape, new_tensor.strides, new_tensor.ndim);

        if (device == Tensor::Device::CPU) {
            new_tensor.data = std::malloc(new_tensor.nbytes);
            assert(new_tensor.data != nullptr);
        } else {
#if defined(USE_CUDA)
            int curr;
            cudaGetDevice(&curr);
            new_tensor.gpu_id = curr;
            cudaMalloc(&new_tensor.data, new_tensor.nbytes);
#elif defined(USE_HIP)
            int curr;
            hipGetDevice(&curr);
            new_tensor.gpu_id = curr;
            hipMalloc(&new_tensor.data, new_tensor.nbytes);
#else
assert(false);
#endif
        }
        return new_tensor;
    }
    /* Free a tensor */
    inline void free_tensor(Tensor* tensor) {
        if (tensor->owns_data) {
            if (tensor->device == Tensor::Device::CPU) {
                std::free(tensor->data);
            } else {
#if defined(USE_CUDA)
                cudaFree(tensor->data);
#elif defined(USE_HIP)
                hipFree(tensor->data);
#else
                assert(false);
#endif
            }
        }
    }

    /* make a shallow "view" copy */
    inline Tensor make_view(const Tensor& parent_tensor) {
        Tensor child_tensor = parent_tensor;
        child_tensor.owns_data = false;
        return child_tensor;
    }

    /* Swap two stride and shape dims, and get a shallow copy */
    inline Tensor transpose(const Tensor& tensor, const uint32_t dim0, const uint32_t dim1) {
        auto transposed = make_view(tensor);
        transposed.owns_data = false;
        std::swap(transposed.shape[dim0], transposed.shape[dim1]);
        std::swap(transposed.strides[dim0], transposed.strides[dim1]);
        transposed.is_contiguous = false;
        return transposed;
    }

    inline Tensor deep_copy(const Tensor& src) {
     Tensor dst = src;
        dst.owns_data = true;
        dst.offset = 0;

        if (src.device == Tensor::Device::CPU) {
            dst.data = std::malloc(src.nbytes);
            assert(dst.data != nullptr);
            std::memcpy(dst.data, src.data, src.nbytes);
        } else {
#if defined(USE_CUDA)
            int curr;
            cudaGetDevice(&curr);
            assert(curr == src.gpu_id);
            cudaMalloc(&dst.data, src.nbytes);
            cudaMemcpy(dst.data, src.data + src.offset, src.nbytes, cudaMemcpyDeviceToDevice);
#elif defined(USE_HIP)
            int curr;
            hipGetDevice(&curr);
            assert(curr == (int)src.gpu_id);
            hipMalloc(&dst.data, src.nbytes);
            hipMemcpy(dst.data, (static_cast<char*>(src.data) + src.offset), src.nbytes, hipMemcpyDeviceToDevice);
#else
            assert(false);
#endif
        }
        return dst;
    }
    inline bool is_same_device(const Tensor& t1, const Tensor& t2) {
        if (t1.device != t2.device) {
            return false;
        }
        if (t1.device == Tensor::Device::CPU && t2.device == Tensor::Device::CPU) {
            return true;
        }
        return (t1.gpu_id == t2.gpu_id);
    }
    inline bool is_same_shape(const Tensor& t1, const Tensor& t2) {
        return (t1.ndim == t2.ndim) && std::equal(t1.shape, t1.shape + t1.ndim, t2.shape);
    }

    inline std::string shape_to_string(const Tensor& tensor) {
        std::ostringstream os;
        os << "{";
        for (uint32_t i = 0; i < tensor.ndim; ++i) {
            os << tensor.shape[i];
            if (i < tensor.ndim - 1) os << ", ";
        }
        os <<"}";
        return os.str();
    }

    inline std::string to_string(const Tensor& tensor) {
        std::ostringstream os;
        os << "Tensor(shape=" << shape_to_string(tensor) << "\nstrides=[";
        for (uint32_t i = 0; i < tensor.ndim; ++i) {
            os << tensor.strides[i];
            if (i < tensor.ndim - 1) os << ", ";
        }
        os << "]\nbytes=[" << tensor.nbytes << "]";
        os << "\ndevice[";
        if (tensor.device == Tensor::Device::CPU) {
            os << "CPU";
        } else {
            os << "GPU:" << tensor.gpu_id;
        }
        os << "]";

        return os.str();
    }
}
