#pragma once

#include <memory>
#include <functional>
#include "Dtypes.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

#define BT_TENSOR_MAX_DIMS 8
#define BT_TENSOR_MAX_PARENTS 3
#include <iostream>
#include <random>

#include "tensor.h"
#include "tensor_convert.h"
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#elif  USE_CUDA
#include <cuda_runtime.h>
#else

#endif

namespace BeanTensor::Tensors::detail {

    template<typename T>
    void fill_random_impl(void* data, const size_t numel, const size_t seed, const double min, const double max) {
        const auto threads = Hardware::CPU().threads;
        auto* ptr = static_cast<T*>(data);
        if constexpr (std::is_integral_v<T>) {

            auto worker = [&](const size_t t_start, const size_t t_end, const size_t l_seed, const double t_min, const double t_max) {
                thread_local std::mt19937 l_rng(l_seed);
                thread_local auto dist = std::uniform_int_distribution(
                static_cast<int64_t>(t_min),
                static_cast<int64_t>(t_max));
                for (size_t i = t_start; i < t_end; ++i) {
                    ptr[i] = static_cast<T>(dist(l_rng));
                }
            };
            if (numel <= threads * 64) {
                worker(0, numel, seed, min, max);
                return;
            }
            std::vector<std::thread> thread_pool(threads);
            const size_t chunk_size = (numel + threads - 1) / threads;
            for (size_t t = 0; t < threads; ++t) {
                size_t t_start = t * chunk_size;
                size_t t_end = std::min(t_start + chunk_size, numel);
                thread_pool[t] = std::thread(worker, t_start, t_end, seed + t, min, max);
            }
            for (size_t t = 0; t < threads; ++t) {
                thread_pool[t].join();
            }
        } else {
            auto worker = [&](const size_t t_start, const size_t t_end, const size_t l_seed, const double t_min, const double t_max) {
                thread_local std::mt19937 l_rng(l_seed);
                thread_local auto dist = std::uniform_real_distribution<>(t_min, t_max);
                for (size_t i = t_start; i < t_end; ++i) {
                    ptr[i] = static_cast<T>(dist(l_rng));
                }
            };
            if (numel <= threads * 64) {
                worker(0, numel, seed, min, max);
                return;
            }
            std::vector<std::thread> thread_pool(threads);
            const size_t chunk_size = (numel + threads - 1) / threads;
            for (size_t t = 0; t < threads; ++t) {
                size_t t_start = t * chunk_size;
                size_t t_end = std::min(t_start + chunk_size, numel);
                thread_pool[t] = std::thread(worker, t_start, t_end, seed + t, min, max);
            }
            for (size_t t = 0; t < threads; ++t) {
                thread_pool[t].join();
            }
        }
    }

    template<typename T>
    void print_recursive(std::ostringstream& os, const T* data,
                     const size_t* shape, const size_t ndim,
                     size_t& idx, const size_t depth) {
        os << "[";
        if (ndim == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                os << data[idx++];
                if (i < shape[0] - 1) os << ", ";
            }
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                print_recursive(os, data, shape + 1, ndim - 1, idx, depth + 1);
                if (i < shape[0] - 1) os << ",\n" << std::string(depth + 1, ' ');
            }
        }
        os << "]";
    }

}


namespace BeanTensor::Tensors {
    enum class Device {
        CPU,
        GPU
    };

    struct Tensor;

    Tensor make_tensor(
        );

    struct Node;

    struct Tensor {

        size_t _shape[BT_TENSOR_MAX_DIMS]{}; // Multinational tensor indexing
        size_t _strides[BT_TENSOR_MAX_DIMS]{}; // Multidimensional tensor indexing
        size_t _ndim = 0; // length of(shape)
        size_t _numel = 0; // Number of elements, flat
        size_t _nbytes = 0; // Number of bytes

        void* _stream = nullptr; // hipStream_t / cudaStream_t (GPU Only)
        uint32_t _gpu_id = 0; //Owner GPU

        bool _owns_data = false; // Is this tensor the owner of its data?
        size_t _offset = 0; // How far in the data is this tensor? (in elements)
        bool _is_contiguous = true; // Is this tensor contiguous in memory?

        Casting::DType _dtype = Casting::DType::Float32; // What datatype is this tensor?

        Device _device = Device::CPU; // What device type owns this tensors' data?

        void* _data{};   // CPU or GPU pointer

        bool _requires_grad = false; // Does this tensor require grad?
        Tensor* _grad = nullptr; // Accumulated gradient of same tensor shape
        Node* _grad_fn = nullptr; // Parent grad functions?
        Tensor* _parents[BT_TENSOR_MAX_PARENTS] = {}; //Parent tensors?
        uint8_t _n_parents = 0; //Parent count (for 2-4)
        bool _is_leaf = true; // Autograd property

        Tensor() = default;
        // Declare only — no body yet
        Tensor(const size_t* shape, uint32_t ndim,
               Casting::DType dtype = Casting::DType::Float32,
               Device device = Device::CPU,
               bool requires_grad = false);
    };


    struct Node {
        void (*_backward_fn)(Tensor* output_grad, Tensor** inputs, size_t n_inputs);
        Tensor* _inputs[BT_TENSOR_MAX_PARENTS];
        size_t _n_inputs;
        Tensor* _saved[BT_TENSOR_MAX_PARENTS];
        size_t _n_saved;
    };

    inline void _compute_strides(const size_t* shape, size_t* strides, const uint32_t ndim) {
        strides[ndim - 1] = 1;
        for (int i = static_cast<int>(ndim) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }

    /* Returns a byte pointer to the start of the tensor's data, accounting for element offset */
    inline std::byte* data_ptr(const Tensor& t) {
        return static_cast<std::byte*>(t._data) + t._offset * Casting::dtype_size(t._dtype);
    }

    /* Get a value at a tensor based on idx (tensor[1][4] = {1, 4})*/
    template<typename T>
    T* at_location(Tensor* tensor, const std::initializer_list<int> idx) {
        assert(static_cast<int>(idx.size()) == tensor->_ndim);
        auto* base = static_cast<T*>(tensor->_data) + tensor->_offset;
        for (unsigned int i = 0; i < tensor->_ndim; ++i) {
            assert(idx.begin()[i] >= 0 && idx.begin()[i] < tensor->_shape[i]);
            base += idx.begin()[i] * tensor->_strides[i];
        }
        return base;
    }

    /* Make a new tensor */
    inline Tensor make_tensor(const size_t* shape,
        const uint32_t ndim,
        const Casting::DType dtype = Casting::DType::Float32,
        const Device device = Device::CPU,
        const bool requires_grad = false
        ) {
        if (ndim == 0 || ndim > BT_TENSOR_MAX_DIMS) {
            throw std::invalid_argument("Tensor dimension exceeds maximum supported dimensions");
        }
        Tensor new_tensor{};
        new_tensor._ndim = ndim;
        new_tensor._dtype = dtype;
        new_tensor._device = device;
        new_tensor._requires_grad = requires_grad;
        new_tensor._owns_data = true;

        new_tensor._numel = 1;
        for (uint32_t i = 0; i < ndim; ++i) {
            new_tensor._shape[i] = shape[i];
            new_tensor._numel *= shape[i];
        }
        new_tensor._nbytes = new_tensor._numel * (Casting::dtype_size(dtype));

        _compute_strides(new_tensor._shape, new_tensor._strides, new_tensor._ndim);

        if (device == Device::CPU) {
            new_tensor._data = std::malloc(new_tensor._nbytes);
            assert(new_tensor._data != nullptr);
        } else {
#if defined(USE_CUDA)
            int curr;
            cudaGetDevice(&curr);
            new_tensor._gpu_id = curr;
            cudaMalloc(&new_tensor._data, new_tensor._nbytes);
#elif defined(USE_HIP)
            int curr;
            hipGetDevice(&curr);
            new_tensor._gpu_id = curr;
            hipMalloc(&new_tensor._data, new_tensor._nbytes);
#else
            assert(false && "Attempted GPU tensor on CPU build");
#endif
        }
        return new_tensor;
    }

    inline Tensor::Tensor(const size_t *shape, const uint32_t ndim, const Casting::DType dtype, const Device device, const bool requires_grad) {
        *this = make_tensor(shape, ndim, dtype, device, requires_grad);
    }


    /* Free a tensor */
    inline void free_tensor(Tensor* tensor) {
        if (tensor->_owns_data) {
            if (tensor->_device == Device::CPU) {
                std::free(tensor->_data);
            } else {
#if defined(USE_CUDA)
                cudaFree(tensor->_data);
#elif defined(USE_HIP)
                hipFree(tensor->_data);
#else
                assert(false);
#endif
            }
        }
    }

    /* make a shallow "view" copy */
    [[nodiscard]] inline Tensor make_view(const Tensor& parent_tensor) {
        Tensor child_tensor = parent_tensor;
        child_tensor._owns_data = false;
        return child_tensor;
    }

    /* Swap two stride and shape dims, and get a shallow copy */
    [[nodiscard]] inline Tensor transpose(const Tensor& tensor, const uint32_t dim0, const uint32_t dim1) {
        auto transposed = make_view(tensor);
        transposed._owns_data = false;
        std::swap(transposed._shape[dim0], transposed._shape[dim1]);
        std::swap(transposed._strides[dim0], transposed._strides[dim1]);
        transposed._is_contiguous = false;
        return transposed;
    }

    [[nodiscard]] inline Tensor deep_copy(const Tensor& src) {
        Tensor dst = src;
        dst._owns_data = true;
        dst._offset = 0;

        if (src._device == Device::CPU) {
            dst._data = std::malloc(src._nbytes);
            assert(dst._data != nullptr);
            std::memcpy(dst._data, data_ptr(src), src._nbytes);
        } else {
#if defined(USE_CUDA)
            int curr;
            cudaGetDevice(&curr);
            assert(curr == src._gpu_id);
            cudaMalloc(&dst._data, src._nbytes);
            cudaMemcpy(dst._data, data_ptr(src), src._nbytes, cudaMemcpyDeviceToDevice);
#elif defined(USE_HIP)
            int curr;
            hipGetDevice(&curr);
            assert(curr == (int)src._gpu_id);
            hipMalloc(&dst._data, src._nbytes);
            hipMemcpy(dst._data, data_ptr(src), src._nbytes, hipMemcpyDeviceToDevice);
#else
            assert(false);
#endif
        }
        return dst;
    }

    [[nodiscard]] inline bool is_same_device(const Tensor& t1, const Tensor& t2) {
        if (t1._device != t2._device) {
            return false;
        }
        if (t1._device == Device::CPU && t2._device == Device::CPU) {
            return true;
        }
        return (t1._gpu_id == t2._gpu_id);
    }

    [[nodiscard]] inline bool is_same_shape(const Tensor& t1, const Tensor& t2) {
        return (t1._ndim == t2._ndim) && std::equal(t1._shape, t1._shape + t1._ndim, t2._shape);
    }

    [[nodiscard]] inline Tensor make_slice(const Tensor& parent, const size_t dim, const size_t index) {
        assert(parent._ndim > 1); // slicing a 1D tensor would produce a 0-dim scalar;
        assert(dim < parent._ndim);
        assert(index < parent._shape[dim]);

        Tensor child = parent;
        child._owns_data = false;

        child._offset = parent._offset + index * parent._strides[dim];

        for (size_t i = dim; i < parent._ndim - 1; ++i) {
            child._shape[i]   = parent._shape[i + 1];
            child._strides[i] = parent._strides[i + 1];
        }
        child._ndim  -= 1;
        child._numel  = 1;
        for (size_t i = 0; i < child._ndim; ++i)
            child._numel *= child._shape[i];
        child._nbytes = child._numel * Casting::dtype_size(child._dtype);

        return child;
    }

    [[nodiscard]] inline std::string shape_to_string(const Tensor& tensor) {
        std::ostringstream os;
        os << "{";
        for (uint32_t i = 0; i < tensor._ndim; ++i) {
            os << tensor._shape[i];
            if (i < tensor._ndim - 1) os << ", ";
        }
        os <<"}";
        return os.str();
    }

    [[nodiscard]] inline std::string to_string(const Tensor& tensor) {
        std::ostringstream os;
        os << "Tensor(shape=" << shape_to_string(tensor) << "\nstrides=[";
        for (uint32_t i = 0; i < tensor._ndim; ++i) {
            os << tensor._strides[i];
            if (i < tensor._ndim - 1) os << ", ";
        }
        os << "]\nbytes=[" << tensor._nbytes << "]";
        os << "\ndevice[";
        if (tensor._device == Device::CPU) {
            os << "CPU";
        } else {
            os << "GPU:" << tensor._gpu_id;
        }
        os << "]";

        return os.str();
    }

    inline void convert_dtype(Tensor& t, const Casting::DType& new_dtype) {
        if (t._dtype == new_dtype) return;
        const size_t new_nbytes = t._numel * Casting::dtype_size(new_dtype);
        void* dst = nullptr;
        if (t._device == Device::CPU) {
            dst = std::malloc(new_nbytes);
            assert(dst != nullptr);
        }
        // TODO: GPU allocation
        detail::launch_convert(t._dtype, new_dtype, t._data, dst, t._numel);
        free_tensor(&t);
        t._data = dst;
        t._dtype = new_dtype;
        t._nbytes = new_nbytes;
        t._owns_data = true;
    }

    inline void set_zero(const Tensor& t) {
        if (t._device == Device::CPU) {
            std::memset(t._data, 0, t._nbytes);
        } else {
            //TODO: Enable GPU Zero
            assert(false);
        }
    }
    inline void set_one(const Tensor& t) {
        if (t._device == Device::CPU) {
            std::memset(t._data, 1, t._nbytes);
        } else {
            // TODO: Enable GPU One
            assert(false);
        }
    }
    inline void set_random(const Tensor& t, const size_t seed = std::random_device()(), const double& min = 0.0, const double& max = 1.0) {
        using FillFn = void(*)(void*, size_t, size_t, double, double);
        using DType = Casting::DType;

        static const std::unordered_map<DType, FillFn> fill_fns = {
            {DType::Int8,    detail::fill_random_impl<int8_t>},
            {DType::Int16,   detail::fill_random_impl<int16_t>},
            {DType::Int32,   detail::fill_random_impl<int32_t>},
            {DType::Int64,   detail::fill_random_impl<int64_t>},
            {DType::UInt8,   detail::fill_random_impl<uint8_t>},
            {DType::UInt16,  detail::fill_random_impl<uint16_t>},
            {DType::UInt32,  detail::fill_random_impl<uint32_t>},
            {DType::UInt64,  detail::fill_random_impl<uint64_t>},
            {DType::Float16, detail::fill_random_impl<Casting::float16_t>},
            {DType::BFloat16, detail::fill_random_impl<Casting::bfloat16_t>},
            {DType::Float32, detail::fill_random_impl<float>},
            {DType::Float64, detail::fill_random_impl<double>}
        };
        if (t._device != Device::CPU) {
            assert(false); //TODO: GPU Support
        }
        const auto it = fill_fns.find(t._dtype);
        assert(it != fill_fns.end());
        it->second(t._data, t._numel, seed, min, max);
    }


    [[nodiscard]] inline std::string print_tensor(Tensor &t) {
        std::ostringstream os;
        size_t idx = 0; //Required isolation for recursive
        if (t._dtype == Casting::DType::Float32) {
            detail::print_recursive(os, static_cast<float*>(t._data), t._shape, t._ndim, idx, 0);
        } else if (t._dtype == Casting::DType::BFloat16) {
            convert_dtype(t, Casting::DType::Float32);
            detail::print_recursive(os, static_cast<float*>(t._data), t._shape, t._ndim, idx, 0);
        } else {
            //TODO: Fix
            assert(false);
        }
        return os.str();
    }
}

