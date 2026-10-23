cmake_minimum_required(VERSION 4.2)

option(USE_HIP "Build with AMD ROCM" ON)
option(USE_CUDA "Build with NVIDIA CUDA" OFF)
option(USE_STRIX_HALO "Build with Xilinx StrixHalo NPU backend" OFF)

if (USE_HIP AND USE_CUDA)
    message(FATAL_ERROR "USE_HIP and USE_CUDA are mutually exclusive — enable only one.")
endif()
if (NOT USE_HIP AND NOT USE_CUDA AND NOT USE_STRIX_HALO)
    message(WARNING "No backend selected — assuming CPU Compilation")
endif()

if ((USE_HIP OR USE_CUDA) AND USE_STRIX_HALO)
    message(WARNING "HIP/CUDA and StrixHalo are mutually enabled. This isn't implicitly a problem, but some CPU pathways may rely on NPU; CPU > GPU calls could be delayed from NPU memory allocation transfers.")
endif()

if (USE_STRIX_HALO)
    message(FATAL_ERROR "StrixHalo backend requested, but not implemented yet. Please disable USE_STRIX_HALO and try again.")
endif()


if (USE_CUDA)
    # CUDA device code is pinned to nvcc 12.9: CUDA 13 dropped sm_70 (Volta),
    # which we still target, and 12.x is the last nvcc line that supports it.
    # The tradeoff: nvcc 12.x maxes out at -std=c++20 (C++23 arrived in 13.0),
    # so CUDA TUs compile as C++20 while host CXX stays C++23. Prefer the 12.9
    # toolkit explicitly when it is installed alongside newer ones.
    if (EXISTS "/usr/local/cuda-12.9" AND NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "/usr/local/cuda-12.9")
    endif()
    IF (NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "NOTFOUND")
        if (EXISTS "/usr/local/cuda-12.9/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
        else()
            set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    ENDIF()
    # nvcc 12.9 only supports host GCC <= 14, but this system defaults to GCC
    # 15, whose libstdc++ headers nvcc's frontend cannot parse. Pin g++-13 as
    # the CUDA host compiler when it is installed.
    if (EXISTS "/usr/bin/g++-13" AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
    endif()
    # glibc >= 2.41 declares C23 math functions (cospi/sinpi/rsqrt + f variants)
    # with noexcept, colliding with CUDA 12.9's crt/math_functions.h. The fix is
    # a one-line-per-function patch adding noexcept(true) to those 6 declarations
    # in the toolkit header (documented in AGENTS.md); -U_GNU_SOURCE is NOT a
    # viable workaround — it breaks libstdc++'s own headers.
    project(BeanTensorV5 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 70 86)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9)
        message(FATAL_ERROR "CUDA backend requires nvcc >= 12.9 (found ${CMAKE_CUDA_COMPILER_VERSION}). Install the CUDA 12.9 toolkit or build with USE_CUDA=OFF.")
    endif()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        message(FATAL_ERROR "CUDA 13 dropped sm_70 support; pin the CUDA 12.9 toolkit (found ${CMAKE_CUDA_COMPILER_VERSION}).")
    endif()
elseif (USE_HIP)
    project(BeanTensorV5 LANGUAGES CXX HIP)
else()
    project(BeanTensorV5 LANGUAGES CXX)
endif()
include(CTest)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# [[nodiscard]] cannot be violated.
add_compile_options(
        $<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:HIP>>:-Werror=unused-result>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror=unused-result>
)

# Errors on all basic "warnings" - my code standard is C++ code standard, not whatever I type and then hoping it works.
set(BEANTENSOR_WERROR_FLAGS
        -Werror=uninitialized
        -Werror=return-type
        -Werror=strict-aliasing
        -Werror=delete-non-virtual-dtor
        -Werror=non-virtual-dtor
        -Werror=implicit-fallthrough
        -Werror=sign-conversion
        -Werror=float-equal
        -Werror=double-promotion
        -Werror=old-style-cast
        -Werror=unused-variable
        -Werror=unused-parameter
        -Werror=unused-but-set-variable
        -Werror=format=2
        -Werror=reorder
)
# -Werror=maybe-uninitialized is GNU-only so let's make it be that.
set(BEANTENSOR_WERROR_FLAGS_GNU ${BEANTENSOR_WERROR_FLAGS} -Werror=maybe-uninitialized -Werror=sequence-point)
set(BEANTENSOR_WERROR_FLAGS_CLANG ${BEANTENSOR_WERROR_FLAGS} -Werror=unsequenced)
if (USE_CUDA)
    set(BEANTENSOR_WERROR_FLAGS_CUDA "")
    foreach (flag IN LISTS BEANTENSOR_WERROR_FLAGS_GNU)
        # nvcc my beloved! - it is rewriting brace-inits into C-style casts during compilation.
        # This triggers the old-style-cast error. Must disable when CUDA touches a file.
        if (NOT flag STREQUAL "-Werror=old-style-cast")
            list(APPEND BEANTENSOR_WERROR_FLAGS_CUDA "-Xcompiler=${flag}")
        endif ()
    endforeach ()
endif ()
add_compile_options(
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:${BEANTENSOR_WERROR_FLAGS_GNU}>"
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:${BEANTENSOR_WERROR_FLAGS_CLANG}>"
        "$<$<COMPILE_LANG_AND_ID:HIP,Clang>:${BEANTENSOR_WERROR_FLAGS_CLANG}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${BEANTENSOR_WERROR_FLAGS_CUDA}>"
)

if (USE_HIP)
    message(STATUS "Backend: HIP/ROCm")
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
elseif(USE_CUDA)
    message(STATUS "Backend: CUDA")
    find_package(CUDAToolkit REQUIRED)
    # nvcc 12.9 is pinned to g++-13 as host compiler, so CMake records gcc-13's
    # libstdc++ dir in the CUDA implicit link directories. That libstdc++
    # predates __cxa_call_terminate (emitted by the system g++ 15 compiling our
    # C++23 TUs), and the implicit -L makes the linker pick it over the system
    # one. Search the CXX compiler's own libstdc++ dir first.
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
            OUTPUT_VARIABLE BEANTENSOR_CXX_LIBSTDCXX_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
    )
    get_filename_component(BEANTENSOR_CXX_LIBSTDCXX_DIR "${BEANTENSOR_CXX_LIBSTDCXX_PATH}" DIRECTORY)
    add_link_options("-L${BEANTENSOR_CXX_LIBSTDCXX_DIR}")
endif()

if (USE_STRIX_HALO)
    message(STATUS "Backend: StrixHaloNPU")
    message(FATAL_ERROR "NotImplementedError")
endif()

set(CORE_SOURCES
        Helpers/CPUThreading.h
        Helpers/Hardware.h
        Intrinsics/ConversionMethods.h
        Intrinsics/Native/DTypeConversions.h
        Intrinsics/Shared/SharedIntrinsics.h
        Intrinsics/AVX2/DTypeConversions.h
        Intrinsics/AVX2/DTypeConversions.cpp
        Intrinsics/AVX512/DTypeConversions.h
        Intrinsics/AVX512/DTypeConversions.cpp
        Intrinsics/Dispatcher.h
        Intrinsics/Dispatcher.cpp
        Tensor/DTypes.h
        Tensor/Tensor.h
        Tensor/Tensor.cpp
        Math/Elementwise/Elementwise.h
        Math/Elementwise/Elementwise.hpp
        Math/Elementwise/ElementwiseShared.h
        Math/Elementwise/ElementwiseCudaShared.h
        Math/Elementwise/ElementwiseHipShared.h
        Math/Elementwise/ElementwiseCPU.cpp
        Math/Elementwise/ElementwiseKernels.h
        Math/Elementwise/ElementwiseAVX2.cpp
        Math/Elementwise/ElementwiseAVX512.cpp
        Math/LinearAlgebra/MatMul/Matmul.h
        Math/LinearAlgebra/MatMul/Matmul.hpp
        Math/LinearAlgebra/MatMul/MatmulCPU.cpp
        Math/LinearAlgebra/MatMul/GemmKernels.h
        Math/LinearAlgebra/MatMul/GemmAVX2.cpp
        Math/LinearAlgebra/MatMul/GemmAVX512.cpp
        Math/Reductions/Reductions.h
        Math/Optimization/Optimization.h
        Math/Optimization/SgdUpdate/SgdUpdate.h
        Math/Optimization/SgdUpdate/SgdUpdate.hpp
        Math/Optimization/SgdUpdate/SgdUpdateKernels.h
        Math/Optimization/SgdUpdate/SgdUpdateCPU.cpp
        Math/Optimization/SgdUpdate/SgdUpdateAVX2.cpp
        Math/Optimization/SgdUpdate/SgdUpdateAVX512.cpp
        Math/Reductions/Sum/Sum.h
        Math/Reductions/Sum/Sum.hpp
        Math/Reductions/Sum/SumKernels.h
        Math/Reductions/Sum/SumCPU.cpp
        Math/Reductions/Sum/SumAVX2.cpp
        Math/Reductions/Sum/SumAVX512.cpp
        Math/Reductions/Argmax/Argmax.h
        Math/Reductions/Argmax/Argmax.hpp
        Math/Reductions/Argmax/ArgmaxCPU.cpp
        Math/Reductions/Mean/Mean.h
        Math/Reductions/Mean/Mean.hpp
        Math/Reductions/Mean/MeanCPU.cpp
        Math/Compare/Compare.h
        Math/Compare/Equal/Equal.h
        Math/Compare/Equal/Equal.hpp
        Math/Compare/Equal/EqualCPU.cpp
        Math/Shape/Reshape.cpp
        Math/Shape/Indexing/Indexing.h
        Math/Shape/Indexing/Indexing.hpp
        Math/Shape/Indexing/IndexingShared.h
        Math/Shape/Indexing/IndexingCPU.cpp
        Math/Shape/Slice.cpp
        Math/Reductions/ReductionsShared.h
        Math/Reductions/ReductionsCudaShared.h
        Math/Reductions/ReductionsHipShared.h
        Math/Optimization/OptimizationShared.h
        Math/Optimization/OptimizationCudaShared.h
        Math/Optimization/OptimizationHipShared.h
        Math/LinearAlgebra/LinearAlgebra.h
        Math/Activations/Activations.h
        Math/LinearAlgebra/BiasAdd/BiasAdd.h
        Math/LinearAlgebra/BiasAdd/BiasAdd.hpp
        Math/LinearAlgebra/BiasAdd/BiasAddKernels.h
        Math/LinearAlgebra/BiasAdd/BiasAddCPU.cpp
        Math/LinearAlgebra/BiasAdd/BiasAddAVX2.cpp
        Math/LinearAlgebra/BiasAdd/BiasAddAVX512.cpp
        Math/Activations/Relu/Relu.h
        Math/Activations/Relu/Relu.hpp
        Math/Activations/Relu/ReluKernels.h
        Math/Activations/Relu/ReluCPU.cpp
        Math/Activations/Relu/ReluAVX2.cpp
        Math/Activations/Relu/ReluAVX512.cpp
        Math/Activations/SoftmaxCE/SoftmaxCE.h
        Math/Activations/SoftmaxCE/SoftmaxCE.hpp
        Math/Activations/SoftmaxCE/SoftmaxCECPU.cpp
        Math/LinearAlgebra/LinearAlgebraShared.h
        Math/LinearAlgebra/LinearAlgebraCudaShared.h
        Math/LinearAlgebra/LinearAlgebraHipShared.h
        Math/Activations/ActivationsShared.h
        Math/Activations/ActivationsCudaShared.h
        Math/Activations/ActivationsHipShared.h
        Math/Autograd/Autograd.h
        Math/Autograd/GradAcc/GradAcc.h
        Math/Autograd/GradAcc/GradAcc.hpp
        Math/Autograd/GradAcc/GradAccKernels.h
        Math/Autograd/GradAcc/GradAccCPU.cpp
        Math/Autograd/GradAcc/GradAccAVX2.cpp
        Math/Autograd/GradAcc/GradAccAVX512.cpp
        Math/Autograd/ReluG/ReluG.h
        Math/Autograd/ReluG/ReluG.hpp
        Math/Autograd/ReluG/ReluGKernels.h
        Math/Autograd/ReluG/ReluGCPU.cpp
        Math/Autograd/ReluG/ReluGAVX2.cpp
        Math/Autograd/ReluG/ReluGAVX512.cpp
        Math/Autograd/BiasAddG/BiasAddG.h
        Math/Autograd/BiasAddG/BiasAddG.hpp
        Math/Autograd/BiasAddG/BiasAddGKernels.h
        Math/Autograd/BiasAddG/BiasAddGCPU.cpp
        Math/Autograd/BiasAddG/BiasAddGAVX2.cpp
        Math/Autograd/BiasAddG/BiasAddGAVX512.cpp
        Math/Autograd/SoftmaxCEG/SoftmaxCEG.h
        Math/Autograd/SoftmaxCEG/SoftmaxCEG.hpp
        Math/Autograd/SoftmaxCEG/SoftmaxCEGCPU.cpp
        Math/Autograd/TransposeLast2/TransposeLast2.h
        Math/Autograd/TransposeLast2/TransposeLast2.hpp
        Math/Autograd/TransposeLast2/TransposeLast2CPU.cpp
        Math/Autograd/GemmG/GemmG.h
        Math/Autograd/GemmG/GemmGCPU.cpp
        Math/Autograd/Backward.cpp
        Math/Autograd/GradDType.h
        Math/Autograd/AutogradShared.h
        Math/Autograd/AutogradCudaShared.h
        Math/Autograd/AutogradHipShared.h
        Math/Autograd/ElementwiseG/ElementwiseG.h
        Math/Autograd/ElementwiseG/ElementwiseG.hpp
        Math/Autograd/ElementwiseG/ElementwiseGShared.h
        Math/Autograd/ElementwiseG/ElementwiseGCPU.cpp
        Math/Autograd/ElementwiseG/ElementwiseGHIP.cpp
        Math/Autograd/ElementwiseG/ElementwiseGCUDA.cu
        Helpers/Bits.h
)

# AVX512/AVX2 conversion kernels are always compiled with AVX512/AVX2 codegen
# (per-file, so other TUs are unaffected); execution is gated at runtime via
# IncludedCPUFeatures() so CPUs lacking AVX512BF16/AVX2 take the scalar fallback.
# The HIP variant of these flags is applied in the USE_HIP block below.

if (USE_CUDA)
    list(APPEND CORE_SOURCES
            Tensor/Ops/TensorOps.cu
            Math/Elementwise/ElementwiseCUDA.cu
            Math/LinearAlgebra/MatMul/MatmulCUDA.cu
            Math/Optimization/SgdUpdate/SgdUpdateCUDA.cu
            Math/Reductions/Sum/SumCUDA.cu
            Math/Reductions/Argmax/ArgmaxCUDA.cu
            Math/Compare/Equal/EqualCUDA.cu
            Math/LinearAlgebra/BiasAdd/BiasAddCUDA.cu
            Math/Activations/Relu/ReluCUDA.cu
            Math/Activations/SoftmaxCE/SoftmaxCECUDA.cu
            Math/Autograd/GradAcc/GradAccCUDA.cu
            Math/Autograd/ReluG/ReluGCUDA.cu
            Math/Autograd/BiasAddG/BiasAddGCUDA.cu
            Math/Autograd/SoftmaxCEG/SoftmaxCEGCUDA.cu
            Math/Autograd/TransposeLast2/TransposeLast2CUDA.cu
            Math/Autograd/ElementwiseG/ElementwiseGCUDA.cu
            Math/Shape/Indexing/IndexingCUDA.cu
    )
elseif (USE_HIP)
    list(APPEND CORE_SOURCES
            Tensor/Ops/TensorOps.cpp
            Math/Elementwise/ElementwiseHIP.cpp
            Math/LinearAlgebra/MatMul/MatmulHIP.cpp
            Math/Optimization/SgdUpdate/SgdUpdateHIP.cpp
            Math/Reductions/Sum/SumHIP.cpp
            Math/Reductions/Argmax/ArgmaxHIP.cpp
            Math/Reductions/Mean/MeanHIP.cpp
            Math/Compare/Equal/EqualHIP.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddHIP.cpp
            Math/Activations/Relu/ReluHIP.cpp
            Math/Activations/SoftmaxCE/SoftmaxCEHIP.cpp
            Math/Autograd/GradAcc/GradAccHIP.cpp
            Math/Autograd/ReluG/ReluGHIP.cpp
            Math/Autograd/BiasAddG/BiasAddGHIP.cpp
            Math/Autograd/SoftmaxCEG/SoftmaxCEGHIP.cpp
            Math/Autograd/TransposeLast2/TransposeLast2HIP.cpp
            Math/Autograd/ElementwiseG/ElementwiseGHIP.cpp
            Math/Shape/Indexing/IndexingHIP.cpp
    )
elseif(USE_STRIX_HALO)
    message(FATAL_ERROR "NotImplementedError - Core Sources")
endif()

add_library(BeanTensorV5Core STATIC ${CORE_SOURCES})
if(USE_CUDA)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_CUDA)
    # nvcc 12.x supports at most -std=c++20; pin it explicitly.
    # -Wno-error=float-equal must come after the -Werror flags: nvcc passes
    # toolkit headers as plain -I (not -isystem), so cuda_bf16.hpp's float
    # comparisons would otherwise trip the project's -Werror=float-equal.
    # float-equal stays an error everywhere except CUDA host passes.
    # --extended-lambda permits explicit __device__ annotations on lambdas
    # (used in SgdUpdateCUDA's int4 lane step); harmless elsewhere.
    target_compile_options(BeanTensorV5Core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20;--extended-lambda;-Xcompiler=-Wno-error=float-equal>
    )
endif()
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_STRIX_HALO)
endif()

if(USE_HIP)
    set_source_files_properties(
            Tensor/Tensor.cpp
            Tensor/Ops/TensorOps.cpp
            Math/LinearAlgebra/MatMul/MatmulCPU.cpp
            Math/Elementwise/ElementwiseHIP.cpp
            Math/LinearAlgebra/MatMul/MatmulHIP.cpp
            Math/LinearAlgebra/MatMul/GemmAVX2.cpp
            Math/LinearAlgebra/MatMul/GemmAVX512.cpp
            Math/Optimization/SgdUpdate/SgdUpdateCPU.cpp
            Math/Optimization/SgdUpdate/SgdUpdateHIP.cpp
            Math/Reductions/Sum/SumCPU.cpp
            Math/Reductions/Sum/SumHIP.cpp
            Math/Reductions/Argmax/ArgmaxCPU.cpp
            Math/Reductions/Argmax/ArgmaxHIP.cpp
            Math/Reductions/Mean/MeanCPU.cpp
            Math/Reductions/Mean/MeanHIP.cpp
            Math/Compare/Equal/EqualCPU.cpp
            Math/Compare/Equal/EqualHIP.cpp
            Math/Shape/Reshape.cpp
            Math/Optimization/SgdUpdate/SgdUpdateAVX2.cpp
            Math/Optimization/SgdUpdate/SgdUpdateAVX512.cpp
            Math/Reductions/Sum/SumAVX2.cpp
            Math/Reductions/Sum/SumAVX512.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddCPU.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddHIP.cpp
            Math/Activations/Relu/ReluCPU.cpp
            Math/Activations/Relu/ReluHIP.cpp
            Math/Activations/SoftmaxCE/SoftmaxCECPU.cpp
            Math/Activations/SoftmaxCE/SoftmaxCEHIP.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddAVX2.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddAVX512.cpp
            Math/Activations/Relu/ReluAVX2.cpp
            Math/Activations/Relu/ReluAVX512.cpp
            Math/Autograd/GradAcc/GradAccCPU.cpp
            Math/Autograd/GradAcc/GradAccHIP.cpp
            Math/Autograd/ReluG/ReluGCPU.cpp
            Math/Autograd/ReluG/ReluGHIP.cpp
            Math/Autograd/BiasAddG/BiasAddGCPU.cpp
            Math/Autograd/BiasAddG/BiasAddGHIP.cpp
            Math/Autograd/SoftmaxCEG/SoftmaxCEGCPU.cpp
            Math/Autograd/SoftmaxCEG/SoftmaxCEGHIP.cpp
            Math/Autograd/TransposeLast2/TransposeLast2CPU.cpp
            Math/Autograd/TransposeLast2/TransposeLast2HIP.cpp
            Math/Autograd/GemmG/GemmGCPU.cpp
            Math/Autograd/Backward.cpp
            Math/Autograd/GradAcc/GradAccAVX2.cpp
            Math/Autograd/GradAcc/GradAccAVX512.cpp
            Math/Autograd/ReluG/ReluGAVX2.cpp
            Math/Autograd/ReluG/ReluGAVX512.cpp
            Math/Autograd/BiasAddG/BiasAddGAVX2.cpp
            Math/Autograd/BiasAddG/BiasAddGAVX512.cpp
            Math/Elementwise/ElementwiseAVX2.cpp
            Math/Elementwise/ElementwiseAVX512.cpp
            Intrinsics/AVX2/DTypeConversions.cpp
            Intrinsics/AVX512/DTypeConversions.cpp
            Intrinsics/Dispatcher.cpp
            Math/Shape/Indexing/IndexingHIP.cpp
            Math/Shape/Indexing/IndexingCPU.cpp
            Math/Shape/Slice.cpp
            Math/Elementwise/ElementwiseCPU.cpp
            Math/Autograd/ElementwiseG/ElementwiseGCPU.cpp
            Math/Autograd/ElementwiseG/ElementwiseGHIP.cpp
            PROPERTIES LANGUAGE HIP
    )
    set_source_files_properties(
            Tensor/Tensor.cpp
            Tensor/Ops/TensorOps.cpp
            Math/LinearAlgebra/MatMul/MatmulCPU.cpp
            Math/Elementwise/ElementwiseHIP.cpp
            Math/LinearAlgebra/MatMul/MatmulHIP.cpp
            Math/Optimization/SgdUpdate/SgdUpdateCPU.cpp
            Math/Optimization/SgdUpdate/SgdUpdateHIP.cpp
            Math/Reductions/Sum/SumCPU.cpp
            Math/Reductions/Sum/SumHIP.cpp
            Math/Reductions/Argmax/ArgmaxCPU.cpp
            Math/Reductions/Argmax/ArgmaxHIP.cpp
            Math/Reductions/Mean/MeanCPU.cpp
            Math/Reductions/Mean/MeanHIP.cpp
            Math/Compare/Equal/EqualCPU.cpp
            Math/Compare/Equal/EqualHIP.cpp
            Math/Shape/Reshape.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddCPU.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddHIP.cpp
            Math/Activations/Relu/ReluCPU.cpp
            Math/Activations/Relu/ReluHIP.cpp
            Math/Activations/SoftmaxCE/SoftmaxCECPU.cpp
            Math/Activations/SoftmaxCE/SoftmaxCEHIP.cpp
            Math/Autograd/GradAcc/GradAccCPU.cpp
            Math/Autograd/GradAcc/GradAccHIP.cpp
            Math/Autograd/ReluG/ReluGCPU.cpp
            Math/Autograd/ReluG/ReluGHIP.cpp
            Math/Autograd/BiasAddG/BiasAddGCPU.cpp
            Math/Autograd/BiasAddG/BiasAddGHIP.cpp
            Math/Autograd/SoftmaxCEG/SoftmaxCEGCPU.cpp
            Math/Autograd/SoftmaxCEG/SoftmaxCEGHIP.cpp
            Math/Autograd/TransposeLast2/TransposeLast2CPU.cpp
            Math/Autograd/TransposeLast2/TransposeLast2HIP.cpp
            Math/Autograd/GemmG/GemmGCPU.cpp
            Math/Autograd/Backward.cpp
            Math/Shape/Indexing/IndexingCPU.cpp
            Math/Shape/Slice.cpp
            Math/Elementwise/ElementwiseAVX2.cpp
            Math/Elementwise/ElementwiseAVX512.cpp
            Math/Elementwise/ElementwiseCPU.cpp
            Math/Autograd/ElementwiseG/ElementwiseGCPU.cpp
            Math/Autograd/ElementwiseG/ElementwiseGHIP.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native"
    )
    # The conversion dispatcher's IncludedCPUFeatures() copy needs every ISA flag
    # it gates on (AVX512F/BF16/DQ, AVX2, F16C); without them its #if defined()
    # guards in Helpers/Hardware.h are all false and the AVX512/AVX2 conversion
    # fast paths never dispatch. Same clobber rule as the AVX512 kernels above:
    # offload arch + host flags in ONE COMPILE_OPTIONS property.
    set_source_files_properties(
            Intrinsics/Dispatcher.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512bw;-mavx512bf16;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c"
    )
    # AVX512 kernels: HIP offload arch + host AVX512 codegen combined (the
    # single COMPILE_OPTIONS property would otherwise clobber one or the other).
    set_source_files_properties(
            Intrinsics/AVX512/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512bf16;-mavx512dq;-mf16c"
    )
    # GEMM's AVX512 kernels additionally need -mavx512bw (16-bit permutes for
    # the dpbf16 pair layouts and packed-lane extract/zip) and -mavx512vnni
    # (vpdpbusd/vpdpwssd integer dot products); the Intrinsics AVX512
    # conversion file needs neither.
    set_source_files_properties(
            Math/LinearAlgebra/MatMul/GemmAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512bw;-mavx512bf16;-mavx512dq;-mavx512vnni;-mf16c"
    )
    # GEMM's AVX2 kernels also need -mfma (register-blocked fmadd path); the
    # Intrinsics AVX2 conversion file does not.
    set_source_files_properties(
            Intrinsics/AVX2/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx2;-mf16c"
    )
    # Elementwise AVX512 kernels need -mavx512bf16 (cvtneps2bf16 bf16 narrowing)
    # and -mavx512bw (epi8/epi16 lane ops + 16-bit masked tails); -mavx512dq
    # covers mullo_epi64. Same clobber rule: offload arch + host flags in ONE
    # COMPILE_OPTIONS property.
    set_source_files_properties(
            Math/Elementwise/ElementwiseAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512bw;-mavx512bf16;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Math/Elementwise/ElementwiseAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx2;-mf16c"
    )
    set_source_files_properties(
            Math/LinearAlgebra/MatMul/GemmAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx2;-mfma;-mf16c"
    )
    # Per-op AVX512 kernels need only AVX512F codegen (fp32/fp64 fma/add);
    # -mavx512dq/-mf16c ride along to match the Intrinsics AVX512 file.
    set_source_files_properties(
            Math/Optimization/SgdUpdate/SgdUpdateAVX512.cpp
            Math/Reductions/Sum/SumAVX512.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddAVX512.cpp
            Math/Activations/Relu/ReluAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512dq;-mf16c"
    )
    # Per-op AVX2 kernels get the same -mfma flag set as GEMM's AVX2 file.
    set_source_files_properties(
            Math/Optimization/SgdUpdate/SgdUpdateAVX2.cpp
            Math/Reductions/Sum/SumAVX2.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddAVX2.cpp
            Math/Activations/Relu/ReluAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx2;-mfma;-mf16c"
    )
    # Autograd kernels share the per-op flag split (fp32/fp64 add/cmp only).
    set_source_files_properties(
            Math/Autograd/GradAcc/GradAccAVX512.cpp
            Math/Autograd/ReluG/ReluGAVX512.cpp
            Math/Autograd/BiasAddG/BiasAddGAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Math/Autograd/GradAcc/GradAccAVX2.cpp
            Math/Autograd/ReluG/ReluGAVX2.cpp
            Math/Autograd/BiasAddG/BiasAddGAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx2;-mfma;-mf16c"
    )
else()
    set_source_files_properties(
            Intrinsics/AVX512/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bf16;-mavx512dq;-mf16c"
    )
    # The conversion dispatcher gates its AVX512/AVX2 fast paths on
    # IncludedCPUFeatures(), whose #if defined() guards in Helpers/Hardware.h
    # only see flags the dispatcher's own TU was compiled with — it needs the
    # full union or the fast paths silently never dispatch.
    set_source_files_properties(
            Intrinsics/Dispatcher.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512bf16;-mavx512dq;-mavx512vnni;-mavx2;-mfma;-mf16c"
    )
    # Same split as the HIP branch: GEMM AVX512 kernels get -mavx512bw and
    # -mavx512vnni.
    set_source_files_properties(
            Math/LinearAlgebra/MatMul/GemmAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512bf16;-mavx512dq;-mavx512vnni;-mf16c"
    )
    set_source_files_properties(
            Intrinsics/AVX2/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c"
    )
    # Same split as the HIP branch: elementwise AVX512 kernels get -mavx512bw
    # (epi8/epi16 lanes + 16-bit masked tails) and -mavx512bf16 (cvtneps2bf16);
    # -mavx512dq covers mullo_epi64.
    set_source_files_properties(
            Math/Elementwise/ElementwiseAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512bf16;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Math/Elementwise/ElementwiseAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c"
    )
    set_source_files_properties(
            Math/LinearAlgebra/MatMul/GemmAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
    )
    # Same split as the HIP branch: per-op AVX512 gets AVX512F codegen only,
    # AVX2 gets -mfma for the sgd fmadd path.
    set_source_files_properties(
            Math/Optimization/SgdUpdate/SgdUpdateAVX512.cpp
            Math/Reductions/Sum/SumAVX512.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddAVX512.cpp
            Math/Activations/Relu/ReluAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Math/Optimization/SgdUpdate/SgdUpdateAVX2.cpp
            Math/Reductions/Sum/SumAVX2.cpp
            Math/LinearAlgebra/BiasAdd/BiasAddAVX2.cpp
            Math/Activations/Relu/ReluAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
    )
    # Same split as the HIP branch for the Autograd kernels.
    set_source_files_properties(
            Math/Autograd/GradAcc/GradAccAVX512.cpp
            Math/Autograd/ReluG/ReluGAVX512.cpp
            Math/Autograd/BiasAddG/BiasAddGAVX512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Math/Autograd/GradAcc/GradAccAVX2.cpp
            Math/Autograd/ReluG/ReluGAVX2.cpp
            Math/Autograd/BiasAddG/BiasAddGAVX2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
    )
endif()
if (USE_CUDA)
    set_source_files_properties(
            Tensor/Ops/TensorOps.cu
            Math/Elementwise/ElementwiseCUDA.cu
            Math/LinearAlgebra/MatMul/MatmulCUDA.cu
            Math/Optimization/SgdUpdate/SgdUpdateCUDA.cu
            Math/Reductions/Sum/SumCUDA.cu
            Math/Reductions/Argmax/ArgmaxCUDA.cu
            Math/Compare/Equal/EqualCUDA.cu
            Math/LinearAlgebra/BiasAdd/BiasAddCUDA.cu
            Math/Activations/Relu/ReluCUDA.cu
            Math/Activations/SoftmaxCE/SoftmaxCECUDA.cu
            Math/Autograd/GradAcc/GradAccCUDA.cu
            Math/Autograd/ReluG/ReluGCUDA.cu
            Math/Autograd/BiasAddG/BiasAddGCUDA.cu
            Math/Autograd/SoftmaxCEG/SoftmaxCEGCUDA.cu
            Math/Autograd/TransposeLast2/TransposeLast2CUDA.cu
            Math/Shape/Indexing/IndexingCUDA.cu
        PROPERTIES LANGUAGE CUDA
    )
endif()
target_include_directories(BeanTensorV5Core PUBLIC ${CMAKE_SOURCE_DIR})

#if(USE_HIP)
#    target_include_directories(BeanTensorV5Core PRIVATE ${HIP_INCLUDE_DIRS})
#endif()

if(USE_HIP)
    target_link_libraries(BeanTensorV5Core PRIVATE hip::device)
    target_link_libraries(BeanTensorV5Core PUBLIC hip::host)
    target_link_libraries(BeanTensorV5Core PUBLIC roc::rocblas)
elseif(USE_CUDA)
    target_link_libraries(BeanTensorV5Core PUBLIC CUDA::cudart)
    target_link_libraries(BeanTensorV5Core PUBLIC CUDA::cublas)
endif()


add_executable(BeanTensorV5 main.cpp)
target_link_libraries(BeanTensorV5 PRIVATE BeanTensorV5Core)
target_compile_options(BeanTensorV5 PRIVATE -O2)

add_executable(BeanTensorV5Accelerated main.cpp)
target_link_libraries(BeanTensorV5Accelerated PRIVATE BeanTensorV5Core)
target_compile_options(BeanTensorV5Accelerated PRIVATE
        -O3
        -mavx512f
        -mavx512bf16
        -mavx512dq
        -mf16c
)

if(BUILD_TESTING)
    enable_testing()
    add_executable(BeanTensorV5Tests tests/tensor_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5Tests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5Tests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5Tests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5Tests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5Tests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5Tests COMMAND BeanTensorV5Tests)

    add_executable(BeanTensorV5MathTests tests/matmul_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5MathTests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5MathTests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5MathTests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5MathTests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5MathTests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5MathTests COMMAND BeanTensorV5MathTests)

    add_executable(BeanTensorV5BasicMathTests tests/basic_ops_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5BasicMathTests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5BasicMathTests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5BasicMathTests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5BasicMathTests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5BasicMathTests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5BasicMathTests COMMAND BeanTensorV5BasicMathTests)

    add_executable(BeanTensorV5ComplexMathTests tests/complex_ops_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5ComplexMathTests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5ComplexMathTests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5ComplexMathTests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5ComplexMathTests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5ComplexMathTests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5ComplexMathTests COMMAND BeanTensorV5ComplexMathTests)

    add_executable(BeanTensorV5AutogradTests tests/autograd_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5AutogradTests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5AutogradTests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5AutogradTests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5AutogradTests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5AutogradTests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5AutogradTests COMMAND BeanTensorV5AutogradTests)

    add_executable(BeanTensorV5MetricsTests tests/metrics_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5MetricsTests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5MetricsTests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5MetricsTests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5MetricsTests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5MetricsTests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5MetricsTests COMMAND BeanTensorV5MetricsTests)

    add_executable(BeanTensorV5ElementwiseTests tests/elementwise_tests.cpp)
    if(USE_HIP)
        target_compile_definitions(BeanTensorV5ElementwiseTests PRIVATE USE_HIP)
    elseif(USE_CUDA)
        target_compile_definitions(BeanTensorV5ElementwiseTests PRIVATE USE_CUDA)
    endif()
    if(USE_STRIX_HALO)
        target_compile_definitions(BeanTensorV5ElementwiseTests PRIVATE USE_STRIX_HALO)
    endif()

    target_link_libraries(BeanTensorV5ElementwiseTests PRIVATE BeanTensorV5Core)
    target_compile_options(BeanTensorV5ElementwiseTests PRIVATE
            -O2
            -mavx2
            -mf16c
    )
    add_test(NAME BeanTensorV5ElementwiseTests COMMAND BeanTensorV5ElementwiseTests)
endif()

if(USE_HIP)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_HIP)
    target_compile_definitions(BeanTensorV5 PRIVATE USE_HIP)
    target_compile_definitions(BeanTensorV5Accelerated PRIVATE USE_HIP)
elseif(USE_CUDA)
    # target_compile_definitions(BeanTensorV5Core PRIVATE USE_CUDA)
    target_compile_definitions(BeanTensorV5 PRIVATE USE_CUDA)
    target_compile_definitions(BeanTensorV5Accelerated PRIVATE USE_CUDA)
endif()
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5 PRIVATE USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Accelerated PRIVATE USE_STRIX_HALO)
endif()

add_custom_target(verify ALL
        COMMAND bash -c "shopt -s nullglob; for f in ${CMAKE_SOURCE_DIR}/Verifications/*.sh; do \
    echo \"[verify] \$(basename \$f)\"; \
    bash \$f ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} || exit 1; \
done"
        COMMENT "running verifications"
        VERBATIM
)
add_dependencies(BeanTensorV5 verify)
add_dependencies(BeanTensorV5Accelerated verify)
add_dependencies(BeanTensorV5Core verify)
