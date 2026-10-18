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
    message(FATAL_ERROR "HIP/CUDA and StrixHalo are mutually exclusive — enable only one.")
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
# -Werror=maybe-uninitialized is GNU-onlyyyyyyyyyyyyyyyyyyy so lets make it be that.
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
        Helpers/Bits.h
)

# AVX512/AVX2 conversion kernels are always compiled with AVX512/AVX2 codegen
# (per-file, so other TUs are unaffected); execution is gated at runtime via
# IncludedCPUFeatures() so CPUs lacking AVX512BF16/AVX2 take the scalar fallback.
# The HIP variant of these flags is applied in the USE_HIP block below.

if (USE_CUDA)
    list(APPEND CORE_SOURCES
            Tensor/Ops/TensorOps.cu
    )
elseif (USE_HIP)
    list(APPEND CORE_SOURCES
            Tensor/Ops/TensorOps.cpp
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
    target_compile_options(BeanTensorV5Core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20;-Xcompiler=-Wno-error=float-equal>
    )
endif()
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_STRIX_HALO)
endif()

if(USE_HIP)
    set_source_files_properties(
            Tensor/Tensor.cpp
            Tensor/Ops/TensorOps.cpp
            Intrinsics/AVX2/DTypeConversions.cpp
            Intrinsics/AVX512/DTypeConversions.cpp
            Intrinsics/Dispatcher.cpp
            PROPERTIES LANGUAGE HIP
    )
    set_source_files_properties(
            Tensor/Tensor.cpp
            Tensor/Ops/TensorOps.cpp
            Intrinsics/Dispatcher.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native"
    )
    # AVX512 kernels: HIP offload arch + host AVX512 codegen combined (the
    # single COMPILE_OPTIONS property would otherwise clobber one or the other).
    set_source_files_properties(
            Intrinsics/AVX512/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx512f;-mavx512bf16;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Intrinsics/AVX2/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native;-mavx2;-mf16c"
    )
else()
    set_source_files_properties(
            Intrinsics/AVX512/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bf16;-mavx512dq;-mf16c"
    )
    set_source_files_properties(
            Intrinsics/AVX2/DTypeConversions.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c"
    )
endif()
if (USE_CUDA)
    set_source_files_properties(
            Tensor/Ops/TensorOps.cu
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
            -mavx512f
            -mavx512bf16
            -mavx512dq
            -mf16c
    )
    add_test(NAME BeanTensorV5Tests COMMAND BeanTensorV5Tests)

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
