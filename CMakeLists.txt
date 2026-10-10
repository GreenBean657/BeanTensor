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
    IF (NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "NOTFOUND")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    ENDIF()
    project(BeanTensorV5 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
elseif (USE_HIP)
    project(BeanTensorV5 LANGUAGES CXX HIP)
else()
    project(BeanTensorV5 LANGUAGES CXX)
endif()


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (USE_HIP)
    message(STATUS "Backend: HIP/ROCm")
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip REQUIRED)
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
        Tensor/DTypes.h
        Tensor/Tensor.h
        Tensor/Tensor.cpp
        scratch/gpu.cpp
        Helpers/Bits.h
)

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

if(USE_HIP)

elseif(USE_CUDA)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_CUDA)
endif()
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_STRIX_HALO)
endif()

if(USE_HIP)
    set_source_files_properties(
            scratch/gpu.cpp
            Tensor/Tensor.cpp
            Tensor/Ops/TensorOps.cpp
            PROPERTIES LANGUAGE HIP
    )
    set_source_files_properties(
            scratch/gpu.cpp
            Tensor/Tensor.cpp
            Tensor/Ops/TensorOps.cpp
            PROPERTIES COMPILE_OPTIONS "--offload-arch=native"
    )
elseif (USE_CUDA)
    set_source_files_properties(
            scratch/gpu.cpp
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
elseif(USE_CUDA)
    target_link_libraries(BeanTensorV5Core PUBLIC CUDA::cudart)
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
        -mf16c
)

if(BUILD_TESTING)
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
            -mf16c
    )
    add_test(NAME BeanTensorV5Tests COMMAND BeanTensorV5Tests)
endif()

if(USE_HIP)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_HIP)
    target_compile_definitions(BeanTensorV5 PRIVATE USE_HIP)
    target_compile_definitions(BeanTensorV5Accelerated PRIVATE USE_HIP)
elseif(USE_CUDA)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_CUDA)
    target_compile_definitions(BeanTensorV5 PRIVATE USE_CUDA)
    target_compile_definitions(BeanTensorV5Accelerated PRIVATE USE_CUDA)
endif()
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Core PRIVATE USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5 PRIVATE USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV5Accelerated PRIVATE USE_STRIX_HALO)
endif()
