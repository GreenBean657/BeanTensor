cmake_minimum_required(VERSION 3.25)

# ─── Backend selection ────────────────────────────────────────────────────────
option(USE_HIP        "Build with AMD HIP/ROCm backend"       OFF)
option(USE_CUDA       "Build with NVIDIA CUDA backend"        OFF)
option(USE_STRIX_HALO "Build with Xilinx StrixHalo NPU backend" OFF)

if(USE_HIP AND USE_CUDA)
    message(FATAL_ERROR "USE_HIP and USE_CUDA are mutually exclusive — enable only one.")
endif()
if(NOT USE_HIP AND NOT USE_CUDA)
    message(WARNING "No backend selected — set either USE_HIP=ON or USE_CUDA=ON.")
endif()

if(USE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "NOTFOUND")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
    project(BeanTensorV4 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
elseif(USE_HIP)
    project(BeanTensorV4 LANGUAGES CXX HIP)
else()
    project(BeanTensorV4 LANGUAGES CXX)
endif()

include(CTest)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── ROCm / HIP ──────────────────────────────────────────────────────────────
if(USE_HIP)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip REQUIRED)
endif()

# ─── CUDA ────────────────────────────────────────────────────────────────────
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ─── XRT (NPU host runtime) ──────────────────────────────────────────────────
if(USE_STRIX_HALO)
    find_path(XRT_INCLUDE_DIR
            NAMES xrt/xrt_device.h
            PATHS /usr/include /opt/xilinx/xrt/include
            REQUIRED
    )
    find_library(XRT_CORE_LIB     NAMES xrt_core
            PATHS /usr/lib/x86_64-linux-gnu /opt/xilinx/xrt/lib REQUIRED)
    find_library(XRT_COREUTIL_LIB NAMES xrt_coreutil
            PATHS /usr/lib/x86_64-linux-gnu /opt/xilinx/xrt/lib REQUIRED)
endif()

# ─── MLIR-AIE / IRON toolchain ───────────────────────────────────────────────
if(USE_STRIX_HALO)
    set(IRONENV_DIR      "/mnt/schoolwork/BeanTensorV4/Compilers/ironenv")
    set(IRONENV_SITE     "${IRONENV_DIR}/lib/python3.10/site-packages")
    set(MLIR_AIE_INSTALL "${IRONENV_SITE}/mlir_aie")
    set(MLIR_AIE_SRC     "/mnt/schoolwork/BeanTensorV4/Compilers/mlir-aie")
    set(LLVM_AIE_DIR     "${IRONENV_SITE}/llvm-aie")
    set(Python3_EXECUTABLE "${IRONENV_DIR}/bin/python3")
    set(AIE_PYTHON_PATH    "${MLIR_AIE_INSTALL}/python")
    set(AIECC              "${AIE_PYTHON_PATH}/aie/compiler/aiecc/main.py")
    set(PEANO_CXX          "${LLVM_AIE_DIR}/bin/clang++")

    set(PEANOWRAP2P_FLAGS
            "-O2"
            "-std=c++20"
            "--target=aie2p-none-unknown-elf"
            "-DNDEBUG"
            "-I${MLIR_AIE_INSTALL}/include"
            "-I${MLIR_AIE_SRC}/aie_kernels"
    )

    find_path(AIE_INCLUDE_DIR
            NAMES aie_api/aie.hpp
            PATHS "${MLIR_AIE_INSTALL}/include"
            REQUIRED
    )
endif()

# ─── Status messages ─────────────────────────────────────────────────────────
if(USE_HIP)
    message(STATUS "Backend     : HIP/ROCm")
elseif(USE_CUDA)
    message(STATUS "Backend     : CUDA")
else()
    message(STATUS "Backend     : None (CPU only)")
endif()
if(USE_STRIX_HALO)
    message(STATUS "XRT include : ${XRT_INCLUDE_DIR}")
    message(STATUS "AIE include : ${AIE_INCLUDE_DIR}")
    message(STATUS "MLIR-AIE    : ${MLIR_AIE_INSTALL}")
    message(STATUS "Peano       : ${PEANO_CXX}")
    message(STATUS "Python      : ${Python3_EXECUTABLE}")
endif()

# ─── Step 1: Python graph → MLIR ─────────────────────────────────────────────
if(USE_STRIX_HALO)
    set(FMA_MLIR   "${CMAKE_CURRENT_BINARY_DIR}/fma.mlir")
    set(FMA_OBJ    "${CMAKE_CURRENT_BINARY_DIR}/fma.o")
    set(FMA_XCLBIN "${CMAKE_CURRENT_BINARY_DIR}/fma.xclbin")
    set(FMA_INSTS  "${CMAKE_CURRENT_BINARY_DIR}/fma_insts.bin")

    add_custom_command(
            OUTPUT  "${FMA_MLIR}"
            COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${AIE_PYTHON_PATH}"
            "VIRTUAL_ENV=${IRONENV_DIR}"
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/NPU/fma_graph.py"
            -d npu2
            -n 1024
            > "${FMA_MLIR}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/NPU/fma_graph.py"
            COMMENT "Generating MLIR from fma_graph.py"
            VERBATIM
    )

    # ─── Step 2: Peano compiles fma.cc → fma.o ───────────────────────────────
    add_custom_command(
            OUTPUT  "${FMA_OBJ}"
            COMMAND ${PEANO_CXX}
            ${PEANOWRAP2P_FLAGS}
            -c "${CMAKE_CURRENT_SOURCE_DIR}/NPU/fma.cc"
            -o "${FMA_OBJ}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/NPU/fma.cc"
            COMMENT "Compiling fma.cc with Peano → fma.o"
            VERBATIM
    )

    # ─── Step 3: aiecc links MLIR + object → xclbin ──────────────────────────
    add_custom_command(
            OUTPUT  "${FMA_XCLBIN}" "${FMA_INSTS}"
            COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${IRONENV_SITE}:${AIE_PYTHON_PATH}"
            "VIRTUAL_ENV=${IRONENV_DIR}"
            "PATH=${IRONENV_DIR}/bin:${LLVM_AIE_DIR}/bin:$ENV{PATH}"
            ${Python3_EXECUTABLE} "${AIECC}"
            "${FMA_MLIR}"
            --aie-generate-xclbin
            --aie-generate-npu-insts
            --no-compile-host
            --no-xchesscc
            --no-xbridge
            --xclbin-name=${FMA_XCLBIN}
            --npu-insts-name=${FMA_INSTS}
            DEPENDS "${FMA_MLIR}" "${FMA_OBJ}"
            COMMENT "Linking fma.mlir + fma.o → fma.xclbin + fma_insts.bin"
            VERBATIM
    )
    add_custom_target(fma_xclbin ALL DEPENDS "${FMA_XCLBIN}" "${FMA_INSTS}")
endif()

# ─── Core static library ─────────────────────────────────────────────────────
# All shared sources live here. Both executables link against this.
set(CORE_SOURCES
        Tensor/Tensor.cpp
        Tensor/TensorOps.cpp
        Ops/Elementwise/Add/cpu_Add.cpp
        Intrinsics/AVX512/DTypeConversions.cpp
        Intrinsics/AVX512/DTypeConversions.h
        Intrinsics/Native/DTypeConversions.h
        Intrinsics/Shared/SharedIntrinsics.h
        Intrinsics/Dispatcher.h
)
if(USE_CUDA)
    list(APPEND CORE_SOURCES Tensor/TensorOps.cu)
endif()
if(USE_STRIX_HALO)
    list(APPEND CORE_SOURCES NPU/NPUBackend.cpp)
endif()

add_library(BeanTensorV4Core STATIC ${CORE_SOURCES})

# ─── GPU language routing ────────────────────────────────────────────────────
if(USE_HIP)
    set_source_files_properties(
            Tensor/Tensor.cpp
            Tensor/TensorOps.cpp
            Intrinsics/AVX512/DTypeConversions.cpp
            PROPERTIES LANGUAGE HIP
    )
elseif(USE_CUDA)
    set_source_files_properties(
            Tensor/Tensor.cpp
            Tensor/TensorOps.cu
            PROPERTIES LANGUAGE CUDA
    )
endif()

# ─── Core compile definitions ────────────────────────────────────────────────
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV4Core PRIVATE
            FMA_XCLBIN_PATH="${FMA_XCLBIN}"
            FMA_INSTS_PATH="${FMA_INSTS}"
    )
endif()
if(USE_HIP)
    target_compile_definitions(BeanTensorV4Core PRIVATE USE_HIP)
elseif(USE_CUDA)
    target_compile_definitions(BeanTensorV4Core PRIVATE USE_CUDA)
endif()
if(USE_STRIX_HALO)
    target_compile_definitions(BeanTensorV4Core PRIVATE USE_STRIX_HALO)
endif()

# ─── Core include directories ────────────────────────────────────────────────
target_include_directories(BeanTensorV4Core PUBLIC ${CMAKE_SOURCE_DIR})
if(USE_STRIX_HALO)
    target_include_directories(BeanTensorV4Core PRIVATE
            ${XRT_INCLUDE_DIR}
            ${AIE_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/NPU
    )
endif()
if(USE_HIP)
    target_include_directories(BeanTensorV4Core PRIVATE ${HIP_INCLUDE_DIRS})
endif()

# ─── Core link libraries ─────────────────────────────────────────────────────
if(USE_STRIX_HALO)
    target_link_libraries(BeanTensorV4Core PRIVATE
            ${XRT_CORE_LIB}
            ${XRT_COREUTIL_LIB}
            uuid
            dl
    )
endif()
if(USE_HIP)
    target_link_libraries(BeanTensorV4Core PRIVATE hip::host)
elseif(USE_CUDA)
    target_link_libraries(BeanTensorV4Core PRIVATE CUDA::cudart)
endif()
if(USE_STRIX_HALO)
    add_dependencies(BeanTensorV4Core fma_xclbin)
endif()

# ─── Executables ─────────────────────────────────────────────────────────────
add_executable(BeanTensorV4 main.cpp)
target_link_libraries(BeanTensorV4 PRIVATE BeanTensorV4Core)
target_compile_options(BeanTensorV4 PRIVATE -O2)

add_executable(BeanTensorV4Accelerated main.cpp)
target_link_libraries(BeanTensorV4Accelerated PRIVATE BeanTensorV4Core)
target_compile_options(BeanTensorV4Accelerated PRIVATE
        -O3
        -mavx512f
        -mavx512bf16
        -mf16c
)

# ─── Tests ───────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(BeanTensorV4Tests
            tests/tensor_tests.cpp
    )
    target_link_libraries(BeanTensorV4Tests PRIVATE BeanTensorV4Core)
    target_compile_options(BeanTensorV4Tests PRIVATE
            -O2
            -mavx512f
            -mavx512bf16
            -mf16c
    )
    add_test(NAME BeanTensorV4Tests COMMAND BeanTensorV4Tests)
endif()