cmake_minimum_required(VERSION 3.25)
project(BeanTensorV4 LANGUAGES CXX HIP)
include(CTest)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── ROCm / HIP ──────────────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED)

# ─── XRT (NPU host runtime) ───────────────────────────────────────────────────
find_path(XRT_INCLUDE_DIR
        NAMES xrt/xrt_device.h
        PATHS /usr/include /opt/xilinx/xrt/include
        REQUIRED
)
find_library(XRT_CORE_LIB     NAMES xrt_core
        PATHS /usr/lib/x86_64-linux-gnu /opt/xilinx/xrt/lib REQUIRED)
find_library(XRT_COREUTIL_LIB NAMES xrt_coreutil
        PATHS /usr/lib/x86_64-linux-gnu /opt/xilinx/xrt/lib REQUIRED)

# ─── MLIR-AIE / IRON toolchain ───────────────────────────────────────────────
set(IRONENV_DIR      "/mnt/schoolwork/BeanTensorV4/Compilers/ironenv")
set(IRONENV_SITE     "${IRONENV_DIR}/lib/python3.10/site-packages")
set(MLIR_AIE_INSTALL "${IRONENV_SITE}/mlir_aie")
set(MLIR_AIE_SRC     "/mnt/schoolwork/BeanTensorV4/Compilers/mlir-aie")
set(LLVM_AIE_DIR     "${IRONENV_SITE}/llvm-aie")
set(Python3_EXECUTABLE "${IRONENV_DIR}/bin/python3")
set(AIE_PYTHON_PATH    "${MLIR_AIE_INSTALL}/python")
set(AIECC              "${AIE_PYTHON_PATH}/aie/compiler/aiecc/main.py")
set(PEANO_CXX          "${LLVM_AIE_DIR}/bin/clang++")

# Matches official PEANOWRAP2P_FLAGS + aie_kernels for AIE_* macros
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

message(STATUS "XRT include : ${XRT_INCLUDE_DIR}")
message(STATUS "AIE include : ${AIE_INCLUDE_DIR}")
message(STATUS "MLIR-AIE    : ${MLIR_AIE_INSTALL}")
message(STATUS "Peano       : ${PEANO_CXX}")
message(STATUS "Python      : ${Python3_EXECUTABLE}")

# ─── Step 1: Python graph → MLIR ─────────────────────────────────────────────
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

# ─── Step 2: Peano compiles fma.cc → fma.o ───────────────────────────────────
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

# ─── Step 3: aiecc links MLIR + object → xclbin ──────────────────────────────
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

# ─── Main executable ──────────────────────────────────────────────────────────
add_executable(BeanTensorV4
        main.cpp
        NPU/NPUBackend.cpp
        Helper/errorcheck.h
        Ops/Elementwise/Add/Add.h
        Ops/Elementwise/Add/cpu_Add.h
        Ops/Elementwise/Add/gpu_Add.h
        Ops/Elementwise/Add/npu_Add.h
        Helper/Broadcast/Broadcast.h
        Helper/Casting/casting_detail.h
        Ops/Elementwise/Add/cpu_Add.cpp
        Tensor/tensor_convert.h
        Tensor/tensor.h
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

target_compile_definitions(BeanTensorV4 PRIVATE
        USE_HIP
        USE_NPU_STRIXHALO
        FMA_XCLBIN_PATH="${FMA_XCLBIN}"
        FMA_INSTS_PATH="${FMA_INSTS}"
)
target_compile_options(BeanTensorV4 PRIVATE
        -mavx512f
        -mavx512bf16
)

include_directories(${CMAKE_SOURCE_DIR})

target_include_directories(BeanTensorV4 PRIVATE
        ${HIP_INCLUDE_DIRS}
        ${XRT_INCLUDE_DIR}
        ${AIE_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/NPU
)

target_link_libraries(BeanTensorV4 PRIVATE
        hip::host
        ${XRT_CORE_LIB}
        ${XRT_COREUTIL_LIB}
        uuid
        dl
)

add_dependencies(BeanTensorV4 fma_xclbin)

# ─── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_executable(BeanTensorV4Tests tests/tensor_tests.cpp)
    include_directories(${CMAKE_SOURCE_DIR})
    add_test(NAME BeanTensorV4Tests COMMAND BeanTensorV4Tests)
endif()