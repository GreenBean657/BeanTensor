cmake_minimum_required(VERSION 3.25)

project(BeanTensorV4 LANGUAGES CXX HIP)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

find_package(hip REQUIRED)

add_executable(BeanTensorV4 main.cpp
        Helper/hipcheck.h
        Ops/Elementwise/Add.h
        Ops/Elementwise/Mul.h
        Ops/Elementwise/Div.h
        Ops/Elementwise/Exp.h
        Ops/Elementwise/Log.h
        Ops/Elementwise/Relu.h
        Ops/Reduction/Sum.h
        Ops/Reduction/Mean.h
        Ops/Reduction/Max.h
        Ops/Reduction/Norm.h
        Ops/Reduction/Softmax.h
        Ops/Algebra/MatMul.h
        Ops/Algebra/Dot.h
        Ops/Algebra/Transpose.h
        Ops/Algebra/SVD.h
        Ops/Algebra/Eigen.h
        Ops/Algebra/Inverse.h
        Ops/Transform/Reshape.h
        Ops/Transform/View.h
        Ops/Transform/Squeeze.h
        Ops/Transform/Permute.h
        Ops/Transform/Concat.h
        Ops/Convolution/Conv1D.h
        Ops/Convolution/Conv2D.h
        Ops/Convolution/Im2Col.h
        Ops/Convolution/Winograd.h
        Ops/Indexing/Gather.h
        Ops/Indexing/Scatter.h
        Ops/Indexing/Slice.h
        Ops/Indexing/Mask.h
        Ops/Indexing/Argmax.h
        deving/test.h
)

# Force CMake to compile HIP-dependent files through the HIP toolchain
set_source_files_properties(main.cpp
        PROPERTIES LANGUAGE HIP)

target_compile_definitions(BeanTensorV4 PRIVATE USE_HIP)
# target_compile_options(BeanTensorV4 PRIVATE -mf16c)

include_directories(${CMAKE_SOURCE_DIR})

target_include_directories(BeanTensorV4 PRIVATE ${HIP_INCLUDE_DIRS})

# AI GENERATED
if(BUILD_TESTING)
    add_executable(BeanTensorV4Tests
            tests/tensor_tests.cpp
    )
    include_directories(${CMAKE_SOURCE_DIR})
    add_test(NAME BeanTensorV4Tests COMMAND BeanTensorV4Tests)
endif()
#END AI GENERATED
target_link_libraries(BeanTensorV4 PRIVATE hip::host)