# Python bindings module. pybind11 is expected from pip, not a system package,
# so its CMake package location is probed through the interpreter itself.
#
# Setup on a PEP 668 (externally managed) box — --user is rejected there just
# like a system-wide install, so use one venv for the interpreter AND the
# packages, consistently:
#
#   python3 -m venv .venv
#   .venv/bin/python -m pip install "pybind11>=3.0" numpy
#
# The venv's interpreter must be the one CMake picks up: either activate the
# venv before configuring (PATH discovery) or pass
# -DPython3_EXECUTABLE=$PWD/.venv/bin/python. Python development headers are
# required (find_package below asks for Development.Module; Debian/Ubuntu
# package: python3-dev).
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE BEANTENSOR_PYBIND11_CMAKEDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BEANTENSOR_PYBIND11_CMAKEDIR_RESULT
)
if (BEANTENSOR_PYBIND11_CMAKEDIR_RESULT OR NOT BEANTENSOR_PYBIND11_CMAKEDIR)
    message(FATAL_ERROR
            "pybind11 is not installed for ${Python3_EXECUTABLE}. Set up one interpreter as documented at the top of Bindings/CMakeLists.txt: python3 -m venv .venv && .venv/bin/python -m pip install \"pybind11>=3.0\" numpy (reconfigure with -DPython3_EXECUTABLE=<venv>/bin/python if the venv is not first on PATH).")
endif ()
set(pybind11_DIR "${BEANTENSOR_PYBIND11_CMAKEDIR}" CACHE PATH "pybind11 CMake package dir (reported by the pip install)" FORCE)
# py::smart_holder (required below for the unique_ptr<Module> ownership
# transfers) ships with pybind11 3.0 — pin the floor here so an older install
# fails configure with an upgrade hint instead of a smart_holder compile error.
find_package(pybind11 3.0 REQUIRED)

# The extension module is a shared object, so the static core it links must
# be position-independent: without -fPIC the core's function-local thread_local
# RNG (Tensor.cpp fill_random_impl) gets local-exec TLS, which ld refuses when
# linking a static archive into a shared library.
set_target_properties(BeanTensorV5Core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(beantensor pybind_module.cpp)
target_link_libraries(beantensor PRIVATE BeanTensorV5Core)

# The root CMakeLists registers the bindings_smoke ctest when the option is
# on; hand the interpreter path up so the test runs the same Python.
set(BEANTENSOR_PYTHON3_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
