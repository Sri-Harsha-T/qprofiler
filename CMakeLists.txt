cmake_minimum_required(VERSION 3.16)
project(qprofiler VERSION 1.0.0 LANGUAGES CXX)

# Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build for benchmarking. Keep debug symbols for profiling
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags
add_compile_options(
    -O3 # full optimization
    -march=native # enable SIMD (AVX2 / SSE4) on build machine
    -ffast-math # allow float reassociation (safe for our kernel, can be omitted)
    -Wall -Wextra
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — multithreaded kernels enabled")
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found — single-threaded fallback")
endif()

# Python + Pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "pybind11: ${pybind11_VERSION}")

# Core C++ Library (static, linked into extension)

add_library(qprofiler_lib STATIC
    src/gates.cpp
    src/profiler.cpp
    src/simulator.cpp
)
set_target_properties(qprofiler_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(qprofiler_lib PUBLIC src)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qprofiler_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Pybind11 extension module
pybind11_add_module(qprofiler_core src/bindings.cpp)
target_link_libraries(qprofiler_core PRIVATE qprofiler_lib)

# Put .so in python/ so that `import qprofiler_core` can work directly
set_target_properties(qprofiler_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

# Standalone benchmark binary
add_executable(qp_bench src/benchmark_main.cpp)
target_link_libraries(qp_bench PRIVATE qprofiler_lib)

# C++ unit tests
enable_testing()
add_executable(test_gates tests/test_gates.cpp)
target_include_directories(test_gates PRIVATE src)
target_link_libraries(test_gates PRIVATE qprofiler_lib)
add_test(NAME GateTests COMMAND test_gates)

