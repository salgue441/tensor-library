cmake_minimum_required(VERSION 3.20)
project(tensor VERSION 0.1.0 LANGUAGES CXX)

# Options
option(TENSOR_BUILD_TESTS "Build test suite" ON)
option(TENSOR_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TENSOR_BUILD_EXAMPLES "Build examples" ON)
option(TENSOR_USE_SIMD "Enable SIMD optimizations" ON)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindTensorDependencies)

# Compiler flags
if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Werror)
endif()

if(TENSOR_USE_SIMD)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2)
    endif()
endif()

# Library target
add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Dependencies
find_package(OpenMP REQUIRED)

# # Source files (if any non-header implementations)
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
#     add_subdirectory(src)
# endif()

# Tests
if(TENSOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
# if(TENSOR_BUILD_EXAMPLES)
#     add_subdirectory(examples)
# endif()

# Benchmarks
if(TENSOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS tensor
    EXPORT tensor-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/tensor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT tensor-targets
    FILE TensorTargets.cmake
    NAMESPACE Tensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    TensorConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    cmake/TensorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TensorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor
)

target_compile_options(tensor INTERFACE -mavx2 -fopenmp)
target_link_libraries(tensor INTERFACE OpenMP::OpenMP_CXX)