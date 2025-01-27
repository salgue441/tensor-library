cmake_minimum_required(VERSION 3.20)

# Project definition
project(tensor_framework 
    VERSION 0.1.0 
    DESCRIPTION "Modern C++ Tensor Framework"
    HOMEPAGE_URL "https://github.com/salgue441/tensor_framework"
    LANGUAGES CXX
)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Project options
option(TF_BUILD_TESTS "Build tests" ON)
option(TF_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TF_BUILD_EXAMPLES "Build examples" ON)
option(TF_USE_CUDA "Enable CUDA support" OFF)
option(TF_USE_BLAS "Enable BLAS support" ON)
option(TF_ENABLE_PROFILING "Enable profiling support" OFF)
option(TF_BUILD_SHARED_LIBS "Build shared libraries" ON)
option(TF_ENABLE_SANITIZERS "Enable sanitizers in debug build" OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable CUDA if requested
if(TF_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find required packages
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
if(TF_USE_BLAS)
    find_package(BLAS REQUIRED)
endif()

# Main library target
add_library(tf)
add_library(tf::tf ALIAS tf)

# Set source files
file(GLOB_RECURSE TF_SOURCES
    "src/core/*.cpp"
    "src/ops/*.cpp"
    "src/math/*.cpp"
    "src/nn/*.cpp"
)

target_sources(tf
    PRIVATE
        ${TF_SOURCES}
)

# Set include directories
target_include_directories(tf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set compile options
if(MSVC)
    target_compile_options(tf PRIVATE
        /W4
        /WX
        /permissive-
        /Zc:__cplusplus
        /utf-8
    )
else()
    target_compile_options(tf PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -fPIC
    )
endif()

# Add compile definitions based on options
if(TF_ENABLE_PROFILING)
    target_compile_definitions(tf PRIVATE TF_PROFILING)
endif()

if(TF_USE_CUDA)
    target_compile_definitions(tf PRIVATE TF_CUDA_ENABLED)
endif()

if(TF_USE_BLAS)
    target_compile_definitions(tf PRIVATE TF_BLAS_ENABLED)
endif()

# Link dependencies
if(TF_USE_BLAS)
    target_link_libraries(tf PUBLIC BLAS::BLAS)
endif()

# Tests
if(TF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(TF_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

# Examples
if(TF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS tf
    EXPORT tf-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/tf
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(
    EXPORT tf-targets
    FILE tf-targets.cmake
    NAMESPACE tf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tf
)