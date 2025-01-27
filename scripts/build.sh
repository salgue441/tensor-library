#!/usr/bin/env bash
set -euo pipefail

# Default values
BUILD_TYPE="Release"
BUILD_TESTS=ON
BUILD_BENCHMARKS=ON
BUILD_EXAMPLES=ON
USE_CUDA=OFF
COMPILER="gcc"
CLEAN=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE_TOOLCHAIN_FILE=""

# Help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help              Show this help message"
  echo "  -t, --type TYPE         Build type (Debug|Release|RelWithDebInfo) [default: Release]"
  echo "  -c, --compiler CC       Compiler to use (gcc|clang) [default: gcc]"
  echo "  --toolchain FILE        CMake toolchain file"
  echo "  --no-tests              Disable building tests"
  echo "  --no-benchmarks         Disable building benchmarks"
  echo "  --no-examples           Disable building examples"
  echo "  --cuda                  Enable CUDA support"
  echo "  --clean                 Clean build directory before building"
  echo "  -j, --jobs N           Number of parallel jobs [default: number of CPU cores]"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    show_help
    exit 0
    ;;
  -t | --type)
    BUILD_TYPE="$2"
    shift 2
    ;;
  -c | --compiler)
    COMPILER="$2"
    shift 2
    ;;
  --toolchain)
    CMAKE_TOOLCHAIN_FILE="$2"
    shift 2
    ;;
  --no-tests)
    BUILD_TESTS=OFF
    shift
    ;;
  --no-benchmarks)
    BUILD_BENCHMARKS=OFF
    shift
    ;;
  --no-examples)
    BUILD_EXAMPLES=OFF
    shift
    ;;
  --cuda)
    USE_CUDA=ON
    shift
    ;;
  --clean)
    CLEAN=true
    shift
    ;;
  -j | --jobs)
    JOBS="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    show_help
    exit 1
    ;;
  esac
done

# Set compiler
if [ "$COMPILER" = "gcc" ]; then
  export CC=gcc
  export CXX=g++
elif [ "$COMPILER" = "clang" ]; then
  export CC=clang
  export CXX=clang++
else
  echo "Unknown compiler: $COMPILER"
  exit 1
fi

# Create and enter build directory
BUILD_DIR="build/${COMPILER}-${BUILD_TYPE,,}"
mkdir -p "$BUILD_DIR"

if [ "$CLEAN" = true ]; then
  echo "Cleaning build directory..."
  rm -rf "$BUILD_DIR"/*
fi

cd "$BUILD_DIR"

# Configure
echo "Configuring with CMake..."
CMAKE_ARGS=(
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DTF_BUILD_TESTS=$BUILD_TESTS"
  "-DTF_BUILD_BENCHMARKS=$BUILD_BENCHMARKS"
  "-DTF_BUILD_EXAMPLES=$BUILD_EXAMPLES"
  "-DTF_USE_CUDA=$USE_CUDA"
  "-DCMAKE_PREFIX_PATH=$HOME/vcpkg/installed/x64-linux"
)

if [ -n "$CMAKE_TOOLCHAIN_FILE" ]; then
  CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE")
fi

cmake ../.. "${CMAKE_ARGS[@]}"

# Build
echo "Building..."
cmake --build . --config "$BUILD_TYPE" -j "$JOBS"

cd ../..
