# Tensor Framework

A modern C++20 tensor computation framework designed for high performance and ease of use. Similar to PyTorch and TensorFlow, but with modern C++ features and design patterns.

## Features

- 🚀 High-performance tensor operations
- 🧮 Automatic differentiation
- 🔄 Dynamic computation graphs
- 📊 Neural network primitives
- 🎯 CUDA acceleration support
- 📈 BLAS integration
- 🧪 Comprehensive test suite
- 📊 Benchmarking tools

## Requirements

- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 19.28+)
- CMake 3.20 or higher
- vcpkg package manager
- CUDA toolkit 11.0+ (optional)
- OpenBLAS or compatible BLAS implementation

## Quick Start

### Installation

1. Clone the repository

```bash
git clone https://github.com/salgue441/tensor_library.git
cd tensor_framework
```

2. Build the project

```bash
# Unix-like systems
./scripts/build.sh

# Windows
.\scripts\build.bat
```

### For custom build options

```bash
./scripts/build.sh -t Debug --cuda -j 8  # Debug build with CUDA support
```

## Basic Usage

```cpp
#include <tf/tensor.hpp>

int main() {
    // Create tensors
    tf::Tensor<float> a({2, 3}, true);  // 2x3 tensor with gradient tracking
    tf::Tensor<float> b({2, 3}, true);

    // Perform operations
    auto c = a + b;
    auto d = c.matmul(b.transpose());

    // Compute gradients
    d.backward();

    return 0;
}
```

## Build Options

| Option            | Description           | Default |
| ----------------- | --------------------- | ------- |
| TF_BUILD_TESTS    | Build the test suite  | ON      |
| TF_BUILD_BENCH    | Build benchmarks      | ON      |
| TF_BUILD_EXAMPLES | Build examples        | ON      |
| TF_USE_CUDA       | Enable CUDA support   | OFF     |
| TF_USE_BLAS       | Enable BLAS support   | ON      |
| TF_USE_PROFILING  | Enable OpenMP support | OFF     |

## Development

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run specific tests
./scripts/test.sh -f TensorTest

# Run benchmarks
./scripts/test.sh -b
```

### Running Examples

```bash
./scripts/run_example.sh tensor_basics
```

### Code Style

This project follows the C++ Core Guidelines and uses clang-format for code formatting. To format your code:

```bash
clang-format -i include/**/*.hpp src/**/*.cpp
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Pytorch - For inspiration and API design
- TensorFlow - For computational graph concepts
- Eigen - For tensor operation optimizations
