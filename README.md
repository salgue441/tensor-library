# C++ Tensor Library

A high-performance, header-only tensor library implementing basic operations similar to PyTorch and TensorFlow.

## Features

- Expression templates for lazy evaluation
- SIMD Optimization for basic operations
- Modern C++20 features
- Thread-safe operations
- Comprehensive test coverage
- Benchmarking suite

## Requirements

- C++20 compliant compiler
- CMake 3.20+
- vcpkg (optional)
- GoogleTest (for testing)
- Google Benchmark

## Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Build Options

- `TENSOR_BUILD_TESTS`: Build test suite (ON/OFF, default: ON)
- `TENSOR_BUILD_BENCHMARKS`: Build benchmarks (ON/OFF, default: OFF)
- `TENSOR_BUILD_EXAMPLES`: Build examples (ON/OFF, default: ON)
- `TENSOR_USE_SIMD`: Enable SIMD optimizations (ON/OFF, default: ON)

## Usage

```cpp
#include <tensor/tensor.hpp>

using namespace tensor;

int main() {
    // Create 2D tensors
    Tensor<float, 2> a({2, 3});
    Tensor<float, 2> b({2, 3});

    // Perform operations
    auto c = a + b;                              // Element-wise addition
    auto d = matrix_multiply(a, b.transpose());  // Matrix multiplication

    return 0;
}
```

## Documentation

Generate documentation nusing Doxygen:

```bash
cd docs
doxygen
```

## License

MIT License - see [LICENSE](LICENSE) for more details.
