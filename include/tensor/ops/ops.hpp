#pragma once

#include "../core/device/device.hpp"

#include <immintrin.h>
#include <functional>

namespace tensor
{
  template <typename T>
  class Tensor;

  namespace ops
  {
    template <typename T>
    class TensorOps
    {
    public:
      // Basic arithmetic operations
      static Tensor<T> add(const Tensor<T> &lhs, const Tensor<T> &rhs);
      static Tensor<T> subtract(const Tensor<T> &lhs, const Tensor<T> &rhs);
      static Tensor<T> multiply(const Tensor<T> &lhs, const Tensor<T> &rhs);
      static Tensor<T> divide(const Tensor<T> &lhs, const Tensor<T> &rhs);

      // Matrix operations
      static Tensor<T> matmul(const Tensor<T> &lhs, const Tensor<T> &rhs);
      static Tensor<T> transpose(const Tensor<T> &tensor);
      static Tensor<T> dot(const Tensor<T> &lhs, const Tensor<T> &rhs);

      // Element-wise operations
      static Tensor<T> abs(const Tensor<T> &tensor);
      static Tensor<T> exp(const Tensor<T> &tensor);
      static Tensor<T> log(const Tensor<T> &tensor);
      static Tensor<T> sqrt(const Tensor<T> &tensor);
      static Tensor<T> pow(const Tensor<T> &tensor, T exponent);

      // Reduction operations
      static T sum(const Tensor<T> &tensor, int axis = -1);
      static T mean(const Tensor<T> &tensor, int axis = -1);
      static T max(const Tensor<T> &tensor, int axis = -1);
      static T min(const Tensor<T> &tensor, int axis = -1);

      // Broadcasting operations
      static Tensor<T> broadcast_to(const Tensor<T> &tensor,
                                    const std::vector<size_t> &shape);

    private:
      static void validate_shapes_match(const Tensor<T> &lhs,
                                        const Tensor<T> &rhs);

      static void validate_matmul_shapes(const Tensor<T> &lhs,
                                         const Tensor<T> &rhs);

      static std::vector<size_t> compute_broadcast_shape(
          const std::vector<size_t> &lhs_shape,
          const std::vector<size_t> &rhs_shape);

      static bool shapes_are_broadcastable(const std::vector<size_t> &lhs_shape,
                                           const std::vector<size_t> &rhs_shape);
    };
  } // namespace ops
} // namespace tensor