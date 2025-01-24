#pragma once

#include "../core/tensor_base.hpp"
#include "../utils/exceptions.hpp"
#include <algorithm>

namespace tensor
{
  /**
   * @brief Matrix multiplication operation
   *
   * @tparam T Numeric type of the tensor
   * @param a First tensor to multiply
   * @param b Second tensor to multiply
   * @return Tensor<T, 2> Result of the matrix multiplication
   * @throw DimensionMismatch If the dimensions of the input tensors are invalid
   */
  template <NumericType T>
  Tensor<T, 2> matrix_multiply(const Tensor<T, 2> &a, const Tensor<T, 2> &b)
  {
    const auto &shape_a = a.shape();
    const auto &shape_b = b.shape();

    if (shape_a[1] != shape_b[0])
    {
      throw DimensionMismatch("Invalid dimensions for matrix operation");
    }

    Tensor<T, 2> result({shape_a[0], shape_b[1]});
    for (std::size_t i = 0; i < shape_a[0]; ++i)
    {
      for (std::size_t j = 0; j < shape_b[1]; ++j)
      {
        T sum = 0;

        for (std::size_t k = 0; k < shape_a[1]; ++k)
        {
          sum += a[i * shape_a[1] + k] * b[k * shape_b[1] + j];
        }

        result[i * shape_b[1] + j] = sum;
      }
    }

    return result;
  };

  /**
   * @brief Matrix transpose operation
   *
   * @tparam T Numeric type of the tensor
   * @param input Input tensor
   * @return Tensor<T, 2> Transposed tensor
   */
  template <NumericType T>
  Tensor<T, 2> transpose(const Tensor<T, 2> &input)
  {
    const auto &shape = input.shape();
    Tensor<T, 2> result({shape[1], shape[0]});

    for (std::size_t i = 0; i < shape[0]; ++i)
    {
      for (std::size_t j = 0; j < shape[1]; ++j)
      {
        result[j * shape[0] + i] = input[i * shape[1] + j];
      }
    }

    return result;
  }
} // namespace tensor