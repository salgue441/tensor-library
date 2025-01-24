#pragma once

#include "../core/expression.hpp"
#include "../core/tensor_base.hpp"
#include <immintrin.h>

namespace tensor
{
  /**
   * @brief Overload of % oeprator for binary operations of two tensors
   *
   * @tparam T Tensor data type
   * @param lhs Left hand side tensor
   * @param rhs Right hand side tensor
   * @return Tensor<T, 2> Result tensor of the binary operation
   * @throw DimensionMismatch If the dimensions of the two tensors do not match
   */
  template <typename T>
  Tensor<T, 2> operator%(const Tensor<T, 2> &lhs, const Tensor<T, 2> &rhs)
  {
    const auto &lhs_shape = lhs.shape();
    const auto &rhs_shape = rhs.shape();

    if (lhs_shape[1] != rhs_shape[0])
    {
      throw DimensionMismatch("Invalid dimensions for matrix multiplication");
    }

    constexpr size_t BLOCK_SIZE = 32;
    Tensor<T, 2> result({lhs_shape[0], rhs_shape[1]});

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < lhs_shape[0]; i += BLOCK_SIZE)
    {
      for (size_t j = 0; j < rhs_shape[1]; j += BLOCK_SIZE)
      {
        for (size_t k = 0; k < lhs_shape[1]; k += BLOCK_SIZE)
        {
          size_t i_end = std::min(i + BLOCK_SIZE, lhs_shape[0]);
          size_t j_end = std::min(j + BLOCK_SIZE, rhs_shape[1]);
          size_t k_end = std::min(k + BLOCK_SIZE, lhs_shape[1]);

          for (size_t ii = i; ii < i_end; ++ii)
          {
            for (size_t jj = j; jj < j_end; jj += 8)
            {
              __m256 sum = _mm256_setzero_ps();

              for (size_t kk = k; kk < k_end; ++kk)
              {
                __m256 a = _mm256_set1_ps(static_cast<float>(lhs[ii * lhs_shape[1] + kk]));
                __m256 b = _mm256_loadu_ps(reinterpret_cast<const float *>(&rhs[kk * rhs_shape[1] + jj]));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
              }

              _mm256_storeu_ps(reinterpret_cast<float *>(&result[ii * rhs_shape[1] + jj]), sum);
            }
          }
        }
      }
    }
    return result;
  }

  /**
   * @brief Overload of * operator for binary operations of a tensor and
   * a scalar
   *
   * @tparam T Tensor data type
   * @param scalar Scalar value
   * @param tensor Tensor object
   * @return Tensor<T, 2> Result tensor of the binary operation
   */
  template <typename T>
  Tensor<T, 2> operator*(T scalar, const Tensor<T, 2> &tensor)
  {
    Tensor<T, 2> result(tensor.shape());
    size_t size = tensor.size();
    size_t i = 0;

    __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i + 8 <= size; i += 8)
    {
      __m256 vec = _mm256_loadu_ps(&tensor[i]);
      _mm256_storeu_ps(&result[i], _mm256_mul_ps(scalar_vec, vec));
    }

    for (; i < size; ++i)
    {
      result[i] = scalar * tensor[i];
    }

    return result;
  };

  /**
   * @brief Overload of * operator for binary operations of a tensor and
   * a scalar
   *
   * @tparam T Tensor data type
   * @param tensor Tensor object
   * @param scalar Scalar value
   * @return Tensor<T, 2> Result tensor of the binary operation
   */
  template <typename T>
  Tensor<T, 2> operator*(const Tensor<T, 2> &tensor, T scalar)
  {
    return scalar * tensor;
  };
} // namespace tensor