#pragma once

#include "../core/tensor_base.hpp"

namespace tensor
{
  /**
   * @brief Dot product of two tensors of the same size
   *
   * @tparam T Tensor type
   * @param a First tensor
   * @param b Second tensor
   * @return T Dot product of the two tensors
   */
  template <typename T>
  T dot_product(const Tensor<T, 2> &a, const Tensor<T, 2> &b)
  {
    if (a.size() != b.size())
      throw DimensionMismatch("Dot product requires tensors of the same size");

    T result = T(0);
    for (size_t i = 0; i < a.size(); ++i)
      result += a[i] * b[i];

    return result;
  }

  /**
   * @brief Cross product of two 3D vectors
   *
   * @tparam T Tensor type
   * @param a First vector
   * @param b Second vector
   * @return Tensor<T, 2> Cross product of the two vectors
   */
  template <typename T>
  Tensor<T, 2> cross_product(const Tensor<T, 2> &a, const Tensor<T, 2> &b)
  {
    if (a.size() != 3 || b.size() != 3)
      throw DimensionMismatch("Cross product requires 3D vectors");

    Tensor<T, 2> result({3, 1});

    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];

    return result;
  }

  /**
   * @brief Outer product of two tensors
   *
   * @tparam T Tensor type
   * @param a First tensor
   * @param b Second tensor
   * @return Tensor<T, 2> Outer product of the two tensors
   */
  template <typename T>
  Tensor<T, 2> outer_product(const Tensor<T, 2> &a, const Tensor<T, 2> &b)
  {
    Tensor<T, 2> result({a.size(), b.size()});

    for (size_t i = 0; i < a.size(); ++i)
      for (size_t j = 0; j < b.size(); ++j)
        result[i * b.size() + j] = a[i] * b[j];

    return result;
  }

  /**
   * @brief Kronecker product of two tensors
   *
   * @tparam T Tensor type
   * @param a First tensor
   * @param b Second tensor
   * @return Tensor<T, 2> Kronecker product of the two tensors
   */
  template <typename T>
  Tensor<T, 2> kronecker_product(const Tensor<T, 2> &a, const Tensor<T, 2> &b)
  {
    Tensor<T, 2> result({a.size() * b.size(), a.size() * b.size()});

    for (size_t i = 0; i < a.size(); ++i)
      for (size_t j = 0; j < b.size(); ++j)
        for (size_t k = 0; k < a.size(); ++k)
          for (size_t l = 0; l < b.size(); ++l)
            result[i * b.size() * a.size() * b.size() + j * a.size() * b.size() + k * b.size() + l] = a[i] * b[j];

    return result;
  }
} // namespace tensor