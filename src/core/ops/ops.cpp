#include <tensor/ops/ops.hpp>
// #include <tensor/core/tensor.hpp>
#include <cmath>

namespace tensor
{
  namespace ops
  {
    /**
     * @brief Validate that the shapes of two tensors match
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @throw ShapeError if the shapes do not match
     *
     * @version 1.0.0
     * @note This function is not exposed to the public API
     */
    template <typename T>
    void TensorOps<T>::validate_shapes_match(const Tensor<T> &lhs,
                                             const Tensor<T> &rhs)
    {
      if (lhs.shape() != rhs.shape())
        throw ShapeError("Tensor shapes do not match" +
                         std::to_string(lhs.shape()) + " != " +
                         std::to_string(rhs.shape()));
    };

    /**
     * @brief Validates matrix multiplication shapes
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @throw ShapeError if the shapes are invalid
     *
     * @version 1.0.0
     * @note This function is not exposed to the public API
     */
    template <typename T>
    void TensorOps<T>::validate_matmul_shapes(const Tensor<T> &lhs,
                                              const Tensor<T> &rhs)
    {
      if (lhs.shape().size() != 2 || rhs.shape().size() != 2)
        throw ShapeError("Matrix multiplication requires 2D tensors");

      if (lhs.shape()[1] != rhs.shape()[0])
        throw ShapeError("Matrix dimensions do not match for multiplication" +
                         std::to_string(lhs.shape()) + " != " +
                         std::to_string(rhs.shape()));
    };

    /**
     * @brief Add two tensors element-wise
     *
     * Enables SIMD if available
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @return Tensor<T> Result of the addition
     *
     * @version 1.0.0
     */
    template <typename T>
    Tensor<T> TensorOps<T>::add(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
      validate_shapes_match(lhs, rhs);

      Tensor<T> result(lhs.shape(), lhs.device());
      const size_t size = lhs.size();

      if constexpr (std::is_same_v<T, float>)
      {
        size_t i = 0;
        for (; i + 8 <= size; i += 8)
        {
          __m256 va = _mm256_loadu_ps(&lhs[i]);
          __m256 vb = _mm256_loadu_ps(&rhs[i]);
          __m256 vr = _mm256_add_ps(va, vb);
          _mm256_storeu_ps(&result[i], vr);
        }

        for (; i < size; ++i)
          result[i] = lhs[i] + rhs[i];
      }
      else
      {
        for (size_t i = 0; i < size; ++i)
          result[i] = lhs[i] + rhs[i];
      }

      return result;
    }

    /**
     * @brief Subtract two tensors element-wise
     *
     * Enables SIMD if available
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @return Tensor<T> Result of the subtraction
     *
     * @version 1.0.0
     */
    template <typename T>
    Tensor<T> TensorOps<T>::subtract(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
      validate_shapes_match(lhs, rhs);

      Tensor<T> result(lhs.shape(), lhs.device());
      const size_t size = lhs.size();

      if constexpr (std::is_same_v<T, float>)
      {
        size_t i = 0;
        for (; i + 8 <= size; i += 8)
        {
          __m256 va = _mm256_loadu_ps(&lhs[i]);
          __m256 vb = _mm256_loadu_ps(&rhs[i]);
          __m256 vr = _mm256_sub_ps(va, vb);
          _mm256_storeu_ps(&result[i], vr);
        }

        for (; i < size; ++i)
          result[i] = lhs[i] - rhs[i];
      }
      else
      {
        for (size_t i = 0; i < size; ++i)
          result[i] = lhs[i] - rhs[i];
      }

      return result;
    }

    /**
     * @brief Multiply two tensors element-wise
     *
     * Enables SIMD if available
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @return Tensor<T> Result of the multiplication
     *
     * @version 1.0.0
     */
    template <typename T>
    Tensor<T> TensorOps<T>::multiply(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
      validate_shapes_match(lhs, rhs);

      Tensor<T> result(lhs.shape(), lhs.device());
      const size_t size = lhs.size();

      if constexpr (std::is_same_v<T, float>)
      {
        size_t i = 0;
        for (; i + 8 <= size; i += 8)
        {
          __m256 va = _mm256_loadu_ps(&lhs[i]);
          __m256 vb = _mm256_loadu_ps(&rhs[i]);
          __m256 vr = _mm256_mul_ps(va, vb);
          _mm256_storeu_ps(&result[i], vr);
        }

        for (; i < size; ++i)
          result[i] = lhs[i] * rhs[i];
      }
      else
      {
        for (size_t i = 0; i < size; ++i)
          result[i] = lhs[i] * rhs[i];
      }

      return result;
    }

    /**
     * @brief Divide two tensors element-wise
     *
     * Enables SIMD if available
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @return Tensor<T> Result of the division
     *
     * @version 1.0.0
     */
    template <typename T>
    Tensor<T> TensorOps<T>::divide(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
      validate_shapes_match(lhs, rhs);

      Tensor<T> result(lhs.shape(), lhs.device());
      const size_t size = lhs.size();

      if constexpr (std::is_same_v<T, float>)
      {
        size_t i = 0;
        for (; i + 8 <= size; i += 8)
        {
          __m256 va = _mm256_loadu_ps(&lhs[i]);
          __m256 vb = _mm256_loadu_ps(&rhs[i]);
          __m256 vr = _mm256_div_ps(va, vb);
          _mm256_storeu_ps(&result[i], vr);
        }

        for (; i < size; ++i)
          result[i] = lhs[i] / rhs[i];
      }
      else
      {
        for (size_t i = 0; i < size; ++i)
          result[i] = lhs[i] / rhs[i];
      }

      return result;
    }

    /**
     * @brief Matmul operation for two tensors
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @return Tensor<T> Result of the matrix multiplication
     */
    template <typename T>
    Tensor<T> TensorOps<T>::matmul(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
      validate_matmul_shapes(lhs, rhs);

      const auto &shape_a = lhs.shape();
      const auto &shape_b = rhs.shape();
      Tensor<T> result({shape_a[0], shape_b[1]}, lhs.device());
      constexpr size_t BLOCK_SIZE = 32;

#pragma omp parallel for collapse(2)
      for (size_t i = 0; i < shape_a[0]; i += BLOCK_SIZE)
      {
        for (size_t j = 0; j < shape_b[1]; j += BLOCK_SIZE)
        {
          for (size_t k = 0; k < shape_a[1]; k += BLOCK_SIZE)
          {
            size_t i_end = std::min(i + BLOCK_SIZE, shape_a[0]);
            size_t j_end = std::min(j + BLOCK_SIZE, shape_b[1]);
            size_t k_end = std::min(k + BLOCK_SIZE, shape_a[1]);

            for (size_t ii = i; ii < i_end; ++ii)
            {
              for (size_t jj = j; jj < j_end; ++jj)
              {
                T sum = T(0);
                for (size_t kk = k; kk < k_end; ++kk)
                  sum += lhs[ii * shape_a[1] + kk] *
                         rhs[kk * shape_b[1] + jj];

                result[ii * shape_b[1] + jj] += sum;
              }
            }
          }
        }
      }

      return result;
    }

    /**
     * @brief Transpose a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @return Tensor<T> Transposed tensor
     */
    template <typename T>
    Tensor<T> TensorOps<T>::transpose(const Tensor<T> &tensor)
    {
      const auto &shape = tensor.shape();
      Tensor<T> result({shape[1], shape[0]}, tensor.device());

      for (size_t i = 0; i < shape[0]; ++i)
      {
        for (size_t j = 0; j < shape[1]; ++j)
          result[j * shape[0] + i] = tensor[i * shape[1] + j];
      }

      return result;
    }

    /**
     * @brief Dot product of two tensors
     *
     * @tparam T Tensor data type
     * @param lhs Left-hand side tensor
     * @param rhs Right-hand side tensor
     * @return Tensor<T> Result of the dot product
     */
    template <typename T>
    Tensor<T> TensorOps<T>::dot(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
      validate_shapes_match(lhs, rhs);

      Tensor<T> result({1}, lhs.device());
      const size_t size = lhs.size();

      for (size_t i = 0; i < size; ++i)
        result[0] += lhs[i] * rhs[i];

      return result;
    }

    /**
     * @brief Compute the absolute value of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @return Tensor<T> Result of the absolute value
     */
    template <typename T>
    Tensor<T> TensorOps<T>::abs(const Tensor<T> &tensor)
    {
      Tensor<T> result(tensor.shape(), tensor.device());
      const size_t size = tensor.size();

      for (size_t i = 0; i < size; ++i)
        result[i] = std::abs(tensor[i]);

      return result;
    }

    /**
     * @brief Compute the exponential of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @return Tensor<T> Result of the exponential
     */
    template <typename T>
    Tensor<T> TensorOps<T>::exp(const Tensor<T> &tensor)
    {
      Tensor<T> result(tensor.shape(), tensor.device());
      const size_t size = tensor.size();

      for (size_t i = 0; i < size; ++i)
        result[i] = std::exp(tensor[i]);

      return result;
    }

    /**
     * @brief Compute the natural logarithm of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @return Tensor<T> Result of the logarithm
     */
    template <typename T>
    Tensor<T> TensorOps<T>::log(const Tensor<T> &tensor)
    {
      Tensor<T> result(tensor.shape(), tensor.device());
      const size_t size = tensor.size();

      for (size_t i = 0; i < size; ++i)
        result[i] = std::log(tensor[i]);

      return result;
    }

    /**
     * @brief Compute the square root of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @return Tensor<T> Result of the square root
     */
    template <typename T>
    Tensor<T> TensorOps<T>::sqrt(const Tensor<T> &tensor)
    {
      Tensor<T> result(tensor.shape(), tensor.device());
      const size_t size = tensor.size();

      for (size_t i = 0; i < size; ++i)
        result[i] = std::sqrt(tensor[i]);

      return result;
    }

    /**
     * @brief Compute the power of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @param exponent Exponent value
     * @return Tensor<T> Result of the power operation
     */
    template <typename T>
    Tensor<T> TensorOps<T>::pow(const Tensor<T> &tensor, T exponent)
    {
      Tensor<T> result(tensor.shape(), tensor.device());
      const size_t size = tensor.size();

      for (size_t i = 0; i < size; ++i)
        result[i] = std::pow(tensor[i], exponent);

      return result;
    }

    /**
     * @brief Compute the sum of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @param axis Axis to compute the sum
     * @return T Result of the sum
     */
    template <typename T>
    T TensorOps<T>::sum(const Tensor<T> &tensor, int axis)
    {
      const auto &shape = tensor.shape();
      const size_t size = tensor.size();

      if (axis == -1)
      {
        T result = T(0);
        for (size_t i = 0; i < size; ++i)
          result += tensor[i];

        return result;
      }

      if (axis < 0 || axis >= shape.size())
        throw AxisError("Invalid axis for sum operation");

      size_t axis_size = shape[axis];
      T result = T(0);

      if (axis == 0)
      {
        for (size_t i = 0; i < size; i += axis_size)
        {
          for (size_t j = 0; j < axis_size; ++j)
            result += tensor[i + j];
        }
      }
      else
      {
        size_t stride = 1;
        for (int i = axis + 1; i < shape.size(); ++i)
          stride *= shape[i];

        for (size_t i = 0; i < size; i += axis_size * stride)
        {
          for (size_t j = 0; j < axis_size; ++j)
          {
            for (size_t k = 0; k < stride; ++k)
              result += tensor[i + j * stride + k];
          }
        }
      }

      return result;
    }

    /**
     * @brief Compute the mean of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @param axis Axis to compute the mean
     * @return T Result of the mean
     */
    template <typename T>
    T TensorOps<T>::mean(const Tensor<T> &tensor, int axis)
    {
      T sum = sum(tensor, axis);
      const auto &shape = tensor.shape();
      size_t size = tensor.size();

      if (axis == -1)
        return sum / size;

      if (axis < 0 || axis >= shape.size())
        throw AxisError("Invalid axis for mean operation");

      size_t axis_size = shape[axis];
      size_t count = size / axis_size;

      return sum / count;
    }

    /**
     * @brief Compute the maximum value of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @param axis Axis to compute the maximum
     * @return T Result of the maximum
     */
    template <typename T>
    T TensorOps<T>::max(const Tensor<T> &tensor, int axis)
    {
      const auto &shape = tensor.shape();
      const size_t size = tensor.size();

      if (axis == -1)
      {
        T result = tensor[0];
        for (size_t i = 1; i < size; ++i)
          result = std::max(result, tensor[i]);

        return result;
      }

      if (axis < 0 || axis >= shape.size())
        throw AxisError("Invalid axis for max operation");

      size_t axis_size = shape[axis];
      T result = tensor[0];

      if (axis == 0)
      {
        for (size_t i = 0; i < size; i += axis_size)
        {
          for (size_t j = 0; j < axis_size; ++j)
            result = std::max(result, tensor[i + j]);
        }
      }
      else
      {
        size_t stride = 1;
        for (int i = axis + 1; i < shape.size(); ++i)
          stride *= shape[i];

        for (size_t i = 0; i < size; i += axis_size * stride)
        {
          for (size_t j = 0; j < axis_size; ++j)
          {
            for (size_t k = 0; k < stride; ++k)
              result = std::max(result, tensor[i + j * stride + k]);
          }
        }
      }

      return result;
    }

    /**
     * @brief Compute the minimum value of a tensor
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @param axis Axis to compute the minimum
     * @return T Result of the minimum
     */
    template <typename T>
    T TensorOps<T>::min(const Tensor<T> &tensor, int axis)
    {
      const auto &shape = tensor.shape();
      const size_t size = tensor.size();

      if (axis == -1)
      {
        T result = tensor[0];
        for (size_t i = 1; i < size; ++i)
          result = std::min(result, tensor[i]);

        return result;
      }

      if (axis < 0 || axis >= shape.size())
        throw AxisError("Invalid axis for min operation");

      size_t axis_size = shape[axis];
      T result = tensor[0];

      if (axis == 0)
      {
        for (size_t i = 0; i < size; i += axis_size)
        {
          for (size_t j = 0; j < axis_size; ++j)
            result = std::min(result, tensor[i + j]);
        }
      }
      else
      {
        size_t stride = 1;
        for (int i = axis + 1; i < shape.size(); ++i)
          stride *= shape[i];

        for (size_t i = 0; i < size; i += axis_size * stride)
        {
          for (size_t j = 0; j < axis_size; ++j)
          {
            for (size_t k = 0; k < stride; ++k)
              result = std::min(result, tensor[i + j * stride + k]);
          }
        }
      }

      return result;
    }

    /**
     * @brief Broadcast a tensor to a new shape
     *
     * @tparam T Tensor data type
     * @param tensor Input tensor
     * @param shape New shape
     * @return Tensor<T> Result of the broadcast
     */
    template <typename T>
    Tensor<T> TensorOps<T>::broadcast_to(const Tensor<T> &tensor,
                                         const std::vector<size_t> &shape)
    {
      if (tensor.shape() == shape)
        return tensor;

      if (!shapes_are_broadcastable(tensor.shape(), shape))
        throw ShapeError("Shapes are not broadcastable");

      Tensor<T> result(shape, tensor.device());
      const size_t size = result.size();
      const size_t ndim = shape.size();
      std::vector<size_t> strides(ndim);
      std::vector<size_t> indices(ndim, 0);

      for (size_t i = 0; i < ndim; ++i)
      {
        size_t stride = 1;
        for (size_t j = i + 1; j < ndim; ++j)
          stride *= shape[j];

        strides[i] = stride;
      }

      for (size_t i = 0; i < size; ++i)
      {
        size_t index = 0;
        for (size_t j = 0; j < ndim; ++j)
          index += indices[j] * strides[j];

        result[i] = tensor[index];
        for (int j = ndim - 1; j >= 0; --j)
        {
          if (indices[j] + 1 < shape[j])
          {
            indices[j]++;
            break;
          }
          else
          {
            indices[j] = 0;
          }
        }
      }

      return result;
    }

    /**
     * @brief Computes the broadcast shape of two tensors
     *
     * @tparam T Tensor data type
     * @param lhs_shape Left-hand side shape
     * @param rhs_shape Right-hand side shape
     * @return std::vector<size_t> Resulting broadcast shape
     */
    template <typename T>
    std::vector<size_t> TensorOps<T>::compute_broadcast_shape(
        const std::vector<size_t> &lhs_shape,
        const std::vector<size_t> &rhs_shape)
    {
      if (shapes_are_broadcastable(lhs_shape, rhs_shape))
        throw ShapeError("Shapes are not broadcastable");

      size_t ndim = std::max(lhs_shape.size(), rhs_shape.size());
      std::vector<size_t> result(ndim);
      auto padded_lhs = lhs_shape;
      auto padded_rhs = rhs_shape;

      while (padded_lhs.size() < ndim)
        padded_lhs.insert(padded_lhs.begin(), 1);

      while (padded_rhs.size() < ndim)
        padded_rhs.insert(padded_rhs.begin(), 1);

      for (size_t i = 0; i < ndim; ++i)
        result[i] = std::max(padded_lhs[i], padded_rhs[i]);

      return result;
    }

    /**
     * @brief Check if two shapes are broadcastable
     *
     * @tparam T Tensor data type
     * @param lhs_shape Left-hand side shape
     * @param rhs_shape Right-hand side shape
     * @return true if the shapes are broadcastable
     * @return false if the shapes are not broadcastable
     */
    template <typename T>
    bool TensorOps<T>::shapes_are_broadcastable(
        const std::vector<size_t> &lhs_shape,
        const std::vector<size_t> &rhs_shape)
    {
      size_t ndim = std::max(lhs_shape.size(), rhs_shape.size());
      auto padded_lhs = lhs_shape;
      auto padded_rhs = rhs_shape;

      while (padded_lhs.size() < ndim)
        padded_lhs.insert(padded_lhs.begin(), 1);

      while (padded_rhs.size() < ndim)
        padded_rhs.insert(padded_rhs.begin(), 1);

      for (size_t i = 0; i < ndim; ++i)
      {
        if (padded_lhs[i] != padded_rhs[i] && padded_lhs[i] != 1 &&
            padded_rhs[i] != 1)
          return false;
      }

      return true;
    }
  } // namespace ops
} // namespace tensor