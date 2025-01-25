#pragma once

#include "../core/tensor_base.hpp"
#include <cmath>

namespace tensor
{
  /**
   * @brief Unary operations for tensor objects
   *
   * @tparam T Numeric type of the tensor
   */
  template <typename T>
  class UnaryOps
  {
  public:
    /**
     * @brief Exponential operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Exponential of the tensor
     */
    static Tensor<T, 2> exp(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::exp(tensor[i]);

      return result;
    }

    /**
     * @brief Logarithm operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Logarithm of the tensor
     */
    static Tensor<T, 2> log(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
      {
        if (tensor[i] <= 0)
          throw std::runtime_error("Logarithm of non-positive number");

        result[i] = std::log(tensor[i]);
      }

      return result;
    }

    /**
     * @brief Sine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Sine of the tensor
     */
    static Tensor<T, 2> sin(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::sin(tensor[i]);

      return result;
    }

    /**
     * @brief Cosine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Cosine of the tensor
     */
    static Tensor<T, 2> cos(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::cos(tensor[i]);

      return result;
    }

    /**
     * @brief Tangent operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Tangent of the tensor
     */
    static Tensor<T, 2> tan(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::tan(tensor[i]);

      return result;
    }

    /**
     * @brief Arcsine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Arcsine of the tensor
     */
    static Tensor<T, 2> asin(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
      {
        if (tensor[i] < -1 || tensor[i] > 1)
          throw std::runtime_error("Arcsine of out-of-range number");

        result[i] = std::asin(tensor[i]);
      }

      return result;
    }

    /**
     * @brief Arccosine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Arccosine of the tensor
     */
    static Tensor<T, 2> acos(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
      {
        if (tensor[i] < -1 || tensor[i] > 1)
          throw std::runtime_error("Arccosine of out-of-range number");

        result[i] = std::acos(tensor[i]);
      }

      return result;
    }

    /**
     * @brief Arctangent operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Arctangent of the tensor
     */
    static Tensor<T, 2> atan(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::atan(tensor[i]);

      return result;
    }

    /**
     * @brief Hyperbolic sine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Hyperbolic sine of the tensor
     */
    static Tensor<T, 2> sinh(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::sinh(tensor[i]);

      return result;
    }

    /**
     * @brief Hyperbolic cosine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Hyperbolic cosine of the tensor
     */
    static Tensor<T, 2> cosh(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::cosh(tensor[i]);

      return result;
    }

    /**
     * @brief Hyperbolic tangent operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Hyperbolic tangent of the tensor
     */
    static Tensor<T, 2> tanh(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::tanh(tensor[i]);

      return result;
    }

    /**
     * @brief Inverse hyperbolic sine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Inverse hyperbolic sine of the tensor
     */
    static Tensor<T, 2> asinh(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::asinh(tensor[i]);

      return result;
    }

    /**
     * @brief Inverse hyperbolic cosine operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Inverse hyperbolic cosine of the tensor
     */
    static Tensor<T, 2> acosh(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
      {
        if (tensor[i] < 1)
          throw std::runtime_error("Inverse hyperbolic cosine of out-of-range number");

        result[i] = std::acosh(tensor[i]);
      }

      return result;
    }

    /**
     * @brief Inverse hyperbolic tangent operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Inverse hyperbolic tangent of the tensor
     */
    static Tensor<T, 2> atanh(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
      {
        if (tensor[i] <= -1 || tensor[i] >= 1)
          throw std::runtime_error("Inverse hyperbolic tangent of out-of-range number");

        result[i] = std::atanh(tensor[i]);
      }

      return result;
    }

    /**
     * @brief Square root operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Square root of the tensor
     */
    static Tensor<T, 2> sqrt(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
      {
        if (tensor[i] < 0)
          throw std::runtime_error("Square root of negative number");

        result[i] = std::sqrt(tensor[i]);
      }

      return result;
    }

    /**
     * @brief Absolute value operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Absolute value of the tensor
     */
    static Tensor<T, 2> abs(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::abs(tensor[i]);

      return result;
    }

    /**
     * @brief Floor operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Floor of the tensor
     */
    static Tensor<T, 2> floor(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::floor(tensor[i]);

      return result;
    }

    /**
     * @brief Ceil operator for tensor objects
     *
     * @tparam T Tensor type
     * @param tensor T Tensor oject
     * @return Tensor<T, 2> Ceil of the tensor
     */
    static Tensor<T, 2> ceil(const Tensor<T, 2> &tensor)
    {
      Tensor<T, 2> result(tensor.shape());
      for (size_t i = 0; i < tensor.size(); ++i)
        result[i] = std::ceil(tensor[i]);

      return result;
    }
  };

} // namespace tensor