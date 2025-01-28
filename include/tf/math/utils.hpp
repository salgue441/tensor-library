#pragma once

#include <tf/core/types.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tf
{
  namespace math
  {
    /**
     * @brief Clamp value to the range [min, max]
     *
     * @tparam T Type of the array
     * @param data Pointer to the array
     * @param min Minimum value
     * @param max Maximum value
     * @return T Clamped value in the range [min, max]
     * @version 1.0.0
     */
    template <typename T>
    inline T clamp(T value, T min, T max)
    {
      return std::max(min, std::min(value, max));
    }

    /**
     * @brief Linear interpolation between two values
     *
     * @tparam T Type of the array
     * @param a First value
     * @param b Second value
     * @param t Interpolation factor
     * @return T Interpolated value
     * @version 1.0.0
     */
    template <typename T>
    inline T lerp(T a, T b, T t)
    {
      return a + t * (b - a);
    }

    // Activation functions
    /**
     * @brief Sigmoid activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @return T Sigmoid value of the input in the range [0, 1]
     * @version 1.0.0
     */
    template <typename T>
    inline T sigmoid(T x)
    {
      return T{1} / (T{1} + std::exp(-x));
    }

    /**
     * @brief Hyperbolic tangent activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @return T Hyperbolic tangent value of the input in the range [-1, 1]
     * @version 1.0.0
     */
    template <typename T>
    inline T tanh(T x)
    {
      return std::tanh(x);
    }

    /**
     * @brief Rectified linear unit (ReLU) activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @return T ReLU value of the input
     * @version 1.0.0
     */
    template <typename T>
    inline T relu(T x)
    {
      return std::max(T{0}, x);
    }

    /**
     * @brief Leaky ReLU activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @param alpha Leaky factor
     * @return T Leaky ReLU value of the input
     * @version 1.0.0
     */
    template <typename T>
    inline T leaky_relu(T x, T alpha = T{0.01})
    {
      return x > T{0} ? x : alpha * x;
    }

    /**
     * @brief Softmax activation function
     *
     * @tparam T Type of the array
     * @param x Input array
     * @return T Softmax value of the input
     * @version 1.0.0
     */
    template <typename T>
    inline T softmax(T x)
    {
      return std::exp(x) / std::exp(x).sum();
    }

    // Derivative of activation functions
    /**
     * @brief Derivative of the sigmoid activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @return T Derivative of the sigmoid function
     * @version 1.0.0
     */
    template <typename T>
    inline T sigmoid_derivative(T x)
    {
      return sigmoid(x) * (T{1} - sigmoid(x));
    }

    /**
     * @brief Derivative of the hyperbolic tangent activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @return T Derivative of the hyperbolic tangent function
     * @version 1.0.0
     */
    template <typename T>
    inline T tanh_derivative(T x)
    {
      return T{1} - std::pow(tanh(x), T{2});
    }

    /**
     * @brief Derivative of the ReLU activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @return T Derivative of the ReLU function
     * @version 1.0.0
     */
    template <typename T>
    inline T relu_derivative(T x)
    {
      return x > T{0} ? T{1} : T{0};
    }

    /**
     * @brief Derivative of the Leaky ReLU activation function
     *
     * @tparam T Type of the array
     * @param x Input value
     * @param alpha Leaky factor
     * @return T Derivative of the Leaky ReLU function
     * @version 1.0.0
     */
    template <typename T>
    inline T leaky_relu_derivative(T x, T alpha = T{0.01})
    {
      return x > T{0} ? T{1} : alpha;
    }

    // Statistic functions
    /**
     * @brief Calculates the mean of the array
     *
     * @tparam Iterator Type of the iterator
     * @param begin Begin iterator
     * @param end End iterator
     * @return auto Mean value of the array
     * @version 1.0.0
     */
    template <typename Iterator>
    auto mean(Iterator begin, Iterator end)
    {
      using T = typename std::iterator_traits<Iterator>::value_type;
      auto size = std::distance(begin, end);

      if (size == 0)
        return T{0};

      return std::accumulate(begin, end, T{0}) / static_cast<T>(size);
    }

    /**
     * @brief Calculates the variance of the array
     *
     * @tparam Iterator Type of the iterator
     * @param begin Begin iterator
     * @param end End iterator
     * @return auto Variance value of the array
     * @version 1.0.0
     */
    template <typename Iterator>
    auto variance(Iterator begin, Iterator end)
    {
      using T = typename std::iterator_traits<Iterator>::value_type;
      auto size = std::distance(begin, end);

      if (size <= 1)
        return T{0};

      T m = mean(begin, end);
      T sum = std::accumulate(begin, end, T{0}, [m](T a, T b)
                              { return a + (b - m) * (b - m); });

      return sum / static_cast<T>(size - 1);
    }

    /**
     * @brief Calculates the standard deviation of the array
     *
     * @tparam Iterator Type of the iterator
     * @param begin Begin iterator
     * @param end End iterator
     * @return auto Standard deviation value of the array
     * @version 1.0.0
     */
    template <typename Iterator>
    auto stddev(Iterator begin, Iterator end)
    {
      return std::sqrt(variance(begin, end));
    }

    /**
     * @brief Calculates the covariance of two arrays
     *
     * @tparam Iterator1 Type of the first iterator
     * @tparam Iterator2 Type of the second iterator
     * @param begin1 Begin iterator of the first array
     * @param end1 End iterator of the first array
     * @param begin2 Begin iterator of the second array
     * @param end2 End iterator of the second array
     * @return auto Covariance value of the two arrays
     * @version 1.0.0
     */
    template <typename Iterator1, typename Iterator2>
    auto covariance(Iterator1 begin1, Iterator1 end1,
                    Iterator2 begin2, Iterator2 end2)
    {
      using T1 = typename std::iterator_traits<Iterator1>::value_type;
      using T2 = typename std::iterator_traits<Iterator2>::value_type;

      auto size1 = std::distance(begin1, end1);
      auto size2 = std::distance(begin2, end2);

      if (size1 != size2 || size1 <= 1)
        return T1{0};

      T1 m1 = mean(begin1, end1);
      T1 m2 = mean(begin2, end2);

      T1 sum = std::inner_product(begin1, end1, begin2, T1{0}, std::plus<T1>(),
                                  [m1, m2](T1 a, T2 b)
                                  { return (a - m1) * (b - m2); });

      return sum / static_cast<T1>(size1 - 1);
    }

    /**
     * @brief Calculates the correlation coefficient of two arrays
     *
     * @tparam Iterator1 Type of the first iterator
     * @tparam Iterator2 Type of the second iterator
     * @param begin1 Begin iterator of the first array
     * @param end1 End iterator of the first array
     * @param begin2 Begin iterator of the second array
     * @return auto Correlation coefficient value of the two arrays
     * @version 1.0.0
     */
    template <typename Iterator1, typename Iterator2>
    auto correlation(Iterator1 begin1, Iterator1 end1,
                     Iterator2 begin2, Iterator2 end2)
    {
      using T1 = typename std::iterator_traits<Iterator1>::value_type;
      using T2 = typename std::iterator_traits<Iterator2>::value_type;

      auto size1 = std::distance(begin1, end1);
      auto size2 = std::distance(begin2, end2);

      if (size1 != size2 || size1 <= 1)
        return T1{0};

      T1 cov = covariance(begin1, end1, begin2, end2);
      T1 std1 = stddev(begin1, end1);
      T1 std2 = stddev(begin2, end2);

      return cov / (std1 * std2);
    }
  } // namespace math
} // namespace tf