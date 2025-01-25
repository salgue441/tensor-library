#pragma once

#include "../core/tensor_base.hpp"
#include <limits>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace tensor
{
  template <typename T>
  class ReductionOps
  {
  public:
    /**
     * @brief Compute the sum of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T Sum of the tensor
     */
    template <size_t N>
    static T sum(const Tensor<T, N> &tensor)
    {
      return std::accumulate(tensor.begin(), tensor.end(), T(0));
    }

    /**
     * @brief Compute the mean of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T Mean of the tensor
     */
    template <size_t N>
    static T mean(const Tensor<T, N> &tensor)
    {
      return sum(tensor) / tensor.size();
    }

    /**
     * @brief Compute the minimum element of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T Minimum element of the tensor
     */
    template <size_t N>
    static T min(const Tensor<T, N> &tensor)
    {
      return *std::min_element(tensor.begin(), tensor.end());
    }

    /**
     * @brief Compute the maximum element of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T Maximum element of the tensor
     */
    template <size_t N>
    static T max(const Tensor<T, N> &tensor)
    {
      return *std::max_element(tensor.begin(), tensor.end());
    }

    /**
     * @brief Compute the argmin of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return size_t Index of the minimum element
     */
    template <size_t N>
    static size_t argmin(const Tensor<T, N> &tensor)
    {
      return std::distance(tensor.begin(), std::min_element(tensor.begin(), tensor.end()));
    }

    /**
     * @brief Compute the argmax of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return size_t Index of the maximum element
     */
    template <size_t N>
    static size_t argmax(const Tensor<T, N> &tensor)
    {
      return std::distance(tensor.begin(), std::max_element(tensor.begin(), tensor.end()));
    }

    /**
     * @brief Compute the L1 norm of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T L1 norm of the tensor
     */
    template <size_t N>
    static T l1_norm(const Tensor<T, N> &tensor)
    {
      return std::accumulate(tensor.begin(), tensor.end(), T(0), [](T a, T b)
                             { return a + std::abs(b); });
    }

    /**
     * @brief Compute the L2 norm of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T L2 norm of the tensor
     */
    template <size_t N>
    static T l2_norm(const Tensor<T, N> &tensor)
    {
      return std::sqrt(std::inner_product(tensor.begin(), tensor.end(), tensor.begin(), T(0)));
    }

    /**
     * @brief Compute the infinity norm of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T Infinity norm of the tensor
     */
    template <size_t N>
    static T infinity_norm(const Tensor<T, N> &tensor)
    {
      return *std::max_element(tensor.begin(), tensor.end(), [](T a, T b)
                               { return std::abs(a) < std::abs(b); });
    }

    /**
     * @brief Compute the Frobenius norm of the tensor
     *
     * @tparam N Number of dimensions
     * @param tensor Tensor
     * @return T Frobenius norm of the tensor
     */
    template <size_t N>
    static T frobenius_norm(const Tensor<T, N> &tensor)
    {
      return std::sqrt(std::inner_product(tensor.begin(), tensor.end(), tensor.begin(), T(0), std::plus<T>(), [](T a, T b)
                                          { return a * b; }));
    }

    /**
     * @brief Compute the mean squared error between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return T Mean squared error between a and b
     */
    template <size_t N>
    static T mean_squared_error(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                { return (a - b) * (a - b); }) /
             a.size();
    }

    /**
     * @brief Compute the cross entropy loss between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return T Cross entropy loss between a and b
     */
    template <size_t N>
    static T cross_entropy_loss(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return -std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                 { return a * std::log(b); });
    }

    /**
     * @brief Compute the KL divergence between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return T KL divergence between a and b
     */
    template <size_t N>
    static T kl_divergence(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                { return a * std::log(a / b); });
    }

    /**
     * @brief Compute the cosine similarity between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return T Cosine similarity between a and b
     */
    template <size_t N>
    static T cosine_similarity(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                { return a * b; }) /
             (l2_norm(a) * l2_norm(b));
    }

    /**
     * @brief Compute the Jaccard similarity between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return T Jaccard similarity between a and b
     */
    template <size_t N>
    static T jaccard_similarity(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                { return std::min(a, b); }) /
             std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                { return std::max(a, b); });
    }

    /**
     * @brief Compute the Hamming distance between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return size_t Hamming distance between a and b
     */
    template <size_t N>
    static size_t hamming_distance(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return std::inner_product(a.begin(), a.end(), b.begin(), size_t(0), std::plus<size_t>(), [](T a, T b)
                                { return a != b; });
    }

    /**
     * @brief Compute the Manhattan distance between two tensors
     *
     * @tparam N Number of dimensions
     * @param a Tensor a
     * @param b Tensor b
     * @return T Manhattan distance between a and b
     */
    template <size_t N>
    static T manhattan_distance(const Tensor<T, N> &a, const Tensor<T, N> &b)
    {
      return std::inner_product(a.begin(), a.end(), b.begin(), T(0), std::plus<T>(), [](T a, T b)
                                { return std::abs(a - b); });
    }
  };
}