#pragma once

#include "../utils/concepts.hpp"
#include "../utils/type_traits.hpp"
#include "expression.hpp"
#include "storage.hpp"

#include <memory>
#include <array>
#include <numeric>
#include <algorithm>

namespace tensor
{
  /**
   * @brief Tensor base class for N-dimensional tensors
   *
   * @tparam T Numeric type of the tensor
   * @tparam N Number of dimensions
   */
  template <NumericType T, size_t N>
  class Tensor : public TensorExpression<Tensor<T, N>, T>
  {
  public:
    using StorageType = TensorStorage<T>;
    using ShapeType = std::array<size_t, N>;
    using value_type = T;
    static constexpr size_t dimensions = N;

    /**
     * @brief Construct a new Tensor object
     *
     * @param shape Shape of the tensor
     */
    explicit Tensor(const ShapeType &shape)
        : shape_(shape), storage_(std::make_shared<StorageType>(compute_size(shape))) {}

    /**
     * @brief Construct a new Tensor object
     *
     * @tparam E Expression type
     * @param expr Expression to evaluate
     */
    template <typename E>
    Tensor(const TensorExpression<E, T> &expr)
        : shape_(expr.derived().shape()), storage_(std::make_shared<StorageType>(expr.size()))
    {
      evaluate(expr);
    }

    /**
     * @brief Operator [] to access elements of the tensor
     *
     * @param i Index of the element
     * @return T& Reference to the element
     */
    T &operator[](size_t i) { return (*storage_)[i]; }

    /**
     * @brief Operator [] to access elements of the tensor
     *
     * @param i Index of the element
     * @return const T& Reference to the element
     */
    const T &operator[](size_t i) const { return (*storage_)[i]; }

    /**
     * @brief Get the shape of the tensor
     *
     * @return const ShapeType& Shape of the tensor
     */
    const ShapeType &shape() const { return shape_; }

    /**
     * @brief Get the size of the tensor
     *
     * @return size_t Size of the tensor
     */
    size_t size() const { return storage_->size(); }

    /**
     * @brief Get the begin iterator of the tensor
     *
     * @return iterator Begin iterator
     */
    template <typename E>
    Tensor &operator=(const TensorExpression<E, T> &expr)
    {
      evaluate(expr);
      return *this;
    }

  private:
    ShapeType shape_;
    std::shared_ptr<StorageType> storage_;

    /**
     * @brief Compute the size of the tensor
     *
     * @param shape Shape of the tensor
     * @return size_t Size of the tensor
     */
    static size_t compute_size(const ShapeType &shape)
    {
      return std::accumulate(shape.begin(), shape.end(),
                             1UL, std::multiplies<size_t>());
    }

    /**
     * @brief Evaluate the expression
     *
     * @tparam E Expression type
     * @param expr Expression to evaluate
     */
    template <typename E>
    void evaluate(const TensorExpression<E, T> &expr)
    {
      const E &e = expr.derived();
      if (e.size() != size())
      {
        throw DimensionMismatch("Expression size mismatch in assignment");
      }

      std::copy(e.begin(), e.end(), storage_->begin());
    }
  };

} // namespace tensor