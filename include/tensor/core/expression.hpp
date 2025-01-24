#pragma once
#include "../utils/concepts.hpp"
#include "../utils/exceptions.hpp"
#include <iterator>

namespace tensor
{
  /**
   * @brief Tensor Expression base class for all tensor expressions
   *
   * @tparam E Expression type
   * @tparam T Numeric type
   */
  template <typename E, NumericType T>
  class TensorExpression
  {
  public:
    using value_type = T;
    using iterator = const T *;
    using const_iterator = const T *;

    /**
     * @brief Derived expression object
     *
     * @return const E& Expression object
     */
    const E &derived() const { return static_cast<const E &>(*this); }

    /**
     * @brief Operator overload [[] to access element at index i
     *
     * @param i Index
     * @return T Element at index i
     */
    T operator[](size_t i) const { return derived()[i]; }

    /**
     * @brief Get the size of the expression
     *
     * @return size_t Size of the expression
     */
    size_t size() const { return derived().size(); }

    /**
     * @brief Get the begin iterator
     *
     * @return iterator Begin iterator
     */
    auto begin() const { return derived().begin(); }

    /**
     * @brief Get the end iterator
     *
     * @return iterator End iterator
     */
    auto end() const { return derived().end(); }
  };

  /**
   * @brief Binary Expression class for binary operations
   *
   * @tparam BinaryOp Binary operation
   * @tparam E1 Expression 1
   * @tparam E2 Expression 2
   * @tparam T Numeric type
   */
  template <typename BinaryOp, typename E1, typename E2, typename T>
  class BinaryExpression : public TensorExpression<BinaryExpression<BinaryOp, E1, E2, T>, T>
  {
    const E1 &lhs_;
    const E2 &rhs_;

  public:
    /**
     * @brief Construct a new Binary Expression object
     *
     * @param lhs Expression 1
     * @param rhs Expression 2
     *
     * @throw DimensionMismatch if the dimensions of the expressions
     *        do not match
     */
    BinaryExpression(const E1 &lhs, const E2 &rhs) : lhs_(lhs), rhs_(rhs)
    {
      if (lhs_.size() != rhs_.size())
      {
        throw DimensionMismatch("Binary operation dimension mismatch");
      }
    }

    /**
     * @brief Operator overload [] to access element at index i
     *
     * @param i Index
     * @return T Element at index i
     */
    T operator[](size_t i) const { return BinaryOp::apply(lhs_[i], rhs_[i]); }

    /**
     * @brief Get the size of the expression
     *
     * @return size_t Size of the expression
     */
    size_t size() const { return lhs_.size(); }

    /**
     * @brief Get the begin iterator
     *
     * @return iterator Begin iterator
     */
    auto begin() const { return iterator(this, 0); }

    /**
     * @brief Get the end iterator
     *
     * @return iterator End iterator
     */
    auto end() const { return iterator(this, size()); }
  };

  /**
   * @brief Unary Expression class for unary operations
   *
   * @tparam UnaryOp Unary operation
   * @tparam E Expression
   * @tparam T Numeric type
   */
  template <typename UnaryOp, typename E, typename T>
  class UnaryExpression : public TensorExpression<UnaryExpression<UnaryOp, E, T>, T>
  {
    const E &expr_;

  public:
    /**
     * @brief Construct a new Unary Expression object
     *
     * @param expr Expression
     */
    explicit UnaryExpression(const E &expr) : expr_(expr) {}

    /**
     * @brief Operator overload [] to access element at index i
     *
     * @param i Index
     * @return T Element at index i
     */
    T operator[](size_t i) const { return UnaryOp::apply(expr_[i]); }

    /**
     * @brief Get the size of the expression
     *
     * @return size_t Size of the expression
     */
    size_t size() const { return expr_.size(); }

    /**
     * @brief Get the begin iterator
     *
     * @return iterator Begin iterator
     */
    auto begin() const { return iterator(this, 0); }

    /**
     * @brief Get the end iterator
     *
     * @return iterator End iterator
     */
    auto end() const { return iterator(this, size()); }
  };

} // namespace tensor