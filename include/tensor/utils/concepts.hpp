#pragma once

#include <concepts>
#include <type_traits>

namespace tensor
{
  template <typename T>
  concept NumericType = std::is_arithmetic_v<T>;

  template <typename T>
  concept TensorExpressionConcept = requires(T t) {
    typename T::value_type;
    { t.derived() } -> std::same_as<const T &>;
    { t.size() } -> std::same_as<size_t>;
    { t[0] } -> std::convertible_to<typename T::value_type>;
  };
} // namespace tensor