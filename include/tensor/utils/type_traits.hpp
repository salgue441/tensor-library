#pragma once

#include <type_traits>

namespace tensor
{
  template <typename T>
  struct is_tensor_expression : std::false_type
  {
  };

  template <typename T>
  static constexpr bool is_tensor_expression_v = is_tensor_expression<T>::value;

  template <typename T>
  struct tensor_value_type
  {
    using type = typename T::value_type;
  };

  template <typename T>
  using tensor_value_type_t = typename tensor_value_type<T>::type;
} // namespace tensor