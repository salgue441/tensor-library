#pragma once
#include "../utils/exceptions.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace tensor
{

  enum class ScalarType
  {
    Uint8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool
  };

  namespace dtype
  {
    /**
     * @brief String representation of the scalar type
     *
     * @param dtype Data type to convert
     * @return std::string_view String representation of the data type
     * @throw TypeError If the data type is unknown
     *
     * @version 1.0.0
     */
    inline std::string_view to_string(ScalarType dtype)
    {
      switch (dtype)
      {
      case ScalarType::Uint8:
        return "uint8";

      case ScalarType::Int8:
        return "int8";

      case ScalarType::Int16:
        return "int16";

      case ScalarType::Int32:
        return "int32";

      case ScalarType::Int64:
        return "int64";

      case ScalarType::Float32:
        return "float32";

      case ScalarType::Float64:
        return "float64";

      case ScalarType::Bool:
        return "bool";

      default:
        throw TypeError("Unknown dtype");
      }
    }

    /**
     * @brief Element size of the scalar type
     *
     * @param dtype Data type to get the size of
     * @return size_t Size of the data type in bytes
     * @throw TypeError If the data type is unknown
     *
     * @version 1.0.0
     */
    inline size_t element_size(ScalarType dtype)
    {
      switch (dtype)
      {
      case ScalarType::Uint8:
        return sizeof(uint8_t);

      case ScalarType::Int8:
        return sizeof(int8_t);

      case ScalarType::Int16:
        return sizeof(int16_t);

      case ScalarType::Int32:
        return sizeof(int32_t);

      case ScalarType::Int64:
        return sizeof(int64_t);

      case ScalarType::Float32:
        return sizeof(float);

      case ScalarType::Float64:
        return sizeof(double);

      case ScalarType::Bool:
        return sizeof(bool);

      default:
        throw TypeError("Unknown dtype");
      }
    }

    /**
     * @brief Check if the scalar type is a floating point type
     *
     * @param dtype Data type to check
     * @return true If the data type is a floating point type
     * @return false If the data type is not a floating point type
     *
     * @version 1.0.0
     */
    inline bool is_floating_point(ScalarType dtype)
    {
      return dtype == ScalarType::Float32 || dtype == ScalarType::Float64;
    }

    /**
     * @brief Check if the scalar type is an integral type
     *
     * @param dtype Data type to check
     * @return true If the data type is an integral type
     * @return false If the data type is not an integral type
     *
     * @version 1.0.0
     */
    inline bool is_integral(ScalarType dtype)
    {
      return dtype != ScalarType::Float32 &&
             dtype != ScalarType::Float64 &&
             dtype != ScalarType::Bool;
    }

    /**
     * @brief Promote two scalar types to a common type
     *
     * @param a First scalar type
     * @param b Second scalar type
     * @return ScalarType Common scalar type
     *
     * @version 1.0.0
     */
    inline ScalarType promote_types(ScalarType a, ScalarType b)
    {
      if (a == b)
        return a;

      if (is_floating_point(a) || is_floating_point(b))
        return ScalarType::Float64;

      if (a == ScalarType::Int64 || b == ScalarType::Int64)
        return ScalarType::Int64;

      return ScalarType::Int32;
    }

  } // namespace dtype

  // Type traits for mapping C++ types to ScalarType
  template <typename T>
  struct TypeToScalar;

  template <>
  struct TypeToScalar<uint8_t>
  {
    static constexpr ScalarType value = ScalarType::Uint8;
  };
  template <>
  struct TypeToScalar<int8_t>
  {
    static constexpr ScalarType value = ScalarType::Int8;
  };
  template <>
  struct TypeToScalar<int16_t>
  {
    static constexpr ScalarType value = ScalarType::Int16;
  };
  template <>
  struct TypeToScalar<int32_t>
  {
    static constexpr ScalarType value = ScalarType::Int32;
  };
  template <>
  struct TypeToScalar<int64_t>
  {
    static constexpr ScalarType value = ScalarType::Int64;
  };
  template <>
  struct TypeToScalar<float>
  {
    static constexpr ScalarType value = ScalarType::Float32;
  };
  template <>
  struct TypeToScalar<double>
  {
    static constexpr ScalarType value = ScalarType::Float64;
  };
  template <>
  struct TypeToScalar<bool>
  {
    static constexpr ScalarType value = ScalarType::Bool;
  };

  template <typename T>
  struct IsScalarType
  {
    static constexpr bool value = false;
  };

#define DEFINE_IS_SCALAR_TYPE(Type)     \
  template <>                           \
  struct IsScalarType<Type>             \
  {                                     \
    static constexpr bool value = true; \
  }

  DEFINE_IS_SCALAR_TYPE(uint8_t);
  DEFINE_IS_SCALAR_TYPE(int8_t);
  DEFINE_IS_SCALAR_TYPE(int16_t);
  DEFINE_IS_SCALAR_TYPE(int32_t);
  DEFINE_IS_SCALAR_TYPE(int64_t);
  DEFINE_IS_SCALAR_TYPE(float);
  DEFINE_IS_SCALAR_TYPE(double);
  DEFINE_IS_SCALAR_TYPE(bool);

#undef DEFINE_IS_SCALAR_TYPE

  template <typename T>
  constexpr bool is_scalar_type_v = IsScalarType<T>::value;

  template <ScalarType S>
  struct ScalarToType;

  template <>
  struct ScalarToType<ScalarType::Uint8>
  {
    using type = uint8_t;
  };
  template <>
  struct ScalarToType<ScalarType::Int8>
  {
    using type = int8_t;
  };
  template <>
  struct ScalarToType<ScalarType::Int16>
  {
    using type = int16_t;
  };
  template <>
  struct ScalarToType<ScalarType::Int32>
  {
    using type = int32_t;
  };
  template <>
  struct ScalarToType<ScalarType::Int64>
  {
    using type = int64_t;
  };
  template <>
  struct ScalarToType<ScalarType::Float32>
  {
    using type = float;
  };
  template <>
  struct ScalarToType<ScalarType::Float64>
  {
    using type = double;
  };
  template <>
  struct ScalarToType<ScalarType::Bool>
  {
    using type = bool;
  };

  template <ScalarType S>
  using scalar_t = typename ScalarToType<S>::type;

} // namespace tensor