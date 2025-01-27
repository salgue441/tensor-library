#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tf
{
  namespace core
  {
    using index_t = std::int64_t;
    using size_t = std::size_t;
    using shape_t = std::vector<index_t>;

    /**
     * @enum DeviceType
     * @brief Device type enumeration
     */
    enum class DeviceType
    {
      CPU,
      CUDA
    };

    /**
     * @enum DataType
     * @brief Data type enumeration
     */
    enum class DataType
    {
      Float32,
      Float64,
      Int32,
      Int64,
      Bool
    };

    /**
     * @enum DataLayout
     * @brief Data layout enumeration
     */
    enum class DataLayout
    {
      RowMajor,
      ColMajor
    };

    // Type traits for data types
    template <DataType T>
    struct DataTypeTraits
    {
    };

    template <>
    struct DataTypeTraits<DataType::Float32>
    {
      using type = float;
      static constexpr const char *name = "float32";
    };

    template <>
    struct DataTypeTraits<DataType::Float64>
    {
      using type = double;
      static constexpr const char *name = "float64";
    };

    template <>
    struct DataTypeTraits<DataType::Int32>
    {
      using type = std::int32_t;
      static constexpr const char *name = "int32";
    };

    template <>
    struct DataTypeTraits<DataType::Int64>
    {
      using type = std::int64_t;
      static constexpr const char *name = "int64";
    };

    template <>
    struct DataTypeTraits<DataType::Bool>
    {
      using type = bool;
      static constexpr const char *name = "bool";
    };
  } // namespace core
} // namespace tf