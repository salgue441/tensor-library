#include <gtest/gtest.h>
#include <tensor/core/dtype.hpp>

using namespace tensor;

TEST(DTypeTest, ToString)
{
  EXPECT_EQ(dtype::to_string(ScalarType::Float32), "float32");
  EXPECT_EQ(dtype::to_string(ScalarType::Int64), "int64");
  EXPECT_EQ(dtype::to_string(ScalarType::Bool), "bool");
}

TEST(DTypeTest, ElementSize)
{
  EXPECT_EQ(dtype::element_size(ScalarType::Float32), sizeof(float));
  EXPECT_EQ(dtype::element_size(ScalarType::Int64), sizeof(int64_t));
  EXPECT_EQ(dtype::element_size(ScalarType::Bool), sizeof(bool));
}

TEST(DTypeTest, TypePredicates)
{
  EXPECT_TRUE(dtype::is_floating_point(ScalarType::Float32));
  EXPECT_TRUE(dtype::is_floating_point(ScalarType::Float64));
  EXPECT_FALSE(dtype::is_floating_point(ScalarType::Int32));

  EXPECT_TRUE(dtype::is_integral(ScalarType::Int32));
  EXPECT_TRUE(dtype::is_integral(ScalarType::Int64));
  EXPECT_FALSE(dtype::is_integral(ScalarType::Float32));
  EXPECT_FALSE(dtype::is_integral(ScalarType::Bool));
}

TEST(DTypeTest, TypePromotion)
{
  // Same type promotions
  EXPECT_EQ(dtype::promote_types(ScalarType::Float32, ScalarType::Float32), ScalarType::Float32);
  EXPECT_EQ(dtype::promote_types(ScalarType::Int32, ScalarType::Int32), ScalarType::Int32);

  // Mixed type promotions
  EXPECT_EQ(dtype::promote_types(ScalarType::Float32, ScalarType::Int32), ScalarType::Float64);
  EXPECT_EQ(dtype::promote_types(ScalarType::Int32, ScalarType::Float32), ScalarType::Float64);
  EXPECT_EQ(dtype::promote_types(ScalarType::Int32, ScalarType::Int64), ScalarType::Int64);
}

TEST(DTypeTest, TypeTraits)
{
  EXPECT_EQ(TypeToScalar<float>::value, ScalarType::Float32);
  EXPECT_EQ(TypeToScalar<int32_t>::value, ScalarType::Int32);
  EXPECT_EQ(TypeToScalar<bool>::value, ScalarType::Bool);

  EXPECT_TRUE((std::is_same_v<ScalarToType<ScalarType::Float32>::type, float>));
  EXPECT_TRUE((std::is_same_v<ScalarToType<ScalarType::Int32>::type, int32_t>));
  EXPECT_TRUE((std::is_same_v<ScalarToType<ScalarType::Bool>::type, bool>));
}

TEST(DTypeTest, IsScalarType)
{
  EXPECT_TRUE(is_scalar_type_v<float>);
  EXPECT_TRUE(is_scalar_type_v<int32_t>);
  EXPECT_TRUE(is_scalar_type_v<bool>);
  EXPECT_FALSE(is_scalar_type_v<std::string>);
  EXPECT_FALSE(is_scalar_type_v<void>);
}

TEST(DTypeTest, ErrorHandling)
{
  // Test invalid dtype
  ScalarType invalid_dtype = static_cast<ScalarType>(999);
  EXPECT_THROW(dtype::to_string(invalid_dtype), TypeError);
  EXPECT_THROW(dtype::element_size(invalid_dtype), TypeError);
}

TEST(DTypeTest, ScalarTypeConversion)
{
  using float32_t = scalar_t<ScalarType::Float32>;
  using int64_t = scalar_t<ScalarType::Int64>;
  using bool_t = scalar_t<ScalarType::Bool>;

  EXPECT_TRUE((std::is_same_v<float32_t, float>));
  EXPECT_TRUE((std::is_same_v<int64_t, int64_t>));
  EXPECT_TRUE((std::is_same_v<bool_t, bool>));
}