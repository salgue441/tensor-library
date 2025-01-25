#include <gtest/gtest.h>
#include <tensor/ops/unary_ops.hpp>
#include "../test_config.hpp"

using namespace tensor;
constexpr float PI = std::numbers::pi_v<float>;

TEST(UnaryOpsTest, Exponential)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 1.0f;
  a[2] = 2.0f;
  a[3] = 3.0f;

  Tensor<float, 2> b = UnaryOps<float>::exp(a);

  EXPECT_FLOAT_EQ(b[0], std::exp(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::exp(1.0f));
  EXPECT_FLOAT_EQ(b[2], std::exp(2.0f));
  EXPECT_FLOAT_EQ(b[3], std::exp(3.0f));
}

TEST(UnaryOpsTest, Logarithm)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;

  Tensor<float, 2> b = UnaryOps<float>::log(a);

  EXPECT_FLOAT_EQ(b[0], std::log(1.0f));
  EXPECT_FLOAT_EQ(b[1], std::log(2.0f));
  EXPECT_FLOAT_EQ(b[2], std::log(3.0f));
  EXPECT_FLOAT_EQ(b[3], std::log(4.0f));
}

TEST(UnaryOpsTest, Sine)
{
  Tensor<float, 2> a({2, 2});
  a[0] = 0.0f;
  a[1] = PI / 2.0f;
  a[2] = PI;
  a[3] = 3.0f * PI / 2.0f;

  auto b = UnaryOps<float>::sin(a);

  EXPECT_NEAR(b[0], std::sin(0.0f), 1e-6f);
  EXPECT_NEAR(b[1], std::sin(PI / 2.0f), 1e-6f);
  EXPECT_NEAR(b[2], std::sin(PI), 1e-6f);
  EXPECT_NEAR(b[3], std::sin(3.0f * PI / 2.0f), 1e-6f);
}

TEST(UnaryOpsTest, Cosine)
{
  Tensor<float, 2> a({2, 2});
  a[0] = 0.0f;
  a[1] = PI / 2.0f;
  a[2] = PI;
  a[3] = 3.0f * PI / 2.0f;

  auto b = UnaryOps<float>::cos(a);

  EXPECT_NEAR(b[0], std::cos(0.0f), 1e-6f);
  EXPECT_NEAR(b[1], std::cos(PI / 2.0f), 1e-6f);
  EXPECT_NEAR(b[2], std::cos(PI), 1e-6f);
  EXPECT_NEAR(b[3], std::cos(3.0f * PI / 2.0f), 1e-6f);
}

TEST(UnaryOpsTest, Tangent)
{
  Tensor<float, 2> a({2, 2});
  a[0] = 0.0f;
  a[1] = PI / 4.0f;
  a[2] = -PI / 4.0f;
  a[3] = PI / 3.0f;

  auto b = UnaryOps<float>::tan(a);

  EXPECT_NEAR(b[0], std::tan(0.0f), 1e-6f);
  EXPECT_NEAR(b[1], std::tan(PI / 4.0f), 1e-6f);
  EXPECT_NEAR(b[2], std::tan(-PI / 4.0f), 1e-6f);
  EXPECT_NEAR(b[3], std::tan(PI / 3.0f), 1e-6f);
}

TEST(UnaryOpsTest, Arcsine)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 0.5f;
  a[3] = 1.0f;

  Tensor<float, 2> b = UnaryOps<float>::asin(a);

  EXPECT_FLOAT_EQ(b[0], std::asin(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::asin(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::asin(0.5f));
  EXPECT_FLOAT_EQ(b[3], std::asin(1.0f));
}

TEST(UnaryOpsTest, Arccosine)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 0.5f;
  a[3] = 1.0f;

  Tensor<float, 2> b = UnaryOps<float>::acos(a);

  EXPECT_FLOAT_EQ(b[0], std::acos(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::acos(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::acos(0.5f));
  EXPECT_FLOAT_EQ(b[3], std::acos(1.0f));
}

TEST(UnaryOpsTest, Arctangent)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 0.5f;
  a[3] = 1.0f;

  Tensor<float, 2> b = UnaryOps<float>::atan(a);

  EXPECT_FLOAT_EQ(b[0], std::atan(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::atan(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::atan(0.5f));
  EXPECT_FLOAT_EQ(b[3], std::atan(1.0f));
}

TEST(UnaryOpsTest, HyperbolicSine)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 1.0f;
  a[2] = 2.0f;
  a[3] = 3.0f;

  Tensor<float, 2> b = UnaryOps<float>::sinh(a);

  EXPECT_FLOAT_EQ(b[0], std::sinh(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::sinh(1.0f));
  EXPECT_FLOAT_EQ(b[2], std::sinh(2.0f));
  EXPECT_FLOAT_EQ(b[3], std::sinh(3.0f));
}

TEST(UnaryOpsTest, HyperbolicCosine)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 1.0f;
  a[2] = 2.0f;
  a[3] = 3.0f;

  Tensor<float, 2> b = UnaryOps<float>::cosh(a);

  EXPECT_FLOAT_EQ(b[0], std::cosh(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::cosh(1.0f));
  EXPECT_FLOAT_EQ(b[2], std::cosh(2.0f));
  EXPECT_FLOAT_EQ(b[3], std::cosh(3.0f));
}

TEST(UnaryOpsTest, HyperbolicTangent)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 0.5f;
  a[2] = 0.9f;
  a[3] = 0.99f;

  Tensor<float, 2> b = UnaryOps<float>::tanh(a);

  EXPECT_FLOAT_EQ(b[0], std::tanh(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::tanh(0.5f));
  EXPECT_FLOAT_EQ(b[2], std::tanh(0.9f));
  EXPECT_FLOAT_EQ(b[3], std::tanh(0.99f));
}

TEST(UnaryOpsTest, InverseHyperbolicSine)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 0.5f;
  a[2] = 0.9f;
  a[3] = 0.99f;

  Tensor<float, 2> b = UnaryOps<float>::asinh(a);

  EXPECT_FLOAT_EQ(b[0], std::asinh(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::asinh(0.5f));
  EXPECT_FLOAT_EQ(b[2], std::asinh(0.9f));
  EXPECT_FLOAT_EQ(b[3], std::asinh(0.99f));
}

TEST(UnaryOpsTest, InverseHyperbolicCosine)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 1.0f;
  a[1] = 1.5f;
  a[2] = 1.9f;
  a[3] = 1.99f;

  Tensor<float, 2> b = UnaryOps<float>::acosh(a);

  EXPECT_FLOAT_EQ(b[0], std::acosh(1.0f));
  EXPECT_FLOAT_EQ(b[1], std::acosh(1.5f));
  EXPECT_FLOAT_EQ(b[2], std::acosh(1.9f));
  EXPECT_FLOAT_EQ(b[3], std::acosh(1.99f));
}

TEST(UnaryOpsTest, InverseHyperbolicTangent)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 0.5f;
  a[2] = 0.9f;
  a[3] = 0.99f;

  Tensor<float, 2> b = UnaryOps<float>::atanh(a);

  EXPECT_FLOAT_EQ(b[0], std::atanh(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::atanh(0.5f));
  EXPECT_FLOAT_EQ(b[2], std::atanh(0.9f));
  EXPECT_FLOAT_EQ(b[3], std::atanh(0.99f));
}

TEST(UnaryOpsTest, Absolute)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 1.0f;
  a[3] = -2.0f;

  Tensor<float, 2> b = UnaryOps<float>::abs(a);

  EXPECT_FLOAT_EQ(b[0], std::abs(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::abs(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::abs(1.0f));
  EXPECT_FLOAT_EQ(b[3], std::abs(-2.0f));
}

TEST(UnaryOpsTest, SquareRoot)
{
  Tensor<float, 2> a({2, 2});

  a[0] = 0.0f;
  a[1] = 1.0f;
  a[2] = 4.0f;
  a[3] = 9.0f;

  Tensor<float, 2> b = UnaryOps<float>::sqrt(a);

  EXPECT_FLOAT_EQ(b[0], std::sqrt(0.0f));
  EXPECT_FLOAT_EQ(b[1], std::sqrt(1.0f));
  EXPECT_FLOAT_EQ(b[2], std::sqrt(4.0f));
  EXPECT_FLOAT_EQ(b[3], std::sqrt(9.0f));
}

TEST(UnaryOpsTest, Abs)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 1.0f;
  a[3] = -2.0f;

  Tensor<float, 2> b = UnaryOps<float>::abs(a);

  EXPECT_FLOAT_EQ(b[0], std::abs(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::abs(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::abs(1.0f));
  EXPECT_FLOAT_EQ(b[3], std::abs(-2.0f));
}

TEST(UnaryOpsTest, Floor)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 1.0f;
  a[3] = -2.0f;

  Tensor<float, 2> b = UnaryOps<float>::floor(a);

  EXPECT_FLOAT_EQ(b[0], std::floor(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::floor(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::floor(1.0f));
  EXPECT_FLOAT_EQ(b[3], std::floor(-2.0f));
}

TEST(UnaryOpsTest, Ceil)
{
  Tensor<float, 2> a({2, 2});

  a[0] = -1.0f;
  a[1] = 0.0f;
  a[2] = 1.0f;
  a[3] = -2.0f;

  Tensor<float, 2> b = UnaryOps<float>::ceil(a);

  EXPECT_FLOAT_EQ(b[0], std::ceil(-1.0f));
  EXPECT_FLOAT_EQ(b[1], std::ceil(0.0f));
  EXPECT_FLOAT_EQ(b[2], std::ceil(1.0f));
  EXPECT_FLOAT_EQ(b[3], std::ceil(-2.0f));
}