#include <gtest/gtest.h>
#include <tensor/core/expression.hpp>
#include <tensor/core/tensor_base.hpp>

using namespace tensor;

// Test operators
struct Add
{
  template <typename T>
  static T apply(T a, T b) { return a + b; }
};

struct Multiply
{
  template <typename T>
  static T apply(T a, T b) { return a * b; }
};

struct Negate
{
  template <typename T>
  static T apply(T a) { return -a; }
};

TEST(ExpressionTest, BinaryExpression)
{
  Tensor<float, 1> a({3});
  Tensor<float, 1> b({3});

  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  b[0] = 2.0f;
  b[1] = 3.0f;
  b[2] = 4.0f;

  BinaryExpression<Add, Tensor<float, 1>, Tensor<float, 1>, float> sum(a, b);
  EXPECT_EQ(sum[0], 3.0f);
  EXPECT_EQ(sum[1], 5.0f);
  EXPECT_EQ(sum[2], 7.0f);

  BinaryExpression<Multiply, Tensor<float, 1>, Tensor<float, 1>, float> prod(a, b);
  EXPECT_EQ(prod[0], 2.0f);
  EXPECT_EQ(prod[1], 6.0f);
  EXPECT_EQ(prod[2], 12.0f);
}

TEST(ExpressionTest, UnaryExpression)
{
  Tensor<float, 1> a({3});
  a[0] = 1.0f;
  a[1] = -2.0f;
  a[2] = 3.0f;

  UnaryExpression<Negate, Tensor<float, 1>, float> neg(a);
  EXPECT_EQ(neg[0], -1.0f);
  EXPECT_EQ(neg[1], 2.0f);
  EXPECT_EQ(neg[2], -3.0f);
}

TEST(ExpressionTest, ChainedExpressions)
{
  Tensor<float, 1> a({3});
  Tensor<float, 1> b({3});
  Tensor<float, 1> c({3});

  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  b[0] = 2.0f;
  b[1] = 3.0f;
  b[2] = 4.0f;
  c[0] = 1.0f;
  c[1] = 1.0f;
  c[2] = 1.0f;

  // Test (a + b) * c
  auto expr1 = BinaryExpression<Add, Tensor<float, 1>, Tensor<float, 1>, float>(a, b);
  auto expr2 = BinaryExpression<Multiply,
                                BinaryExpression<Add, Tensor<float, 1>, Tensor<float, 1>, float>,
                                Tensor<float, 1>, float>(expr1, c);

  EXPECT_EQ(expr2[0], 3.0f); // (1 + 2) * 1
  EXPECT_EQ(expr2[1], 5.0f); // (2 + 3) * 1
  EXPECT_EQ(expr2[2], 7.0f); // (3 + 4) * 1
}

TEST(ExpressionTest, DimensionMismatch)
{
  Tensor<float, 1> a({2});
  Tensor<float, 1> b({3});

  EXPECT_THROW(
      (BinaryExpression<Add, Tensor<float, 1>, Tensor<float, 1>, float>(a, b)),
      DimensionMismatch);
}