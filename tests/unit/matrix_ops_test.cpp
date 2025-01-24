#include <gtest/gtest.h>
#include <tensor/core/tensor_base.hpp>
#include <tensor/ops/binary_ops.hpp>
#include "../test_config.hpp"

using namespace tensor;

TEST(MatrixOpsTest, MatrixMultiplication)
{
  Tensor<float, 2> a({2, 2});
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;

  Tensor<float, 2> b({2, 2});
  b[0] = 2.0f;
  b[1] = 0.0f;
  b[2] = 1.0f;
  b[3] = 3.0f;

  auto c = a % b;
  EXPECT_FLOAT_EQ(c[0], 4.0f);
  EXPECT_FLOAT_EQ(c[1], 6.0f);
  EXPECT_FLOAT_EQ(c[2], 10.0f);
  EXPECT_FLOAT_EQ(c[3], 12.0f);
}

TEST(MatrixOpsTest, ScalarMultiplication)
{
  Tensor<float, 2> a({2, 2});
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;

  auto scaled = 2.0f * a;
  EXPECT_FLOAT_EQ(scaled[0], 2.0f);
  EXPECT_FLOAT_EQ(scaled[1], 4.0f);
  EXPECT_FLOAT_EQ(scaled[2], 6.0f);
  EXPECT_FLOAT_EQ(scaled[3], 8.0f);
}

TEST(MatrixOpsTest, DimensionMismatch)
{
  Tensor<float, 2> a({2, 2});
  Tensor<float, 2> b({3, 2});
  EXPECT_THROW(a % b, DimensionMismatch);
}