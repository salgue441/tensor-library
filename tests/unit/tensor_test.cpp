#include <gtest/gtest.h>
#include <tensor/core/tensor_base.hpp>
#include <tensor/ops/matrix_ops.hpp>

using namespace tensor;

TEST(TensorTest, Construction)
{
  Tensor<float, 2> tensor({2, 3});
  EXPECT_EQ(tensor.size(), 6);
  auto shape = tensor.shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
}

TEST(TensorTest, ElementAccess)
{
  Tensor<int, 2> tensor({2, 2});
  tensor[0] = 1;
  tensor[1] = 2;
  tensor[2] = 3;
  tensor[3] = 4;

  EXPECT_EQ(tensor[0], 1);
  EXPECT_EQ(tensor[3], 4);
}

TEST(TensorTest, MatrixOperations)
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

  auto c = matrix_multiply(a, b);
  EXPECT_EQ(c[0], 4.0f);  // 1*2 + 2*1
  EXPECT_EQ(c[1], 6.0f);  // 1*0 + 2*3
  EXPECT_EQ(c[2], 10.0f); // 3*2 + 4*1
  EXPECT_EQ(c[3], 12.0f); // 3*0 + 4*3
}

TEST(TensorTest, Transpose)
{
  Tensor<int, 2> a({2, 3});
  for (size_t i = 0; i < 6; ++i)
  {
    a[i] = i;
  }

  auto t = transpose(a);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 2);
  EXPECT_EQ(t[0], 0);
  EXPECT_EQ(t[1], 3);
  EXPECT_EQ(t[2], 1);
  EXPECT_EQ(t[3], 4);
  EXPECT_EQ(t[4], 2);
  EXPECT_EQ(t[5], 5);
}