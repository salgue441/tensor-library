#include <gtest/gtest.h>
#include <tensor/ops/reduction_ops.hpp>
#include "../test_config.hpp"

using namespace tensor;

TEST(ReductionOpsTest, Sum)
{
  Tensor<int, 2> a({2, 2});
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 4;

  EXPECT_EQ(ReductionOps<int>::sum(a), 10);
}