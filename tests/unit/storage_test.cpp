#include <gtest/gtest.h>
#include <tensor/core/storage.hpp>

using namespace tensor;

TEST(StorageTest, Construction)
{
  TensorStorage<float> storage(5);
  EXPECT_EQ(storage.size(), 5);
  EXPECT_FALSE(storage.empty());

  TensorStorage<int> storage2(3, 42);
  EXPECT_EQ(storage2.size(), 3);
  EXPECT_EQ(storage2[0], 42);
  EXPECT_EQ(storage2[2], 42);
}

TEST(StorageTest, ElementAccess)
{
  TensorStorage<int> storage(3);
  storage[0] = 1;
  storage[1] = 2;
  storage[2] = 3;

  EXPECT_EQ(storage[0], 1);
  EXPECT_EQ(storage.at(1), 2);
  EXPECT_THROW(storage.at(3), std::out_of_range);
}

TEST(StorageTest, Iterators)
{
  TensorStorage<int> storage(3);
  storage[0] = 1;
  storage[1] = 2;
  storage[2] = 3;

  int sum = 0;
  for (const auto &val : storage)
  {
    sum += val;
  }
  EXPECT_EQ(sum, 6);

  auto it = storage.begin();
  EXPECT_EQ(*it, 1);

  ++it;
  EXPECT_EQ(*it, 2);
}

TEST(StorageTest, Modifiers)
{
  TensorStorage<int> storage(2);
  storage[0] = 1;
  storage[1] = 2;

  storage.resize(3);
  EXPECT_EQ(storage.size(), 3);

  storage.clear();
  EXPECT_TRUE(storage.empty());
}