// tests/unit/common_test.cpp
#include <gtest/gtest.h>
#include <tf/core/common.hpp>
#include <tf/core/macros.hpp>

using namespace tf::core;

namespace test
{

  TEST(ShapeTest, Construction)
  {
    Shape s1;
    EXPECT_TRUE(s1.empty());
    EXPECT_EQ(s1.rank(), 0);

    Shape s2({2, 3, 4});
    EXPECT_EQ(s2.rank(), 3);
    EXPECT_EQ(s2[0], 2);
    EXPECT_EQ(s2[1], 3);
    EXPECT_EQ(s2[2], 4);
  }

  TEST(ShapeTest, NumElements)
  {
    Shape s1({2, 3, 4});
    EXPECT_EQ(s1.num_elements(), 24);

    Shape s2({5, 1, 3});
    EXPECT_EQ(s2.num_elements(), 15);
  }

  TEST(ShapeTest, Broadcasting)
  {
    Shape s1({1, 3});
    Shape s2({2, 3});
    Shape s3({4, 2, 3});

    EXPECT_TRUE(s1.is_broadcastable_to(s2));
    EXPECT_TRUE(s1.is_broadcastable_to(s3));
    EXPECT_TRUE(s2.is_broadcastable_to(s3));

    EXPECT_FALSE(s2.is_broadcastable_to(s1));
    EXPECT_FALSE(s3.is_broadcastable_to(s1));
  }

  TEST(ShapeTest, ToString)
  {
    Shape s({2, 3, 4});
    EXPECT_EQ(s.to_string(), "(2, 3, 4)");
  }

  TEST(MemoryTest, AllocateAndCopy)
  {
    auto ptr1 = Memory<int>::allocate(5);
    auto ptr2 = Memory<int>::allocate(5);

    for (int i = 0; i < 5; ++i)
    {
      ptr1[i] = i;
    }

    Memory<int>::copy(ptr2.get(), ptr1.get(), 5);

    for (int i = 0; i < 5; ++i)
    {
      EXPECT_EQ(ptr2[i], i);
    }
  }

  TEST(MemoryTest, Fill)
  {
    auto ptr = Memory<int>::allocate(5);
    Memory<int>::fill(ptr.get(), 5, 42);

    for (int i = 0; i < 5; ++i)
    {
      EXPECT_EQ(ptr[i], 42);
    }
  }

  TEST(ScopeGuardTest, Execution)
  {
    bool executed = false;
    {
      ScopeGuard guard([&]()
                       { executed = true; });
      EXPECT_FALSE(executed);
    }
    EXPECT_TRUE(executed);
  }

  TEST(ScopeGuardTest, Move)
  {
    bool executed = false;
    {
      ScopeGuard guard1([&]()
                        { executed = true; });
      ScopeGuard guard2(std::move(guard1));
      EXPECT_FALSE(executed);
    }
    EXPECT_TRUE(executed);
  }

  TEST(MacroTest, Alignment)
  {
    TF_ALIGNED int value = 42;
    EXPECT_TRUE(is_aligned(&value, TF_ALIGNMENT));
  }

  TEST(MacroTest, ScopeExit)
  {
    bool executed = false;
    {
      TF_SCOPE_EXIT(executed = true);
      EXPECT_FALSE(executed);
    }
    EXPECT_TRUE(executed);
  }

  TEST(MacroTest, Version)
  {
    std::string version = TF_VERSION_STRING;
    EXPECT_FALSE(version.empty());
  }

} // namespace test
