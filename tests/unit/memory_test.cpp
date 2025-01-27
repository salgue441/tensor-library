#include <gtest/gtest.h>
#include <tf/utils/memory.hpp>
#include <thread>
#include <array>

using namespace tf::utils;

namespace test
{
  /**
   * @brief Test fixture for memory tests.
   *
   */
  class MemoryTest : public ::testing::Test
  {
  protected:
    void SetUp() override
    {
      MemoryTracker::instance().reset_stats();
    }
  };

  TEST_F(MemoryTest, AlignmentFunctions)
  {
    char buffer[1024];
    for (size_t alignment = 2; alignment <= 128; alignment *= 2)
    {
      char *aligned = align_pointer(buffer, alignment);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned) % alignment, 0);
    }
  }

  TEST_F(MemoryTest, MemoryPoolBasicOperations)
  {
    MemoryPool pool(1024); // 1KB initial size

    // Test initial state
    EXPECT_GE(pool.total_size(), 1024);
    EXPECT_GT(pool.max_block_size(), 0);
    EXPECT_GT(pool.num_blocks(), 0);

    // Test allocation
    void *ptr1 = pool.allocate(256);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr1) % DEFAULT_ALIGNMENT, 0);

    // Test deallocation
    pool.deallocate(ptr1);
  }

  TEST_F(MemoryTest, MemoryPoolGrowth)
  {
    MemoryPool pool(1024);
    size_t initial_blocks = pool.num_blocks();

    // Force pool growth
    void *ptr = pool.allocate(2048);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GT(pool.num_blocks(), initial_blocks);
    EXPECT_GE(pool.total_size(), 2048);
  }

  TEST_F(MemoryTest, MemoryPoolMultipleAllocations)
  {
    MemoryPool pool;
    std::vector<void *> ptrs;
    const size_t num_allocs = 5;

    // Multiple allocations
    for (size_t i = 0; i < num_allocs; ++i)
    {
      void *ptr = pool.allocate(1024);
      EXPECT_NE(ptr, nullptr);
      ptrs.push_back(ptr);
    }

    // Multiple deallocations
    for (void *ptr : ptrs)
      EXPECT_NO_THROW(pool.deallocate(ptr));
  }

  TEST_F(MemoryTest, MemoryTrackerBasicOperations)
  {
    auto &tracker = MemoryTracker::instance();
    size_t initial_count = tracker.allocation_count();

    // Track allocation
    int *ptr = new int(42);
    tracker.track_allocation(ptr, sizeof(int));

    EXPECT_EQ(tracker.allocation_count(), initial_count + 1);
    EXPECT_EQ(tracker.total_allocated(), sizeof(int));
    EXPECT_EQ(tracker.active_allocations(), 1);

    // Track deallocation
    tracker.track_deallocation(ptr);
    delete ptr;

    EXPECT_EQ(tracker.deallocation_count(), initial_count + 1);
    EXPECT_EQ(tracker.active_allocations(), 0);
  }

  TEST_F(MemoryTest, TrackedPointerBasicOperations)
  {
    {
      TrackedPointer<int> ptr(new int(42));
      EXPECT_TRUE(ptr);
      EXPECT_EQ(*ptr, 42);
      EXPECT_EQ(MemoryTracker::instance().active_allocations(), 1);
    }
    EXPECT_EQ(MemoryTracker::instance().active_allocations(), 0);
  }

  TEST_F(MemoryTest, TrackedPointerAdvancedOperations)
  {
    TrackedPointer<int> ptr(new int(42));

    // Test dereferencing
    EXPECT_EQ(*ptr, 42);
    *ptr = 24;
    EXPECT_EQ(*ptr, 24);

    // Test release
    int *raw_ptr = ptr.release();
    EXPECT_FALSE(ptr);
    EXPECT_EQ(*raw_ptr, 24);
    delete raw_ptr;

    // Test reset
    ptr.reset(new int(100));
    EXPECT_TRUE(ptr);
    EXPECT_EQ(*ptr, 100);
  }

  TEST_F(MemoryTest, ThreadSafetyMemoryPool)
  {
    MemoryPool pool;
    const int num_threads = 10;
    const int ops_per_thread = 100;
    std::vector<std::thread> threads;

    auto worker = [&]()
    {
      std::vector<void *> ptrs;
      for (int i = 0; i < ops_per_thread; ++i)
      {
        void *ptr = pool.allocate(64);
        ptrs.push_back(ptr);
      }

      for (void *ptr : ptrs)
        pool.deallocate(ptr);
    };

    for (int i = 0; i < num_threads; ++i)
      threads.emplace_back(worker);

    for (auto &thread : threads)
      thread.join();
  }

  TEST_F(MemoryTest, ThreadSafetyMemoryTracker)
  {
    const int num_threads = 10;
    const int ops_per_thread = 100;
    std::vector<std::thread> threads;

    auto worker = [&]()
    {
      for (int i = 0; i < ops_per_thread; ++i)
      {
        TrackedPointer<int> ptr(new int(i));
        EXPECT_EQ(*ptr, i);
      }
    };

    for (int i = 0; i < num_threads; ++i)
      threads.emplace_back(worker);

    for (auto &thread : threads)
      thread.join();
  }
}