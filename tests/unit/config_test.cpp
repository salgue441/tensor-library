#include <gtest/gtest.h>
#include <tf/core/config.hpp>
#include <thread>
#include <future>

using namespace tf::core;

namespace test
{
  /**
   * @brief Test fixture for the configuration class
   *
   */
  class ConfigTest : public ::testing::Test
  {
  protected:
    void SetUp() override
    {
      // Reset configuration to defaults before each test
      config().set_default_device(DeviceType::CPU);
      config().set_memory_fraction(0.9f);
      config().set_num_threads(4);
      config().set_debug_mode(false);
    }
  };

  TEST_F(ConfigTest, DefaultValues)
  {
    EXPECT_EQ(config().default_device(), DeviceType::CPU);
    EXPECT_FLOAT_EQ(config().memory_fraction(), 0.9f);
    EXPECT_EQ(config().num_threads(), 4);
    EXPECT_FALSE(config().debug_mode());
  }

  TEST_F(ConfigTest, DeviceConfiguration)
  {
    config().set_default_device(DeviceType::CUDA);
    EXPECT_EQ(config().default_device(), DeviceType::CUDA);
  }

  TEST_F(ConfigTest, MemoryConfiguration)
  {
    config().set_memory_fraction(0.5f);
    EXPECT_FLOAT_EQ(config().memory_fraction(), 0.5f);

    EXPECT_THROW(config().set_memory_fraction(-0.1f), ValueError);
    EXPECT_THROW(config().set_memory_fraction(1.1f), ValueError);
  }

  TEST_F(ConfigTest, ThreadConfiguration)
  {
    config().set_num_threads(8);
    EXPECT_EQ(config().num_threads(), 8);

    EXPECT_THROW(config().set_num_threads(0), ValueError);
    EXPECT_THROW(config().set_num_threads(-1), ValueError);
  }

  TEST_F(ConfigTest, DebugMode)
  {
    config().set_debug_mode(true);
    EXPECT_TRUE(config().debug_mode());
  }

  TEST_F(ConfigTest, CustomOptions)
  {
    // Test setting and getting custom options
    config().set_option("custom_int", 42);
    EXPECT_EQ(config().get_option("custom_int", 0), 42);

    std::string test_str = "test";
    config().set_option("custom_string", test_str);
    EXPECT_EQ(config().get_option("custom_string", std::string("")), "test");

    // Test default values
    EXPECT_EQ(config().get_option("nonexistent", 100), 100);
  }

  TEST_F(ConfigTest, TypeSafety)
  {
    config().set_option("value", 42);
    EXPECT_THROW(config().get_option<std::string>("value", ""), TypeError);
  }

  TEST_F(ConfigTest, ThreadSafety)
  {
    const int num_threads = 10;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_threads; ++i)
    {
      futures.push_back(std::async(std::launch::async, [i]()
                                   {
            config().set_option(std::to_string(i), i);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            EXPECT_EQ(config().get_option(std::to_string(i), -1), i); }));
    }

    for (auto &future : futures)
    {
      future.get();
    }
  }

  TEST_F(ConfigTest, ConfigGuard)
  {
    config().set_option("test", 1);
    {
      TF_WITH_CONFIG("test", 2);
      EXPECT_EQ(config().get_option("test", 0), 2);
    }
    EXPECT_EQ(config().get_option("test", 0), 1);
  }

  TEST_F(ConfigTest, MultipleGuards)
  {
    config().set_option("test1", 1);
    config().set_option("test2", std::string("a"));

    {
      TF_WITH_CONFIG("test1", 2);
      EXPECT_EQ(config().get_option("test1", 0), 2);

      {
        TF_WITH_CONFIG("test2", std::string("b"));
        EXPECT_EQ(config().get_option("test2", std::string("")), "b");
      }

      EXPECT_EQ(config().get_option("test2", std::string("")), "a");
    }

    EXPECT_EQ(config().get_option("test1", 0), 1);
  }
} // namespace test