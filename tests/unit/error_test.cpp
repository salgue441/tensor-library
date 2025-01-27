#include <gtest/gtest.h>
#include <tf/core/error.hpp>

using namespace tf::core;

namespace test
{
  TEST(ErrorTest, BasicException)
  {
    try
    {
      throw Exception("Test error message");
    }
    catch (const Exception &e)
    {
      EXPECT_EQ(e.message(), "Test error message");
      EXPECT_TRUE(std::string(e.what())
                      .find("Test error message") != std::string::npos);
      EXPECT_TRUE(std::string(e.what())
                      .find("ErrorTest") != std::string::npos);
    }
  }

  TEST(ErrorTest, SpecificExceptions)
  {
    EXPECT_THROW(throw ShapeError("Invalid shape"), ShapeError);
    EXPECT_THROW(throw DeviceError("Device not available"), DeviceError);
    EXPECT_THROW(throw MemoryError("Out of memory"), MemoryError);
    EXPECT_THROW(throw TypeError("Invalid type"), TypeError);
    EXPECT_THROW(throw IndexError("Index out of bounds"), IndexError);
    EXPECT_THROW(throw NotImplementedError("Not implemented"),
                 NotImplementedError);
  }

  TEST(ErrorTest, CheckMacros)
  {
    EXPECT_THROW(TF_CHECK(false, Exception, "Check failed"), Exception);
    EXPECT_THROW(TF_CHECK_SHAPE(false, "Invalid shape"), ShapeError);
    EXPECT_THROW(TF_CHECK_DEVICE(false, "Device error"), DeviceError);
    EXPECT_THROW(TF_CHECK_MEMORY(false, "Memory error"), MemoryError);
    EXPECT_THROW(TF_CHECK_TYPE(false, "Type error"), TypeError);
    EXPECT_THROW(TF_CHECK_INDEX(false, "Index error"), IndexError);

    // Check that valid conditions don't throw
    EXPECT_NO_THROW(TF_CHECK(true, Exception, "Should not throw"));
    EXPECT_NO_THROW(TF_CHECK_SHAPE(true, "Should not throw"));
  }

  TEST(ErrorTest, ExceptionInformation)
  {
    try
    {
      TF_CHECK_SHAPE(false, "Test shape error");
      FAIL() << "Expected ShapeError";
    }
    catch (const ShapeError &e)
    {
      std::string what = e.what();

      EXPECT_TRUE(what.find("error_test.cpp") != std::string::npos);
      EXPECT_TRUE(what.find("Test shape error") != std::string::npos);
      EXPECT_EQ(e.message(), "Test shape error");
      EXPECT_FALSE(e.file().empty());
      EXPECT_GT(e.line(), 0);
      EXPECT_FALSE(e.function().empty());
    }
  }

  /**
   * @brief Test helper function to demonstrate source location tracking
   * 
   */
  void throw_in_function()
  {
    TF_CHECK_TYPE(false, "Error in function");
  }

  TEST(ErrorTest, SourceLocation)
  {
    try
    {
      throw_in_function();
      FAIL() << "Expected TypeError";
    }
    catch (const TypeError &e)
    {
      EXPECT_TRUE(std::string(e.what()).find("throw_in_function") != std::string::npos);
      EXPECT_EQ(e.message(), "Error in function");
    }
  }
}