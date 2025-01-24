#pragma once
#include <gtest/gtest.h>

inline void AssertTensorsEqual(const auto &expected, const auto &actual)
{
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_NEAR(expected[i], actual[i], 1e-5);
  }
}

inline void AssertShapesEqual(const auto &expected, const auto &actual)
{
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], actual[i]);
  }
}