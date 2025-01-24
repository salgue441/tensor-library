#pragma once

#define ASSERT_TENSORS_EQUAL(expected, actual) \
  ASSERT_EQ(expected.size(), actual.size());   \
  for (size_t i = 0; i < expected.size(); ++i) \
  {                                            \
    ASSERT_NEAR(expected[i], actual[i], 1e-5); \
  }

#define ASSERT_SHAPES_EQUAL(expected, actual)  \
  ASSERT_EQ(expected.size(), actual.size());   \
  for (size_t i = 0; i < expected.size(); ++i) \
  {                                            \
    ASSERT_EQ(expected[i], actual[i]);         \
  }