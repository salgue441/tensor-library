#include <gtest/gtest.h>
#include <tf/core/types.hpp>

using namespace tf::core;

namespace test
{
  TEST(TypesTest, DataTypeTraits)
  {
    EXPECT_EQ(std::string(DataTypeTraits<DataType::Float32>::name), "float32");
    EXPECT_TRUE((std::is_same<DataTypeTraits<DataType::Float32>::type,
                              float>::value));

    EXPECT_EQ(std::string(DataTypeTraits<DataType::Float64>::name), "float64");
    EXPECT_TRUE((std::is_same<DataTypeTraits<DataType::Float64>::type,
                              double>::value));

    EXPECT_EQ(std::string(DataTypeTraits<DataType::Int32>::name), "int32");
    EXPECT_TRUE((std::is_same<DataTypeTraits<DataType::Int32>::type,
                              std::int32_t>::value));

    EXPECT_EQ(std::string(DataTypeTraits<DataType::Int64>::name), "int64");
    EXPECT_TRUE((std::is_same<DataTypeTraits<DataType::Int64>::type,
                              std::int64_t>::value));

    EXPECT_EQ(std::string(DataTypeTraits<DataType::Bool>::name), "bool");
    EXPECT_TRUE((std::is_same<DataTypeTraits<DataType::Bool>::type,
                              bool>::value));
  }

  TEST(TypesTest, ShapeType)
  {
    shape_t shape{1, 2, 3};

    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 3);
  }

  TEST(TypesTest, DeviceType)
  {
    DeviceType device = DeviceType::CPU;
    EXPECT_EQ(device, DeviceType::CPU);

    device = DeviceType::CUDA;
    EXPECT_EQ(device, DeviceType::CUDA);
  }

  TEST(TypesTest, Layout)
  {
    DataLayout layout = DataLayout::RowMajor;
    EXPECT_EQ(layout, DataLayout::RowMajor);

    layout = DataLayout::ColMajor;
    EXPECT_EQ(layout, DataLayout::ColMajor);
  }
}