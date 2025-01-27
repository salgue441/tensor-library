#include <gtest/gtest.h>
#include <tensor/core/device/device.hpp>

using namespace tensor;

TEST(DeviceTest, DefaultConstruction)
{
  Device device;

  EXPECT_TRUE(device.is_cpu());
  EXPECT_EQ(device.index(), -1);
  EXPECT_EQ(device.type(), DeviceType::CPU);
}

TEST(DeviceTest, CPUConstruction)
{
  Device device = Device::CPU();

  EXPECT_TRUE(device.is_cpu());
  EXPECT_FALSE(device.is_cuda());
  EXPECT_EQ(device.index(), -1);
}

TEST(DeviceTest, CUDAConstruction)
{
#ifdef USE_CUDA
  Device device = Device::CUDA(0);
  EXPECT_TRUE(device.is_cuda());
  EXPECT_FALSE(device.is_cpu());
  EXPECT_EQ(device.index(), 0);

#else
  EXPECT_THROW(Device::CUDA(0), DeviceError);
#endif
}

TEST(DeviceTest, InvalidDeviceIndices)
{
  EXPECT_THROW(Device(DeviceType::CPU, 0), DeviceError);
  EXPECT_THROW(Device(DeviceType::CUDA, -1), DeviceError);

#ifdef USE_CUDA
  int device_count;
  cudaGetDeviceCount(&device_count);

  EXPECT_THROW(Device(DeviceType::CUDA, device_count), DeviceError);
#endif
}

TEST(DeviceTest, StringRepresentation)
{
  Device cpu = Device::CPU();
  EXPECT_EQ(cpu.to_string(), "cpu");

#ifdef USE_CUDA
  Device cuda = Device::CUDA(0);
  EXPECT_EQ(cuda.to_string(), "cuda:0");
#endif
}

TEST(DeviceTest, Equality)
{
  Device cpu1 = Device::CPU();
  Device cpu2 = Device::CPU();
  EXPECT_EQ(cpu1, cpu2);

#ifdef USE_CUDA
  Device cuda1 = Device::CUDA(0);
  Device cuda2 = Device::CUDA(0);
  Device cuda3 = Device::CUDA(1);

  EXPECT_EQ(cuda1, cuda2);
  EXPECT_NE(cuda1, cuda3);
  EXPECT_NE(cuda1, cpu1);
#endif
}