#include <gtest/gtest.h>
#include <tensor/core/device/device_memory.hpp>
#include <numeric>

using namespace tensor;

TEST(DeviceMemoryTest, CPUAllocation)
{
  auto &memory = DeviceMemory::instance();
  Device cpu = Device::CPU();

  void *ptr = memory.allocate(1024, cpu);
  ASSERT_NE(ptr, nullptr);

  uint8_t *data = static_cast<uint8_t *>(ptr);
  std::iota(data, data + 1024, 0);

  for (int i = 0; i < 1024; ++i)
    EXPECT_EQ(data[i], static_cast<uint8_t>(i));

  memory.deallocate(ptr, cpu);
}

TEST(DeviceMemoryTest, CPUZeroSizeAllocation)
{
  auto &memory = DeviceMemory::instance();
  Device cpu = Device::CPU();

  void *ptr = memory.allocate(0, cpu);
  EXPECT_EQ(ptr, nullptr);
}

TEST(DeviceMemoryTest, CPUMemoryCopy)
{
  auto &memory = DeviceMemory::instance();
  Device cpu = Device::CPU();

  const size_t size = 1024;
  void *src = memory.allocate(size, cpu);
  uint8_t *src_data = static_cast<uint8_t *>(src);
  std::iota(src_data, src_data + size, 0);

  void *dst = memory.allocate(size, cpu);
  memory.copy_to_host(dst, src, size, cpu);

  // Verify the data
  uint8_t *dst_data = static_cast<uint8_t *>(dst);
  for (size_t i = 0; i < size; ++i)
    EXPECT_EQ(dst_data[i], static_cast<uint8_t>(i));

  memory.deallocate(src, cpu);
  memory.deallocate(dst, cpu);
}

TEST(DeviceMemoryTest, MemoryGuard)
{
  Device cpu = Device::CPU();
  const size_t size = 1024;

  {
    MemoryGuard guard(size, cpu);

    EXPECT_NE(guard.get(), nullptr);
    EXPECT_EQ(guard.size(), size);

    uint8_t *data = static_cast<uint8_t *>(guard.get());
    std::fill(data, data + size, 0xFF);
  }
}

#ifdef USE_CUDA
TEST(DeviceMemoryTest, CUDAAllocation)
{
  auto &memory = DeviceMemory::instance();
  Device cuda = Device::CUDA(0);

  void *d_ptr = memory.allocate(1024, cuda);
  ASSERT_NE(d_ptr, nullptr);

  void *h_ptr = memory.allocate(1024, Device::CPU());
  uint8_t *h_data = static_cast<uint8_t *>(h_ptr);
  std::iota(h_data, h_data + 1024, 0);

  memory.copy_to_device(d_ptr, h_ptr, 1024, cuda);

  void *h_result = memory.allocate(1024, Device::CPU());
  memory.copy_to_host(h_result, d_ptr, 1024, cuda);

  uint8_t *result_data = static_cast<uint8_t *>(h_result);
  for (int i = 0; i < 1024; ++i)
    EXPECT_EQ(result_data[i], static_cast<uint8_t>(i));

  memory.deallocate(d_ptr, cuda);
  memory.deallocate(h_ptr, Device::CPU());
  memory.deallocate(h_result, Device::CPU());
}

TEST(DeviceMemoryTest, CUDAPeerCopy)
{
  auto &memory = DeviceMemory::instance();
  int device_count;
  cudaGetDeviceCount(&device_count);

  if (device_count > 1)
  {
    Device cuda0 = Device::CUDA(0);
    Device cuda1 = Device::CUDA(1);

    void *d0_ptr = memory.allocate(1024, cuda0);
    void *h_ptr = memory.allocate(1024, Device::CPU());
    uint8_t *h_data = static_cast<uint8_t *>(h_ptr);
    std::iota(h_data, h_data + 1024, 0);

    memory.copy_to_device(d0_ptr, h_ptr, 1024, cuda0);

    void *d1_ptr = memory.allocate(1024, cuda1);
    memory.peer_copy(d1_ptr, cuda1, d0_ptr, cuda0, 1024);

    void *h_result = memory.allocate(1024, Device::CPU());
    memory.copy_to_host(h_result, d1_ptr, 1024, cuda1);

    uint8_t *result_data = static_cast<uint8_t *>(h_result);
    for (int i = 0; i < 1024; ++i)
      EXPECT_EQ(result_data[i], static_cast<uint8_t>(i));

    memory.deallocate(d0_ptr, cuda0);
    memory.deallocate(d1_ptr, cuda1);
    memory.deallocate(h_ptr, Device::CPU());
    memory.deallocate(h_result, Device::CPU());
  }
}
#endif

TEST(DeviceMemoryTest, MemoryPool)
{
  auto &memory = DeviceMemory::instance();
  Device cpu = Device::CPU();

  void *ptr1 = memory.allocate(1024, cpu);
  ASSERT_NE(ptr1, nullptr);

  memory.deallocate(ptr1, cpu);

  void *ptr2 = memory.allocate(1024, cpu);
  EXPECT_EQ(ptr1, ptr2);

  memory.deallocate(ptr2, cpu);
}