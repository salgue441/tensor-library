#include <gtest/gtest.h>
#include <tensor/core/device/device_properties.hpp>
#include <thread>
#include <vector>

using namespace tensor;

TEST(DevicePropertiesTest, CPUProperties)
{
  const auto &props = DeviceProperties::instance().get_info(Device::CPU());

  EXPECT_EQ(props.warp_size, 1);
  EXPECT_EQ(props.max_threads_per_block, std::thread::hardware_concurrency());
  EXPECT_STREQ(props.name, "CPU");
}
  
TEST(DevicePropertiesTest, CachingBehavior)
{
  auto &properties = DeviceProperties::instance();
  Device cpu = Device::CPU();

  const auto &props1 = properties.get_info(cpu);
  const auto &props2 = properties.get_info(cpu);

  EXPECT_EQ(&props1, &props2);
}

TEST(DevicePropertiesTest, ThreadSafety)
{
  const int num_threads = 10;
  std::vector<std::thread> threads;
  std::vector<const DeviceInfo *> results(num_threads);

  for (int i = 0; i < num_threads; ++i)
  {
    threads.emplace_back([&results, i]()
                         {
            const auto& props = DeviceProperties::instance().get_info(Device::CPU());
            results[i] = &props; });
  }

  for (auto &thread : threads)
    thread.join();

  for (int i = 1; i < num_threads; ++i)
    EXPECT_EQ(results[0], results[i]);
}

#ifdef USE_CUDA
TEST(DevicePropertiesTest, GPUProperties)
{
  const auto &props = DeviceProperties::instance().get_info(Device::CUDA(0));

  EXPECT_GT(props.memory_capacity, 0);
  EXPECT_GT(props.max_threads_per_block, 0);
  EXPECT_GT(props.warp_size, 0);
  EXPECT_GT(props.max_shared_memory, 0);

  EXPECT_GT(props.max_grid_size[0], 0);
  EXPECT_GT(props.max_grid_size[1], 0);
  EXPECT_GT(props.max_grid_size[2], 0);

  EXPECT_GT(props.max_block_size[0], 0);
  EXPECT_GT(props.max_block_size[1], 0);
  EXPECT_GT(props.max_block_size[2], 0);

  EXPECT_GT(props.compute_capability_major, 0);
  EXPECT_GE(props.compute_capability_minor, 0);
  EXPECT_STRNE(props.name, "");
}

TEST(DevicePropertiesTest, MultiGPUProperties)
{
  int device_count;
  cudaGetDeviceCount(&device_count);

  if (device_count > 1)
  {
    const auto &props0 = DeviceProperties::instance().get_info(Device::CUDA(0));
    const auto &props1 = DeviceProperties::instance().get_info(Device::CUDA(1));

    EXPECT_NE(&props0, &props1);
    EXPECT_GT(props0.memory_capacity, 0);
    EXPECT_GT(props1.memory_capacity, 0);
    EXPECT_STRNE(props0.name, props1.name);
  }
}
#endif

TEST(DevicePropertiesTest, InvalidDevice)
{
#ifdef USE_CUDA
  int device_count;

  cudaGetDeviceCount(&device_count);
  Device invalid_cuda(DeviceType::CUDA, device_count);
  EXPECT_THROW(DeviceProperties::instance().get_info(invalid_cuda), DeviceError);
#endif
}