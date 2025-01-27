#include <tensor/core/device/device_properties.hpp>
#include <thread>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor
{
  /**
   * @brief Returns the device information
   *
   * @param device The device
   * @return const DeviceInfo& The device information
   * @version 1.0.0
   */
  const DeviceInfo &DeviceProperties::get_info(const Device &device)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_device_info.find(device);
    if (it == m_device_info.end())
    {
      auto [new_it, inserted] = m_device_info.emplace(
          device, DeviceInfo{});

      init_info(device, new_it->second);
      return new_it->second;
    }

    return it->second;
  }

  /**
   * @brief Initializes the device information
   *
   * @param device The device
   * @param info The device information
   * @version 1.0.0
   */
  void DeviceProperties::init_info(const Device &device, DeviceInfo &info)
  {
    if (device.is_cpu())
    {
      info.memory_capacity = 0;
      info.max_threads_per_block = std::thread::hardware_concurrency();
      info.warp_size = 1;
      info.max_shared_memory = 0;
      info.max_grid_size = {1, 1, 1};
      info.max_block_size = {1, 1, 1};
      info.compute_capability_major = 0;
      info.compute_capability_minor = 0;
      info.unified_addressing = false;
      std::strncpy(info.name, "CPU", sizeof(info.name) - 1);
      info.name[sizeof(info.name) - 1] = '\0';

      return;
    }

#ifdef USE_CUDA
    if (device.is_cuda())
    {
      cudaDeviceProp prop;
      cudaError_t err = cudaGetDeviceProperties(&prop, device.index());
      if (err != cudaSuccess)
        throw DeviceError("Failed to get CUDA device properties: " +
                          std::string(cudaGetErrorString(err)));

      info.memory_capacity = prop.totalGlobalMem;
      info.max_threads_per_block = prop.maxThreadsPerBlock;
      info.warp_size = prop.warpSize;
      info.max_shared_memory = prop.sharedMemPerBlock;
      info.max_grid_size = {
          static_cast<size_t>(prop.maxGridSize[0]),
          static_cast<size_t>(prop.maxGridSize[1]),
          static_cast<size_t>(prop.maxGridSize[2])};
      info.max_block_size = {
          static_cast<size_t>(prop.maxThreadsDim[0]),
          static_cast<size_t>(prop.maxThreadsDim[1]),
          static_cast<size_t>(prop.maxThreadsDim[2])};
      info.compute_capability_major = prop.major;
      info.compute_capability_minor = prop.minor;
      info.unified_addressing = prop.unifiedAddressing;
      std::strncpy(info.name, prop.name, sizeof(info.name) - 1);
      info.name[sizeof(info.name) - 1] = '\0';
    }
#endif
  }
} // namespace tensor