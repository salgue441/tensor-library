#pragma once

#include "device.hpp"

#include <array>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace tensor
{
  /**
   * @struct DeviceInfo
   * @brief Cache-aligned structure to store device properties.
   *
   * @since 1.0.0
   */
  struct alignas(64) DeviceInfo
  {
    size_t memory_capacity{0};
    size_t max_threads_per_block{0};
    size_t warp_size{0};
    size_t max_shared_memory{0};
    std::array<size_t, 3> max_grid_size{0, 0, 0};
    std::array<size_t, 3> max_block_size{0, 0, 0};
    int compute_capability_major{0};
    int compute_capability_minor{0};
    bool unified_addressing{false};
    char name[256]{0};
  };

  /**
   * @class DeviceProperties
   * @brief Singleton class to store device properties.
   *
   * @since 1.0.0
   */
  class DeviceProperties
  {
  public:
    static DeviceProperties &instance()
    {
      static DeviceProperties instance;
      return instance;
    }

    const DeviceInfo &get_info(const Device &device);

  private:
    std::unordered_map<Device, DeviceInfo> m_device_info;
    std::mutex m_mutex;

    DeviceProperties() = default;
    DeviceProperties(const DeviceProperties &) = delete;
    DeviceProperties &operator=(const DeviceProperties &) = delete;

    // Methods
    void init_info(const Device &device, DeviceInfo &info);
  };
}