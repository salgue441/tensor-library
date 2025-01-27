#pragma once

#include "device_types.hpp"
#include <thread>

namespace tensor
{
  /**
   * @class DeviceContext
   * @brief Context for a device to manage device-specific resources
   *
   * @since 1.0.0
   */
  class DeviceContext
  {
  public:
    static DeviceContext &get_instance();

    const Device &current_device() const;
    void set_device(const Device &device);

    // CPU affinity management
    void set_cpu_affinity(int cpu_id);
    void reset_cpu_affinity();

    void synchronize() const;

  private:
    DeviceContext();
    Device m_current_device;
    int m_previous_cpu_affinity;
  };

  /**
   * @class DeviceGuard
   * @brief RAII guard for setting the current device
   *
   * @since 1.0.0
   */
  class DeviceGuard
  {
  public:
    explicit DeviceGuard(const Device &device);
    ~DeviceGuard();

  private:
    const Device &m_previous_device;
  };
} // namespace tensor