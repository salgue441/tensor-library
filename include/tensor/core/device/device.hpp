#pragma once

#include "device_types.hpp"
#include "../../utils/exceptions.hpp"

#include <string>
#include <functional>

namespace tensor
{
  /**
   * @class Device
   * @brief Base class for device management and operations
   *
   * @since 1.0.0
   */
  class Device
  {
  public:
    Device() : m_type(DeviceType::CPU), m_index(-1) {}
    explicit Device(DeviceType type, int index = 0);

    /**
     * @brief Creates a device using CPU type
     *
     * @return Device instance
     * @since 1.0.0
     */
    static Device CPU() { return Device(); }

    /**
     * @brief Creates a device using CUDA type
     *
     * @param index CUDA device index
     * @return Device instance
     * @since 1.0.0
     */
    static Device CUDA(int index = 0)
    {
      return Device(DeviceType::CUDA, index);
    }

    // Properties
    /**
     * @brief Type of the device (CPU or CUDA)
     *
     * @return DeviceType Type of the device
     * @since 1.0.0
     */
    DeviceType type() const noexcept { return m_type; }

    /**
     * @brief Index of the device
     *
     * @return int Index of the device
     * @since 1.0.0
     */
    int index() const noexcept { return m_index; }

    // Operator
    /**
     * @brief Operator== for comparing two devices
     *
     * @param other Device to compare
     * @return bool True if the devices are equal, false otherwise
     * @version 1.0.0
     */
    bool operator==(const Device &other) const noexcept
    {
      return m_type == other.m_type && m_index == other.m_index;
    }

    /**
     * @brief Operator!= for comparing two devices
     *
     * @param other Device to compare
     * @return bool True if the devices are not equal, false otherwise
     * @version 1.0.0
     */
    bool operator!=(const Device &other) const noexcept
    {
      return !(*this == other);
    }

    // Methods
    /**
     * @brief Checks if the device type is CPU
     *
     * @return bool True if the device is CPU, false otherwise
     * @since 1.0.0
     */
    bool is_cpu() const noexcept { return m_type == DeviceType::CPU; }

    /**
     * @brief Checks if the device type is CUDA
     *
     * @return bool True if the device is CUDA, false otherwise
     * @since 1.0.0
     */
    bool is_cuda() const noexcept { return m_type == DeviceType::CUDA; }

    /**
     * @brief Converts the device information to a string
     *
     * @return The string representation of the device
     * @since 1.0.0
     */
    std::string to_string() const
    {
      if (is_cpu())
        return "cpu";

      return "cuda:" + std::to_string(m_index);
    }

  private:
    DeviceType m_type;
    int m_index;

    void validate_device() const;
  };
} // namespace tensor

// hash function for Device class
namespace std
{
  template <>
  struct hash<tensor::Device>
  {
    size_t operator()(const tensor::Device &device) const noexcept
    {
      return static_cast<size_t>(device.type()) ^
             (static_cast<size_t>(device.index()) << 1);
    }
  };
}