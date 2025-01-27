#include <tensor/core/device/device.hpp>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor
{
  /**
   * @brief Constructor of the Device class.
   *
   * @param type The type of the device.
   * @param index The id of the device.
   *
   * @since 1.0.0
   */
  Device::Device(DeviceType type, int index)
      : m_type(type), m_index(index)
  {
    validate_device();

    if (m_type == DeviceType::CPU)
      m_index = -1;
  }

  /**
   * @brief Validates the device.
   *
   * @throw DeviceError if the device is invalid.
   *
   * @version 1.0.0
   */
  void Device::validate_device() const
  {
    if (m_type == DeviceType::CPU && m_index != -1)
      throw DeviceError("CPU device index must be -1");

    if (m_type == DeviceType::CUDA)
    {
      if (m_index < 0)
        throw DeviceError("CUDA device index must be non-negative");

#ifdef USE_CUDA
      int device_count;
      cudaError_t err = cudaGetDeviceCount(&device_count);

      if (err != cudaSuccess)
        throw DeviceError("Failed to get CUDA device count");

      if (m_index >= device_count)
        throw DeviceError("Invalid CUDA device index: " + std::to_string(m_index));
#else
      throw DeviceError("CUDA support not enabled");
#endif
    }
  }
} // namespace tensor