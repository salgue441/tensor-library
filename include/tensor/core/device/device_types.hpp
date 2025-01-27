#pragma once

#include <cstdint>

namespace tensor
{
  /**
   * @enum DeviceType
   * @brief Basic types of devices
   *
   * @since 1.0.0
   */
  enum class DeviceType : uint8_t
  {
    CPU = 0,
    CUDA = 1
  };
} // namespace tensor