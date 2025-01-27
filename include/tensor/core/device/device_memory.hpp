#pragma once

#include "device.hpp"

#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>

namespace tensor
{
  /**
   * @struct MemoryBlock
   * @brief Structure to store memory block information.
   *
   * @since 1.0.0
   */
  struct MemoryBlock
  {
    void *ptr{nullptr};
    size_t size{0};
    bool in_use{false};
  };

  /**
   * @class DeviceMemory
   * @brief Class to manage device memory.
   *
   * @since 1.0.0
   */
  class DeviceMemory
  {
  public:
    static DeviceMemory &instance()
    {
      static DeviceMemory instance;
      return instance;
    }

    // Methods
    void *allocate(size_t size, const Device &device);
    void deallocate(void *ptr, const Device &device);

    void copy_to_host(void *dst, const void *src, size_t size,
                      const Device &device);
    void copy_to_device(void *dst, const void *src, size_t size,
                        const Device &device);
    void peer_copy(void *dst, const Device &dst_device,
                   const void *src, const Device &src_device,
                   size_t size);

  private:
    std::unordered_map<Device, std::vector<MemoryBlock>> m_memory_pools;
    std::mutex m_mutex;

    // Class methods
    DeviceMemory() = default;
    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &operator=(const DeviceMemory &) = delete;

    // Methods
    void *get_from_pool(size_t size, const Device &device);
    void return_to_pool(void *ptr, size_t size, const Device &device);
  };

  /**
   * @class MemoryGuard
   * @brief Class to manage device memory.
   *
   * @since 1.0.0
   */
  class MemoryGuard
  {
  public:
    MemoryGuard(size_t size, const Device &device);
    ~MemoryGuard();

    // Methods
    /**
     * @brief Get the pointer to the memory block
     *
     * @return void* Pointer to the memory block
     * @since 1.0.0
     */
    void *get() const noexcept { return m_ptr; }

    /**
     * @brief Get the size of the memory block
     *
     * @return size_t Size of the memory block
     * @since 1.0.0
     */
    size_t size() const noexcept { return m_size; }

    /**
     * @brief Get the device of the memory block
     *
     * @return Device Device of the memory block
     * @since 1.0.0
     */
    const Device &device() const noexcept { return m_device; }

  private:
    Device m_device;
    void *m_ptr;
    size_t m_size;
  };
} // namespace tensor