#include <tensor/core/device/device_memory.hpp>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor
{
  /**
   * @brief Allocates memory on the device.
   *
   * @param size The size of the memory to allocate
   * @param device The device
   * @return void* A pointer to the allocated memory
   * @throw DeviceError If the memory allocation fails
   *
   * @version 1.0.0
   */
  void *DeviceMemory::allocate(size_t size, const Device &device)
  {
    if (size == 0)
      return nullptr;

    void *ptr = get_from_pool(size, device);
    if (ptr)
      return ptr;

    if (device.is_cpu())
    {
#ifdef _WIN32
      ptr = _aligned_malloc(size, 64);
      if (!ptr)
        throw DeviceError("Failed to allocate memory on the CPU");

#else
      if (posix_memalign(&ptr, 64, size) != 0)
        throw DeviceError("Failed to allocate memory on the CPU");
#endif
    }

#ifdef USE_CUDA
    else
    {
      cudaError_t err = cudaMalloc(&ptr, size);
      if (err != cudaSuccess)
        throw DeviceError("Failed to allocate memory on the CUDA device");
    }
#endif

    return ptr;
  }

  /**
   * @brief Deallocates memory on the device.
   *
   * @param ptr A pointer to the memory to deallocate
   * @param device The device
   *
   * @version 1.0.0
   */
  void DeviceMemory::deallocate(void *ptr, const Device &device)
  {
    if (!ptr)
      return;

    if (device.is_cpu())
    {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
#ifdef USE_CUDA
    else
    {
      cudaError_t err = cudaFree(ptr);
      if (err != cudaSuccess)
      {
        throw DeviceError("CUDA memory deallocation failed: " +
                          std::string(cudaGetErrorString(err)));
      }
    }
#endif
  }

  /**
   * @brief Copies memory from the device to the host.
   *
   * @param dst A pointer to the destination memory
   * @param src A pointer to the source memory
   * @param size The size of the memory to copy
   * @param device The device
   * @throw DeviceError If the memory copy fails
   *
   * @version 1.0.0
   */
  void DeviceMemory::copy_to_host(void *dst, const void *src, size_t size,
                                  const Device &device)
  {
    if (size == 0)
      return;

    if (device.is_cpu())
      std::memcpy(dst, src, size);

#ifdef USE_CUDA
    else
    {
      cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
        throw DeviceError("CUDA to host memory copy failed: " +
                          std::string(cudaGetErrorString(err)));
    }
#endif
  }

  /**
   * @brief Copies memory from the host to the device.
   *
   * @param dst A pointer to the destination memory
   * @param src A pointer to the source memory
   * @param size The size of the memory to copy
   * @param device The device
   * @throw DeviceError If the memory copy fails
   *
   * @version 1.0.0
   */
  void DeviceMemory::copy_to_device(void *dst, const void *src, size_t size, const Device &device)
  {
    if (size == 0)
      return;

    if (device.is_cpu())
    {
      std::memcpy(dst, src, size);
    }
#ifdef USE_CUDA
    else
    {
      cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
      {
        throw DeviceError("Host to CUDA memory copy failed: " +
                          std::string(cudaGetErrorString(err)));
      }
    }
#endif
  }

  /**
   * @brief Gets from the memory pool.
   *
   * @param size The size of the memory block
   * @param device The device
   * @return void* A pointer to the memory block
   * @throw DeviceError If the memory allocation fails
   *
   * @version 1.0.0
   */
  void *DeviceMemory::get_from_pool(size_t size, const Device &device)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &pool = m_memory_pools[device];
    auto it = std::find_if(pool.begin(), pool.end(),
                           [size](const MemoryBlock &block)
                           { return !block.in_use && block.size >= size; });

    if (it != pool.end())
    {
      it->in_use = true;
      return it->ptr;
    }

    return nullptr;
  }

  /**
   * @brief Returns the memory block to the pool.
   *
   * @param ptr A pointer to the memory block
   * @param size The size of the memory block
   * @param device The device
   *
   * @version 1.0.0
   */
  void DeviceMemory::return_to_pool(void *ptr, size_t size,
                                    const Device &device)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &pool = m_memory_pools[device];
    auto it = std::find_if(pool.begin(), pool.end(),
                           [ptr](const MemoryBlock &block)
                           {
                             return block.ptr == ptr;
                           });

    if (it != pool.end())
      it->in_use = false;

    else
      pool.push_back({ptr, size, false});
  }

  // MemoryGuard implementation
  MemoryGuard::MemoryGuard(size_t size, const Device &device)
      : m_device(device), m_size(size)
  {
    m_ptr = DeviceMemory::instance().allocate(size, device);
  }

  MemoryGuard::~MemoryGuard()
  {
    DeviceMemory::instance().deallocate(m_ptr, m_device);
  }
} // namespace tensor