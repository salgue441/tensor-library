#pragma once

#include <tf/core/types.hpp>
#include <tf/core/error.hpp>
#include <tf/core/macros.hpp>

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <thread>

namespace tf
{
  namespace utils
  {
    // Forward declarations
    class MemoryPool;
    class MemoryTracker;

    inline constexpr size_t DEFAULT_ALIGNMENT = 64;

    /**
     * @brief Allocate memory with the specified size and alignment.
     *
     * @tparam T Type of the memory to allocate.
     * @param ptr Pointer to the memory.
     * @param alignment Alignment of the memory.
     * @return T* Pointer to the allocated memory.
     */
    template <typename T>
    inline T *align_pointer(T *ptr, size_t alignment = DEFAULT_ALIGNMENT)
    {
      std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr);
      std::uintptr_t aligned_address = (address + alignment - 1) & ~(alignment - 1);

      return reinterpret_cast<T *>(aligned_address);
    }

    /**
     * @struct MemoryBlock
     * @brief Structure representing a memory block.
     */
    struct MemoryBlock
    {
      void *ptr;
      size_t size;
      size_t alignment;
      bool in_use;
    };

    /**
     * @class MemoryPool
     * @brief Class representing a memory pool.
     */
    class MemoryPool
    {
    public:
      explicit MemoryPool(size_t initial_size = 1024 * 1024)
          : m_total_size(0), m_max_block_size(0)
      {
        grow(initial_size);
      }

      ~MemoryPool()
      {
        for (MemoryBlock &block : m_blocks)
          ::operator delete(block.ptr);
      }

      /**
       * @brief Allocates memory with the specified size and alignment.
       *
       * @param size Size of the memory to allocate.
       * @param alignment Alignment of the memory.
       * @return void* Pointer to the allocated memory.
       */
      void *allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (MemoryBlock &block : m_blocks)
        {
          if (!block.in_use && block.size >= size)
          {
            block.in_use = true;
            return align_pointer(static_cast<char *>(block.ptr), alignment);
          }
        }

        size_t grow_size = std::max(size, m_total_size / 2);
        grow(grow_size);

        m_blocks.back().in_use = true;
        return align_pointer(static_cast<char *>(m_blocks.back().ptr),
                             alignment);
      }

      /**
       * @brief Deallocates memory.
       *
       * @param ptr Pointer to the memory to deallocate.
       */
      void deallocate(void *ptr)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (MemoryBlock &block : m_blocks)
        {
          if (block.ptr == ptr)
          {
            block.in_use = false;
            return;
          }
        }
      }

      // Statistics
      size_t total_size() const { return m_total_size; }
      size_t max_block_size() const { return m_max_block_size; }
      size_t num_blocks() const { return m_blocks.size(); }

    private:
      std::vector<MemoryBlock> m_blocks;
      size_t m_total_size;
      size_t m_max_block_size;
      mutable std::mutex m_mutex;

      /**
       * @brief Grows the memory pool by the specified size.
       *
       * @param min_size Minimum size to grow the memory pool.
       */
      void grow(size_t min_size)
      {
        size_t size = std::max(min_size, static_cast<size_t>(64));
        void *ptr = ::operator new(size);

        m_blocks.push_back({ptr, size, DEFAULT_ALIGNMENT, false});

        m_total_size += size;
        m_max_block_size = std::max(m_max_block_size, size);
      }
    };

    /**
     * @class MemoryTracker
     * @brief Class representing a memory tracker.
     */
    class MemoryTracker
    {
    public:
      /**
       * @brief Gets the memory tracker instance.
       *
       * @return MemoryTracker& Memory tracker instance.
       */
      static MemoryTracker &instance()
      {
        static MemoryTracker instance;
        return instance;
      }

      /**
       * @brief Tracks a memory allocation.
       *
       * @param ptr Pointer to the memory.
       * @param size Size of the memory.
       */
      void track_allocation(void *ptr, size_t size)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_allocations[ptr] = AllocationInfo{size, std::this_thread::get_id()};
        m_total_allocated += size;
        m_allocation_count++;
      }

      /**
       * @brief Tracks a memory deallocation.
       *
       * @param ptr Pointer to the memory.
       */
      void track_deallocation(void *ptr)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        auto it = m_allocations.find(ptr);
        if (it != m_allocations.end())
        {
          m_total_allocated -= it->second.size;
          m_deallocation_count++;
          m_allocations.erase(it);
        }
      }

      // Statistics
      size_t total_allocated() const { return m_total_allocated; }
      size_t allocation_count() const { return m_allocation_count; }
      size_t deallocation_count() const { return m_deallocation_count; }
      size_t active_allocations() const { return m_allocations.size(); }

      /**
       * @brief Resets the memory tracker statistics.
       */
      void reset_stats()
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_total_allocated = 0;
        m_allocation_count = 0;
        m_deallocation_count = 0;

        m_allocations.clear();
      }

    private:
      struct AllocationInfo
      {
        size_t size;
        std::thread::id thread_id;
      };

      std::unordered_map<void *, AllocationInfo> m_allocations;
      std::atomic<size_t> m_total_allocated{0};
      std::atomic<size_t> m_allocation_count{0};
      std::atomic<size_t> m_deallocation_count{0};
      mutable std::mutex m_mutex;

      MemoryTracker() = default;
    };

    /**
     * @class TrackedPointer
     * @brief RAII class for tracking memory allocations.
     */
    template <typename T>
    class TrackedPointer
    {
    public:
      explicit TrackedPointer(T *ptr = nullptr) : m_ptr(ptr)
      {
        if (m_ptr)
          MemoryTracker::instance().track_allocation(m_ptr, sizeof(T));
      }

      ~TrackedPointer()
      {
        if (m_ptr)
        {
          MemoryTracker::instance().track_deallocation(m_ptr);
          delete m_ptr;
        }
      }

      // Pointer operations
      T *get() const { return m_ptr; }
      T &operator*() const { return *m_ptr; }
      T *operator->() const { return m_ptr; }
      operator bool() const { return m_ptr != nullptr; }

      // Transfer ownership
      /**
       * @brief Releases the pointer.
       *
       * @return T* Pointer that was released.
       */
      T *release()
      {
        T *tmp = m_ptr;

        m_ptr = nullptr;
        return tmp;
      }

      /**
       * @brief Resets the pointer.
       *
       * @param ptr  Pointer to reset to.
       */
      void reset(T *ptr = nullptr)
      {
        if (m_ptr)
        {
          MemoryTracker::instance().track_deallocation(m_ptr);
          delete m_ptr;
        }

        m_ptr = ptr;
        if (m_ptr)
          MemoryTracker::instance().track_allocation(m_ptr, sizeof(T));
      }

    private:
      T *m_ptr;
    };

  } // namespace utils
} // namespace tf