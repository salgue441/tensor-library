#pragma once

#include <tf/core/types.hpp>
#include <tf/core/error.hpp>
#include <cstddef>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <memory>
#include <algorithm>
#include <numeric>

namespace tf
{
  namespace core
  {
    class Shape
    {
    public:
      using value_type = index_t;
      using container_type = std::vector<value_type>;
      using iterator = container_type::iterator;
      using const_iterator = container_type::const_iterator;

      Shape() = default;
      Shape(std::initializer_list<value_type> dims) : m_dims(dims) {}
      explicit Shape(const container_type &dims) : m_dims(dims) {}

      // Access
      /**
       * @brief Operator [] to access the shape dimensions
       *
       * @param idx Index of the dimension
       * @return value_type Dimension value
       */
      value_type &operator[](size_t idx) { return m_dims[idx]; }

      /**
       * @brief Operator [] to access the shape dimensions
       *
       * @param idx Index of the dimension
       * @return value_type Dimension value
       */
      const value_type &operator[](size_t idx) const { return m_dims[idx]; }

      // Iterators
      /**
       * @brief Get the begin iterator of the shape dimensions
       *
       * @return const_iterator Begin iterator
       */
      iterator begin() { return m_dims.begin(); }

      /**
       * @brief Get the end iterator of the shape dimensions
       *
       * @return const_iterator End iterator
       */
      iterator end() { return m_dims.end(); }

      /**
       * @brief Get the begin iterator of the shape dimensions
       *
       * @return const_iterator Begin iterator
       */
      const_iterator begin() const { return m_dims.begin(); }

      /**
       * @brief Get the end iterator of the shape dimensions
       *
       * @return const_iterator End iterator
       */
      const_iterator end() const { return m_dims.end(); }

      // Properties
      /**
       * @brief Get the number of dimensions
       *
       * @return size_t Number of dimensions
       */
      size_t rank() const { return m_dims.size(); }

      /**
       * @brief Returns true if the shape is empty
       *
       * @return true If the shape is empty, false otherwise
       */
      bool empty() const { return m_dims.empty(); }

      /**
       * @brief Get the total number of elements in the shape
       *
       * @return size_t Total number of elements
       */
      value_type num_elements() const
      {
        return std::accumulate(m_dims.begin(), m_dims.end(),
                               value_type{1},
                               std::multiplies<value_type>());
      }

      /**
       * @brief String representation of the shape
       *
       * @return std::string String representation of the shape
       */
      std::string to_string() const
      {
        std::ostringstream oss;

        oss << "(";
        for (size_t i = 0; i < m_dims.size(); ++i)
        {
          if (i > 0)
            oss << ", ";

          oss << m_dims[i];
        }

        oss << ")";
        return oss.str();
      }

      // Comparison
      /**
       * @brief Equality operator
       *
       * @param other Other shape to compare
       * @return true If the shapes are equal, false otherwise
       */
      bool operator==(const Shape &other) const
      {
        return m_dims == other.m_dims;
      }

      /**
       * @brief Inequality operator
       *
       * @param other Other shape to compare
       * @return true If the shapes are not equal, false otherwise
       */
      bool operator!=(const Shape &other) const
      {
        return !(*this == other);
      }

      // Broadcasting
      /**
       * @brief Checks if the shape is broadcastable to another shape
       *
       * @param other Other shape to check
       * @return true If the shape is broadcastable, false otherwise
       */
      bool is_broadcastable_to(const Shape &other) const
      {
        if (rank() > other.rank())
          return false;

        auto it1 = m_dims.rbegin();
        auto it2 = other.m_dims.rbegin();

        while (it1 != m_dims.rend() && it2 != other.m_dims.rend())
        {
          if (*it1 != 1 && *it1 != *it2)
            return false;

          ++it1;
          ++it2;
        }

        return true;
      }

    private:
      container_type m_dims;
    };

    template <typename T>
    class Memory
    {
    public:
      /**
       * @brief Allocates memory of a given size
       *
       * @param size Size of the memory to allocate
       * @return std::shared_ptr<T[]> Shared pointer to the allocated memory
       */
      static std::shared_ptr<T[]> allocate(size_t size)
      {
        return std::make_shared<T[]>(size);
      }

      /**
       * @brief Copies memory from one location to another
       *
       * @param dst Destination memory location
       * @param src Source memory location
       * @param size Size of the memory to copy
       */
      static void copy(T *dst, const T *src, size_t size)
      {
        std::copy_n(src, size, dst);
      }

      /**
       * @brief Fills memory with a given value
       *
       * @param ptr Memory location
       * @param size Size of the memory to fill
       * @param value Value to fill the memory with
       */
      static void fill(T *ptr, size_t size, const T &value)
      {
        std::fill_n(ptr, size, value);
      }
    };

    /**
     * @class ScopeGuard
     * @brief RAII-style scope guard
     */
    class ScopeGuard
    {
    public:
      template <typename F>
      explicit ScopeGuard(F &&f) : m_func(std::forward<F>(f)) {}

      ~ScopeGuard()
      {
        if (m_func)
          m_func();
      }

      // Disable copy
      ScopeGuard(const ScopeGuard &) = delete;
      ScopeGuard &operator=(const ScopeGuard &) = delete;

      // Enable move
      /**
       * @brief Move construct a new Scope Guard object
       *
       * @param other Other ScopeGuard object
       */
      ScopeGuard(ScopeGuard &&other) noexcept
          : m_func(std::move(other.m_func))
      {
        other.m_func = nullptr;
      }

      /**
       * @brief Move assign a new Scope Guard object
       *
       * @param other Other ScopeGuard object
       * @return ScopeGuard& Reference to the new ScopeGuard object
       */
      ScopeGuard &operator=(ScopeGuard &&other) noexcept
      {
        if (this != &other)
        {
          m_func = std::move(other.m_func);
          other.m_func = nullptr;
        }

        return *this;
      }

    private:
      std::function<void()> m_func;
    };

    // Utility functions
    /**
     * @brief Gets the name of a type
     *
     * @tparam T Type to get the name of
     * @return std::string Name of the type
     */
    template <typename T>
    std::string type_name()
    {
      return typeid(T).name();
    }

    /**
     * @brief Checks if a pointer is aligned to a given alignment value
     *
     * @tparam T Type of the pointer
     * @param ptr Pointer to check
     * @param alignment Alignment value
     * @return true If the pointer is aligned
     * @return false If the pointer is not aligned
     */
    template <typename T>
    bool is_aligned(const T *ptr, size_t alignment)
    {
      return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
    }

    /**
     * @brief Aligns a pointer to a given alignment value
     *
     * @tparam T Type of the pointer
     * @param ptr Pointer to align
     * @param alignment Alignment value
     * @return T*  Aligned pointer
     */
    template <typename T>
    T *align_pointer(T *ptr, size_t alignment)
    {
      std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
      std::uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);

      return reinterpret_cast<T *>(aligned);
    }
  } // namespace core
} // namespace tf