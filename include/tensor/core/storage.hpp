#pragma once

#include "../utils/concepts.hpp"
#include "../utils/exceptions.hpp"
#include <vector>
#include <memory>

namespace tensor
{
  template <NumericType T>
  class TensorStorage
  {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    /**
     * @brief Construct a new Tensor Storage object
     *
     * @param size Size of the storage
     */
    explicit TensorStorage(size_type size) : data_(size) {}

    /**
     * @brief Construct a new Tensor Storage object
     *
     * @param size Size of the storage
     * @param value Value to initialize the storage
     */
    TensorStorage(size_type size, const T &value) : data_(size, value) {}

    /**
     * @brief Construct a new Tensor Storage object
     *
     * @tparam InputIt Iterator type
     * @param first Iterator to the first element
     * @param last Iterator to the last element
     */
    template <typename InputIt>
    TensorStorage(InputIt first, InputIt last) : data_(first, last) {}

    // Element access
    /**
     * @brief Operator [] to access elements of the storage
     *
     * @param pos Position of the element
     * @return T& Reference to the element
     */
    T &operator[](size_type pos) { return data_[pos]; }

    /**
     * @brief Operator [] to access elements of the storage
     *
     * @param pos Position of the element
     * @return const T& Reference to the element
     */
    const T &operator[](size_type pos) const { return data_[pos]; }

    /**
     * @brief Access element at position pos
     *
     * @param pos Position of the element
     * @return T& Reference to the element
     */
    T &at(size_type pos) { return data_.at(pos); }

    /**
     * @brief Access element at position pos
     *
     * @param pos Position of the element
     * @return const T& Reference to the element
     */
    const T &at(size_type pos) const { return data_.at(pos); }

    // Iterators
    /**
     * @brief Get an iterator to the beginning
     *
     * @return iterator Iterator to the beginning
     */
    iterator begin() noexcept { return data_.begin(); }

    /**
     * @brief Get a const iterator to the beginning
     *
     * @return const_iterator Const iterator to the beginning
     */
    const_iterator begin() const noexcept { return data_.begin(); }

    /**
     * @brief Get an iterator to the end
     *
     * @return iterator Iterator to the end
     */
    iterator end() noexcept { return data_.end(); }

    /**
     * @brief Get a const iterator to the end
     *
     * @return const_iterator Const iterator to the end
     */
    const_iterator end() const noexcept { return data_.end(); }

    // Capacity
    /**
     * @brief Check if the storage is empty
     *
     * @return true If the storage is empty
     * @return false If the storage is not empty
     */
    bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Get the size of the storage
     *
     * @return size_type Size of the storage
     */
    size_type size() const noexcept { return data_.size(); }

    /**
     * @brief Resize the storage to the given size
     *
     * @return size_type Capacity of the storage
     */
    void resize(size_type count) { data_.resize(count); }

    /**
     * @brief Resize the storage to the given size
     *
     * @return size_type Capacity of the storage
     */
    void reserve(size_type new_cap) { data_.reserve(new_cap); }

    // Modifiers
    /**
     * @brief Clear the storage
     *
     */
    void clear() noexcept { data_.clear(); }

    /**
     * @brief Insert an element at the end
     *
     * @tparam Args Types of the arguments
     * @param args Arguments to construct the element
     */
    template <typename... Args>
    void emplace_back(Args &&...args)
    {
      data_.emplace_back(std::forward<Args>(args)...);
    }

    // Memory management
    /**
     * @brief Swap the storage with another storage
     *
     * @param other Storage to swap with
     */
    void swap(TensorStorage &other) noexcept
    {
      data_.swap(other.data_);
    }

    // Raw data access
    /**
     * @brief Get a pointer to the data
     *
     * @return T* Pointer to the data
     */
    T *data() noexcept { return data_.data(); }

    /**
     * @brief Get a const pointer to the data
     *
     * @return const T* Const pointer to the data
     */
    const T *data() const noexcept { return data_.data(); }

  private:
    std::vector<T> data_;
  };
} // namespace tensor