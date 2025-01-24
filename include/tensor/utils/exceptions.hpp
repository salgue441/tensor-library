#pragma once

#include <stdexcept>
#include <string>

namespace tensor
{
  /**
   * @brief TensorException base class for all custom exceptions
   *
   */
  class TensorException : public std::runtime_error
  {
  public:
    /**
     * @brief Construct a new Tensor Exception object
     *
     * @param msg Message to be displayed
     */
    explicit TensorException(const std::string &msg) : std::runtime_error(msg) {}
  };

  /**
   * @brief Exception thrown when tensor dimensions do not match
   *
   */
  class DimensionMismatch : public TensorException
  {
  public:
    explicit DimensionMismatch(const std::string &msg) : TensorException(msg) {}
  };

  /**
   * @brief Exception thrown when tensor shapes do not match
   *
   */
  class ShapeError : public TensorException
  {
  public:
    explicit ShapeError(const std::string &msg) : TensorException(msg) {}
  };

  /**
   * @brief Exception thrown when tensor memory allocation fails
   *
   */
  class MemoryError : public TensorException
  {
  public:
    explicit MemoryError(const std::string &msg) : TensorException(msg) {}
  };
} // namespace tensor