#pragma once

#include <stdexcept>
#include <string>

namespace tensor
{
  /**
   * @class TensorException
   * @brief Extends runtime error for tensor related exceptions
   *
   * @version 1.0.0
   */
  class TensorException : public std::runtime_error
  {
  public:
    explicit TensorException(const std::string &message)
        : std::runtime_error(message) {};
  };

  /**
   * @class DeviceError
   * @brief Exception for device related errors
   *
   * @version 1.0.0
   */
  class DeviceError : public TensorException
  {
  public:
    explicit DeviceError(const std::string &message)
        : TensorException("Device error: " + message) {}
  };

  /**
   * @class TypeError
   * @brief Exception for type related errors
   *
   * @version 1.0.0
   */
  class TypeError : public TensorException
  {
  public:
    explicit TypeError(const std::string &message)
        : TensorException("Type error: " + message) {}
  };

  /**
   * @class StorageError
   * @brief Exception for storage related errors
   *
   * @version 1.0.0
   */
  class StorageError : public TensorException
  {
  public:
    explicit StorageError(const std::string &message)
        : TensorException("Storage error: " + message) {}
  };

  /**
   * @class IndexError
   * @brief Exception for index related errors
   *
   * @version 1.0.0
   */
  class IndexError : public TensorException
  {
  public:
    explicit IndexError(const std::string &message)
        : TensorException("Index error: " + message) {}
  };

  /**
   * @class ShapeError
   * @brief Exception for shape related errors
   *
   * @version 1.0.0
   */
  class ShapeError : public TensorException
  {
  public:
    explicit ShapeError(const std::string &message)
        : TensorException("Shape error: " + message) {}
  };

  /**
   * @class NotImplementedError
   * @brief Exception for not implemented errors
   *
   * @version 1.0.0
   */
  class NotImplementedError : public TensorException
  {
  public:
    explicit NotImplementedError(const std::string &message)
        : TensorException("Not implemented: " + message) {}
  };
} // namespace tensor