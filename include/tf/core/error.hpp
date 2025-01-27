#pragma once

#include <stdexcept>
#include <string>
#include <source_location>
#include <format>

namespace tf
{
  namespace core
  {
    /**
     * @class Exception
     * @brief base Exception class
     */
    class Exception : public std::runtime_error
    {
    public:
      explicit Exception(const std::string &message,
                         const std::source_location &location = std::source_location::current())
          : std::runtime_error(format_message(message, location)),
            m_message(message),
            m_file(location.file_name()),
            m_line(location.line()),
            m_function(location.function_name()) {}

      // Accessors
      /**
       * @brief Gets the message of the exception
       *
       * @return std::string Message of the exception
       */
      const std::string &message() const noexcept { return m_message; }

      /**
       * @brief Gets the file where the exception was thrown
       *
       * @return std::string File where the exception was thrown
       */
      const std::string &file() const noexcept { return m_file; }

      /**
       * @brief Gets the line where the exception was thrown
       *
       * @return size_t Line where the exception was thrown
       */
      size_t line() const noexcept { return m_line; }

      /**
       * @brief Gets the function where the exception was thrown
       *
       * @return std::string Function where the exception was thrown
       */
      const std::string &function() const noexcept { return m_function; }

    private:
      std::string m_message;
      std::string m_file;
      size_t m_line;
      std::string m_function;

      /**
       * @brief Formats the message with the location information
       *
       * @param message Message to format
       * @param location Location information
       * @return std::string Formatted message
       */
      std::string format_message(const std::string &message,
                                 const std::source_location &location)
      {
        return std::format("{}:{} in {}:{}",
                           location.file_name(),
                           location.line(),
                           location.function_name(),
                           message);
      }
    };

    /**
     * @class ShapeError
     * @brief Exception thrown when a shape error occurs
     */
    class ShapeError : public Exception
    {
    public:
      using Exception::Exception;
    };

    /**
     * @class DeviceError
     * @brief Exception thrown when a device error occurs
     */
    class DeviceError : public Exception
    {
    public:
      using Exception::Exception;
    };

    /**
     * @class MemoryError
     * @brief Exception thrown when a memory error occurs
     */
    class MemoryError : public Exception
    {
    public:
      using Exception::Exception;
    };

    /**
     * @class TypeError
     * @brief Exception thrown when a type error occurs
     */
    class TypeError : public Exception
    {
    public:
      using Exception::Exception;
    };

    /**
     * @class IndexError
     * @brief Exception thrown when an index error occurs
     */
    class IndexError : public Exception
    {
    public:
      using Exception::Exception;
    };

    /**
     * @class NotImplementedError
     * @brief Exception thrown when a method is not implemented
     */
    class NotImplementedError : public Exception
    {
    public:
      using Exception::Exception;
    };

    /**
     * @class ValueError
     * @brief Exception thrown when a value error occurs
     */
    class ValueError : public Exception
    {
    public:
      using Exception::Exception;
    };

    // Macros
    /**
     * @brief Checks a condition and throws an exception if it is false
     *
     * @param condition Condition to check
     * @param exception_type Type of exception to throw
     * @param message Message of the exception
     */
#define TF_CHECK(condition, exception_type, message) \
  do                                                 \
  {                                                  \
    if (!(condition))                                \
    {                                                \
      throw exception_type(message);                 \
    }                                                \
  } while (0)

    /**
     * @brief Checks a condition and throws a ShapeError if it is false
     *
     * @param condition Condition to check
     * @param message Message of the exception
     */
#define TF_CHECK_SHAPE(condition, message) \
  TF_CHECK(condition, ShapeError, message)

    /**
     * @brief Checks a condition and throws a DeviceError if it is false
     *
     * @param condition Condition to check
     * @param message Message of the exception
     */
#define TF_CHECK_DEVICE(condition, message) \
  TF_CHECK(condition, DeviceError, message)

    /**
     * @brief Checks a condition and throws a MemoryError if it is false
     *
     * @param condition Condition to check
     * @param message Message of the exception
     */
#define TF_CHECK_MEMORY(condition, message) \
  TF_CHECK(condition, MemoryError, message)

    /**
     * @brief Checks a condition and throws a TypeError if it is false
     *
     * @param condition Condition to check
     * @param message Message of the exception
     */
#define TF_CHECK_TYPE(condition, message) \
  TF_CHECK(condition, TypeError, message)

    /**
     * @brief Checks a condition and throws an IndexError if it is false
     *
     * @param condition Condition to check
     * @param message Message of the exception
     */
#define TF_CHECK_INDEX(condition, message) \
  TF_CHECK(condition, IndexError, message)

    /**
     * @brief Throws a NotImplementedError
     *
     * @param message Message of the exception
     */
#define TF_NOT_IMPLEMENTED(message) \
  throw NotImplementedError(message)
  } // namespace core

} // namespace tf