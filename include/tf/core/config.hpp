#pragma once

#include <tf/core/types.hpp>
#include <tf/core/error.hpp>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tf
{
  namespace core
  {
    // Forward declaration
    class ConfigOption;
    class Configuration;

    /**
     * @class Configuration
     * @brief Global configuration singleton class
     */
    class Configuration
    {
    public:
      /**
       * @brief Gets the default instance of the configuration
       *
       * @return Configuration& reference to the configuration instance
       */
      static Configuration &instance()
      {
        static Configuration instance;
        return instance;
      }

      // Device configuration
      /**
       * @brief Sets the default device
       *
       * @param device Device type
       */
      void set_default_device(DeviceType device)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_default_device = device;
      }

      /**
       * @brief Gets the default device
       *
       * @return DeviceType Default device type
       */
      DeviceType default_device() const
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_default_device;
      }

      // Memory configuration
      /**
       * @brief Sets the memory fraction
       *
       * @param fraction Memory fraction
       * @throw ValueError if the fraction is not in the range (0, 1]
       */
      void set_memory_fraction(float fraction)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        TF_CHECK(fraction > 0.0f && fraction <= 1.0f, ValueError,
                 "Memory fraction must be in the range (0, 1]");

        m_memory_fraction = fraction;
      }

      /**
       * @brief Gets the memory fraction
       *
       * @return float Memory fraction
       */
      float memory_fraction() const
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_memory_fraction;
      }

      // Threading configuration
      /**
       * @brief Sets the number of threads
       *
       * @param num_threads Number of threads
       * @throw ValueError if the number of threads is not positive
       */
      void set_num_threads(int num_threads)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        TF_CHECK(num_threads > 0, ValueError,
                 "Number of threads must be positive");

        m_num_threads = num_threads;
      }

      /**
       * @brief Gets the number of threads
       *
       * @return int Number of threads
       */
      int num_threads() const
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_num_threads;
      }

      // Debug configuration
      /**
       * @brief Sets the debug mode
       *
       * @param debug_mode Debug mode flag
       */
      void set_debug_mode(bool debug_mode)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_debug_mode = debug_mode;
      }

      /**
       * @brief Gets the debug mode
       *
       * @return bool Debug mode flag
       */
      bool debug_mode() const
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_debug_mode;
      }

      // Custom options
      /**
       * @brief Sets a custom option value by name
       *
       * @tparam T Type of the option
       * @param name Name of the option
       * @param default_value Default value of the option
       * @return T Value of the option
       */
      template <typename T>
      void set_option(const std::string &name, const T &value)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_options[name] = std::make_shared<ConfigOptionsImpl<T>>(value);
      }

      /**
       * @brief Gets a custom option value by name
       *
       * @tparam T Type of the option
       * @param name Name of the option
       * @param default_value Default value of the option
       * @return T Value of the option
       */
      template <typename T>
      T get_option(const std::string &name, const T &default_value) const
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_options.find(name);

        if (it == m_options.end())
          return default_value;

        auto option = std::dynamic_pointer_cast<ConfigOptionsImpl<T>>(it->second);
        if (!option)
          throw TypeError("Invalid option type");

        return option->value();
      }

    private:
      mutable std::mutex m_mutex;
      DeviceType m_default_device;
      float m_memory_fraction;
      int m_num_threads;
      bool m_debug_mode;

      // Constructor
      Configuration()
          : m_default_device(DeviceType::CPU), m_memory_fraction(0.9f),
            m_num_threads(4), m_debug_mode(false) {}

      Configuration(const Configuration &) = delete;
      Configuration &operator=(const Configuration &) = delete;
      Configuration(Configuration &&) = delete;
      Configuration &operator=(Configuration &&) = delete;

      // Base class for configuration options
      class ConfigOption
      {
      public:
        virtual ~ConfigOption() = default;
      };

      // Templated implementation for configuration options
      template <typename T>
      class ConfigOptionsImpl : public ConfigOption
      {
      public:
        explicit ConfigOptionsImpl(const T &value) : m_value(value) {}
        T value() const { return m_value; }

      private:
        T m_value;
      };

      std::unordered_map<std::string, std::shared_ptr<ConfigOption>> m_options;
    };

    /**
     * @brief Global configuration accessor functions
     *
     * @return Configuration& Reference to the global configuration
     */
    inline Configuration &config()
    {
      return Configuration::instance();
    }

    /**
     * @class ConfigGuard
     * @brief RAII guard for configuration changes
     */
    template <typename T>
    class ConfigGuard
    {
    public:
      ConfigGuard(const std::string &name, const T &value)
          : m_name(name), m_old_value(config().get_option<T>(name, value))
      {
        config().set_option(name, value);
      }

      ~ConfigGuard()
      {
        try
        {
          config().set_option(m_name, m_old_value);
        }
        catch (...)
        {
        }
      }

    private:
      std::string m_name;
      T m_old_value;
    };

// Macros
#define TF_CONFIG Configuration::instance()
#define TF_WITH_CONFIG(name, value) \
  auto _config_guard##__LINE__ = ConfigGuard(name, value)

  } // namespace core
} // namespace tf