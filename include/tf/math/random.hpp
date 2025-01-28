#pragma once

#include <tf/core/types.hpp>
#include <tf/core/error.hpp>

#include <random>
#include <chrono>
#include <mutex>
#include <thread>
#include <memory>

namespace tf
{
  namespace math
  {
    /**
     * @class RandomGenerator
     * @brief Thread-safe random number generator with various distributions
     */
    class RandomGenerator
    {
    public:
      static RandomGenerator &instance()
      {
        static RandomGenerator instance;
        return instance;
      }

      /**
       * @brief Set the seed object for the random number generator
       *
       * @param seed The seed value
       * @version 1.0.0
       */
      void set_seed(uint64_t seed)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_generator.seed(seed);
      }

      /**
       * @brief Uniform distribution random number generator
       *
       * @tparam T Type of the random number
       * @param min Minimum value
       * @param max Maximum value
       * @return T Random number in the range [min, max]
       * @version 1.0.0
       */
      template <typename T>
      T uniform(T min = T{0}, T max = T{1})
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_integral_v<T>)
        {
          std::uniform_int_distribution<T> dist(min, max);
          return dist(m_generator);
        }
        else
        {
          std::uniform_real_distribution<T> dist(min, max);
          return dist(m_generator);
        }
      }

      /**
       * @brief Normal distribution random number generator
       *
       * @tparam T Type of the random number
       * @param mean Mean value
       * @param stddev Standard deviation value
       * @return T Random number from the normal distribution
       * @version 1.0.0
       */
      template <typename T>
      T normal(T mean = T{0}, T stddev = T{1})
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        std::normal_distribution<T> dist(mean, stddev);
        return dist(m_generator);
      }

      /**
       * @brief Bernoulli distribution random number generator
       *
       * @param p Probability of success
       * @return bool Random boolean value
       * @version 1.0.0
       */
      bool bernoulli(double p = 0.5)
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::bernoulli_distribution dist(p);

        return dist(m_generator);
      }

      /**
       * @brief Generate array of random values
       *
       * @tparam T Type of the random number
       * @param data Pointer to the array
       * @param size Size of the array
       * @param min Minimum value
       * @param max Maximum value
       */
      template <typename T>
      void fill_uniform(T *data, size_t size, T min = T{0}, T max = T{1})
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_integral_v<T>)
        {
          std::uniform_int_distribution<T> dist(min, max);
          for (size_t i = 0; i < size; ++i)
            data[i] = dist(m_generator);
        }
        else
        {
          std::uniform_real_distribution<T> dist(min, max);
          for (size_t i = 0; i < size; ++i)
            data[i] = dist(m_generator);
        }
      }

      /**
       * @brief Generate array of random values from normal distribution
       *
       * @tparam T Type of the random number
       * @param data Pointer to the array
       * @param size Size of the array
       * @param mean Mean value
       * @param stddev Standard deviation value
       */
      template <typename T>
      void fill_normal(T *data, size_t size, T mean = T{0}, T stddev = T{1})
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        std::normal_distribution<T> dist(mean, stddev);
        for (size_t i = 0; i < size; ++i)
          data[i] = dist(m_generator);
      }

    private:
      mutable std::mutex m_mutex;
      std::mt19937_64 m_generator;

      RandomGenerator()
      {
        auto seed = std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count();

        m_generator.seed(static_cast<uint64_t>(seed));
      }
    };
  } // namespace math
} // namespace tf