#pragma once

#include <tf/core/config.hpp>

// Version information
#define TF_MAJOR_VERSION 0
#define TF_MINOR_VERSION 1
#define TF_PATCH_VERSION 0

// Compiler detection
#if defined(_MSC_VER)
#define TF_COMPILER_MSVC
#elif defined(__GNUC__)
#define TF_COMPILER_GCC
#elif defined(__clang__)
#define TF_COMPILER_CLANG
#endif

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
#define TF_PLATFORM_WINDOWS
#elif defined(__linux__)
#define TF_PLATFORM_LINUX
#elif defined(__APPLE__)
#define TF_PLATFORM_MACOS
#endif

// Export / Import macros
#if defined(TF_PLATFORM_WINDOWS)
#if defined(TF_BUILD_SHARED)
#define TF_API __declspec(dllexport)
#elif defined(TF_USE_SHARED)
#define TF_API __declspec(dllimport)
#else
#define TF_API
#endif
#else
#define TF_API __attribute__((visibility("default")))
#endif

// Function attributes
#if defined(TF_COMPILER_MSVC)
#define TF_FORCE_INLINE __forceinline
#define TF_NOINLINE __declspec(noinline)
#else
#define TF_FORCE_INLINE __attribute__((always_inline)) inline
#define TF_NOINLINE __attribute__((noinline))
#endif

// Debug macros
#if defined(NDEBUG)
#define TF_DEBUG_MODE 0
#else
#define TF_DEBUG_MODE 1
#endif

#define TF_DEBUG_ONLY(x) \
  do                     \
  {                      \
    if (TF_DEBUG_MODE)   \
    {                    \
      x;                 \
    }                    \
  } while (0)

// Assert macros
#define TF_ASSERT(condition, message)          \
  do                                           \
  {                                            \
    if (!(condition))                          \
    {                                          \
      throw tf::core::AssertionError(message); \
    }                                          \
  } while (0)

#define TF_DEBUG_ASSERT(condition, message) \
  TF_DEBUG_ONLY(TF_ASSERT(condition, message))

// Memory alignment
#define TF_ALIGNMENT 16
#define TF_ALIGNED alignas(TF_ALIGNMENT)

// CUDA support
#if defined(TF_USE_CUDA)
#define TF_CUDA_CALLABLE __host__ __device__
#else
#define TF_CUDA_CALLABLE
#endif

// Function declaration helpers
#define TF_NODISCARD [[nodiscard]]
#define TF_DEPRECATED [[deprecated]]
#define TF_MAYBE_UNUSED [[maybe_unused]]

// Namespace helpers
#define TF_NAMESPACE_BEGIN \
  namespace tf             \
  {
#define TF_NAMESPACE_END }

// String manipulation
#define TF_STRINGIFY(x) #x
#define TF_STRINGIFY_MACRO(x) TF_STRINGIFY(x)

// Version string
#define TF_VERSION_STRING              \
  TF_STRINGIFY_MACRO(TF_MAJOR_VERSION) \
  "." TF_STRINGIFY_MACRO(TF_MINOR_VERSION) "." TF_STRINGIFY_MACRO(TF_PATCH_VERSION)

// Unused variable
#define TF_UNUSED(x) (void)(x)

// Prevent copy and move
#define TF_DISABLE_COPY(Class)   \
  Class(const Class &) = delete; \
  Class &operator=(const Class &) = delete

#define TF_DISABLE_MOVE(Class) \
  Class(Class &&) = delete;    \
  Class &operator=(Class &&) = delete

#define TF_DISABLE_COPY_AND_MOVE(Class) \
  TF_DISABLE_COPY(Class);               \
  TF_DISABLE_MOVE(Class)

// Singleton helper
#define TF_DECLARE_SINGLETON(Class) \
public:                             \
  static Class &instance()          \
  {                                 \
    static Class instance;          \
    return instance;                \
  }                                 \
                                    \
private:                            \
  Class();                          \
  TF_DISABLE_COPY_AND_MOVE(Class)

// Performance helpers
#define TF_LIKELY(x) __builtin_expect(!!(x), 1)
#define TF_UNLIKELY(x) __builtin_expect(!!(x), 0)

// Scope guard helper
#define TF_SCOPE_EXIT(x) \
  auto TF_MAYBE_UNUSED _scope_exit##__LINE__ = tf::core::ScopeGuard([&]() { x; })