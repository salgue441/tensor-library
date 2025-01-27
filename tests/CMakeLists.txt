enable_testing()

# Find GTest
find_package(GTest REQUIRED)

# Create test executable
add_executable(tensor_tests
    unit/core/dtype_test.cpp
    unit/core/device_test.cpp
    unit/core/device_properties_test.cpp
    unit/core/device_memory_test.cpp
)

# Link dependencies
target_link_libraries(tensor_tests
    PRIVATE
        tensor_core
        GTest::GTest
        GTest::Main
)

# Add test directory to include path
target_include_directories(tensor_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register tests
include(GoogleTest)
gtest_discover_tests(tensor_tests)