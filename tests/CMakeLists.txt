enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

# Unit test
add_executable(tensor_tests
    unit/storage_test.cpp
    unit/tensor_test.cpp
    unit/expression_test.cpp
)

target_link_libraries(tensor_tests
    PRIVATE
        tensor
        GTest::GTest
        GTest::Main
)

target_include_directories(tensor_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(tensor_tests)