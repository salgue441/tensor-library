file(GLOB_RECURSE UNIT_TEST_SOURCES "*.cpp")

add_executable(unit_tests
    ${UNIT_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        tf
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Use add_test instead of gtest_discover_tests for now
add_test(NAME unit_tests COMMAND unit_tests)