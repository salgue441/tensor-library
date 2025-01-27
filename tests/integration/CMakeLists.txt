file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "*.cpp")

add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})
target_link_libraries(integration_tests PRIVATE ${TEST_COMMON_LIBS})
gtest_discover_tests(integration_tests)