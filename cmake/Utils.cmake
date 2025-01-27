function(tf_add_test)
    cmake_parse_arguments(TEST
        ""
        "NAME;DIRECTORY"
        "SOURCES;DEPENDENCIES"
        ${ARGN}
    )

    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            tf
            GTest::gtest
            GTest::gtest_main
            ${TEST_DEPENDENCIES}
    )

    if(TEST_DIRECTORY)
        set_target_properties(${TEST_NAME}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${TEST_DIRECTORY}"
        )
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 120
        WILL_FAIL FALSE
    )
endfunction()

# Function to add a benchmark executable
function(tf_add_benchmark)
    cmake_parse_arguments(BENCH
        ""
        "NAME;DIRECTORY"
        "SOURCES;DEPENDENCIES"
        ${ARGN}
    )

    add_executable(${BENCH_NAME} ${BENCH_SOURCES})
    target_link_libraries(${BENCH_NAME}
        PRIVATE
            tf
            benchmark::benchmark
            benchmark::benchmark_main
            ${BENCH_DEPENDENCIES}
    )

    if(BENCH_DIRECTORY)
        set_target_properties(${BENCH_NAME}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${BENCH_DIRECTORY}"
        )
    endif()
endfunction()

# Function to add an example executable
function(tf_add_example)
    cmake_parse_arguments(EXAMPLE
        ""
        "NAME;DIRECTORY"
        "SOURCES;DEPENDENCIES"
        ${ARGN}
    )

    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_NAME}
        PRIVATE
            tf
            ${EXAMPLE_DEPENDENCIES}
    )

    if(EXAMPLE_DIRECTORY)
        set_target_properties(${EXAMPLE_NAME}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIRECTORY}"
        )
    endif()
endfunction()

# Function to install headers
function(tf_install_headers)
    cmake_parse_arguments(INSTALL
        ""
        "DESTINATION"
        "HEADERS"
        ${ARGN}
    )

    foreach(header ${INSTALL_HEADERS})
        get_filename_component(dir ${header} DIRECTORY)
        install(
            FILES ${header}
            DESTINATION ${INSTALL_DESTINATION}/${dir}
        )
    endforeach()
endfunction()

# Function to configure version header
function(tf_configure_version)
    configure_file(
        ${PROJECT_SOURCE_DIR}/include/tf/version.hpp.in
        ${PROJECT_BINARY_DIR}/include/tf/version.hpp
        @ONLY
    )
endfunction()