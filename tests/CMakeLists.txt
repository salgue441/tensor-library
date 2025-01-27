# tests/CMakeLists.txt
if(NOT TF_BUILD_TESTS)
    return()
endif()

# Enable testing
enable_testing()

# Find GTest package
find_package(GTest CONFIG REQUIRED)
include(GoogleTest) 

# Add the unit tests subdirectory
add_subdirectory(unit)