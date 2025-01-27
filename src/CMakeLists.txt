add_library(tensor_core
    core/device/device.cpp
    core/device/device_properties.cpp
    core/device/device_memory.cpp
)

target_include_directories(tensor_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(tensor_core PUBLIC USE_CUDA)
    target_link_libraries(tensor_core PUBLIC CUDA::cudart)
endif()

# Add OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tensor_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Platform-specific flags
if(MSVC)
    target_compile_options(tensor_core PRIVATE /W4 /arch:AVX2)
else()
    target_compile_options(tensor_core PRIVATE -Wall -Wextra -mavx2)
endif()