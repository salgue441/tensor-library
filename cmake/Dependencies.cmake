include(FetchContent)

# Function to handle external dependencies
function(tf_setup_dependencies)
    # Find required packages
    find_package(GTest CONFIG REQUIRED)
    find_package(benchmark CONFIG REQUIRED)
    find_package(fmt CONFIG REQUIRED)
    find_package(spdlog CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(CLI11 CONFIG REQUIRED)

    # Optional CUDA support
    if(TF_USE_CUDA)
        find_package(CUDA REQUIRED)
        if(CUDA_FOUND)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()
    endif()

    # Optional BLAS support
    if(TF_USE_BLAS)
        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            add_definitions(-DTF_USE_BLAS)
        endif()
    endif()

    # Handle third-party libraries not available in vcpkg
    if(TF_BUILD_TESTS)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG master
        )
        FetchContent_MakeAvailable(rapidcheck)
    endif()
endfunction()