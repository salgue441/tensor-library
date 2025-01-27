include(CheckCXXCompilerFlag)

function(tf_set_compiler_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4             # Warning level 4
            /WX             # Treat warnings as errors
            /permissive-    # Standards conformance
            /Zc:__cplusplus # Enable proper __cplusplus macro
            /utf-8          # Set source and execution character sets to UTF-8
            /volatile:iso   # Volatile behavior according to C++ standard
            /Zc:preprocessor # Enable standard-conforming preprocessor
            /EHsc          # Enable C++ exception handling
            /MP            # Multi-processor compilation
            /bigobj        # Increase number of sections in .obj file
        )
        
        if(TF_CUDA_SUPPORT)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:
                    -arch=sm_60
                    --expt-relaxed-constexpr
                    --extended-lambda
                >
            )
        endif()
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wno-unused-parameter
            -Wno-missing-field-initializers
            -fPIC
            -march=native
        )

        if(TF_CUDA_SUPPORT)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:
                    -arch=sm_60
                    --expt-relaxed-constexpr
                    --extended-lambda
                >
            )
        endif()
    endif()

    # Set C++20 features
    target_compile_features(${target} PRIVATE cxx_std_20)

    # Add defines based on configuration
    if(TF_ENABLE_PROFILING)
        target_compile_definitions(${target} PRIVATE TF_PROFILING)
    endif()

    if(TF_USE_CUDA)
        target_compile_definitions(${target} PRIVATE TF_CUDA_ENABLED)
    endif()

    if(TF_USE_BLAS)
        target_compile_definitions(${target} PRIVATE TF_BLAS_ENABLED)
    endif()

    # Enable IPO/LTO if available
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT error)
    if(ipo_supported)
        set_target_properties(${target} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()
endfunction()