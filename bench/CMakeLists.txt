if(NOT TF_BUILD_BENCHMARKS)
    return()
endif()

file(GLOB_RECURSE BENCHMARK_SOURCES "*.cpp")

add_executable(benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(benchmarks PRIVATE
    tf::tf
    benchmark::benchmark
    benchmark::benchmark_main
)
