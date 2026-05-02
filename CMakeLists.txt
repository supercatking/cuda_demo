cmake_minimum_required(VERSION 3.24)

project(cuda_demo LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_gemm_demo
    src/gemm/main.cu
)

add_executable(cuda_stream_overlap_demo
    src/stream_overlap/main.cu
)

set_target_properties(cuda_gemm_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_stream_overlap_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(cuda_gemm_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_options(cuda_stream_overlap_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
