cmake_minimum_required(VERSION 3.24)

project(cuda_demo LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function(add_cuda_demo target source)
    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endfunction()

add_cuda_demo(cuda_gemm_demo src/gemm/main.cu)
add_cuda_demo(cuda_stream_overlap_demo src/stream_overlap/main.cu)
add_cuda_demo(cuda_vector_add_demo src/vector_add/main.cu)
add_cuda_demo(cuda_reduction_demo src/reduction/main.cu)
