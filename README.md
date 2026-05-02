# cuda_demo

A collection of compact C++/CUDA demos.

## Demos

- `src/gemm/`: tiled shared-memory GEMM for `C = A x B`, with sampled CPU validation.
- `src/stream_overlap/`: pinned host memory plus `cudaMemcpyAsync` and multiple CUDA streams to overlap host-device copies with kernel execution.

## Requirements

- NVIDIA CUDA Toolkit
- CMake 3.24+
- A C++ compiler supported by `nvcc`

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

```powershell
.\build\Release\cuda_gemm_demo.exe
.\build\Release\cuda_stream_overlap_demo.exe
```

Each program exits with code `0` when validation passes.
