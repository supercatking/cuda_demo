# cuda_demo

A compact C++/CUDA GEMM demo under `src/gemm/` that computes `C = A x B` with a tiled shared-memory CUDA kernel, validates sampled GPU output elements against CPU reference calculations, and prints timing plus numerical error.

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
```

The program exits with code `0` when the CUDA result matches the CPU reference within tolerance.
