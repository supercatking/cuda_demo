#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kElements = 1 << 24;
constexpr int kThreadsPerBlock = 256;
constexpr float kTolerance = 1.0e-2f;

void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " CUDA call failed: " +
            expression + " -> " + cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void reduceSumKernel(const float* input, float* partialSums, int count) {
    __shared__ float shared[kThreadsPerBlock];

    const int thread = threadIdx.x;
    const int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (index < count) {
        sum += input[index];
    }
    if (index + blockDim.x < count) {
        sum += input[index + blockDim.x];
    }

    shared[thread] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            shared[thread] += shared[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        partialSums[blockIdx.x] = shared[0];
    }
}

std::vector<float> makeInput() {
    std::vector<float> input(kElements);
    for (int i = 0; i < kElements; ++i) {
        input[i] = static_cast<float>((i % 17) + 1) * 0.125f;
    }
    return input;
}

double cpuSum(const std::vector<float>& input) {
    return std::accumulate(input.begin(), input.end(), 0.0);
}

}  // namespace

int main() {
    try {
        int device = 0;
        CHECK_CUDA(cudaSetDevice(device));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        std::cout << "CUDA device: " << prop.name << "\n";
        std::cout << "Elements: " << kElements << "\n";

        const std::vector<float> hostInput = makeInput();
        const int blocks = (kElements + (kThreadsPerBlock * 2 - 1)) / (kThreadsPerBlock * 2);
        std::vector<float> hostPartials(blocks, 0.0f);

        float* deviceInput = nullptr;
        float* devicePartials = nullptr;
        CHECK_CUDA(cudaMalloc(&deviceInput, static_cast<size_t>(kElements) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&devicePartials, static_cast<size_t>(blocks) * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(deviceInput, hostInput.data(), static_cast<size_t>(kElements) * sizeof(float),
                              cudaMemcpyHostToDevice));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        reduceSumKernel<<<blocks, kThreadsPerBlock>>>(deviceInput, devicePartials, kElements);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        CHECK_CUDA(cudaMemcpy(hostPartials.data(), devicePartials, static_cast<size_t>(blocks) * sizeof(float),
                              cudaMemcpyDeviceToHost));

        const double gpuSum = std::accumulate(hostPartials.begin(), hostPartials.end(), 0.0);
        const double expected = cpuSum(hostInput);
        const double error = std::fabs(expected - gpuSum);
        const double bandwidthGbps = (static_cast<double>(kElements) * sizeof(float)) / (elapsedMs * 1.0e6);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Blocks: " << blocks << "\n";
        std::cout << "Kernel time: " << elapsedMs << " ms\n";
        std::cout << "Read bandwidth: " << bandwidthGbps << " GB/s\n";
        std::cout << "CPU sum: " << expected << "\n";
        std::cout << "GPU sum: " << gpuSum << "\n";
        std::cout << "Absolute error: " << error << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(deviceInput));
        CHECK_CUDA(cudaFree(devicePartials));

        if (error > kTolerance) {
            std::cerr << "Validation FAILED\n";
            return EXIT_FAILURE;
        }

        std::cout << "Validation PASSED\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
