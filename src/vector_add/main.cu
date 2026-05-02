#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kElements = 1 << 24;
constexpr int kThreadsPerBlock = 256;
constexpr float kTolerance = 1.0e-6f;

void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " CUDA call failed: " +
            expression + " -> " + cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        c[index] = a[index] + b[index];
    }
}

void initialize(std::vector<float>& a, std::vector<float>& b) {
    for (int i = 0; i < kElements; ++i) {
        a[i] = static_cast<float>(i % 1000) * 0.001f;
        b[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
    }
}

float maxAbsError(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
    float maxError = 0.0f;
    for (int i = 0; i < kElements; ++i) {
        const float expected = a[i] + b[i];
        maxError = std::max(maxError, std::fabs(expected - c[i]));
    }
    return maxError;
}

}  // namespace

int main() {
    try {
        int device = 0;
        CHECK_CUDA(cudaSetDevice(device));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        std::cout << "CUDA device: " << prop.name << "\n";
        std::cout << "Vector elements: " << kElements << "\n";

        std::vector<float> hostA(kElements);
        std::vector<float> hostB(kElements);
        std::vector<float> hostC(kElements, 0.0f);
        initialize(hostA, hostB);

        const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);
        float* deviceA = nullptr;
        float* deviceB = nullptr;
        float* deviceC = nullptr;
        CHECK_CUDA(cudaMalloc(&deviceA, bytes));
        CHECK_CUDA(cudaMalloc(&deviceB, bytes));
        CHECK_CUDA(cudaMalloc(&deviceC, bytes));

        CHECK_CUDA(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        const int blocks = (kElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
        CHECK_CUDA(cudaEventRecord(start));
        vectorAddKernel<<<blocks, kThreadsPerBlock>>>(deviceA, deviceB, deviceC, kElements);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        CHECK_CUDA(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

        const float error = maxAbsError(hostA, hostB, hostC);
        const double bandwidthGbps = (3.0 * bytes) / (elapsedMs * 1.0e6);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Kernel time: " << elapsedMs << " ms\n";
        std::cout << "Effective bandwidth: " << bandwidthGbps << " GB/s\n";
        std::cout << "Max absolute error: " << error << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(deviceA));
        CHECK_CUDA(cudaFree(deviceB));
        CHECK_CUDA(cudaFree(deviceC));

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
