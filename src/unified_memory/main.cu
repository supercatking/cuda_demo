#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kElements = 1 << 24;
constexpr int kThreadsPerBlock = 256;
constexpr float kAlpha = 3.0f;
constexpr float kTolerance = 1.0e-6f;

void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " CUDA call failed: " +
            expression + " -> " + cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void saxpyKernel(float alpha, const float* x, float* y, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        y[index] = alpha * x[index] + y[index];
    }
}

void initialize(float* x, float* y) {
    for (int i = 0; i < kElements; ++i) {
        x[i] = static_cast<float>(i % 2048) * 0.0005f;
        y[i] = static_cast<float>((i * 5) % 2048) * 0.00025f;
    }
}

float maxAbsError(const float* x, const float* originalY, const float* y) {
    float maxError = 0.0f;
    for (int i = 0; i < kElements; ++i) {
        const float expected = kAlpha * x[i] + originalY[i];
        maxError = std::max(maxError, std::fabs(expected - y[i]));
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
        std::cout << "Elements: " << kElements << "\n";
        std::cout << "Unified addressing: " << prop.unifiedAddressing
                  << ", managed memory: " << prop.managedMemory << "\n";

        const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);

        float* x = nullptr;
        float* y = nullptr;
        float* originalY = nullptr;
        CHECK_CUDA(cudaMallocManaged(&x, bytes));
        CHECK_CUDA(cudaMallocManaged(&y, bytes));
        CHECK_CUDA(cudaMallocManaged(&originalY, bytes));

        initialize(x, y);
        std::copy(y, y + kElements, originalY);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        const int blocks = (kElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
        CHECK_CUDA(cudaEventRecord(start));
        saxpyKernel<<<blocks, kThreadsPerBlock>>>(kAlpha, x, y, kElements);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        CHECK_CUDA(cudaDeviceSynchronize());

        const float error = maxAbsError(x, originalY, y);
        const double bandwidthGbps = (3.0 * bytes) / (elapsedMs * 1.0e6);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Kernel time: " << elapsedMs << " ms\n";
        std::cout << "Effective bandwidth: " << bandwidthGbps << " GB/s\n";
        std::cout << "Max absolute error: " << error << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(x));
        CHECK_CUDA(cudaFree(y));
        CHECK_CUDA(cudaFree(originalY));

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
