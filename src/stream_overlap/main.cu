#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kNumStreams = 4;
constexpr int kElements = 1 << 24;
constexpr int kElementsPerStream = kElements / kNumStreams;
constexpr int kThreadsPerBlock = 256;
constexpr float kScale = 2.5f;
constexpr float kBias = 1.25f;
constexpr float kTolerance = 1.0e-5f;

void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " CUDA call failed: " +
            expression + " -> " + cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void scaleBiasKernel(const float* input, float* output, int count, float scale, float bias) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] * scale + bias;
    }
}

void initialize(float* input, int count) {
    for (int i = 0; i < count; ++i) {
        input[i] = static_cast<float>(i % 1024) * 0.001f;
    }
}

float maxAbsError(const float* input, const float* output, int count) {
    float maxError = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float expected = input[i] * kScale + kBias;
        maxError = std::max(maxError, std::fabs(expected - output[i]));
    }
    return maxError;
}

float runSingleStream(const float* hostInput, float* hostOutput, float* deviceInput, float* deviceOutput) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);
    const int blocks = (kElements + kThreadsPerBlock - 1) / kThreadsPerBlock;

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    scaleBiasKernel<<<blocks, kThreadsPerBlock>>>(deviceInput, deviceOutput, kElements, kScale, kBias);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsedMs;
}

float runMultiStream(const float* hostInput, float* hostOutput, float* deviceInput, float* deviceOutput) {
    cudaStream_t streams[kNumStreams]{};
    for (int i = 0; i < kNumStreams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const size_t chunkBytes = static_cast<size_t>(kElementsPerStream) * sizeof(float);
    const int blocks = (kElementsPerStream + kThreadsPerBlock - 1) / kThreadsPerBlock;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kNumStreams; ++i) {
        const int offset = i * kElementsPerStream;
        CHECK_CUDA(cudaMemcpyAsync(deviceInput + offset, hostInput + offset, chunkBytes,
                                   cudaMemcpyHostToDevice, streams[i]));
        scaleBiasKernel<<<blocks, kThreadsPerBlock, 0, streams[i]>>>(
            deviceInput + offset, deviceOutput + offset, kElementsPerStream, kScale, kBias);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, chunkBytes,
                                   cudaMemcpyDeviceToHost, streams[i]));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    for (cudaStream_t stream : streams) {
        CHECK_CUDA(cudaStreamDestroy(stream));
    }
    return elapsedMs;
}

}  // namespace

int main() {
    try {
        int device = 0;
        CHECK_CUDA(cudaSetDevice(device));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        std::cout << "CUDA device: " << prop.name << "\n";
        std::cout << "Elements: " << kElements << ", streams: " << kNumStreams << "\n";
        std::cout << "asyncEngineCount: " << prop.asyncEngineCount << "\n";

        const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);

        float* hostInput = nullptr;
        float* singleOutput = nullptr;
        float* streamOutput = nullptr;
        CHECK_CUDA(cudaMallocHost(&hostInput, bytes));
        CHECK_CUDA(cudaMallocHost(&singleOutput, bytes));
        CHECK_CUDA(cudaMallocHost(&streamOutput, bytes));

        float* deviceInput = nullptr;
        float* deviceOutput = nullptr;
        CHECK_CUDA(cudaMalloc(&deviceInput, bytes));
        CHECK_CUDA(cudaMalloc(&deviceOutput, bytes));

        initialize(hostInput, kElements);

        const float singleMs = runSingleStream(hostInput, singleOutput, deviceInput, deviceOutput);
        const float singleError = maxAbsError(hostInput, singleOutput, kElements);

        CHECK_CUDA(cudaMemset(deviceInput, 0, bytes));
        CHECK_CUDA(cudaMemset(deviceOutput, 0, bytes));

        const float streamMs = runMultiStream(hostInput, streamOutput, deviceInput, deviceOutput);
        const float streamError = maxAbsError(hostInput, streamOutput, kElements);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Single stream time: " << singleMs << " ms\n";
        std::cout << "Multi-stream time: " << streamMs << " ms\n";
        std::cout << "Speedup: " << (singleMs / streamMs) << "x\n";
        std::cout << "Single stream max error: " << singleError << "\n";
        std::cout << "Multi-stream max error: " << streamError << "\n";

        CHECK_CUDA(cudaFree(deviceInput));
        CHECK_CUDA(cudaFree(deviceOutput));
        CHECK_CUDA(cudaFreeHost(hostInput));
        CHECK_CUDA(cudaFreeHost(singleOutput));
        CHECK_CUDA(cudaFreeHost(streamOutput));

        if (singleError > kTolerance || streamError > kTolerance) {
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
