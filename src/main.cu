#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTileSize = 16;
constexpr int kM = 256;
constexpr int kN = 256;
constexpr int kK = 256;
constexpr float kTolerance = 1.0e-2f;

void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " CUDA call failed: " +
            expression + " -> " + cudaGetErrorString(result));
    }
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void gemmKernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float aTile[kTileSize][kTileSize];
    __shared__ float bTile[kTileSize][kTileSize];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + kTileSize - 1) / kTileSize; ++tile) {
        const int tiledCol = tile * kTileSize + threadIdx.x;
        const int tiledRow = tile * kTileSize + threadIdx.y;

        aTile[threadIdx.y][threadIdx.x] = (row < m && tiledCol < k) ? a[row * k + tiledCol] : 0.0f;
        bTile[threadIdx.y][threadIdx.x] = (tiledRow < k && col < n) ? b[tiledRow * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < kTileSize; ++i) {
            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> makeRandomMatrix(int rows, int cols, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> matrix(static_cast<size_t>(rows) * cols);
    for (float& value : matrix) {
        value = dist(rng);
    }
    return matrix;
}

std::vector<float> cpuGemm(const std::vector<float>& a, const std::vector<float>& b, int m, int n, int k) {
    std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    return c;
}

float maxAbsError(const std::vector<float>& expected, const std::vector<float>& actual) {
    float maxError = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        maxError = std::max(maxError, std::fabs(expected[i] - actual[i]));
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
        std::cout << "Matrix sizes: A(" << kM << "x" << kK << ") * B(" << kK << "x" << kN
                  << ") = C(" << kM << "x" << kN << ")\n";

        const auto hostA = makeRandomMatrix(kM, kK, 7);
        const auto hostB = makeRandomMatrix(kK, kN, 13);
        std::vector<float> hostC(static_cast<size_t>(kM) * kN, 0.0f);

        const size_t bytesA = hostA.size() * sizeof(float);
        const size_t bytesB = hostB.size() * sizeof(float);
        const size_t bytesC = hostC.size() * sizeof(float);

        float* deviceA = nullptr;
        float* deviceB = nullptr;
        float* deviceC = nullptr;
        CHECK_CUDA(cudaMalloc(&deviceA, bytesA));
        CHECK_CUDA(cudaMalloc(&deviceB, bytesB));
        CHECK_CUDA(cudaMalloc(&deviceC, bytesC));

        CHECK_CUDA(cudaMemcpy(deviceA, hostA.data(), bytesA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(deviceB, hostB.data(), bytesB, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(deviceC, 0, bytesC));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        const dim3 block(kTileSize, kTileSize);
        const dim3 grid((kN + block.x - 1) / block.x, (kM + block.y - 1) / block.y);

        gemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, kM, kN, kK);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemset(deviceC, 0, bytesC));

        CHECK_CUDA(cudaEventRecord(start));
        gemmKernel<<<grid, block>>>(deviceA, deviceB, deviceC, kM, kN, kK);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float gpuMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpuMs, start, stop));
        CHECK_CUDA(cudaMemcpy(hostC.data(), deviceC, bytesC, cudaMemcpyDeviceToHost));

        const auto cpuStart = std::chrono::high_resolution_clock::now();
        const auto referenceC = cpuGemm(hostA, hostB, kM, kN, kK);
        const auto cpuStop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> cpuMs = cpuStop - cpuStart;

        const float error = maxAbsError(referenceC, hostC);
        const double gflops = (2.0 * kM * kN * kK) / (gpuMs * 1.0e6);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "GPU time: " << gpuMs << " ms (" << gflops << " GFLOP/s)\n";
        std::cout << "CPU reference time: " << cpuMs.count() << " ms\n";
        std::cout << "Max absolute error: " << error << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(deviceA));
        CHECK_CUDA(cudaFree(deviceB));
        CHECK_CUDA(cudaFree(deviceC));

        if (error > kTolerance) {
            std::cerr << "Validation FAILED: error exceeds tolerance " << kTolerance << "\n";
            return EXIT_FAILURE;
        }

        std::cout << "Validation PASSED\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
