#include <stdio.h>

__global__ void addKernel(int *result) {
    int i = threadIdx.x;
    result[i] = i * i;
}

int main() {
    const int N = 10;
    int result[N] = {0};
    int *d_result;

    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&d_result, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Zero out GPU memory first
    cudaMemset(d_result, 0, N * sizeof(int));

    // Run kernel
    addKernel<<<1, N>>>(d_result);

    // Check kernel launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back
    err = cudaMemcpy(result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print
    printf("GPU computed squares:\n");
    for (int i = 0; i < N; i++) {
        printf("  %d^2 = %d\n", i, result[i]);
    }

    cudaFree(d_result);
    return 0;
}