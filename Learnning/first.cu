#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>  // for atoi


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void unifiedMemExample(int vectorLength);

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if (argc >=2)
    {
        // Take arguments
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);
    return 0;
}
void initArray(float* arr, int n)
{
    for(int i = 0; i < n; i++)
    {
        arr[i] = static_cast<float>(i);  // or any pattern you want
    }
}
void serialVecAdd(float* A, float* B, float* C, int n)
{
    for(int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* a, float* b, int n)
{
    const float epsilon = 1e-5f;

    for(int i = 0; i < n; i++)
    {
        if(fabs(a[i] - b[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

void unifiedMemExample(int vectorLength)
{
    float *A, *B, *C;
    float* comparisonResult = (float*)malloc(vectorLength * sizeof(float));

    // Use unified memoy to allocate buffers
    cudaMallocManaged(&A, vectorLength * sizeof(float));
    cudaMallocManaged(&B, vectorLength * sizeof(float));
    cudaMallocManaged(&C, vectorLength * sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int threads = 256;
    int blocks = (vectorLength + threads - 1) / threads;

    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);

    // Makes the CPU wait for the GPU to finish (blocks CPU)
    cudaDeviceSynchronize();

    serialVecAdd(A, B, comparisonResult, vectorLength);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
        printf("Unified Memory: CPU and GPU answers match\n");
    else
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}