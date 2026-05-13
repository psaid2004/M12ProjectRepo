// System includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
// #define NUM_OF_PARTITIONS 8
namespace mm{
    static float *d_A, *d_B, *d_C;
    static float *h_C;
    static float *h_B;
    static float *h_A;
    static int block_size, chunkSize, chunkSizeC;
    static dim3 dimsA(512, 512, 1);
    static dim3 dimsB(512, 512, 1); //NEEDS TO BE DIVISIBLE BY BLOCK SIZE (REMEMBER FOR OTHER KERNELS)
    static dim3 dimsC, threads, grid;
    static unsigned int, size_A, mem_size_A, size_B, mem_size_B, mem_size_C,

// ===================== REPLACEMENTS =====================

// Replacement for checkCudaErrors
#define checkCudaErrors(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ===================== KERNEL =====================

template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// ===================== UTIL =====================

void ConstantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

// ===================== MAIN COMPUTE =====================

int MatrixMultiply(int partitions, cudaStream_t stream, int block_size, const dim3 &dimsA, const dim3 &dimsB)
{    

    // printf("Launching Matrix Multiplication Kernel\n");

    // cudaStream_t streams[NUM_OF_PARTITIONS]; 
    for (int i = 0; i < partitions; i++){
        int offset = i * chunkSize;
       

        checkCudaErrors(cudaMemcpyAsync(d_A + offset, h_A + offset, chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream));
        // checkCudaErrors(cudaMemcpyAsync(d_B, h_B, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]));


        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C + offset, d_A + offset, d_B, dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C + offset, d_A + offset, d_B, dimsA.x, dimsB.x);
        }

        checkCudaErrors(cudaMemcpyAsync(h_C + offset, d_C + offset, chunkSizeC * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }   
    // printf("Matrix Multiplication Kernel Ended\n");

    return 0;
}

int setup(int partitions, cudaStream_t stream){


    block_size = 32;

    // Default if no values passed
  
    size_A = dimsA.x * dimsA.y;
    mem_size_A = sizeof(float) * size_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

    size_B = dimsB.x * dimsB.y;
    mem_size_B = sizeof(float) * size_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));


    // Fill with dummy data
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);
   

    dimsC = dim3(dimsB.x, dimsA.y, 1);
    mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc(&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc(&d_C, mem_size_C));

    chunkSize = dimsA.x * (dimsA.y/partitions); 
    chunkSizeC = dimsB.x * (dimsA.y / partitions);

    
    threads = dim3(block_size, block_size);
    grid = dim3(dimsB.x / threads.x, (dimsA.y / threads.y)/partitions);

    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    return 0;
    


}


int launcher(int partitions, cudaStream_t stream)
{


   

    // checkCudaErrors(cudaProfilerStart());
    int result = MatrixMultiply(partitions, stream, block_size, dimsA, dimsB);
    // checkCudaErrors(cudaProfilerStop());


    return result;
}

int cleaner(void){
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
}

// ===================== MAIN =====================

// int main(int argc, char **argv)
// {
//     printf("[Matrix Multiply Using CUDA] - Starting...\n");

//     // int dev = findCudaDevice(argc, (const char **)argv);

//     int block_size = 32;

//     // Default if no values passed
//     dim3 dimsA(512, 512, 1);
//     dim3 dimsB(512, 512, 1); //NEEDS TO BE DIVISIBLE BY BLOCK SIZE (REMEMBER FOR OTHER KERNELS)


//     checkCudaErrors(cudaProfilerStart());
//     int result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
//     checkCudaErrors(cudaProfilerStop());

//     return result;
// }