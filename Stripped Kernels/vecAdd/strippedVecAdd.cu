/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>


// #define NUM_OF_PARTITIONS 2

namespace vecAdd{
 
    static float* h_A;
    static float* h_B;
    static float* h_C;
    static float* d_A;
    static float* d_B;
    static float* d_C;
    static int numElements, size, chunkSize, threadsPerBlock, blocksPerGrid;


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }

}

int setup(int partitions, cudaStream_t stream){

    // Error code to check return values for CUDA calls
    // Print the vector length to be used, and compute its size
    numElements = 1000;
    size        = numElements;
    chunkSize = size/partitions;
    // printf("[Vector addition of %d elements]\n", numElements);

    cudaMallocHost(&h_A, size * sizeof(float));
    cudaMallocHost(&h_B, size * sizeof(float));
    cudaMallocHost(&h_C, size * sizeof(float));


    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    threadsPerBlock = 256;
    blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    return 0;
}

/**
 * Host main routine
 */
int launcher(int partitions, cudaStream_t stream)
{

    // printf("Launching vecAdd Kernel\n");

    // SHOULD THIS BE IN THE SETUP OR HERE IS FINE?
    //Copy what is in array in the CPU to the array in the GPU
    cudaMemcpyAsync(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice, stream);

   
    for (int i = 0; i < partitions; i++){
            int offset = i *chunkSize;
            // Run the kernel
            vectorAdd<<<blocksPerGrid,threadsPerBlock, 0, stream>>>(d_A + offset, d_B + offset, d_C +offset, chunkSize);


    }

    //SAME QUESTION AS ABOVE, SHOULD THIS BE IN THE CLEANER OR HERE IS FINE AND IS THE STREAM SYNC NEEDED?
    // cudaStreamSynchronize(stream);
    // Wait for all threads in streams[i] to finish before running this
    cudaMemcpyAsync(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // printf("VecAdd Kernel Ended\n");

    return 0;
}

int cleaner(void){
    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return 0;
}
}