
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float*C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        // printf("i = %d\n", i);

        C[i] = A[i] + B[i];
    }

}

int main(){
    int gpu_device_index = 0;

    cudaSetDevice(gpu_device_index);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_device_index);

    printf("GPU name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    cudaDevSmResource result[2] {{}, {}};




    size_t size = 1024*1000;
    int numstreams = 4;
    int chunkSize = size/numstreams;

    float* h_A;
    float* h_B;
    float* h_C;


    cudaMallocHost(&h_A, size * sizeof(float));
    cudaMallocHost(&h_B, size * sizeof(float));
    cudaMallocHost(&h_C, size * sizeof(float));

    for(size_t i = 0; i < size; i++){
        h_A[i] =  static_cast<float>(i);
        h_B[i] =  static_cast<float>(i);
        h_C[i] =  0.0f;
    }

    float* d_A;
    float* d_B;
    float* d_C;


    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    
    cudaEvent_t start[4], stop[4];
    for (int i = 0; i < numstreams; i++) {
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    }


    cudaStream_t streams[4];    
    for (int i = 0; i < numstreams; i++){
            int offset = i *chunkSize;
            cudaStreamCreate(&streams[i]);



            //Copy what is in array in the CPU to the array in the GPU
            cudaMemcpyAsync(d_A + offset, h_A + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice,
                    streams[i]);

            cudaMemcpyAsync(d_B + offset, h_B + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice,
                    streams[i]);

            int threadsPerBlock = 256;
            int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
            cudaEventRecord(start[i], streams[i]);

            // Run the kernel
            vecAdd<<<blocksPerGrid,threadsPerBlock, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C +offset, chunkSize);
            cudaEventRecord(stop[i], streams[i]);


            // Wait for all threads in streams[i] to finish before running this
            cudaMemcpyAsync(h_C + offset, d_C + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    streams[i]);


    }



    cudaDeviceSynchronize();
    for (int i = 0; i < numstreams; i++) {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start[i], stop[i]);
        printf("Stream %d kernel time: %.3f ms\n", i, milliseconds);
    }
    // destroy streams
    for (int i = 0; i < numstreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    for (size_t i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
   // Cleanup
    for (int i = 0; i < numstreams; i++) {
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
        cudaStreamDestroy(streams[i]);
    }
}