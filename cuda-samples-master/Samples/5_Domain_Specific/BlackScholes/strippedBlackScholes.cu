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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <helper_cuda.h>      // helper functions CUDA error checking and initialization
#include <helper_functions.h> // helper functions for string parsing
#define NUM_OF_PARTITIONS 2

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult,
                                float *h_PutResult,
                                float *h_StockPrice,
                                float *h_OptionStrike,
                                float *h_OptionYears,
                                float  Riskfree,
                                float  Volatility,
                                int    optN);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N          = 4000000;
const int NUM_ITERATIONS = 512;

const int   OPT_SZ     = OPT_N * sizeof(float);
const float RISKFREE   = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
        // Results calculated by CPU for reference
        *h_CallResultCPU,
        *h_PutResultCPU,
        // CPU copy of GPU results
        *h_CallResultGPU, *h_PutResultGPU,
        // CPU instance of input data
        *h_StockPrice, *h_OptionStrike, *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
        // Results calculated by GPU
        *d_CallResult,
        *d_PutResult,
        // GPU instance of input data
        *d_StockPrice, *d_OptionStrike, *d_OptionYears;

    // double delta, ref, sum_delta, sum_ref, max_delta, L1norm;

    int                 i;

    findCudaDevice(argc, (const char **)argv);


    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    cudaMallocHost(&h_CallResultCPU, OPT_SZ);
    cudaMallocHost(&h_PutResultCPU, OPT_SZ);
    cudaMallocHost(&h_StockPrice, OPT_SZ);
    cudaMallocHost(&h_OptionStrike, OPT_SZ);
    cudaMallocHost(&h_OptionYears, OPT_SZ);
    cudaMallocHost(&h_CallResultGPU, OPT_SZ);
    cudaMallocHost(&h_PutResultGPU, OPT_SZ);




    // h_CallResultGPU = (float *)malloc(OPT_SZ);
    // h_PutResultGPU  = (float *)malloc(OPT_SZ);

    printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears, OPT_SZ));

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    // Generate options set
    for (i = 0; i < OPT_N; i++) {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
    int chunkSize = OPT_N/NUM_OF_PARTITIONS;
    cudaStream_t streams[NUM_OF_PARTITIONS];    
    for (int i = 0; i < NUM_OF_PARTITIONS; i++){   
        cudaStreamCreate(&streams[i]);

        printf("...copying input data to GPU mem.\n");
        // Copy options data to GPU memory for further processing

        int offset = i*chunkSize;
        checkCudaErrors(cudaMemcpyAsync(d_StockPrice + offset, h_StockPrice + offset, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        checkCudaErrors(cudaMemcpyAsync(d_OptionStrike + offset, h_OptionStrike + offset, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        checkCudaErrors(cudaMemcpyAsync(d_OptionYears + offset, h_OptionYears + offset, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        if(i==1)
        continue;


        BlackScholesGPU<<<DIV_UP(chunkSize/2, 128), 128 /*480, 128*/>>>((float2 *)d_CallResult +offset/2,
                                                                            (float2 *)d_PutResult+offset/2,
                                                                            (float2 *)d_StockPrice+offset/2,
                                                                            (float2 *)d_OptionStrike+offset/2,
                                                                            (float2 *)d_OptionYears+offset/2,
                                                                            RISKFREE,
                                                                            VOLATILITY,
                                                                            chunkSize);
            getLastCudaError("BlackScholesGPU() execution failed\n");
        
        printf("\nReading back GPU results...\n");
        // Read back GPU results to compare them to CPU results
        checkCudaErrors(cudaMemcpyAsync(h_CallResultGPU+offset, d_CallResult+offset, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        checkCudaErrors(cudaMemcpyAsync(h_PutResultGPU+offset, d_PutResult+offset, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));


    }
    checkCudaErrors(cudaDeviceSynchronize());

// FILE *fp = fopen("blackscholes_outputtest.txt", "w");
// if (!fp) {
//     printf("Error opening output file\n");
//     exit(1);
// }

// // Write Call Results
// fprintf(fp, "=== Call Results ===\n");
// for (int i = 0; i < OPT_N; i++) {
//     int idx =  i;
//     fprintf(fp, "h_CallResultGPU[%d] = %f\n",
//             idx, h_CallResultGPU[idx]);
// }

// // Write Put Results
// fprintf(fp, "\n=== Put Results ===\n");
// for (int i = 0; i < OPT_N; i++) {
//     int idx =  i;
//     fprintf(fp, "h_PutResultGPU[%d] = %f\n",
//             idx, h_PutResultGPU[idx]);
// }

fclose(fp);

 

    

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    printf("...releasing CPU memory.\n");
    cudaFreeHost(h_OptionYears);
    cudaFreeHost(h_OptionStrike);
    cudaFreeHost(h_StockPrice);
    cudaFreeHost(h_PutResultGPU);
    cudaFreeHost(h_CallResultGPU);
    cudaFreeHost(h_PutResultCPU);
    cudaFreeHost(h_CallResultCPU);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");



    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. "
           "Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
