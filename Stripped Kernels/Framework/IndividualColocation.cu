#include "../vecAdd/strippedVecAdd.cu"
#include "../matrixMultiplication/strippedMatrixMultiplication.cu"
#include "../srad_v2/strippedsradKernel1.cu"
#include "../srad_v2/strippedsradKernel2.cu"
#include "../nn/strippednn.cu"
#include "../hotspot/strippedhotspot.cu"




#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#define COLOC_SIZE 2


typedef int (*SetupFn)(int partitions, cudaStream_t stream);
typedef int (*LauncherFn)(int partitions, cudaStream_t stream);
typedef int (*CleanerFn)(void);


typedef struct {
    const char *name;
    SetupFn setup;
    LauncherFn launcher;
    CleanerFn cleaner;
} KernelEntry;



KernelEntry kernel_registry[] = {
    { "vecAdd", vecAdd::setup, vecAdd::launcher, vecAdd::cleaner},
    { "matrixMultiplication", mm::setup, mm::launcher, mm::cleaner},
    { "sradv2_1", sradv2_1::setup, sradv2_1::launcher, sradv2_1::cleaner},
    { "sradv2_2", sradv2_2::setup, sradv2_2::launcher, sradv2_2::cleaner},
    { "nn", nn::setup, nn::launcher, nn::cleaner},
    {"hotspot", hotspot::setup, hotspot::launcher, hotspot::cleaner},

  
};
int num_kernels = sizeof(kernel_registry) / sizeof(kernel_registry[0]);



LauncherFn find_launcher(const char *name)
{
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].launcher;

    fprintf(stderr, "ERROR: kernel  launcher'%s' not found\n", name);
    return NULL;
}

CleanerFn find_cleaner(const char *name)
{
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].cleaner;

    fprintf(stderr, "ERROR: kernel cleaner'%s' not found\n", name);
    return NULL;
}

SetupFn find_setup(const char *name)
{
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].setup;

    fprintf(stderr, "ERROR: kernel setup'%s' not found\n", name);
    return NULL;
}


void colocateKernels(const char *kernel1, const char *kernel2, int partitions_A, int partitions_B, cudaStream_t streamA, cudaStream_t streamB){

    SetupFn setupA = find_setup(kernel1);
    SetupFn setupB = find_setup(kernel2);
    LauncherFn launcherA = find_launcher(kernel1);
    LauncherFn launcherB = find_launcher(kernel2);
    CleanerFn cleanerA = find_cleaner(kernel1);
    CleanerFn cleanerB = find_cleaner(kernel2);


    if (!launcherA || !launcherB || !cleanerA || !cleanerB || !setupA || !setupB) {
        printf("Error finding kernel functions");
        exit(0);
    }


    setupA(partitions_A, streamA);

    setupB(partitions_B, streamB);
    
    cudaEvent_t globalStart, globalStop;
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);

    cudaEvent_t startA, stopA, startB, stopB;
    cudaEventCreate(&startA);    cudaEventCreate(&stopA);
    cudaEventCreate(&startB);    cudaEventCreate(&stopB);

    cudaStream_t syncStream;
    cudaStreamCreate(&syncStream);

    cudaDeviceSynchronize();

    cudaEventRecord(globalStart, streamA);
    cudaEventRecord(startA, streamA);
    cudaEventRecord(startB, streamB);

    launcherA(partitions_A, streamA);
    launcherB(partitions_B, streamB);

    cudaEventRecord(stopA, streamA);
    cudaEventRecord(stopB, streamB);

    // syncStream waits for both streams before recording globalStop
    cudaStreamWaitEvent(syncStream, stopA, 0);
    cudaStreamWaitEvent(syncStream, stopB, 0);
    cudaEventRecord(globalStop, syncStream);

    cudaEventSynchronize(globalStop);

    cudaDeviceSynchronize();
 

    float msA, msB, msTotal;
    cudaEventElapsedTime(&msA,     startA,      stopA);
    cudaEventElapsedTime(&msB,     startB,      stopB);
    cudaEventElapsedTime(&msTotal, globalStart, globalStop);

    printf("%-30s %.3f ms\n", kernel1,    msA);
    printf("%-30s %.3f ms\n", kernel2,    msB);
    printf("%-30s %.3f ms\n", "Total wall time", msTotal);

    // If msTotal << msA + msB, the kernels overlapped
    printf("Overlap efficiency: %.1f%%\n", 
           (msA + msB - msTotal) / (msA + msB) * 100.0f);


    // printf("Finished launching and synchronizing both kernels, now cleaning up...\n");    
    cleanerA();
    cleanerB();

}

int main(void){
    const char *kernels_to_colocate[] =  {"vecAdd", "hotspot"};

    int partitions_A = 2;
    int partitions_B = 2;


    // Initialize streams
    cudaStream_t streams[COLOC_SIZE];    
    for (int i = 0; i < COLOC_SIZE; i++){
            cudaStreamCreate(&streams[i]);
    }

    colocateKernels(kernels_to_colocate[0], kernels_to_colocate[1], partitions_A, partitions_B, streams[0], streams[1]);

      for (int i = 0; i < COLOC_SIZE; i++){
        cudaStreamDestroy(streams[i]);
      }

    return 0;
    

    }