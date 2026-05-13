#include "../vecAdd/strippedVecAdd.cu"
#include "../matrixMultiplication/strippedMatrixMultiplication.cu"


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#define COLOC_SIZE 2


typedef int (*SetupFn)(int partitions, cudaStream_t stream);
typedef int (*LauncherFn)(int partitions, cudaStream_t stream);
typedef int (*CleanerFn)();


typedef struct {
    const char *name;
    SetupFn setup;
    LauncherFn launcher;
    CleanerFn cleaner;
} KernelEntry;



KernelEntry kernel_registry[] = {
    { "vecAdd",         vecAdd::setup, vecAdd::launcher, vecAdd::cleaner},
    { "matrixMultiplication", mm::setup, mm::launcher, mm::cleaner},
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
            return kernel_registry[i].cleaner;

    fprintf(stderr, "ERROR: kernel cleaner'%s' not found\n", name);
    return NULL;
}


void colocateKernels(const char *kernel1, const char *kernel2, int partitions_A, int partitions_B, cudaStream_t streamA, cudaStream_t streamB){

    SetupFn setupA = find_setup(kernel1);
    SetupFn setupB = find_setup(kernel2);
    LauncherFn launcherA = find_launcher(kernel1);
    LauncherFn launcherB = find_launcher(kernel2);
    CleanerFn cleanerA = find_cleaner(kernel1);
    CleanerFn cleanerB = find_cleaner(kernel2);

    //Split setup from launch?
    if (!launcherA || !launcherB){
        printf("Error finding kernel Launcher");
        exit(0);
    }


    launcherA(partitions_A, streamA);

    launcherB(partitions_B, streamB);

    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);
    
    cleanerA();
    cleanerB();

}

int main(void){
    const char *kernels_to_colocate[] =  {"vecAdd", "matrixMultiplication"};



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