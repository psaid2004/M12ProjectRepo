#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Max blocks per multiprocessor: %d\n",
           prop.maxThreadsPerBlock);

    return 0;
}