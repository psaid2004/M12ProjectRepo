#include <stdio.h>

__global__ void test() {
    printf("GPU thread %d running\n", threadIdx.x);
}

int main() {
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    fflush(stdout);

    printf("CPU done\n");
}