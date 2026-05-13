// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);

void print_matrix_region(const char* name, float* M, int rows, int cols, int max_r, int max_c) {
    printf("=== %s (top-left %dx%d) ===\n", name, max_r, max_c);
    for (int i = 0; i < max_r; i++) {
        for (int j = 0; j < max_c; j++) {
            printf("%0.6f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest(argc, argv);
    return EXIT_SUCCESS;
}

void runTest(int argc, char** argv) 
{
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

#ifdef GPU
    float *J_cuda, *C_cuda;
    float *E_C, *W_C, *N_C, *S_C;
#endif

    unsigned int r1, r2, c1, c2;

    if (argc == 9)
    {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);

        if ((rows % 16 != 0) || (cols % 16 != 0)) {
            fprintf(stderr, "rows and cols must be multiples of 16\n");
            exit(1);
        }

        r1 = atoi(argv[3]);
        r2 = atoi(argv[4]);
        c1 = atoi(argv[5]);
        c2 = atoi(argv[6]);
        lambda = atof(argv[7]);
        niter = atoi(argv[8]);
    }
    else {
        fprintf(stderr, "Invalid arguments\n");
        exit(1);
    }

    size_I = rows * cols;
    size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    // Host allocations
    I = (float*)malloc(sizeof(float) * size_I);
    J = (float*)malloc(sizeof(float) * size_I);

    // NEW: host buffers for outputs
    float *h_C   = (float*)malloc(sizeof(float) * size_I);
    float *h_E_C = (float*)malloc(sizeof(float) * size_I);
    float *h_W_C = (float*)malloc(sizeof(float) * size_I);
    float *h_N_C = (float*)malloc(sizeof(float) * size_I);
    float *h_S_C = (float*)malloc(sizeof(float) * size_I);

#ifdef GPU
    // Device allocations
    cudaMalloc((void**)&J_cuda, sizeof(float) * size_I);
    cudaMalloc((void**)&C_cuda, sizeof(float) * size_I);
    cudaMalloc((void**)&E_C, sizeof(float) * size_I);
    cudaMalloc((void**)&W_C, sizeof(float) * size_I);
    cudaMalloc((void**)&N_C, sizeof(float) * size_I);
    cudaMalloc((void**)&S_C, sizeof(float) * size_I);
#endif

    printf("Randomizing the input matrix\n");
    random_matrix(I, rows, cols);

    for (int k = 0; k < size_I; k++) {
        J[k] = exp(I[k]);
    }

    printf("Start the SRAD main loop\n");

    for (iter = 0; iter < niter; iter++) {
        sum = 0; sum2 = 0;

        for (int i = r1; i <= r2; i++) {
            for (int j = c1; j <= c2; j++) {
                tmp = J[i * cols + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }

        meanROI = sum / size_R;
        varROI = (sum2 / size_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

#ifdef GPU
        int block_x = cols / BLOCK_SIZE;
        int block_y = rows / BLOCK_SIZE;

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(block_x, block_y);

        cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

        srad_cuda_1<<<dimGrid, dimBlock>>>(
            E_C, W_C, N_C, S_C,
            J_cuda, C_cuda,
            cols, rows, q0sqr
        );

        cudaDeviceSynchronize();

        // Copy ALL outputs back
        cudaMemcpy(J,     J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C,   C_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_E_C, E_C,    sizeof(float) * size_I, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_W_C, W_C,    sizeof(float) * size_I, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_N_C, N_C,    sizeof(float) * size_I, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_S_C, S_C,    sizeof(float) * size_I, cudaMemcpyDeviceToHost);
#endif
    }

    // Print results
    print_matrix_region("J",    J,     rows, cols, 5, 5);
    print_matrix_region("C",    h_C,   rows, cols, 5, 5);
    print_matrix_region("E_C",  h_E_C, rows, cols, 5, 5);
    print_matrix_region("W_C",  h_W_C, rows, cols, 5, 5);
    print_matrix_region("N_C",  h_N_C, rows, cols, 5, 5);
    print_matrix_region("S_C",  h_S_C, rows, cols, 5, 5);

    printf("Computation Done\n");

    // Free host memory
    free(I);
    free(J);
    free(h_C);
    free(h_E_C);
    free(h_W_C);
    free(h_N_C);
    free(h_S_C);

#ifdef GPU
    cudaFree(J_cuda);
    cudaFree(C_cuda);
    cudaFree(E_C);
    cudaFree(W_C);
    cudaFree(N_C);
    cudaFree(S_C);
#endif
}

void random_matrix(float *I, int rows, int cols) {
    srand(7);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            I[i * cols + j] = rand() / (float)RAND_MAX;
        }
    }
}