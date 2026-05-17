// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
// #include "srad_kernel.cu"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
namespace sradv2_2 {
	static int rows, cols, size_I, size_R, chunkSize;
	static float q0sqr, lambda, sum, sum2, tmp, meanROI,varROI ;
	static float *h_I, *h_J, *d_J, *d_C, *d_E_C, *d_W_C, *d_N_C, *d_S_C, *h_C, *h_E_C, *h_W_C, *h_N_C, *h_S_C;
	static dim3 dimBlock, dimGrid;





	void generate_dummy_data(float *h_C, float *h_E_C, float *h_W_C, float *h_N_C, float *h_S_C, int size_I){
		// Generate dummy data
		for (int k = 0; k < size_I; k++) {
			h_C[k]   = rand() / (float)RAND_MAX;
			h_E_C[k] = rand() / (float)RAND_MAX;
			h_W_C[k] = rand() / (float)RAND_MAX;
			h_N_C[k] = rand() / (float)RAND_MAX;
			h_S_C[k] = rand() / (float)RAND_MAX;
		}
	}



	int launcher(int partitions, cudaStream_t stream) 
	{


		CUDA_CHECK(cudaMemcpyAsync(d_J, h_J, sizeof(float) *  cols*rows, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_C , h_C, sizeof(float) *  cols*rows, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_E_C, h_E_C, sizeof(float) *  cols*rows, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_W_C, h_W_C, sizeof(float) *  cols*rows, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_N_C , h_N_C, sizeof(float) *  cols*rows, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_S_C, h_S_C, sizeof(float) *  cols*rows, cudaMemcpyHostToDevice, stream));

		for (int i = 0; i < partitions; i ++){

			//Copy data from main memory to device memory
			int offset = i * chunkSize;

			srad_cuda_2<<<dimGrid, dimBlock, 0, stream>>>(d_E_C + offset, d_W_C + offset, d_N_C + offset, d_S_C + offset, d_J + offset, d_C + offset, cols, rows/partitions, lambda, q0sqr, chunkSize);
		}
		
		// CUDA_CHECK(cudaStreamSynchronize(stream));
		
			//Copy data from device memory to main memory
		CUDA_CHECK(cudaMemcpyAsync(h_J, d_J , sizeof(float) * cols * rows, cudaMemcpyDeviceToHost, stream));


		return 0;
	}


	void random_matrix(float *I, int rows, int cols){
		
		srand(7);
		
		for( int i = 0 ; i < rows ; i++){
			for ( int j = 0 ; j < cols ; j++){
			I[i * cols + j] = rand()/(float)RAND_MAX ;
			}
		}

	}

	int setup(int partitions, cudaStream_t stream){
		unsigned int r1, r2, c1, c2;
		
		
	
		rows = 2048;  //number of rows in the domain
		cols = 2048;  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
			fprintf(stderr, "rows and cols must be multiples of 16\n");
			exit(1);
		}
		r1   = 0;  //y1 position of the speckle
		r2   = 127;  //y2 position of the speckle
		c1   = 0;  //x1 position of the speckle
		c2   = 127;  //x2 position of the speckle
		lambda = 0.5;

		size_I = cols * rows;
		size_R = (r2-r1+1)*(c2-c1+1);   

		CUDA_CHECK(cudaMallocHost(&h_I, size_I * sizeof(float)));
		CUDA_CHECK(cudaMallocHost(&h_J, size_I * sizeof(float)));
		CUDA_CHECK(cudaMallocHost(&h_C, size_I * sizeof(float)));


		CUDA_CHECK(cudaMallocHost(&h_E_C, size_I * sizeof(float)));
		CUDA_CHECK(cudaMallocHost(&h_W_C, size_I * sizeof(float)));
		CUDA_CHECK(cudaMallocHost(&h_N_C, size_I * sizeof(float)));
		CUDA_CHECK(cudaMallocHost(&h_S_C, size_I * sizeof(float)));




		//Allocate device memory
		CUDA_CHECK(cudaMalloc((void**)& d_J, sizeof(float)* size_I));
		CUDA_CHECK(cudaMalloc((void**)& d_C, sizeof(float)* size_I));
		CUDA_CHECK(cudaMalloc((void**)& d_E_C, sizeof(float)* size_I));
		CUDA_CHECK(cudaMalloc((void**)& d_W_C, sizeof(float)* size_I));
		CUDA_CHECK(cudaMalloc((void**)& d_S_C, sizeof(float)* size_I));
		CUDA_CHECK(cudaMalloc((void**)& d_N_C, sizeof(float)* size_I));

		
		// printf("Randomizing the input matrix\n");
		//Generate a random matrix
		random_matrix(h_I, rows, cols);

		for (int k = 0;  k < size_I; k++ ) {
			h_J[k] = (float)exp(h_I[k]) ;
		}
		// printf("Start the SRAD main loop\n");
	
		// REMOVED FOR LOOP (DOING ONLY 1 iteration)
		// Loop over region of interest
		sum=0; sum2=0;
		for (int i=r1; i<=r2; i++) {
			for (int j=c1; j<=c2; j++) {
				tmp   = h_J[i * cols + j];
				sum  += tmp ;
				sum2 += tmp*tmp;
			}
		}
		meanROI = sum / size_R;
		varROI  = (sum2 / size_R) - meanROI*meanROI;
		q0sqr   = varROI / (meanROI*meanROI);
		generate_dummy_data(h_C ,h_E_C ,h_W_C , h_N_C, h_S_C, size_I);


		//Currently the input size must be divided by 16 - the block size
		int block_x = cols/BLOCK_SIZE ;
		int block_y = rows/BLOCK_SIZE ;


		dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
		dimGrid = dim3(block_x , block_y/partitions);

		chunkSize = cols * (rows/partitions);
		return 0;

	}

	int cleaner(void){
		
		cudaFreeHost(h_I);
		cudaFreeHost(h_J);
		cudaFreeHost(h_C);
		cudaFreeHost(h_E_C);
		cudaFreeHost(h_W_C);
		cudaFreeHost(h_N_C);
		cudaFreeHost(h_S_C);

		cudaFree(d_C);
		cudaFree(d_J);
		cudaFree(d_E_C);
		cudaFree(d_W_C);
		cudaFree(d_N_C);
		cudaFree(d_S_C);
		return 0;

	}
}