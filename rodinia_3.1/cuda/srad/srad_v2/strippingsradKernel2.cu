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

#define NUM_OF_PARTITIONS 2

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void random_matrix(float *I, int rows, int cols);
void generate_dummy_data(float *h_C, float *h_E_C, float *h_W_C, float *h_N_C, float *h_S_C, int size_I);
void runTest( int argc, char** argv);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

    return EXIT_SUCCESS;
}

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


void
runTest( int argc, char** argv) 
{
    int rows, cols, size_I, size_R;
    float *h_I, *h_J, q0sqr, lambda, sum, sum2, tmp, meanROI,varROI ;

#ifdef GPU
	
	float *d_J;
    float *d_C;
	float *d_E_C, *d_W_C, *d_N_C, *d_S_C;

#endif

	unsigned int r1, r2, c1, c2;
	float *h_C;
    
	
 
	if (argc == 9)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]);

		
	}
	float* h_E_C;
    float* h_W_C;
    float* h_N_C;
    float* h_S_C;


	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	h_I = (float *)malloc( size_I * sizeof(float) );
    h_J = (float *)malloc( size_I * sizeof(float) );
	h_C  = (float *)malloc(sizeof(float)* size_I) ;


	CUDA_CHECK(cudaMallocHost(&h_E_C, size_I * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_W_C, size_I * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_N_C, size_I * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_S_C, size_I * sizeof(float)));



#ifdef GPU

	//Allocate device memory
     CUDA_CHECK(cudaMalloc((void**)& d_J, sizeof(float)* size_I));
     CUDA_CHECK(cudaMalloc((void**)& d_C, sizeof(float)* size_I));
	 CUDA_CHECK(cudaMalloc((void**)& d_E_C, sizeof(float)* size_I));
	 CUDA_CHECK(cudaMalloc((void**)& d_W_C, sizeof(float)* size_I));
	 CUDA_CHECK(cudaMalloc((void**)& d_S_C, sizeof(float)* size_I));
	 CUDA_CHECK(cudaMalloc((void**)& d_N_C, sizeof(float)* size_I));

	
#endif 

	// IDK WHAT DO WITH THIS (HOW DOES IT IMPACT PROFILLING)
	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(h_I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	h_J[k] = (float)exp(h_I[k]) ;
    }
	printf("Start the SRAD main loop\n");
   
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

	printf("%d block x\n", block_x);
	printf("%d block y\n", block_y);
	printf("%d cols \n", cols);
	printf("%d rows\n", rows);


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y/NUM_OF_PARTITIONS);

	int chunkSize = cols * (rows/NUM_OF_PARTITIONS);
    cudaStream_t streams[NUM_OF_PARTITIONS];

	for (int i = 0; i < NUM_OF_PARTITIONS; i ++){
        cudaStreamCreate(&streams[i]);

		//Copy data from main memory to device memory
		int offset = i * chunkSize;
		CUDA_CHECK(cudaMemcpyAsync(d_J + offset, h_J + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, streams[i]));
		CUDA_CHECK(cudaMemcpyAsync(d_C + offset, h_C + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, streams[i]));
		CUDA_CHECK(cudaMemcpyAsync(d_E_C + offset, h_E_C + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, streams[i]));
		CUDA_CHECK(cudaMemcpyAsync(d_W_C + offset, h_W_C + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, streams[i]));
		CUDA_CHECK(cudaMemcpyAsync(d_N_C + offset, h_N_C + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, streams[i]));
		CUDA_CHECK(cudaMemcpyAsync(d_S_C + offset, h_S_C + offset, sizeof(float) * chunkSize, cudaMemcpyHostToDevice, streams[i]));

		srad_cuda_2<<<dimGrid, dimBlock, 0, streams[i]>>>(d_E_C + offset, d_W_C + offset, d_N_C + offset, d_S_C + offset, d_J + offset, d_C + offset, cols, rows/NUM_OF_PARTITIONS, lambda, q0sqr, chunkSize);

		CUDA_CHECK(cudaMemcpyAsync(h_J + offset, d_J + offset, sizeof(float) * chunkSize, cudaMemcpyDeviceToHost, streams[i]));

	}

    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < 5 ; i++){
		for ( int j = 0 ; j < 5 ; j++){
         printf("%.5f ", h_C[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 
    print_matrix_region("J",    h_J,     rows, cols, 5, 5);
    print_matrix_region("C",    h_C,   rows, cols, 5, 5);
    print_matrix_region("E_C",  h_E_C, rows, cols, 5, 5);
    print_matrix_region("W_C",  h_W_C, rows, cols, 5, 5);
    print_matrix_region("N_C",  h_N_C, rows, cols, 5, 5);
    print_matrix_region("S_C",  h_S_C, rows, cols, 5, 5);

	printf("Computation Done\n");

	free(h_I);
	free(h_J);

#ifdef GPU
    cudaFree(d_C);
	cudaFree(d_J);
	cudaFree(d_E_C);
	cudaFree(d_W_C);
	cudaFree(d_N_C);
	cudaFree(d_S_C);
#endif 
	free(h_C);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

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

