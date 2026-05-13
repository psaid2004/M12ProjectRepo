/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>

// #include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 256
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 4
#endif

#define NUM_OF_PARTITIONS 2

int Size;


FILE *fp;

void ForwardSub();
__global__ void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);

// unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float* coe = new float[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int main(int argc, char *argv[])
{
  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    int i;
    char flag;
    
    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              Size = atoi(argv[i]);
	      printf("Create matrix internally in parse, size = %d \n", Size);
              break;
            case 'f': // platform
              i++;
	      printf("Read file from %s \n", argv[i]);
              break;
            case 'q': // quiet
              break;
	  }
      }
    }

    
    // run kernels
    ForwardSub();   

}
 

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t, int row_offset)
{
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    // global row
    int global_xidx = xidx + row_offset;

    if(global_xidx + 1 + t >= Size) return;
    if(yidx >= Size - t) return;

    a_cuda[Size*(global_xidx+1+t)+(yidx+t)] -= m_cuda[Size*(global_xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];

    if(yidx == 0){
        b_cuda[global_xidx+1+t] -= m_cuda[Size*(global_xidx+1+t)+t] * b_cuda[t];
    }
}
/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub() // WHERE IS MEMORY BEING ALLOCATED FOR HOST
{

    int size =Size;
	float* a;
    float* b;
    float* m;


    cudaMallocHost(&a, Size * Size * sizeof(float));
    cudaMallocHost(&b, Size * sizeof(float));
    cudaMallocHost(&m, Size * Size * sizeof(float));

    create_matrix(a, Size);

    for (int j =0; j< size; j++)
  	b[j]=1.0;
    
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            m[i * Size + j] =
                (float)rand() / (float)RAND_MAX;
        }
    }
	float *m_cuda,*a_cuda,*b_cuda;
	
	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	cudaMalloc((void **) &b_cuda, Size * sizeof(float));

	

  	int chunkRows = Size / NUM_OF_PARTITIONS;


	// DOES IT NEED TO BE THE HERE?
	cudaMemcpy(m_cuda, m, Size*Size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a_cuda, a, Size*Size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_cuda, b, Size*sizeof(float),      cudaMemcpyHostToDevice);


	dim3 dimBlock2(BLOCK_SIZE_XY, BLOCK_SIZE_XY);
	dim3 dimGrid2(chunkRows / BLOCK_SIZE_XY, Size / BLOCK_SIZE_XY);

	cudaStream_t streams[NUM_OF_PARTITIONS];
	for (int i = 0; i < NUM_OF_PARTITIONS; i++) {
		int row_offset = i * chunkRows;
		cudaStreamCreate(&streams[i]);

		Fan2<<<dimGrid2, dimBlock2, 0, streams[i]>>>(m_cuda, a_cuda, b_cuda, Size, Size, 0, row_offset);
		cudaMemcpyAsync(b + row_offset, b_cuda + row_offset, chunkRows * sizeof(float),cudaMemcpyDeviceToHost, streams[i] );

	}

	
	cudaDeviceSynchronize();

	// PrintAry(b, Size);


 	cudaFree(m_cuda);
	cudaFree(a_cuda);
  	cudaFreeHost(m);
  	cudaFreeHost(a);
  	cudaFreeHost(b);
}

void InitMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+Size*i+j);
		}
	}  
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2f ", ary[i]);
	}
	printf("\n\n");
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

