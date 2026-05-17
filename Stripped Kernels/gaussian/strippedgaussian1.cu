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

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
__global__ void Fan1(float *m, float *a, int Size, int t);
__global__ void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

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
 


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(float* m) 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;

}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t, int offset)
{   
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);
  int current_row = threadIdx.x + blockIdx.x * blockDim.x + t + 1 + offset;
  // printf("%d\n", current_row);

	if(current_row >= Size) return;
	int local_row = current_row - offset;
  m_cuda[Size * local_row + t] = a_cuda[Size * current_row + t] / a_cuda[Size*t+t];
}


/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub()
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
    InitPerRun(m);

    float *m_cuda,*a_cuda;
	
	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	
	
	int block_size,grid_size;
	
	block_size = MAXBLOCKSIZE;
    printf("%d\n", Size/block_size);
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
    printf("%d\n", grid_size);


  int chunkRows = Size / NUM_OF_PARTITIONS;
  int chunkSize = chunkRows * Size;

  
	dim3 dimBlock(block_size);
  dim3 dimGrid((chunkRows / block_size) + (!(chunkRows % block_size) ? 0 : 1));

  cudaMemcpy(a_cuda, a, Size*Size*sizeof(float), cudaMemcpyHostToDevice);


  // ONLY EVER CHANGES THE FIRST COLLUM is that allowed?
	cudaStream_t streams[NUM_OF_PARTITIONS];
	for (int i =0; i < NUM_OF_PARTITIONS; i++) {
    int row_offset = i * chunkRows;
    int offset_m = i * chunkSize;
    cudaStreamCreate(&streams[i]);
		
		// copy memory to GPU
    cudaMemcpyAsync(m_cuda + offset_m, m + offset_m, chunkSize*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
		Fan1<<<dimGrid,dimBlock,0, streams[i]>>>(m_cuda + offset_m,a_cuda, Size,0, row_offset);
    cudaMemcpyAsync(m + offset_m, m_cuda + offset_m, chunkSize*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);


	}
	cudaDeviceSynchronize();

  printf("hi");
  PrintMat(m, Size, Size);


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

