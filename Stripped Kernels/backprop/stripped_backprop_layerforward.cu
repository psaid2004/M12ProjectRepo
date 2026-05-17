

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
// #include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"
#define NUM_OF_PARTITIONS 2

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

// double gettime() {
//   struct timeval t;
//   gettimeofday(&t,NULL);
//   return t.tv_sec+t.tv_usec*1e-6;
// }

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  

  

  cudaMallocHost(&input_weights_one_dim, (in + 1)* (hid + 1) * sizeof(float));
  cudaMallocHost(&input_weights_prev_one_dim, (in + 1)* (hid + 1) * sizeof(float));
  cudaMallocHost(&partial_sum, num_blocks * WIDTH * sizeof(float));



  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }
  
  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  
  
#endif


#ifdef GPU
  int blocksPerPartition = num_blocks / NUM_OF_PARTITIONS;
  dim3  grid( 1 , blocksPerPartition);
  dim3  threads(16 , 16); 
  printf("Performing GPU computation\n");
  int inputRowsPerPartition = (in / NUM_OF_PARTITIONS); 
  int chunkSize_input_weights = inputRowsPerPartition * (hid+1); 
  int chunkSize_partial = blocksPerPartition * WIDTH;

  cudaMemcpy(input_cuda, net->input_units, (in+1)*sizeof(float), cudaMemcpyHostToDevice);

    
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
   cudaStream_t streams[NUM_OF_PARTITIONS];    
    for (int i = 0; i < NUM_OF_PARTITIONS; i++){
        int offset_input   = i * inputRowsPerPartition;    
        int offset_weights = i * chunkSize_input_weights; 
        int offset_partial = i * chunkSize_partial;       
        cudaStreamCreate(&streams[i]);

        cudaMemcpyAsync(input_hidden_cuda + offset_weights, input_weights_one_dim + offset_weights, chunkSize_input_weights * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
        
        
        bpnn_layerforward_CUDA<<< grid, threads, 0, streams[i] >>>(input_cuda +offset_input,
                                                    output_hidden_cuda,
                                                    input_hidden_cuda + offset_weights,
                                                    hidden_partial_sum + offset_partial,
                                                    in/NUM_OF_PARTITIONS,
                                                    hid);
                
        cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        
        cudaMemcpyAsync(partial_sum + offset_partial, hidden_partial_sum + offset_partial, chunkSize_partial * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        }
        cudaDeviceSynchronize();

//   FILE *fp = fopen("outputtest.txt", "w");
//     if (!fp) {
//         printf("Error opening output file\n");
//         exit(1);
//     }

//   printf("hid %d\n", hid);
//   for (int j = 1; j <= hid; j++) {
//     sum = 0.0;
//     for (int k = 0; k < num_blocks; k++) {
//       fprintf(fp, "%f\n", partial_sum[k * hid + j - 1]);
//       sum += partial_sum[k * hid + j-1] ;
//     }
// // 	sum += net->input_weights[0][j];
// // 	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
//   }
  #endif


#ifdef GPU

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
//   cudaFree(input_prev_weights_cuda);
//   cudaFree(hidden_delta_cuda);
  
  cudaFreeHost(partial_sum);
  cudaFreeHost(input_weights_one_dim);
  cudaFreeHost(input_weights_prev_one_dim);

#endif   
  
  
  

}
