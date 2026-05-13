

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
  float out_err, hid_err;
  
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
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
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


  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);



  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);


  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);



#ifdef GPU

  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

  int blocksPerPartition = num_blocks / NUM_OF_PARTITIONS;
  int chunkSize_weights = blocksPerPartition * HEIGHT * (hid + 1);
  int chunkSize_input = blocksPerPartition * HEIGHT;

  dim3  grid( 1 , blocksPerPartition);
  dim3  threads(16 , 16);
  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid+1)*sizeof(float), cudaMemcpyHostToDevice);
 
   cudaStream_t streams[NUM_OF_PARTITIONS];    

  for (int i =0; i < NUM_OF_PARTITIONS; i++){
    int offset_weights = i * chunkSize_weights;
    int offset_input   = i * chunkSize_input;
    cudaStreamCreate(&streams[i]);

    cudaMemcpyAsync(input_hidden_cuda + offset_weights, input_weights_one_dim + offset_weights, chunkSize_weights * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

    cudaMemcpyAsync(input_prev_weights_cuda + offset_weights, input_weights_prev_one_dim + offset_weights, chunkSize_weights * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

    cudaMemcpyAsync(input_cuda + offset_input, net->input_units + offset_input, chunkSize_input * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

    bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
                                                    hid, 
                                                    input_cuda + offset_input, 
                                                    in,
                                                    input_hidden_cuda + offset_weights, 
                                                    input_prev_weights_cuda + offset_weights
                                                    );

    cudaMemcpyAsync(net->input_units + offset_input, input_cuda + offset_input, chunkSize_input * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

    cudaMemcpyAsync(input_weights_one_dim + offset_weights, input_hidden_cuda + offset_weights, chunkSize_weights * sizeof(float), cudaMemcpyDeviceToHost, streams[i]); 

  }

  cudaDeviceSynchronize();

// FILE *fp = fopen("outputtest.txt", "w");
// if (!fp) {
//     printf("Error opening output file\n");
//     exit(1);
// }

// // Write input units
// fprintf(fp, "=== input_units ===\n");
// for (int i = 0; i <= in; i++) {
//     fprintf(fp, "input_units[%d] = %f\n", i, net->input_units[i]);
// }

// // Write input weights
// fprintf(fp, "=== input_weights_one_dim ===\n");
// for (int i = 0; i <= in; i++) {
//     for (int j = 0; j <= hid; j++) {
//         fprintf(fp, "weights[%d][%d] = %f\n", i, j, input_weights_one_dim[i * (hid+1) + j]);
//     }
// }

// fclose(fp);
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  cudaFreeHost(partial_sum);
  cudaFreeHost(input_weights_one_dim);
  cudaFreeHost(input_weights_prev_one_dim);;

#endif   
  
  
  

}
