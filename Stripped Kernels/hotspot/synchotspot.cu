#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 20
#define MAX_PD     (3.0e6)
#define PRECISION  0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI       100
#define FACTOR_CHIP 0.5
#define EXPAND_RATE 2


#define NUM_OF_PARTITIONS 4

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

float t_chip      = 0.0005f;
float chip_height = 0.016f;
float chip_width  = 0.016f;

__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset 
							   int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step, 
                               float time_elapsed, int row_offset){
	
        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

	float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;
        
	int bx = blockIdx.x;
        int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	
	step_div_Cap=step/Cap;
	
	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int yidx = blkY+ty + row_offset;
	int xidx = blkX+tx;

        // load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
       
	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
	}
	__syncthreads();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
        
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
	       	         (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
		             (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
		             (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];		
      }
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_size> <pyramid_height> <sim_time> "
                        "<temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    const int grid_rows       = atoi(argv[1]);
    const int grid_cols       = grid_rows;
    const int pyramid_height  = atoi(argv[2]);
    const int total_iterations= atoi(argv[3]);
    const int size            = grid_rows * grid_cols;

    //  Pyramid parameters 
    const int borderCols    = pyramid_height * EXPAND_RATE / 2;
    const int borderRows    = pyramid_height * EXPAND_RATE / 2;
    const int smallBlockCol = BLOCK_SIZE - pyramid_height * EXPAND_RATE;
    const int smallBlockRow = BLOCK_SIZE - pyramid_height * EXPAND_RATE;
    const int blockCols     = (grid_cols + smallBlockCol - 1) / smallBlockCol;
    const int blockRows     = (grid_rows + smallBlockRow - 1) / smallBlockRow;
    printf("%d grid rows and cols\n",grid_cols );
    printf("%d pyramid_height\n",pyramid_height );
    printf("%d smallBlockCol\n",smallBlockCol );


    //  Host memory allocate
    float *h_temp, *h_power, *h_out;
    cudaMallocHost(&h_temp,  size * sizeof(float));
    cudaMallocHost(&h_power, size * sizeof(float));
    cudaMallocHost(&h_out,   size * sizeof(float));

    // Read inputs
    FILE *ft = fopen(argv[4], "r"), *fp = fopen(argv[5], "r");
    for (int i = 0; i < size; i++) { fscanf(ft, "%f", &h_temp[i]);  }
    for (int i = 0; i < size; i++) { fscanf(fp, "%f", &h_power[i]); }
    fclose(ft); fclose(fp);

    // Device memory allocate
    float *d_temp[2], *d_power;
    cudaMalloc(&d_temp[0], size * sizeof(float));
    cudaMalloc(&d_temp[1], size * sizeof(float));
    cudaMalloc(&d_power,   size * sizeof(float));

    const float grid_height  = chip_height / grid_rows;
    const float grid_width   = chip_width  / grid_cols;
    const float Cap          = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    const float Rx           = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    const float Ry           = grid_height / (2.0f * K_SI * t_chip * grid_width);
    const float Rz           = t_chip      / (K_SI  * grid_height  * grid_width);
    const float max_slope    = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    const float step         = PRECISION / max_slope;
    const float time_elapsed = 0.001f;


    
    int src = 0, dst = 1;



    cudaMemcpy(d_temp[0], h_temp,  grid_rows * grid_cols * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_temp[1], h_temp ,  grid_rows * grid_cols *  sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_power, h_power, grid_rows * grid_cols * sizeof(float), cudaMemcpyHostToDevice );


    printf("%d blockCols\n", blockCols);
    printf("%d blockCols per partitions\n", blockCols/NUM_OF_PARTITIONS);
    printf("%d blockrows per partitions\n", blockRows);

    const dim3 dimGrid(blockCols, blockRows/NUM_OF_PARTITIONS);

    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    // --- Kernel launch loop: one launch per time step, ping-pong buffers ---
    int chunkSize = (grid_rows / NUM_OF_PARTITIONS) * grid_cols;


    for (int t = 0; t < total_iterations; t += pyramid_height) {

        for(int i = 0; i < NUM_OF_PARTITIONS; i++){
        int offset = i * chunkSize;
        int row_offset = i * (grid_rows/NUM_OF_PARTITIONS);
        int partitionRows = grid_rows / NUM_OF_PARTITIONS;


        calculate_temp<<<dimGrid, dimBlock, 0>>>(
            min(pyramid_height, total_iterations - t),
            d_power + offset, d_temp[src] + offset, d_temp[dst] + offset,
            grid_cols, partitionRows,
            borderCols, borderRows,
            Cap, Rx, Ry, Rz, step, time_elapsed, row_offset);
        }
    
        src ^= 1; dst ^= 1;   // swap ping-pong indices
    }
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_temp[src], grid_rows * grid_cols * sizeof(float), cudaMemcpyDeviceToHost);
    


    // Write output
    FILE *fo = fopen(argv[6], "w");
    for (int i = 0; i < size; i++) fprintf(fo, "%d\t%g\n", i, h_out[i]);
    fclose(fo);

    cudaFree(d_temp[0]); cudaFree(d_temp[1]); cudaFree(d_power);
    cudaFreeHost(h_temp); cudaFreeHost(h_power); cudaFreeHost(h_out);
    return 0;
}