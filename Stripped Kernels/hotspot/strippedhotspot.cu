#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_hotspot 20
#define MAX_PD     (3.0e6)
#define PRECISION  0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI       100
#define FACTOR_CHIP 0.5
#define EXPAND_RATE 2



#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

namespace hotspot{

static float t_chip      = 0.0005f;
static float chip_height = 0.016f;
static float chip_width  = 0.016f;    
static float *h_temp, *h_power, *h_out, *d_temp[2], *d_power;
static float grid_height, grid_width, Cap, Rx, Ry, Rz, max_slope, step, time_elapsed;
static int grid_rows, grid_cols, pyramid_height, total_iterations, size, borderCols, borderRows,smallBlockCol, smallBlockRow, blockCols, blockRows, chunkSize;    
static dim3 dimGrid, dimBlock;




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
	
        __shared__ float temp_on_cuda[BLOCK_SIZE_hotspot][BLOCK_SIZE_hotspot];
        __shared__ float power_on_cuda[BLOCK_SIZE_hotspot][BLOCK_SIZE_hotspot];
        __shared__ float temp_t[BLOCK_SIZE_hotspot][BLOCK_SIZE_hotspot]; // saving temparary temperature result

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
	int small_block_rows = BLOCK_SIZE_hotspot-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE_hotspot-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE_hotspot-1;
        int blkXmax = blkX+BLOCK_SIZE_hotspot-1;

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
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE_hotspot-1-(blkYmax-grid_rows+1) : BLOCK_SIZE_hotspot-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE_hotspot-1-(blkXmax-grid_cols+1) : BLOCK_SIZE_hotspot-1;

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
            if( IN_RANGE(tx, i+1, BLOCK_SIZE_hotspot-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE_hotspot-i-2) &&  \
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

int setup(int partitions, cudaStream_t stream){

    grid_rows       = 512;
    grid_cols       = grid_rows;
    pyramid_height  = 2;
    total_iterations= 1;
    size            = grid_rows * grid_cols;

    //  Pyramid parameters 
    borderCols    = pyramid_height * EXPAND_RATE / 2;
    borderRows    = pyramid_height * EXPAND_RATE / 2;
    smallBlockCol = BLOCK_SIZE_hotspot - pyramid_height * EXPAND_RATE;
    smallBlockRow = BLOCK_SIZE_hotspot - pyramid_height * EXPAND_RATE;
    blockCols     = (grid_cols + smallBlockCol - 1) / smallBlockCol;
    blockRows     = (grid_rows + smallBlockRow - 1) / smallBlockRow;
 


    //  Host memory allocate
    cudaMallocHost(&h_temp,  size * sizeof(float));
    cudaMallocHost(&h_power, size * sizeof(float));
    cudaMallocHost(&h_out,   size * sizeof(float));

    chdir("../hotspot/");

    FILE *ft = fopen("data/temp_512", "r");
    FILE *fp = fopen("data/power_512", "r");

    if (!ft) { fprintf(stderr, "Could not open temp file\n"); exit(1); }
    if (!fp) { fprintf(stderr, "Could not open power file\n"); exit(1); }

    for (int i = 0; i < size; i++) { fscanf(ft, "%f", &h_temp[i]);  }
    for (int i = 0; i < size; i++) { fscanf(fp, "%f", &h_power[i]); }
    fclose(ft); fclose(fp);

    chdir("../Framework/");

    // Device memory allocate
    cudaMalloc(&d_temp[0], size * sizeof(float));
    cudaMalloc(&d_temp[1], size * sizeof(float));
    cudaMalloc(&d_power,   size * sizeof(float));

    grid_height  = chip_height / grid_rows;
    grid_width   = chip_width  / grid_cols;
    Cap          = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    Rx           = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    Ry           = grid_height / (2.0f * K_SI * t_chip * grid_width);
    Rz           = t_chip      / (K_SI  * grid_height  * grid_width);
    max_slope    = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    step         = PRECISION / max_slope;
    time_elapsed = 0.001f;



    chunkSize = (grid_rows / partitions) * grid_cols;

    
    printf("%d blockCols\n", blockCols);
    printf("%d blockCols per partitions\n", blockCols/partitions);
    printf("%d blockrows per partitions\n", blockRows);

    dimGrid = dim3(blockCols, blockRows/partitions);

    dimBlock = dim3(BLOCK_SIZE_hotspot, BLOCK_SIZE_hotspot);
    return 0;
}


int launcher(int partitions, cudaStream_t stream){
    
    int src = 0, dst = 1;


    cudaMemcpyAsync(d_temp[0], h_temp,  grid_rows*grid_cols * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_temp[1], h_temp,  grid_rows*grid_cols * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_power, h_power, grid_rows*grid_cols * sizeof(float), cudaMemcpyHostToDevice, stream);



    for(int i = 0; i < partitions; i++){
        int offset = i * chunkSize;
        int row_offset = i * (grid_rows/partitions);
        int partitionRows = grid_rows / partitions;


        calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(
            min(pyramid_height, total_iterations),
            d_power + offset, d_temp[src] + offset, d_temp[dst] + offset,
            grid_cols, partitionRows,
            borderCols, borderRows,
            Cap, Rx, Ry, Rz, step, time_elapsed, row_offset);
    }
    

    
    cudaMemcpyAsync(h_out, d_temp[src], grid_rows * grid_cols * sizeof(float), cudaMemcpyDeviceToHost, stream);

  
    return 0;
}


int cleaner(void){
    // Write output
    FILE *fo = fopen("output----.out", "w");
    for (int i = 0; i < size; i++) fprintf(fo, "%d\t%g\n", i, h_out[i]);
    fclose(fo);

    cudaFree(d_temp[0]); cudaFree(d_temp[1]); cudaFree(d_power);
    cudaFreeHost(h_temp); cudaFreeHost(h_power); cudaFreeHost(h_out);
    return 0;
}
}