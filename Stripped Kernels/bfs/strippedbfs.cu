/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512
#define NUM_OF_PARTITIONS 1


int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

    char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	// if(!fp)
	// {
	// 	printf("Error Reading graph file\n");
	// 	return;
	// }

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	printf("Original amount of blocks per kernel %d\n", num_of_blocks);

	printf("Blocks per kernel %d\n", num_of_blocks/NUM_OF_PARTITIONS);

	printf("Threads per block %d \n", num_of_threads_per_block);
    int chunkSize = no_of_nodes/NUM_OF_PARTITIONS;


	// allocate host memory --(SHOULD THIS BE CUDAMALLOCHOST?)------
	// Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	// bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	// bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	// bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);


	Node* h_graph_nodes;
	bool *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
    cudaMallocHost(&h_graph_nodes, no_of_nodes * sizeof(Node));
	cudaMallocHost(&h_graph_mask, no_of_nodes * sizeof(bool));
	cudaMallocHost(&h_updating_graph_mask, no_of_nodes * sizeof(bool));
	cudaMallocHost(&h_graph_visited, no_of_nodes * sizeof(bool));



	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges;
	cudaMallocHost(&h_graph_edges, edge_list_size * sizeof(int));

	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    
	 // allocate mem for the result on host side
	int* h_cost;
	cudaMallocHost(&h_cost, no_of_nodes * sizeof(int));

	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;

	printf("Read File\n");

    //-------------------Allocate memory for host variables ----------------------


	//Allocate memory for Device variables
	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
    
    
    //Copy the Edge List to device Memory
	int* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;

    //Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;

    bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;

    //Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;

	
	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);

    //make a bool to check if the execution is over
	bool *d_over;
	cudaMalloc( (void**) &d_over, sizeof(bool));

    printf("Allocated memory for CPU\n");

    // Graph edges is not alligned with nodes indexes, so this has to be shared
    cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;


    cudaStream_t streams[NUM_OF_PARTITIONS];
    
  	// setup execution parameters
	dim3  grid( num_of_blocks/NUM_OF_PARTITIONS, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);


    for(int i = 0; i <NUM_OF_PARTITIONS; i++){
        cudaStreamCreate(&streams[i]);

        int offset = i * chunkSize;

        //Alocate memory per kernel
        cudaMemcpyAsync( d_graph_nodes + offset, h_graph_nodes + offset, sizeof(Node)*chunkSize, cudaMemcpyHostToDevice,streams[i]) ;
        cudaMemcpyAsync( d_graph_mask + offset, h_graph_mask + + offset, sizeof(bool)*chunkSize, cudaMemcpyHostToDevice, streams[i]) ;
    	cudaMemcpyAsync( d_updating_graph_mask + offset, h_updating_graph_mask + offset, sizeof(bool)*chunkSize, cudaMemcpyHostToDevice, streams[i]) ;
        cudaMemcpyAsync( d_graph_visited + offset, h_graph_visited + offset, sizeof(bool)*chunkSize, cudaMemcpyHostToDevice, streams[i]) ;
        cudaMemcpyAsync( d_cost + offset, h_cost + offset, sizeof(int)*chunkSize, cudaMemcpyHostToDevice, streams[i]) ; 
    }


    // --- BFS loop ---
    int k = 0;
    bool stop;

	cudaEvent_t bfs_start, bfs_stop;
	cudaEventCreate(&bfs_start);
	cudaEventCreate(&bfs_stop);
    cudaEventRecord(bfs_start);
	do {
        stop = false;
        cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);

        // Launch the kernels partitioned across streams
        for (int i = 0; i < NUM_OF_PARTITIONS; i++) {
            int offset = i * chunkSize;
            Kernel<<<grid, threads, 0, streams[i]>>>(d_graph_nodes + offset, d_graph_edges,
                d_graph_mask + offset, d_updating_graph_mask,
                d_graph_visited, d_cost, chunkSize, offset);

			Kernel2<<<grid, threads,0, streams[i]>>>(d_graph_mask + offset, d_updating_graph_mask + offset, d_graph_visited + offset, d_over, chunkSize);
			// cudaStreamSynchronize(streams[i]);

		}

		// wait for all Kernel partitions (ASK ABOUT HOW CUDASTREAMSYNCHRONIZE WORKS ----------------)
		for (int i = 0; i < NUM_OF_PARTITIONS; i++){
			cudaStreamSynchronize(streams[i]);
		}

		// All streams must modify the same values, thus not ASYNC
        cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
        k++;
    } while (stop);
    printf("Kernel Executed %d times\n",k);
	

	cudaEventRecord(bfs_stop);
	cudaEventSynchronize(bfs_stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, bfs_start, bfs_stop);

	printf("Total BFS GPU time: %f ms\n", ms);

	// copy result from device to host (maybe this as well, but only on the last iteration?)
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;

	//Store the result into a file
	FILE *fpo = fopen("resultStrippedtest.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in resultStripped.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
}
