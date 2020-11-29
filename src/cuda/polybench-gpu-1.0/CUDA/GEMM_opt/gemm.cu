/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 512
#define NJ 512
#define NK 512

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 16 
#define DIM_THREAD_BLOCK_Y 16 

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
      		A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	__shared__ DATA_TYPE s_a[256]; // 16 x 16 
	__shared__ DATA_TYPE s_b[256];

	int bx = blockIdx.x; 
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// A : NI x NK
	// B : NK x NJ
	int aBegin = NK * 16 * by;
	int aEnd = aBegin + NK - 1; 
	int aStep = 16;

	int bBegin = bx * 16;
	int bStep = 16 * NJ;

	float sum = 0.f;

	int aa, bb;
	for(aa = aBegin, bb = bBegin; aa <= aEnd; aa += aStep, bb += bStep)
	{
		// load titled sub-matrices into local memory	
		s_a[ty * 16 + tx] = a[aa + ty * NK + tx];
		s_b[ty * 16 + tx] = b[bb + ty * NJ + tx];
		
		__syncthreads();

		int k;
#pragma unroll
		for(k = 0; k < 16; ++k)
		{
			sum += s_a[ty * 16 + k] * s_b[k * 16 + tx];
		}

		__syncthreads();
	}

	c[row * NJ + col] = ALPHA * sum + c[row * NJ + col] * BETA;
}


void gemmCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	//double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NJ)/ ((float)block.x) )),(size_t)(ceil( ((float)NI)/ ((float)block.y) )));

    cudaError_t error;

	cudaEvent_t start, stop;

	error = cudaEventCreate(&start);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", 
		cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", 
		cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", 
		cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}



	gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu);

	// Record the stop event
	error = cudaEventRecord(stop, NULL);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", 
		cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", 
		cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float sgemm_msec = 0.f;
	error = cudaEventElapsedTime(&sgemm_msec, start, stop);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", 
		cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}



	// C := alpha*op( A )*op( B ) + beta*C
	// GEMM performs 5 floating point operations for one data output
	double flops_sgemm = 5.f * (double) NI * (double) NJ * (double) NK;

	double gigaFlops = (flops_sgemm * 1.0e-9f) / (sgemm_msec / 1000.f);

	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
			gigaFlops,
			sgemm_msec,
			flops_sgemm,
			block.x * block.y);


	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;  
	DATA_TYPE* C_outputFromGpu; 

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
	C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

	init(A, B, C);
	
	GPU_argv_init();
	
	gemmCuda(A, B, C, C_outputFromGpu);

	t_start = rtclock();	
	gemm(A, B, C);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(C, C_outputFromGpu);

	free(A);
	free(B);  
	free(C);  
	free(C_outputFromGpu); 

    	return 0;
}

