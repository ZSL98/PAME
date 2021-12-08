// This program performs sum reduction with an optimization
// removing warp bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>


#define CHECK_CUDA(cond) check_cuda(cond, __LINE__)
#define SIZE 256
#define SHMEM_SIZE 256

void check_cuda(cudaError_t status, std::size_t line)
{
    if(status != cudaSuccess)
    {
        std::cout << cudaGetErrorString(status) << '\n';
        std::cout << "Line: " << line << '\n';
        throw 0;
    }
}

__global__ void max_reduction_resnet(float *v, float *v_r) {
	__shared__ int partial_sum[1000];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = max(partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block

	if (threadIdx.x == 0 && partial_sum[0] > 0.8f) {
		v_r[blockIdx.x] = 1;
	}
}

__global__ void max_reduction_posenet(float *v, float *v_r) {
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float max_p = 0;
	for (int s = 0; s < 96*96; s++){
		float tmp = v[9216 * blockIdx.x + threadIdx.x + 16 * s];
		if (tmp > max_p) {
			max_p = tmp;
		}
	}
	// v_r[tid] = max_p;

	__shared__ int result[32];

	// Initalize the shared memory to 0
	if (threadIdx.x == 0) {
		result[blockIdx.x] = 0;
	}
	__syncthreads();

	if (max_p > 0.8f) {
		atomicAdd(&result[blockIdx.x], 1);
	}
	__syncthreads();

	if (threadIdx.x == 0 && result[blockIdx.x] > 8) {
		v_r[blockIdx.x] = 1;
	}
}

__global__ void max_reduction_openseg(float *v, float *v_r) {
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = bid * blockDim.x + threadIdx.x;
	float max_p = 0;
	for (int s = 0; s < 19; s++){
		float tmp = v[tid*16*19 + s];
		if (tmp > max_p) {
			max_p = tmp;
		}
	}
	// v_r[tid] = max_p;

	__shared__ int result[8];

	// Initalize the shared memory to 0
	if (threadIdx.x == 0 && blockIdx.y == 0) {
		// printf("%d", blockIdx.y);
		result[blockIdx.x] = 0;
	}
	__syncthreads();

	if (max_p > 0.8f) {
		atomicAdd(&result[blockIdx.x], 1);
	}
	__syncthreads();

	if (threadIdx.x == 0 && blockIdx.y == 0 && result[blockIdx.x] > 1) {
		v_r[blockIdx.x] = 1;
	}
}

__global__ void check_posenet(float *v, int *v_r, int *result) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (v[tid] > 0.8f) {
		atomicAdd(&result[blockIdx.x], 1);
	}
	if (threadIdx.x == 0 && result[blockIdx.x] > 8) {
		v_r[blockIdx.x] = 1;
	}
}


__global__ void sum_reduction(float *v, float *v_r) {
	// Allocate shared memory
	__shared__ float partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = bid * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid * 16];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			// partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
			partial_sum[threadIdx.x] = max(partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(float *v, int n) {
	for (int i = 0; i < n; i++) {
		// v[i] = rand() % 10;
		v[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
}

int main() {
	// Vector size
	int n = 32 * 1000;
	// int n = 32 * 96 * 96 * 16;
	// int n = 8 * 2048 * 1024 * 19;
	size_t bytes = n * sizeof(float);

	// Original vector and result vector
	float *h_v, *h_v_r;
	float *d_v, *d_v_r;

	// Allocate memory
	h_v = (float*)malloc(bytes);
	h_v_r = (float*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// Initialize vector
	initialize_vector(h_v, n);

	// Copy to device
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// TB Size
	int TB_SIZE = 256;

	// Grid Size (No padding)
	// int GRID_SIZE = 32;

	// dim3 threads(256);
	dim3 blocks(8, 512);

	// Call kernel
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	CHECK_CUDA(cudaEventRecord(start));
	max_reduction_posenet<<<32, 16>>> (d_v, d_v_r);
	CHECK_CUDA(cudaEventRecord(stop));
	CHECK_CUDA(cudaEventSynchronize(stop));

	float current_time = 0;
	CHECK_CUDA(cudaEventElapsedTime(&current_time, start, stop));

	std::cout << "Elapsed Time: " << current_time << "ms\n";

	// sum_reduction<<<1, TB_SIZE>>> (d_v_r, d_v_r);

	// Copy to host;
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	printf("Accumulated result is %f \n", h_v_r[0]);
	// scanf("Press enter to continue: ");
	// assert(h_v_r[0] == 65536);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}