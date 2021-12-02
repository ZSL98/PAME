#include "check_exit.cuh"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <iostream>

void cls_copy_list(float* exitSrc, int* output_vector, float threshold, int length, int batch_size)
{
    int test = 0;
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < batch_size; i += blockDim.x * gridDim.x) {
    for (int i = 0; i < batch_size; i++) {
        thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(exitSrc + i*length);
        float max = *(thrust::max_element(d_ptr, d_ptr +length));
        if (max < threshold) {
            // output_vector[i] = 1;
            // cudaMemset(output_vector+i, 1, sizeof(int));
            cudaMemcpyAsync(output_vector + i*sizeof(int), &test, sizeof(int), cudaMemcpyHostToDevice);
        }
    }
}


__global__ void max_reduction_resnet(float *v, int *v_r) {
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

void max_reduction_r(float *v, int *v_r) {
    max_reduction_resnet<<<32, 1000>>> (v, v_r);
}