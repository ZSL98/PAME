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

