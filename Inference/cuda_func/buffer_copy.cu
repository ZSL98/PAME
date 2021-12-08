#include "buffer_copy.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(cond) check_cuda(cond, __LINE__)

void check_cuda(cudaError_t status, std::size_t line)
{
    if(status != cudaSuccess)
    {
        std::cout << cudaGetErrorString(status) << '\n';
        std::cout << "Line: " << line << '\n';
        throw 0;
    }
}

__global__ void copy_kernel(float* output, const float* input, int N, const int* stepOverList, int singleVol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < N; i += blockDim.x * gridDim.x) {

        // int map_i = stepOverList[i/singleVol]*singleVol+i%singleVol;

        // output[i] = input[i];
        // int cur_num = int(i/singleVol);
        // int stored_num = 0;
        // if (indicator[cur_num] == 1) {
        //     for (int j = 0; j < cur_num; j++) {
        //         stored_num += indicator[j];
        //     }
        //     output[i % singleVol + singleVol * stored_num] = input[i];
        // }
    if (i < N) {
        int map_i = stepOverList[i/singleVol]*singleVol+i%singleVol;
        output[i] = input[map_i];
    }
}

void buffercopy(float* d_vector_dest, const float* d_vector_src, int sz, const int* stepOverList, int singleVol, const cudaStream_t& stream)
{
    // int grid_size = 0, block_size = 0;
    // CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, copy_kernel, 0));
    // std::cout << "Grid size: " << grid_size << "  Block size: " << block_size << std::endl;

    copy_kernel<<<singleVol, sz/singleVol, 0, stream>>>(d_vector_dest, d_vector_src, sz, stepOverList, singleVol);
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void foo()
{
    printf("CUDA!\n");
}

void useCUDA()
{
    foo<<<1,5>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
}