#ifndef BUFFER_COPY_CUH
#define BUFFER_COPY_CUH

#include <stdio.h>

extern "C" 
void buffercopy(float* d_vector_dest, const float* d_vector_src, int sz, const int* stepOverList, int singleVol, const cudaStream_t& stream);
void useCUDA();

#endif