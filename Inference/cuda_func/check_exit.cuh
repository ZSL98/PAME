#ifndef CHECK_EXIT_CUH
#define CHECK_EXIT_CUH

#include <stdio.h>

extern "C" 
void cls_copy_list(float* exitSrc, int* output_vector, float threshold, int length, int batch_size);
void max_reduction_r(float *v, int *v_r);
void max_reduction_p(float *v, float *v_r);
void max_reduction_o(float *v, float *v_r);
void generate_fake_copy_list(int last_stage_length, int length_copy, int *fake_copy_list);

#endif