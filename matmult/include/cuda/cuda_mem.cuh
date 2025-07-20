#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

void vector_copy_2host(float* h_vec, float* d_vec, int numElements);

void vector_malloc_cuda(float* d_vec, int numElements);

void vector_copy_cuda(float* h_vec, float* d_vec, int numElements);

void vector_copy_cuda(int* h_vec, int* d_vec, int numElements);

void vector_free_cuda(float* d_vector);

void vector_free_cuda(int* d_vector);

#endif