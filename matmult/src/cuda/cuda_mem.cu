#include "../../include/cuda/cuda_mem.cuh"
#include "../../include/cuda/cuda_check.cuh"

void vector_copy_cuda(float* h_vec, float* d_vec, int numElements) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_vec, h_vec, elements * sizeof(float), cudaMemcpyHostToDevice));
}

void vector_copy_cuda(int* h_vec, int* d_vec, int numElements) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_vec, h_vec, elements * sizeof(int), cudaMemcpyHostToDevice));
}

void vector_free_cuda(float* d_vector){
    CHECK_CUDA_ERROR(cudaFree(d_vector));
}

void vector_free_cuda(int* d_vector){
    CHECK_CUDA_ERROR(cudaFree(d_vector));
}
