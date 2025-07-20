#include "../../include/cuda/ell_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"

void allocate_ell_memory_cuda(float** d_values, int** d_colsIndeces, int numRows, int maxColsPerRow) {
    CHECK_CUDA_ERROR(cudaMalloc((void**)d_values, numRows * maxColsPerRow * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)d_colsIndeces, numRows * maxColsPerRow * sizeof(int)));
}
