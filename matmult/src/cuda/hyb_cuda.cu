#include "../../include/cuda/hyb_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/ell_cuda.cuh"

void allocate_hyp_memory_cuda(float** d_ell_values, int** d_ell_col_indices, int ell_elements, //
                              float** d_coo_values, int** d_coo_row_indices, int** d_coo_col_indices, int coo_elements) {
    CHECK_CUDA_ERROR(cudaMalloc(d_ell_values, ell_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(d_ell_col_indices, ell_elements * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc(d_coo_values, coo_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(d_coo_row_indices, coo_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(d_coo_col_indices, coo_elements * sizeof(int)));
}