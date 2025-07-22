#ifndef BSR_CUDA_CUH
#define BSR_CUDA_CUH

void bsr_matrix_multiply_cuda(const float* A_values,
                              const int* A_colIndices,
                              const int* A_rowPointers,
                              const float* B_values,
                              const int* B_colIndices,
                              const int* B_rowPointers,
                              float* output,
                              int outputRows,
                              int outputCols);

#endif