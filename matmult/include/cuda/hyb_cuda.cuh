#ifndef HYB_CUDA_CUH
#define HYB_CUDA_CUH

void allocate_hyp_memory_cuda(float** d_ell_values, int** d_ell_col_indices, int ell_elements, //
                              float** d_coo_values, int** d_coo_row_indices, int** d_coo_col_indices, int coo_elements);

void hyb_matrix_multiply_launch(const float* A_ellValues, const int* A_ellColIndices,
                                const float* B_ellValues, const int* B_ellColIndices,
                                float* C, int numRowsC, int numColsC,
                                int maxNumNonZeroA, int maxNumNonZeroB,
                                const float* A_cooValues, const int* A_cooRowIndices, const int* A_cooColIndices,
                                const float* B_cooValues, const int* B_cooRowIndices, const int* B_cooColIndices,
                                int A_cooElements, int B_cooElements);
#endif