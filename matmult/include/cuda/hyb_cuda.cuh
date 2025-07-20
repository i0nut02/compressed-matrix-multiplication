#ifndef HYB_CUDA_CUH
#define HYB_CUDA_CUH

void allocate_hyp_memory_cuda(float** d_ell_values, int** d_ell_col_indices, int ell_elements, //
                              float** d_coo_values, int** d_coo_row_indices, int** d_coo_col_indices, int coo_elements);

#endif