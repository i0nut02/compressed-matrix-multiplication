#ifndef ELL_CUDA_CUH
#define ELL_CUDA_CUH

void allocate_ell_memory_cuda(float** d_values, int** d_colsIndeces, int numRows, int maxColsPerRow);

void vector_copy_cuda(float* h_vec, float* d_vec, int elements);

void vector_copy_cuda(int* h_vec, int* d_vec, int elements);

void cuda_ell_matrix_mult();

void cuda_ell_free();

#endif