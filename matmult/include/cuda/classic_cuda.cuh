#ifndef CLASSIC_CUDA_CUH
#define CLASSIC_CUDA_CUH

#define TILE_SIZE 16

void classic_matrix_multiply_cuda(const float* A_values, 
                                const float* B_values,
                                float* C_values,
                                int numRowsA,
                                int numColsA,
                                int numColsB);

#endif