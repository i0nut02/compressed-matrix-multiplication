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

__global__ void hyb_matrix_multiply_kernel(const float* A_ellValues, const int* A_ellColIndices,
                                         const float* B_ellValues, const int* B_ellColIndices,
                                         float* C, int numRowsC, int numColsC,
                                         int maxNumNonZeroA, int maxNumNonZeroB,
                                         const float* A_cooValues, const int* A_cooRowIndices, const int* A_cooColIndices,
                                         const float* B_cooValues, const int* B_cooRowIndices, const int* B_cooColIndices,
                                         int A_cooElements, int B_cooElements) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int) id / numRowsC;
    int col = (int) id % numRowsC;
    
    float sum = 0.0f;
    
    if (row < numRowsC) {
        for (int i = 0; i < maxNumNonZeroA; i++) {
            int idxA = row * maxNumNonZeroA + i;
            if (A_values[idxA] == 0) {
                continue;
            }
            for (int j = 0; j < maxNumNonZeroB; j++) {
                int idxB = col * maxNumNonZeroB + j;
                if (B_values[idxB] == 0) {
                    break;
                }
                if (B_colIndices[idxB] > A_colIndices[idxA]) {
                    break; // we will obtain just bigger indices of the actual col index of A
                }
                if (B_colIndices[idxB] == A_colIndices[idxA]) {
                    sum += B_values[idxB] * A_values[idxA];
                }
            }
        }
        for (int i=0; i < A_cooElements; i++) {
            if (A_rowIndices[i] == row) {
                for (int j=0; j < B_cooElements; j++) {
                    if (B_rowIndices[j] == row && B_colIndices[j] == A_colIndices[i]) {
                        sum += A_cooValues[i] * B_cooValues[j];
                    }
                }
            }
        }
        C[row * numColsC + col] = sum;
    }
}

void hyb_matrix_multiply_launch(const float* A_ellValues, const int* A_ellColIndices,
                                const float* B_ellValues, const int* B_ellColIndices,
                                float* C, int numRowsC, int numColsC,
                                int maxNumNonZeroA, int maxNumNonZeroB,
                                const float* A_cooValues, const int* A_cooRowIndices, const int* A_cooColIndices,
                                const float* B_cooValues, const int* B_cooRowIndices, const int* B_cooColIndices,
                                int A_cooElements, int B_cooElements)
{
    int threadsPerBlock = 256;
    
    int totalElementsC = numRowsC * numColsC;
    
    int numBlocks = (totalElementsC + threadsPerBlock - 1) / threadsPerBlock;
    
    _hyb_matrix_multiply_kernel_impl<<<numBlocks, threadsPerBlock>>>(
        A_ellValues, A_ellColIndices,
        B_ellValues, B_ellColIndices,
        C, numRowsC, numColsC,
        maxNumNonZeroA, maxNumNonZeroB,
        A_cooValues, A_cooRowIndices, A_cooColIndices,
        B_cooValues, B_cooRowIndices, B_cooColIndices,
        A_cooElements, B_cooElements
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}