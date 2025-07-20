#include "../../include/cuda/ell_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"

void allocate_ell_memory_cuda(float** d_values, int** d_colsIndeces, int numRows, int maxColsPerRow) {
    CHECK_CUDA_ERROR(cudaMalloc(d_values, numRows * maxColsPerRow * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(d_colsIndeces, numRows * maxColsPerRow * sizeof(int)));
}

__global__ void ell_matrix_multiply_kernel(const float* A_values, const int* A_colIndices,
                                         const float* B_values, const int* B_colIndices,
                                         float* C, int numRowsC, int numColsC,
                                         int maxNumNonZeroA, int maxNumNonZeroB) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int) id / numRowsC;
    int col = (int) id % numRowsC;
    
    int j = 0; // colB[j] > colA[i] => B[k] < colA[i+1] for k < j
    float sum = 0.0f;
    
    if (row <= numRowsC) {
        for (int i = 0; i < maxNumNonZeroA; i++) {
            int idxA = row * maxNumNonZeroA + i;
            if (A_values[idxA] == 0) {
                continue;
            }
            for (; j < maxNumNonZeroB; j++) {
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
        C[row * numColsC + col] = sum;
    }
}

void ell_matrix_multiply_cuda(const float* A_values, const int* A_colIndices,
                            const float* B_values, const int* B_colIndices,
                            float* C_values, int numRowsC, int numColsC,
                            int maxNumNonZeroA, int maxNumNonZeroB) 
{
    int threadsPerBlock = 256;
    int numBlocks = (numRowsC * numColsC + threadsPerBlock - 1) / threadsPerBlock;
    
    ell_matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
        A_values, A_colIndices,
        B_values, B_colIndices,
        C_values, numRowsC, numColsC,
        maxNumNonZeroA, maxNumNonZeroB
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}