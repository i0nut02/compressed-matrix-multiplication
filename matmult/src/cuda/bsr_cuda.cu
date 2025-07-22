#include "../../include/cuda/bsr_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"

#define TILE_SIZE 32
__global__ void bsr_matrix_multiply_kernel(const float* A_values,
                              const int* A_colIndices,
                              const int* A_rowPointers,
                              const float* B_values,
                              const int* B_colIndices,
                              const int* B_rowPointers,
                              float* output,
                              int outputRows,
                              int outputCols)

    __shared__  float tileA[TILE_SIZE * TILE_SIZE];
    __shared__  float tileB[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    int A_block, B_block = 0;
    while true {
        if (A_rowPointers[row+1] - A_rowPointers[row] == A_block) {
            break;
        }
        if (B_rowPointers[row+1] - B_rowPointers[row] == B_block) {
            break;
        }
        A_colIdx = A_colIndices[A_rowPointers[row] + A_block];
        B_colIdx = B_colIndices[B_rowPointers[row] + B_block];
        if (A_colIdx == B_colIdx) {
            tileA[threadId] = A_values[A_rowPointers[row] * TILE_SIZE + threadId];
            tileB[threadId] = B_values[B_rowPointers[row] * TILE_SIZE + threadId];

            __syncthreads();

            for (int k = 0; k < TILE_SIZE * TILE_SIZE; k++) {
                sum += tileA[threadIdx.y * TILE_SIZE + k] * tileB[threadIdx.x * TILE_SIZE + k];
            }
            __syncthreads();
        } else if (A_colIdx > B_colIdx) {
            B_block++;
        } else {
            A_block++;
        }
    }
    if (row >= outputRows || col >= outputCols) {
        return;
    }
    C[row * outputCols + col] = sum;
    return;

void bsr_matrix_multiply_cuda(const float* A_values,
                              const int* A_colIndices,
                              const int* A_rowPointers,
                              const float* B_values,
                              const int* B_colIndices,
                              const int* B_rowPointers,
                              float* output,
                              int outputRows,
                              int outputCols);
{
    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    dim3 gridDim(
        (outputCols + blockDim.x - 1) / blockDim.x,
        (outputRows + blockDim.y - 1) / blockDim.y
    );

    bsr_matrix_multiply_kernel<<<gridDim, blockDim>>>(
        A_values,
        A_colIndices,
        A_rowPointers,
        B_values,
        B_colIndices,
        B_rowPointers,
        output,
        outputRows,
        outputCols
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}