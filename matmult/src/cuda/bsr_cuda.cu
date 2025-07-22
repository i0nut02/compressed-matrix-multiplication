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
{
    __shared__  float tileA[TILE_SIZE * TILE_SIZE];
    __shared__  float tileB[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row >= outputRows || col >= outputCols) {
        return;
    }

    int A_block_idx = 0;
    int B_block_idx = 0;

    while (true) {
        if (A_block_idx >= (A_rowPointers[row / TILE_SIZE + 1] - A_rowPointers[row / TILE_SIZE])) {
            break;
        }
        if (B_block_idx >= (B_rowPointers[row / TILE_SIZE + 1] - B_rowPointers[row / TILE_SIZE])) {
            break;
        }

        int A_colIdx_block = A_colIndices[A_rowPointers[row / TILE_SIZE] + A_block_idx];
        int B_colIdx_block = B_colIndices[B_rowPointers[row / TILE_SIZE] + B_block_idx];

        if (A_colIdx_block == B_colIdx_block) {
            int A_block_value_start_index = (A_rowPointers[row] + A_block_idx) * (TILE_SIZE * TILE_SIZE);
            int B_block_value_start_index = (B_rowPointers[row] + B_block_idx) * (TILE_SIZE * TILE_SIZE);

            tileA[threadId] = A_values[A_block_value_start_index + threadId];
            tileB[threadId] = B_values[B_block_value_start_index + threadId];

            __syncthreads();

            for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
                sum += tileA[threadIdx.y * TILE_SIZE + k_inner] * tileB[k_inner * TILE_SIZE + threadIdx.x];
            }
            __syncthreads();

            A_block_idx++;
            B_block_idx++;
        } else if (A_colIdx_block > B_colIdx_block) {
            B_block_idx++;
        } else {
            A_block_idx++;
        }
    }
    output[row * outputCols + col] = sum;

}

void bsr_matrix_multiply_cuda(const float* A_values,
                              const int* A_colIndices,
                              const int* A_rowPointers,
                              const float* B_values,
                              const int* B_colIndices,
                              const int* B_rowPointers,
                              float* output,
                              int outputRows,
                              int outputCols)
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
