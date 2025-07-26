#include "../../include/cuda/bsr_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"

#define TILE_SIZE 16

__global__ void bsr_matrix_multiply_kernel(const float* A_values,
                                          const int* A_colIndices,
                                          const int* A_rowPointers,
                                          const float* B_values,
                                          const int* B_colIndices,
                                          const int* B_rowPointers,
                                          float* output,
                                          int outputRows,
                                          int outputCols,
                                          int BSR_block_size)
{
    __shared__  float tileA[TILE_SIZE * TILE_SIZE];
    __shared__  float tileB[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    tileA[threadId] = 0.0f;
    tileB[threadId] = 0.0f;

    float sum = 0.0f;

    if (row >= outputRows || col >= outputCols) {
        return;
    }

    int A_row_idx = row / BSR_block_size;
    int B_row_idx = col / BSR_block_size;

    int A_block_col_idx = 0;
    int B_block_col_idx = 0;

    int A_num_blocks_row = A_rowPointers[A_row_idx + 1] - A_rowPointers[A_row_idx];
    int B_num_blocks_row = B_rowPointers[B_row_idx + 1] - B_rowPointers[B_row_idx];

    int A_offset = A_rowPointers[A_row_idx];
    int B_offset = B_rowPointers[B_row_idx];

    int A_lastColIndices = A_colIndices[A_offset + A_block_col_idx];
    int B_lastColIndices = B_colIndices[B_offset + B_block_col_idx];

    for (int t = 0; t < (outputRows + TILE_SIZE - 1) / TILE_SIZE; t++) {
        while (A_block_col_idx < A_num_blocks_row && A_lastColIndices < (t * (TILE_SIZE / BSR_block_size) + threadIdx.x / BSR_block_size)) {
            A_block_col_idx++;
            A_lastColIndices = A_block_col_idx < A_num_blocks_row ? A_colIndices[A_offset + A_block_col_idx] : 0;
        }
        tileA[threadId] = 0.0f;
        if (A_block_col_idx < A_num_blocks_row && A_lastColIndices == (t * (TILE_SIZE / BSR_block_size) + threadIdx.x / BSR_block_size)) {
            tileA[threadId] = A_values[(A_offset + A_block_col_idx) * BSR_block_size * BSR_block_size + BSR_block_size * (threadIdx.y % BSR_block_size) + (threadIdx.x % BSR_block_size)];
            A_block_col_idx++;
            A_lastColIndices = A_block_col_idx < A_num_blocks_row ? A_colIndices[A_offset + A_block_col_idx] : 0; 
        }

        while (B_block_col_idx < B_num_blocks_row && B_lastColIndices < t * (TILE_SIZE / BSR_block_size) + threadIdx.x / BSR_block_size) {
            B_block_col_idx++;
            B_lastColIndices = B_block_col_idx < B_num_blocks_row ? B_colIndices[B_offset + B_block_col_idx] : 0;
        }
        tileB[threadId] = 0.0f;
        if (B_block_col_idx < B_num_blocks_row && B_colIndices[B_offset + B_block_col_idx] == t * (TILE_SIZE / BSR_block_size) + threadIdx.x / BSR_block_size) {
            tileB[threadId] = B_values[(B_offset + B_block_col_idx) * BSR_block_size * BSR_block_size + BSR_block_size * (threadIdx.y % BSR_block_size) + (threadIdx.x % BSR_block_size)];
            B_block_col_idx++;
            B_lastColIndices = B_block_col_idx < B_num_blocks_row ? B_colIndices[B_offset + B_block_col_idx] : 0;
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
            sum += tileA[threadIdx.y * TILE_SIZE + k_inner] * tileB[threadIdx.y * TILE_SIZE + k_inner];
        }
        __syncthreads();
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
                              int outputCols,
                              int BSR_block_size)
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
        outputCols,
        BSR_block_size
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}