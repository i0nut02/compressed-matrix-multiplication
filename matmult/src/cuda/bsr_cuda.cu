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
                                          int outputCols,
                                          int BSR_block_size_val)
{
    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (globalRow >= outputRows || globalCol >= outputCols) {
        return;
    }

    float sum = 0.0f;

    int A_block_row_idx = globalRow / BSR_block_size_val;
    int B_block_col_idx_target = globalCol / BSR_block_size_val;

    int local_row_in_bsr_block = globalRow % BSR_block_size_val;
    int local_col_in_bsr_block = globalCol % BSR_block_size_val;

    for (int k_idx_A_ptr = A_rowPointers[A_block_row_idx]; k_idx_A_ptr < A_rowPointers[A_block_row_idx + 1]; k_idx_A_ptr++) {
        int A_block_k_col = A_colIndices[k_idx_A_ptr];

        int B_start_ptr_for_k = B_rowPointers[A_block_k_col];
        int B_end_ptr_for_k = B_rowPointers[A_block_k_col + 1];

        for (int k_idx_B_ptr = B_start_ptr_for_k; k_idx_B_ptr < B_end_ptr_for_k; ++k_idx_B_ptr) {
            int B_block_j_col = B_colIndices[k_idx_B_ptr];

            if (B_block_j_col == B_block_col_idx_target) {
                long long A_block_data_offset = (long long)k_idx_A_ptr * (BSR_block_size_val * BSR_block_size_val);
                long long B_block_data_offset = (long long)k_idx_B_ptr * (BSR_block_size_val * BSR_block_size_val);

                float current_block_product_contribution = 0.0f;
                for (int k_local = 0; k_local < BSR_block_size_val; ++k_local) {
                    float A_elem = A_values[A_block_data_offset + (local_row_in_bsr_block * BSR_block_size_val + k_local)];
                    float B_elem = B_values[B_block_data_offset + (k_local * BSR_block_size_val + local_col_in_bsr_block)];
                    current_block_product_contribution += A_elem * B_elem;
                }
                sum += current_block_product_contribution;
            }
        }
    }
    output[globalRow * outputCols + globalCol] = sum;
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