#include "../../include/cuda/classic_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"

#define TILE_SIZE 16

__global__ void classic_matrix_multiply_kernel(const float* A_values,
                                             const float* B_values,
                                             float* C_values,
                                             int numRowsA,
                                             int numColsA,
                                             int numColsB) 
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    if (row >= numRowsA || col >= numColsB) 
        return;

    for (int t = 0; t < (numColsA + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (t * TILE_SIZE + threadIdx.x < numColsA) {
            tile_A[threadIdx.y][threadIdx.x] = A_values[row * numColsA + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_SIZE + threadIdx.y < numColsA) {
            tile_B[threadIdx.y][threadIdx.x] = B_values[(t * TILE_SIZE + threadIdx.y) * numColsB + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    C_values[row * numColsB + col] = sum;
}

void classic_matrix_multiply_cuda(const float* A_values,
                                const float* B_values,
                                float* C_values,
                                int numRowsA,
                                int numColsA,
                                int numColsB) 
{
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (numColsB + TILE_SIZE - 1) / TILE_SIZE,
        (numRowsA + TILE_SIZE - 1) / TILE_SIZE
    );

    classic_matrix_multiply_kernel<<<gridDim, blockDim>>>(
        A_values,
        B_values,
        C_values,
        numRowsA,
        numColsA,
        numColsB
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}