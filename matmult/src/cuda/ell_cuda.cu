#include "../../include/cuda/ell_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"

#define TILE_SIZE 16

void allocate_ell_memory_cuda(float** d_values, int** d_colsIndeces, int numRows, int maxColsPerRow) {
    CHECK_CUDA_ERROR(cudaMalloc(d_values, numRows * maxColsPerRow * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(d_colsIndeces, numRows * maxColsPerRow * sizeof(int)));
}

__global__ void ell_matrix_multiply_kernel(const float* A_values, const int* A_colIndices,
                                         const float* B_values, const int* B_colIndices,
                                         float* C, int numRowsC, int numColsC,
                                         int maxNumNonZeroA, int maxNumNonZeroB) 
{
    __shared__  float tileA[TILE_SIZE * TILE_SIZE];
    __shared__  float tileB[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    
    tileA[threadId] = 0.0f;
    tileB[threadId] = 0.0f;
    
    float sum = 0.0f;

    if (row >= numRowsC || col >= numColsC) {
        return;
    }

    int Aj = 0;
    int Bj = 0;

    for (int t=0; t < (numRowsC + TILE_SIZE -1) / TILE_SIZE; t++) {
        while (Aj < maxNumNonZeroA && A_colIndices[row * maxNumNonZeroA + Aj] < t * TILE_SIZE + threadIdx.x) {
            Aj++;
        }
        tileA[threadId] = 0.0f;
        if (Aj < maxNumNonZeroA && A_colIndices[row * maxNumNonZeroA + Aj] == t * TILE_SIZE + threadIdx.x) {
            tileA[threadId] = A_values[row * maxNumNonZeroA + Aj];
        }

        while (Bj < maxNumNonZeroB && B_colIndices[col * maxNumNonZeroB + Bj] < t * TILE_SIZE + threadIdx.x) {
            Bj++;
        }
        tileB[threadId] = 0.0f;
        if (Bj < maxNumNonZeroB && B_colIndices[col * maxNumNonZeroB + Bj] == t * TILE_SIZE + threadIdx.x) {
            tileB[threadId] = B_values[row * maxNumNonZeroB + Bj];
        }
        
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileB[threadIdx.x * TILE_SIZE + k] * tileA[threadIdx.y * TILE_SIZE + k];
        }
        __syncthreads();
    }
    C[row * numColsC + col] = sum;
}

void ell_matrix_multiply_cuda(const float* A_values, const int* A_colIndices,
                            const float* B_values, const int* B_colIndices,
                            float* C_values, int numRowsC, int numColsC,
                            int maxNumNonZeroA, int maxNumNonZeroB) 
{
    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    dim3 gridDim(
        (numColsC + blockDim.x - 1) / blockDim.x,
        (numRowsC + blockDim.y - 1) / blockDim.y
    );
    ell_matrix_multiply_kernel<<<gridDim, blockDim>>>(
        A_values, A_colIndices,
        B_values, B_colIndices,
        C_values, numRowsC, numColsC,
        maxNumNonZeroA, maxNumNonZeroB
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}