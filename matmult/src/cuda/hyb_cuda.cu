#include "../../include/cuda/hyb_cuda.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/ell_cuda.cuh"

#define TILE_SIZE 16

void allocate_hyp_memory_cuda(float** d_ell_values, int** d_ell_col_indices, int ell_elements,
                              float** d_coo_values, int** d_coo_row_indices, int** d_coo_col_indices, int coo_elements) {
    CHECK_CUDA_ERROR(cudaMalloc(d_ell_values, ell_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(d_ell_col_indices, ell_elements * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc(d_coo_values, coo_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(d_coo_row_indices, coo_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(d_coo_col_indices, coo_elements * sizeof(int)));
}

__global__ void _hyb_matrix_multiply_kernel_impl(const float* A_ellValues, const int* A_ellColIndices,
                                                 const float* B_ellValues, const int* B_ellColIndices,
                                                 float* C, int numRowsC, int numColsC,
                                                 int maxNumNonZeroA, int maxNumNonZeroB,
                                                 const float* A_cooValues, const int* A_cooRowIndices, const int* A_cooColIndices,
                                                 const float* B_cooValues, const int* B_cooRowIndices, const int* B_cooColIndices,
                                                 int A_cooElements, int B_cooElements)
{
    __shared__ float sA_slice[TILE_SIZE * TILE_SIZE];
    __shared__ float sB_slice[TILE_SIZE * TILE_SIZE];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    sA_slice[threadId] = 0.0f;
    sB_slice[threadId] = 0.0f;

    float sum_C_element = 0.0f;

    if (globalRow >= numRowsC || globalCol >= numColsC) {
        return;
    }

    int Aj = 0;
    int Bj = 0;

    for (int t = 0; t < (numRowsC + TILE_SIZE - 1) / TILE_SIZE; t++) {
        while (Aj < maxNumNonZeroA && A_ellColIndices[globalRow * maxNumNonZeroA + Aj] < t * TILE_SIZE + threadIdx.x) {
            Aj++;
        }
        sA_slice[threadId] = 0.0f;
        if (Aj < maxNumNonZeroA && A_ellColIndices[globalRow * maxNumNonZeroA + Aj] == t * TILE_SIZE + threadIdx.x) {
            sA_slice[threadId] = A_ellValues[globalRow * maxNumNonZeroA + Aj];
        }

        while (Bj < maxNumNonZeroB && B_ellColIndices[globalCol * maxNumNonZeroB + Bj] < t * TILE_SIZE + threadIdx.x) {
            Bj++;
        }
        sB_slice[threadId] = 0.0f;
        if (Bj < maxNumNonZeroB && B_ellColIndices[globalCol * maxNumNonZeroB + Bj] == t * TILE_SIZE + threadIdx.x) {
            sB_slice[threadId] = B_ellValues[globalCol * maxNumNonZeroB + Bj];
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum_C_element += sB_slice[threadIdx.x * TILE_SIZE + k] * sA_slice[threadIdx.y * TILE_SIZE + k];
        }
        __syncthreads();
    }

    for (int i = 0; i < A_cooElements; i++) {
        if (A_cooRowIndices[i] == globalRow) {
            for (int j = 0; j < B_cooElements; j++) {
                if (B_cooRowIndices[j] == globalCol && B_cooColIndices[j] == A_cooColIndices[i]) {
                    sum_C_element += A_cooValues[i] * B_cooValues[j];
                }
            }
        }
    }

    C[globalRow * numColsC + globalCol] = sum_C_element;
}

void hyb_matrix_multiply_launch(const float* A_ellValues, const int* A_ellColIndices,
                                const float* B_ellValues, const int* B_ellColIndices,
                                float* C, int numRowsC, int numColsC,
                                int maxNumNonZeroA, int maxNumNonZeroB,
                                const float* A_cooValues, const int* A_cooRowIndices, const int* A_cooColIndices,
                                const float* B_cooValues, const int* B_cooRowIndices, const int* B_cooColIndices,
                                int A_cooElements, int B_cooElements)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    dim3 numBlocks(
        (numColsC + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (numRowsC + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

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
