#include "../../include/matmult/bsr_mat_mult.hpp"
#include "../../include/cuda/bsr_cuda.cuh"

#include <iostream>

BsrMatMult::BsrMatMult(BsrFormat& a_, BsrFormat& b_) 
    : MatMult(a_, b_) {}

void BsrMatMult::multiply() {
    std::cout << "Multiplying two BSR matrices..." << std::endl;
    BsrFormat& bsrA = static_cast<BsrFormat&>(a);
    BsrFormat& bsrB = static_cast<BsrFormat&>(b);

    bsr_matrix_multiply_cuda(
        bsrA.d_values,
        bsrA.d_column,
        bsrA.d_rowPointers,
        bsrB.d_values,
        bsrB.d_column,
        bsrB.d_rowPointers,
        d_c,
        outputRows,
        outputCols,
        bsrA.blockSize
    );
}