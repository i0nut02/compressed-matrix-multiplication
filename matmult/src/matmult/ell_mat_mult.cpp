#include "../../include/matmult/ell_mat_mult.hpp"
#include "../../include/cuda/ell_cuda.cuh"

#include <iostream>

EllMatMult::EllMatMult(EllFormat& a_, EllFormat& b_) 
    : MatMult(a_, b_) {}

void EllMatMult::multiply() {
    std::cout << "Multiplying two ELL matrices..." << std::endl;
    EllFormat& ellA = static_cast<EllFormat&>(a);
    EllFormat& ellB = static_cast<EllFormat&>(b);

    ell_matrix_multiply_cuda(
        ellA.d_values,
        ellA.d_colIndeces,
        ellB.d_values,
        ellB.d_colIndeces,
        d_c,
        ellA.numRows,
        ellB.numRows,
        ellA.ellNumCols,
        ellB.ellNumCols
    );
}