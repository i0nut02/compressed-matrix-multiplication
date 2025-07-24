#include "../../include/matmult/classic_mat_mult.hpp"
#include "../../include/cuda/classic_cuda.cuh"

#include <iostream>

ClassicMatMult::ClassicMatMult(ClassicFormat& a_, ClassicFormat& b_) 
    : MatMult(a_, b_) {}

void ClassicMatMult::multiply() {
    std::cout << "Multiplying two CLASSIC matrices..." << std::endl;
    ClassicFormat& classicA = static_cast<ClassicFormat&>(a);
    ClassicFormat& classicB = static_cast<ClassicFormat&>(b);

    classic_matrix_multiply_cuda(
        classicA.d_values,
        classicB.d_values,
        d_c,
        a.numRows,
        a.numCols,
        b.numCols
    );
}