#include "../../include/matmult/mat_mult.hpp"
#include "../../include/matmult/ell_mat_mult.hpp"
#include "../../include/matmult/hyb_mat_mult.hpp"

#include "../../include/formats/ell.hpp"
#include "../../include/formats/hyb.hpp"
#include "../../include/formats/matrix_format.hpp"

#include "../../include/errors/errors_code.hpp"


#include <iostream>

void MatMult::cudaMemoryAllocation() {
    a.cudaMemoryAllocation();
    b.cudaMemoryAllocation();
}

void MatMult::cudaMemoryFree() {
    a.cudaMemoryFree();
    b.cudaMemoryFree();
}

MatMult* MatMult::create(MatrixFormat& a_, MatrixFormat& b_) {
    if (typeid(a_) != typeid(b_)) {
        std::cout << "A and B must be on the same format for matrix multiplication" << std::endl;
        exit(INPUT_ERROR);
    }
    if (typeid(a_) == typeid(EllFormat)) {
        return new EllMatMult(static_cast<EllFormat&>(a_), static_cast<EllFormat&>(b_));
    } else if (typeid(a_) == typeid(HybFormat)) {
        return new HybMatMult(static_cast<HybFormat&>(a_), static_cast<HybFormat&>(b_));
    } else {
        std::cout << "Unsupported matrix format for multiplication" << std::endl;
        exit(INPUT_ERROR);
    }
}