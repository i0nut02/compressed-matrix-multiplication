#include "../../include/matmult/mat_mult.hpp"
#include "../../include/matmult/ell_mat_mult.hpp"
#include "../../include/matmult/hyb_mat_mult.hpp"
#include "../../include/matmult/classic_mat_mult.hpp"

#include "../../include/formats/ell.hpp"
#include "../../include/formats/hyb.hpp"
#include "../../include/formats/matrix_format.hpp"
#include "../../include/formats/classic.hpp"


#include "../../include/errors/errors_code.hpp"

#include "../../include/cuda/cuda_mem.cuh"

#include <iostream>

void MatMult::cudaMemoryAllocation() {
    int outputElements = a.numRows * b.numRows;
    c = (float*) calloc(outputElements, sizeof(float));
    a.cudaMemoryAllocation();
    b.cudaMemoryAllocation();
    vector_malloc_cuda(&d_c, outputElements);
    vector_copy_cuda(c, d_c, outputElements);
}

void MatMult::cudaMemoryFree() {
    vector_copy_2host(c, d_c, a.numRows * b.numRows);
    a.cudaMemoryFree();
    b.cudaMemoryFree();
    vector_free_cuda(d_c);
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
    } else if (typeid(a_) == typeid(ClassicFormat)) {
        return new ClassicMatMult(static_cast<ClassicFormat&>(a_), static_cast<ClassicFormat&>(b_));
    } else {
        std::cout << "Unsupported matrix format for multiplication" << std::endl;
        exit(INPUT_ERROR);
    }
}

void MatMult::printOutput() {
    for (int i=0; i < outputRows; i++) {
        for (int j=0; j < outputCols; j++) {
            std::cout << c[i * outputCols + j] << " ";
        }
        std::cout << std::endl;
    }
}