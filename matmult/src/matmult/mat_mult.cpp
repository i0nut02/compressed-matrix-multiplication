#include "../../include/matmult/mat_mult.hpp"
#include "../../include/errors/errors_code.hpp"

#include <iostream>

MatMult::MatMult(MatrixFormat& a_, MatrixFormat& b_)
    : a(std::shared_ptr<MatrixFormat>(&a_, [](MatrixFormat*){})),
      b(std::shared_ptr<MatrixFormat>(&b_, [](MatrixFormat*){})) {}

void MatMult::cudaMemoryAllocation() {
    a->cudaMemoryAllocation();
    b->cudaMemoryAllocation();
}

void MatMult::cudaMemoryFree() {
    a->cudaMemoryFree();
    b->cudaMemoryFree();
}

MatMult* MatMult::create(const MatrixFormat& a_, const MatrixFormat& b_) {
    if (typeid(a_) != typeid(b_)) {
        std::cout << "A and B must be on the same format for matrix multiplication" << std::endl;
        exit(INPUT_ERROR);
    }
    return new MatMult(const_cast<MatrixFormat&>(a_), const_cast<MatrixFormat&>(b_));
}