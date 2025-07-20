#include "../../include/matmult/ell_mat_mult.hpp"
#include <iostream>

EllMatMult::EllMatMult(EllFormat& a_, EllFormat& b_)
    : MatMult(a_, b_) {}

void EllMatMult::multiply() {
    std::cout << "Multiplying two ELL matrices..." << std::endl;
}