#include "../../include/matmult/hyb_mat_mult.hpp"
#include <iostream>

HybMatMult::HybMatMult(HybFormat& a_, HybFormat& b_)
    : MatMult(a_, b_) {}

void HybMatMult::multiply() {
    std::cout << "Multiplying two HYB matrices..." << std::endl;
}