#ifndef HYB_MAT_MULT_HPP
#define HYB_MAT_MULT_HPP

#include "mat_mult.hpp"
#include "../formats/hyb.hpp"

// Concrete matrix multiplication class for HYB format
class HybMatMult : public MatMult {
public:
    HybMatMult(HybFormat& a_, HybFormat& b_);

    void multiply() override;
};

#endif