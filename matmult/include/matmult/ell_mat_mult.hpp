#ifndef ELL_MAT_MULT_HPP
#define ELL_MAT_MULT_HPP

#include "mat_mult.hpp"
#include "../formats/ell.hpp"

// Concrete matrix multiplication class for ELL format
class EllMatMult : public MatMult {
public:
    EllMatMult(const EllFormat& a_, const EllFormat& b_);

    void multiply() override;
};

#endif