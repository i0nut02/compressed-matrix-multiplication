#ifndef ELL_MAT_MULT_HPP
#define ELL_MAT_MULT_HPP

#include "mat_mult.hpp"
#include "../formats/ell.hpp"

class EllMatMult : public MatMult {
public:
    EllMatMult(EllFormat& a_, EllFormat& b_);

    void multiply() override;
};

#endif