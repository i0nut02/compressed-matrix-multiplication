#ifndef BSR_MAT_MULT_HPP
#define BSR_MAT_MULT_HPP

#include "mat_mult.hpp"
#include "../formats/bsr.hpp"

class BsrMatMult : public MatMult {
public:
    BsrMatMult(BsrFormat& a_, BsrFormat& b_);

    void multiply() override;
};

#endif