#ifndef CLASSIC_MAT_MULT_HPP
#define CLASSIC_MAT_MULT_HPP

#include "mat_mult.hpp"
#include "../formats/classic.hpp"

class ClassicMatMult : public MatMult {
public:
    ClassicMatMult(ClassicFormat& a_, ClassicFormat& b_);

    void multiply() override;
};

#endif