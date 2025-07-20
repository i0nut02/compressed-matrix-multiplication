#ifndef MAT_MULT_HPP
#define MAT_MULT_HPP

#include "../formats/matrix_format.hpp"
#include "../formats/ell.hpp"
#include "../formats/hyb.hpp"

#include <memory>

class MatMult {
    protected:
        MatrixFormat& a;
        MatrixFormat& b;
        float* c;
        float* d_c;
    public:
        MatMult(MatrixFormat& a_, MatrixFormat& b_) : a(a_), b(b_) {}

        void cudaMemoryAllocation();

        void cudaMemoryFree();

        virtual void multiply() = 0;

        static MatMult* create(MatrixFormat& a, MatrixFormat& b);

        virtual ~MatMult() = default;
};

#endif