#ifndef MAT_MULT_HPP
#define MAT_MULT_HPP

#include "../formats/matrix_format.hpp"
#include "../formats/ell.hpp"
#include "../formats/hyb.hpp"

#include <memory>

class MatMult {
    protected:
        const MatrixFormat& a;
        const MatrixFormat& b;
    public:
        MatMult(const MatrixFormat& a_, const MatrixFormat& b_) : a(a_), b(b_) {}

        void cudaMemoryAllocation();

        void cudaMemoryFree();

        virtual void multiply() = 0;

        static MatMult* create(const MatrixFormat& a, const MatrixFormat& b);

        virtual ~MatMult() = default;
};

#endif