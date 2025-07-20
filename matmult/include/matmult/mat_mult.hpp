#ifndef MAT_MULT_HPP
#define MAT_MULT_HPP

#include "../formats/matrix_format.hpp"
#include "../formats/ell.hpp"
#include "../formats/hyb.hpp"

#include <memory>

class MatMult {
    private:
        std::shared_ptr<MatrixFormat> a;
        std::shared_ptr<MatrixFormat> b;

        MatMult(MatrixFormat& a, MatrixFormat& b);
    public:
        void cudaMemoryAllocation(){};

        void cudaMemoryFree(){};

        virtual void multiply() = 0;

        static MatMult* create(MatrixFormat& a, MatrixFormat& b);
};

#endif