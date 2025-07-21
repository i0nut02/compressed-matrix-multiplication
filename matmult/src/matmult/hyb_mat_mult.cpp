#include "../../include/matmult/hyb_mat_mult.hpp"
#include "../../include/cuda/hyb_cuda.cuh"
#include <iostream>

HybMatMult::HybMatMult(HybFormat& a_, HybFormat& b_)
    : MatMult(a_, b_) {}

void HybMatMult::multiply() {
    std::cout << "Multiplying two HYB matrices..." << std::endl;
    HybFormat& hybA = static_cast<HybFormat&>(a);
    HybFormat& hybB = static_cast<HybFormat&>(b);

    hyb_matrix_multiply_launch(
            hybA.d_ellValues,
            hybA.d_ellColIndices,
            hybB.d_ellValues,
            hybB.d_ellColIndices,
            d_c,
            hybA.numRows,
            hybB.numCols,
            hybA.ellCols,
            hybB.ellCols,
            hybA.d_cooValues,
            hybA.d_cooRowIndices,
            hybA.d_cooColIndices,
            hybB.d_cooValues,
            hybB.d_cooRowIndices,
            hybB.d_cooColIndices,
            hybA.cooElements,
            hybB.cooElements
        );
}