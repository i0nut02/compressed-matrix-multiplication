#ifndef ELL_HPP
#define ELL_HPP

#include <string>
#include <vector>

#include "matrix_format.hpp"

class EllFormat : public MatrixFormat {
    private:
        int numRows;
        int numCols;
        float* values;
        int* colIndices;
    public:
        void initFromMatrix(std::vector<std::vector<int>> m) override;

        ~EllFormat() override;
};

#endif