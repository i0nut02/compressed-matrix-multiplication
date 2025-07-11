#ifndef HYB_HPP
#define HYB_HPP

#include <string>
#include <vector>

#include "matrix_format.hpp"

class HybFormat : public MatrixFormat {
    private:
        int numRows;
        int ellCols;

        float* ellValues;
        int* ellColIndices;

        int* cooRowIndices;
        int* cooColIndices;
        float* cooValues;

        void fillEll(std::vector<int> numNonZeroPerRow);

        void fillCoo();
        std::vector<int> getNumNonZeroPerRow(std::vector<std::vector<int>> m);

        int HybFormat::getEllCols(std::vector<int> numNonZeroPerRow);
    public:
        void initFromMatrix(std::vector<std::vector<int>> m) override;

        ~HybFormat() override;
};

#endif