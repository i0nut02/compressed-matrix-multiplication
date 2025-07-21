#ifndef HYB_HPP
#define HYB_HPP

#include <string>
#include <vector>

#include "matrix_format.hpp"

class HybFormat : public MatrixFormat {
    private:
        float* ellValues;
        int* ellColIndices;

        int* cooRowIndices;
        int* cooColIndices;
        float* cooValues;

        std::vector<int> getNumNonZeroPerRow(std::vector<std::vector<float>> m);

        int getEllCols(std::vector<int> numNonZeroPerRow);
        int remainingElements(std::vector<int> numNonZeroPerRow, int k);
    public:
        HybFormat();

        int ellCols;
        int cooElements;

        float* d_ellValues;
        int* d_ellColIndices;

        int* d_cooRowIndices;
        int* d_cooColIndices;
        float* d_cooValues;

        
        void initFromMatrix(std::vector<std::vector<float>> m) override;

        virtual void writeToFile(const std::string& filepath) const;

        friend std::ostream& operator<<(std::ostream& os, const HybFormat& hyb);

        void cudaMemoryAllocation() override;

        void cudaMemoryFree() override;

        ~HybFormat() override;
};

#endif