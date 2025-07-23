#ifndef BSR_HPP
#define BSR_HPP

#include <string>
#include <vector>

#include <ostream>

#include "matrix_format.hpp"

class BsrFormat : public MatrixFormat {
    private:
        std::vector<float> values;
        std::vector<int> colIndices;
        std::vector<int> rowPointers;

        int blockSize;
    public:
        BsrFormat(int blockSize);

        float* d_values;
        int* d_column;
        int* d_rowPointers;
        
        void initFromMatrix(std::vector<std::vector<float>> m) override;

        virtual void writeToFile(const std::string& filepath) const;

        friend std::ostream& operator<<(std::ostream& os, const BsrFormat& ell);

        void cudaMemoryAllocation() override;

        void cudaMemoryFree() override;

        ~BsrFormat() override;
};

#endif