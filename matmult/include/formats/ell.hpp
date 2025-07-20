#ifndef ELL_HPP
#define ELL_HPP

#include <string>
#include <vector>

#include <ostream>

#include "matrix_format.hpp"

class EllFormat : public MatrixFormat {
    private:
        int numRows;
        int numCols;
        float* values;
        int* colIndices;
    public:
        EllFormat();

        float* d_values;
        int* d_colIndeces;
        
        void initFromMatrix(std::vector<std::vector<float>> m) override;

        virtual void writeToFile(const std::string& filepath) const;

        friend std::ostream& operator<<(std::ostream& os, const EllFormat& ell);

        void cudaMemoryAllocation() override;

        void cudaMemoryFree() override;

        ~EllFormat() override;
};

#endif