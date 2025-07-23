#ifndef MATRIX_FORMAT_HPP
#define MATRIX_FORMAT_HPP

#include <string>
#include <vector>

class MatrixFormat {
    public:
        int numRows;
        int numCols;
        
        virtual void initFromFile(std::string filepath, bool transpose);

        virtual void initFromMatrix(std::vector<std::vector<float>> m) = 0;

        virtual void writeToFile(const std::string& filepath) const = 0;

        virtual void cudaMemoryAllocation() = 0;

        virtual void cudaMemoryFree() = 0;

        virtual ~MatrixFormat() = default;
};

#endif