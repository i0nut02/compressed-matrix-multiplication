#ifndef MATRIX_FORMAT_HPP
#define MATRIX_FORMAT_HPP

#include <string>
#include <vector>

class MatrixFormat {
    public:
        virtual void initFromFile(std::string file){};

        virtual void initFromMatrix(std::vector<std::vector<int>> m) = 0;

        virtual ~MatrixFormat() {}
};

#endif