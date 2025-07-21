#ifndef CLASSIC_HPP
#define CLASSIC_HPP

#include <string>
#include <vector>

#include <ostream>

#include "matrix_format.hpp"

class ClassicFormat : public MatrixFormat {
    public:
        ClassicFormat();

        float* values;
        float* d_values;
        
        void initFromMatrix(std::vector<std::vector<float>> m) override;

        virtual void writeToFile(const std::string& filepath) const;

        friend std::ostream& operator<<(std::ostream& os, const ClassicFormat& classic);

        void cudaMemoryAllocation() override;

        void cudaMemoryFree() override;

        ~ClassicFormat() override;
};

#endif