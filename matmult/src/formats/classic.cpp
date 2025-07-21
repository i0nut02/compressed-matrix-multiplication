#include "../../include/formats/classic.hpp"
#include "../../include/errors/errors_code.hpp"
#include "../../include/cuda/classic_cuda.cuh"
#include "../../include/cuda/cuda_mem.cuh"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

ClassicFormat::ClassicFormat(){}

void ClassicFormat::initFromMatrix(std::vector<std::vector<float>> m) {
    numRows = m.size();
    numCols = m[0].size();
    
    values = (float*) calloc(numCols * numRows, sizeof(float));

    if (values == NULL) {
        exit(MEMORY_ALLOCATION_ERROR);
    }

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            values[i * numCols + j] = m[i][j];
        }
    }
}

void ClassicFormat::cudaMemoryAllocation() {
    vector_malloc_cuda(&d_values, numRows * numCols);
    vector_copy_cuda(values, d_values, numRows * numCols);
}

void ClassicFormat::cudaMemoryFree() {
    vector_free_cuda(d_values);
}

void ClassicFormat::writeToFile(const std::string& filepath) const {
    std::ofstream outFile(filepath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        exit(FILE_WRITE_ERROR);
    }

    outFile << *this;

    outFile.close();
    std::cout << "Format successfully written to: " << filepath << std::endl;
}

std::ostream& operator<<(std::ostream& os, const ClassicFormat& classic) {
    os << "--- Classic (Dense Row-Major) Matrix Format ---" << std::endl;
    os << "Number of Rows (numRows): " << classic.numRows << std::endl;
    os << "Number of Columns (numCols): " << classic.numCols << std::endl;
    os << std::endl;

    os << "Values (values):" << std::endl;
    if (classic.values && classic.numRows > 0 && classic.numCols > 0) {
        for (int i = 0; i < classic.numRows; ++i) {
            for (int j = 0; j < classic.numCols; ++j) {
                os << classic.values[i * classic.numCols + j] << "\t";
            }
            os << std::endl;
        }
    } else {
        os << "No values to display or matrix is empty." << std::endl;
    }
    os << std::endl;

    os << "----------------------------" << std::endl;

    return os;
}

ClassicFormat::~ClassicFormat() {
    free(values);

    values = nullptr;
}