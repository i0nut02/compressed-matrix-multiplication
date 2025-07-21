#include "../../include/formats/ell.hpp"
#include "../../include/errors/errors_code.hpp"
#include "../../include/cuda/ell_cuda.cuh"
#include "../../include/cuda/cuda_mem.cuh"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

EllFormat::EllFormat(){}

void EllFormat::initFromMatrix(std::vector<std::vector<float>> m) {
    numRows = m.size();
    numCols = m[0].size();
    
    int maxNumNonZeroRow = 0;
    for (const auto& row : m) {
        int currNumNonZeroRow = 0;
        for (auto el : row) {
            if (el != 0) {
                currNumNonZeroRow += 1;
            }
        } 
        maxNumNonZeroRow = std::max(maxNumNonZeroRow, currNumNonZeroRow);
    }
    numRows = m.size();
    ellNumCols = maxNumNonZeroRow;

    values = (float*)calloc(numRows*ellNumCols, sizeof(float));
    colIndices = (int*)calloc(numRows*ellNumCols, sizeof(int));

    if (values == NULL || colIndices == NULL) {
        exit(MEMORY_ALLOCATION_ERROR);
    }

    for (long unsigned int i=0; i < m.size(); i++) {
        int columnToFill = 0;
        for (long unsigned int j=0; j < m[i].size(); j++) {
            if (m[i][j] != 0) {
                values[i * ellNumCols + columnToFill] = m[i][j];
                colIndices[i * ellNumCols + columnToFill] = j;
                columnToFill += 1;
            }
        }
    }
}

void EllFormat::cudaMemoryAllocation() {
    allocate_ell_memory_cuda(&d_values, &d_colIndeces, numRows, ellNumCols);
    vector_copy_cuda(values, d_values, numRows * ellNumCols);
    vector_copy_cuda(colIndices, d_colIndeces, numRows * ellNumCols);
}

void EllFormat::cudaMemoryFree() {
    vector_free_cuda(d_values);
    vector_free_cuda(d_colIndeces);
}

void EllFormat::writeToFile(const std::string& filepath) const {
    std::ofstream outFile(filepath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        exit(FILE_WRITE_ERROR);
    }

    outFile << *this;

    outFile.close();
    std::cout << "Format successfully written to: " << filepath << std::endl;
}

std::ostream& operator<<(std::ostream& os, const EllFormat& ell) {
    os << "--- ELL Matrix Format ---" << std::endl;
    os << "Number of Rows (numRows): " << ell.numRows << std::endl;
    os << "Number of ELL Columns (numCols): " << ell.ellNumCols << std::endl;
    os << std::endl;

    os << "Values (values):" << std::endl;
    if (ell.values && ell.numRows > 0 && ell.ellNumCols > 0) {
        for (int i = 0; i < ell.numRows; ++i) {
            for (int j = 0; j < ell.ellNumCols; ++j) {
                os << ell.values[i * ell.ellNumCols + j] << "\t";
            }
            os << std::endl;
        }
    } else {
        os << "No values to display or matrix is empty." << std::endl;
    }
    os << std::endl;

    os << "Column Indices (colIndices):" << std::endl;
    if (ell.colIndices && ell.numRows > 0 && ell.ellNumCols > 0) {
        for (int i = 0; i < ell.numRows; ++i) {
            for (int j = 0; j < ell.ellNumCols; ++j) {
                os << ell.colIndices[i * ell.ellNumCols + j] << "\t";
            }
            os << std::endl;
        }
    } else {
        os << "No column indices to display or matrix is empty." << std::endl;
    }
    os << std::endl;

    os << "----------------------------" << std::endl;

    return os;
}

EllFormat::~EllFormat() {
    free(values);
    free(colIndices);

    values = nullptr;
    colIndices = nullptr;
}