#include "../../include/formats/hyb.hpp"
#include "../../include/errors/errors_code.hpp"
#include "../../include/cuda/hyb_cuda.cuh"
#include "../../include/cuda/cuda_mem.cuh"

#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <fstream>


HybFormat::HybFormat(){}

void HybFormat::initFromMatrix(std::vector<std::vector<float>> m) {
    numRows = m.size();
    numCols = m[0].size();
    {
        std::vector<int> numNonZeroPerRow = getNumNonZeroPerRow(m);
        ellCols = getEllCols(numNonZeroPerRow);
        cooElements = remainingElements(numNonZeroPerRow, ellCols);
    }

    ellValues = (float*)calloc(ellCols * numRows, sizeof(float));
    ellColIndices = (int*)calloc(ellCols * numRows, sizeof(int));

    cooRowIndices = (int*)calloc(cooElements, sizeof(int));
    cooColIndices = (int*)calloc(cooElements, sizeof(int));
    cooValues = (float*)calloc(cooElements, sizeof(float));

    int cooIndex = 0;
    for (long unsigned int i=0; i < m.size(); i++) {
        int ellColIndex = 0;
        for (long unsigned int j=0; j < m[i].size(); j++) {
            if (m[i][j] == 0) {
                continue;
            }
            if (ellColIndex < ellCols) {
                ellValues[i * ellCols + ellColIndex] = m[i][j];
                ellColIndices[i * ellCols + ellColIndex] = j;
                ellColIndex++;
            } else {
                cooRowIndices[cooIndex] = i;
                cooColIndices[cooIndex] = j;
                cooValues[cooIndex] = m[i][j];
            }
        }
    }
}

void HybFormat::cudaMemoryAllocation() {
    allocate_hyp_memory_cuda(&d_ellValues, &d_ellColIndices, numRows * ellCols,
                             &d_cooValues, &d_cooRowIndices, &d_cooColIndices, cooElements);

    vector_copy_cuda(ellValues, d_ellValues, numRows * ellCols);
    vector_copy_cuda(ellColIndices, d_ellColIndices, numRows * ellCols);

    vector_copy_cuda(cooValues, d_cooValues, cooElements);
    vector_copy_cuda(cooRowIndices, d_cooRowIndices, cooElements);
    vector_copy_cuda(cooColIndices, d_cooColIndices, cooElements);
}

void HybFormat::cudaMemoryFree() {
    vector_free_cuda(d_cooColIndices);
    vector_free_cuda(d_cooRowIndices);
    vector_free_cuda(d_cooValues);
    vector_free_cuda(d_ellColIndices);
    vector_free_cuda(d_ellValues);
}

void HybFormat::writeToFile(const std::string& filepath) const {
    std::ofstream outFile(filepath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        exit(FILE_WRITE_ERROR);
    }

    // Call the polymorphic printToStream method
    outFile << this;

    outFile.close();
    std::cout << "Format successfully written to: " << filepath << std::endl;
}

int HybFormat::remainingElements(std::vector<int> numNonZeroPerRow, int k) {
    int res = 0;
    for (auto el : numNonZeroPerRow) {
        res += std::max(0, el - k);
    }
    return res;
}

int HybFormat::getEllCols(std::vector<int> numNonZeroPerRow) {
    int numNonZero = std::accumulate(numNonZeroPerRow.begin(), numNonZeroPerRow.end(), 0);
    std::sort(numNonZeroPerRow.begin(), numNonZeroPerRow.end());

    long unsigned int currSum = 0;
    long unsigned int n = numNonZeroPerRow.size();
    for (long unsigned int i=0; i < n; i++) {
        if ((currSum + (n -i) * numNonZeroPerRow[i]) > (long unsigned int) (0.95 * numNonZero)) { 
            return numNonZeroPerRow[i];
        }
        currSum += numNonZeroPerRow[i];
    }
    return numNonZeroPerRow[n-1];
}

std::vector<int> HybFormat::getNumNonZeroPerRow(std::vector<std::vector<float>> m) {
    std::vector<int> numNonZeroPerRow = std::vector<int>(m.size());

    for (long unsigned int i=0; i < m.size(); i++) {
        int currNumNonZeroRow = 0;
        for (long unsigned int j=0; j < m[i].size(); j++) {
            if (m[i][j] != 0) {
                currNumNonZeroRow += 1;
            }
        }
        numNonZeroPerRow[i] = currNumNonZeroRow;
    }
    return numNonZeroPerRow;
}

std::ostream& operator<<(std::ostream& os, const HybFormat& hyb) {
    os << "--- HYBRID Matrix Format ---" << std::endl;
    os << "Number of Rows (numRows): " << hyb.numRows << std::endl;
    os << "ELL Columns (ellCols): " << hyb.ellCols << std::endl;
    os << "Actual COO Elements: " << hyb.cooElements << std::endl; // Use the new member
    os << std::endl;

    os << "--- ELL Part ---" << std::endl;
    os << "Values:" << std::endl;
    for (int i = 0; i < hyb.numRows; ++i) {
        for (int j = 0; j < hyb.ellCols; ++j) {
            os << hyb.ellValues[i * hyb.ellCols + j] << "\t";
        }
        os << std::endl;
    }
    os << std::endl;

    os << "Column Indices (ellColIndices):" << std::endl;
    for (int i = 0; i < hyb.numRows; ++i) {
        for (int j = 0; j < hyb.ellCols; ++j) {
            os << hyb.ellColIndices[i * hyb.ellCols + j] << "\t";
        }
        os << std::endl;
    }
    os << std::endl;

    os << "--- COO Part ---" << std::endl;
    os << "Row Indices (cooRowIndices): [";
    for (int i = 0; i < hyb.cooElements; ++i) {
        os << hyb.cooRowIndices[i] << (i == (hyb.cooElements - 1) ? "" : ", ");
    }
    os << "]" << std::endl;

    os << "Column Indices (cooColIndices): [";
    for (int i = 0; i < hyb.cooElements; ++i) {
        os << hyb.cooColIndices[i] << (i == (hyb.cooElements - 1) ? "" : ", ");
    }
    os << "]" << std::endl;

    os << "Values (cooValues): [";
    for (int i = 0; i < hyb.cooElements; ++i) {
        os << hyb.cooValues[i] << (i == (hyb.cooElements - 1) ? "" : ", ");
    }
    os << "]" << std::endl;

    os << "----------------------------" << std::endl;

    return os; // Return the ostream reference
}

HybFormat::~HybFormat() {
    free(ellValues);
    free(ellColIndices);
    free(cooRowIndices);
    free(cooColIndices);
    free(cooValues);

    ellValues = nullptr;
    ellColIndices = nullptr;
    cooRowIndices = nullptr;
    cooColIndices = nullptr;
    cooValues = nullptr;
}