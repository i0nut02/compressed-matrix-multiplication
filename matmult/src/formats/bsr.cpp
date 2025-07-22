#include "../../include/formats/bsr.hpp"
#include "../../include/errors/errors_code.hpp"
#include "../../include/cuda/bsr_cuda.cuh"
#include "../../include/cuda/cuda_mem.cuh"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cmath>
#include <sstream>

BsrFormat::BsrFormat(){
    blockSize = 32;
    rowPointers.push_back(0);
}

void BsrFormat::initFromMatrix(std::vector<std::vector<float>> m) {
    numRows = m.size();
    numCols = m[0].size();
    
    for (int blockX = 0; blockX < (numRows + blockSize -1)/ blockSize; blockX++) {
        int blocksInCurrentBlockRow = 0;

        for (int blockY = 0; blockY < (numCols + blockSize -1) / blockSize; blockY++) {
            std::vector<float> actBlock;
            actBlock.reserve(blockSize * blockSize);

            int startRow = blockX * blockSize;
            int startCol = blockY * blockSize;

            bool nonZeroFound = false;

            for (int deltaX = 0; deltaX < blockSize; deltaX++) {
                for (int deltaY = 0; deltaY < blockSize; deltaY++) {
                    if ((startRow + deltaX) < numRows && (startCol + deltaY) < numCols) {
                        actBlock.push_back(m[startRow + deltaX][startCol + deltaY]);
                        if (m[startRow + deltaX][startCol + deltaY] != 0.0f) {
                            nonZeroFound = true;
                        }
                    } else {
                        actBlock.push_back(0);
                    }
                }
            }
            if (nonZeroFound) {
                values.insert(values.end(), actBlock.begin(), actBlock.end());
                colIndices.push_back(blockY);
                blocksInCurrentBlockRow++;
            }
        }
        rowPointers.push_back(rowPointers.back() + blocksInCurrentBlockRow);
    }
    
}

void BsrFormat::cudaMemoryAllocation() {
    vector_malloc_cuda(&d_values, values.size());
    vector_malloc_cuda(&d_column, colIndices.size());
    vector_malloc_cuda(&d_rowPointers, rowPointers.size());

    vector_copy_cuda(values.data(), d_values, values.size());
    vector_copy_cuda(colIndices.data(), d_column, colIndices.size());
    vector_copy_cuda(rowPointers.data(), d_rowPointers, rowPointers.size());
}

void BsrFormat::cudaMemoryFree() {
    vector_free_cuda(d_values);
    vector_free_cuda(d_column);
    vector_free_cuda(d_rowPointers);
}

void BsrFormat::writeToFile(const std::string& filepath) const {
    std::ofstream outFile(filepath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        exit(FILE_WRITE_ERROR);
    }

    outFile << *this;

    outFile.close();
    std::cout << "Format successfully written to: " << filepath << std::endl;
}

std::ostream& operator<<(std::ostream& os, const BsrFormat& bsr) {
    os << "--- BSR (Block Sparse Row) Matrix Format ---" << std::endl;
    os << "Original Number of Rows (numRows): " << bsr.numRows << std::endl;
    os << "Original Number of Columns (numCols): " << bsr.numCols << std::endl;
    os << "Block Size (blockSize): " << bsr.blockSize << std::endl;
    os << std::endl;

    os << "Values (values): [";
    for (size_t i = 0; i < bsr.values.size(); ++i) {
        os << std::fixed << std::setprecision(2) << bsr.values[i];
        if (i < bsr.values.size() - 1) {
            os << ", ";
        }
        if ((i + 1) % (bsr.blockSize * bsr.blockSize) == 0 && (i + 1) < bsr.values.size()) {
            os << "| ";
        }
    }
    os << "]" << std::endl;
    os << "  (Each | separates the data for a single block of size "
       << bsr.blockSize << "x" << bsr.blockSize << ")" << std::endl;
    os << std::endl;

    os << "Column Indices of Blocks (colIndices): [";
    for (size_t i = 0; i < bsr.colIndices.size(); ++i) {
        os << bsr.colIndices[i];
        if (i < bsr.colIndices.size() - 1) {
            os << ", ";
        }
    }
    os << "]" << std::endl;
    os << "  (These are the block column indices for each stored block)" << std::endl;
    os << std::endl;

    os << "Row Pointers (rowPointers): [";
    for (size_t i = 0; i < bsr.rowPointers.size(); ++i) {
        os << bsr.rowPointers[i];
        if (i < bsr.rowPointers.size() - 1) {
            os << ", ";
        }
    }
    os << "]" << std::endl;
    os << "  (Indicates the start index of blocks for each block row in 'values' and 'colIndices')" << std::endl;
    os << "  (The size of rowPointers is (number of block rows + 1))" << std::endl;
    os << std::endl;

    os << "--- Reconstructed (Padded) Dense Matrix (Conceptual View) ---" << std::endl;
    if (bsr.numRows > 0 && bsr.numCols > 0) {
        int paddedNumRows = static_cast<int>(std::ceil(static_cast<float>(bsr.numRows) / bsr.blockSize)) * bsr.blockSize;
        int paddedNumCols = static_cast<int>(std::ceil(static_cast<float>(bsr.numCols) / bsr.blockSize)) * bsr.blockSize;

        int max_width = 1;
        for (float val : bsr.values) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << val;
            max_width = std::max(max_width, static_cast<int>(oss.str().length()));
        }

        std::vector<std::vector<float>> dense_matrix(paddedNumRows, std::vector<float>(paddedNumCols, 0.0f));

        int block_count = 0;
        for (size_t i = 0; i < bsr.rowPointers.size() - 1; ++i) {
            int start_block_idx = bsr.rowPointers[i];
            int end_block_idx = bsr.rowPointers[i+1];

            for (int k = start_block_idx; k < end_block_idx; ++k) {
                int block_col = bsr.colIndices[k];
                int start_row_in_matrix = i * bsr.blockSize;
                int start_col_in_matrix = block_col * bsr.blockSize;

                for (int r = 0; r < bsr.blockSize; ++r) {
                    for (int c = 0; c < bsr.blockSize; ++c) {
                        if (start_row_in_matrix + r < paddedNumRows &&
                            start_col_in_matrix + c < paddedNumCols) {
                            dense_matrix[start_row_in_matrix + r][start_col_in_matrix + c] =
                                bsr.values[block_count * (bsr.blockSize * bsr.blockSize) + (r * bsr.blockSize + c)];
                        }
                    }
                }
                block_count++;
            }
        }

        for (int i = 0; i < paddedNumRows; ++i) {
            for (int j = 0; j < paddedNumCols; ++j) {
                os << std::fixed << std::setprecision(2) << std::setw(max_width + 1) << dense_matrix[i][j];
            }
            os << std::endl;
        }
    } else {
        os << "No data available to reconstruct dense matrix." << std::endl;
    }

    os << "----------------------------" << std::endl;

    return os;
}

BsrFormat::~BsrFormat() {}