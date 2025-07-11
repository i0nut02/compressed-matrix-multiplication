#include "../include/ell.hpp"
#include "../include/errors_code.hpp"

#include <iostream>

EllFormat::EllFormat(){}

void EllFormat::initFromMatrix(std::vector<std::vector<int>> m) {
    int maxNumNonZeroRow = 0;
    for (auto row : m) {
        int currNumNonZeroRow = 0;
        for (auto el : row) {
            if (el != 0) {
                currNumNonZeroRow += 1;
            }
        } 
        maxNumNonZeroRow = std::max(maxNumNonZeroRow, currNumNonZeroRow);
    }
    numRows = m.size();
    numCols = maxNumNonZeroRow;

    values = (float*)calloc(numRows*numCols, sizeof(float));
    colIndices = (int*)calloc(numRows*numCols, sizeof(int));

    if (values == NULL || colIndices == NULL) {
        exit(MEMORY_ALLOCATION_ERROR);
    }

    for (int i=0; i < m.size(); i++) {
        int columnToFill = 0;
        for (int j=0; j < m[i].size(); j++) {
            if (m[i][j] != 0) {
                values[i * numCols + columnToFill] = m[i][j];
                colIndices[i * numCols + columnToFill] = j;
                columnToFill += 1;
            }
        }
    }
}

EllFormat::~EllFormat() {
    free(values);
    free(colIndices);
}