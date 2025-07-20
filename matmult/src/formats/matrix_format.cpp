#include "../../include/formats/matrix_format.hpp"
#include "../../include/errors/errors_code.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

void MatrixFormat::initFromFile(std::string filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open matrix file: " << filepath << std::endl;
        exit(FILE_READ_ERROR);
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty or invalid header in file: " << filepath << std::endl;
        exit(FILE_READ_ERROR);
    }
    std::stringstream ss_header(line);
    int numRows_file, numCols_file;
    if (!(ss_header >> numRows_file >> numCols_file)) {
        std::cerr << "Error: Invalid dimensions in header of file: " << filepath << std::endl;
        exit(FILE_READ_ERROR);
    }

    if (numRows_file <= 0 || numCols_file <= 0) {
        std::cerr << "Error: Non-positive dimensions (" << numRows_file << "x" << numCols_file << ") in file: " << filepath << std::endl;
        exit(FILE_READ_ERROR);
    }

    std::vector<std::vector<float>> matrix(numRows_file, std::vector<float>(numCols_file, 0));

    int line_num = 1;
    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss_data(line);
        int rowIndex, colIndex;
        float value_float;

        if (!(ss_data >> rowIndex >> colIndex >> value_float)) {
            std::cerr << "Warning: Skipping invalid line " << line_num << " in file: " << filepath << std::endl;
            continue;
        }

        if (rowIndex < 0 || rowIndex >= numRows_file || colIndex < 0 || colIndex >= numCols_file) {
            std::cerr << "Warning: Skipping out-of-bounds entry at line " << line_num << " ("
                      << rowIndex << "," << colIndex << ") in file: " << filepath << std::endl;
            continue;
        }

        matrix[rowIndex][colIndex] = value_float;
    }

    file.close();
    initFromMatrix(matrix);
}