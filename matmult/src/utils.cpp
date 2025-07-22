#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

std::vector<std::vector<float>> readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open matrix file " << filename << "\n";
        return {};
    }

    int rows, cols;
    file >> rows >> cols;

    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: Invalid matrix dimensions in " << filename << "\n";
        return {};
    }

    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(file >> matrix[i][j])) {
                std::cerr << "Error: Not enough data in matrix file " << filename << " at row " << i << ", col " << j << "\n";
                return {};
            }
        }
    }
    file.close();
    return matrix;
}

void writeMatrixToFile(const std::string& filename, const std::vector<std::vector<float>>& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << filename << "\n";
        return;
    }

    if (matrix.empty()) {
        std::cerr << "Warning: Attempted to write an empty matrix to " << filename << "\n";
        return;
    }

    file << matrix.size() << " " << matrix[0].size() << "\n";

    for (const auto& row : matrix) {
        for (float val : row) {
            file << val << " ";
        }
        file << "\n";
    }
    file.close();
}