#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include "../include/formats/matrix_format.hpp"
#include "../include/formats/ell.hpp"
#include "../include/formats/hyb.hpp"
#include "../include/matmult/mat_mult.hpp"

void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Create test matrices with floats
    std::vector<std::vector<float>> matrixA = {
        {1.0f, 0.0f, 0.0f, 2.5f},
        {0.0f, 3.2f, 0.0f, 0.0f},
        {0.0f, 0.0f, 4.1f, 0.0f},
        {5.7f, 0.0f, 0.0f, 6.3f}
    };

    std::vector<std::vector<float>> matrixB = {
        {1.5f, 0.0f, 2.2f, 0.0f},
        {0.0f, 3.7f, 0.0f, 0.0f},
        {0.0f, 0.0f, 4.8f, 0.0f},
        {5.1f, 0.0f, 0.0f, 6.9f}
    };

    std::cout << "Input Matrix A:\n";
    printMatrix(matrixA);
    std::cout << "\nInput Matrix B:\n";
    printMatrix(matrixB);

    EllFormat A;
    EllFormat B;
    
    A.initFromMatrix(matrixA);
    B.initFromMatrix(matrixB);

    // Perform multiplication
    std::cout << "\nPerforming multiplication...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    MatMult* C = MatMult::create(A, B);
    C->multiply();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Get and print result
    std::cout << "\nResult Matrix C:\n";
    
    std::cout << "\nMultiplication took " << elapsed.count() << " ms\n";

    return 0;
}