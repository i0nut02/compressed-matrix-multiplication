#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include "../include/formats/matrix_format.hpp"
#include "../include/formats/ell.hpp"
#include "../include/formats/hyb.hpp"

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Create test matrices
    std::vector<std::vector<int>> matrixA = {
        {1, 0, 0, 2},
        {0, 3, 0, 0},
        {0, 0, 4, 0},
        {5, 0, 0, 6}
    };

    std::vector<std::vector<int>> matrixB = {
        {1, 0, 2, 0},
        {0, 3, 0, 0},
        {0, 0, 4, 0},
        {5, 0, 0, 6}
    };

    // Test with ELL format
    std::cout << "Testing ELL Format:\n";
    std::cout << "-------------------\n";
    {
        std::unique_ptr<MatrixFormat> A(new EllFormat());
        std::unique_ptr<MatrixFormat> B(new EllFormat());
        std::unique_ptr<MatrixFormat> C(new EllFormat());

        std::cout << "Initializing matrices...\n";
        A->initFromMatrix(matrixA);
        B->initFromMatrix(matrixB);

        auto start = std::chrono::high_resolution_clock::now();
        C->multiply(*A, *B);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "ELL multiplication took " << elapsed.count() << " ms\n";
    }

    std::cout << "\n";

    // Test with HYB format
    std::cout << "Testing HYB Format:\n";
    std::cout << "-------------------\n";
    {
        std::unique_ptr<MatrixFormat> A(new HybFormat());
        std::unique_ptr<MatrixFormat> B(new HybFormat());
        std::unique_ptr<MatrixFormat> C(new HybFormat());

        std::cout << "Initializing matrices...\n";
        A->initFromMatrix(matrixA);
        B->initFromMatrix(matrixB);

        auto start = std::chrono::high_resolution_clock::now();
        C->multiply(*A, *B);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "HYB multiplication took " << elapsed.count() << " ms\n";
    }

    return 0;
}