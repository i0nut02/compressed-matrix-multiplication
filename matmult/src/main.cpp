#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

#include "../include/formats/matrix_format.hpp"
#include "../include/formats/ell.hpp"
#include "../include/formats/hyb.hpp"
#include "../include/formats/classic.hpp"
#include "../include/formats/bsr.hpp"
#include "../include/matmult/mat_mult.hpp"

#include "../include/errors/errors_code.hpp"

#include "../include/utils.hpp"

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <matrixA_file> <matrixB_file> <output_path> <format> <log_file>\n";
        std::cerr << "  <matrixA_file>: Path to the input file for Matrix A.\n";
        std::cerr << "  <matrixB_file>: Path to the input file for Matrix B.\n";
        std::cerr << "  <output_path>: Path where the result matrix C will be saved.\n";
        std::cerr << "  <format>: The matrix storage format (e.g., 'ELL', 'HYB').\n";
        std::cerr << "  <log_file>: Path to the file where timing information will be logged.\n";
        return INPUT_ERROR;
    }

    std::string matrixA_file = argv[1];
    std::string matrixB_file = argv[2];
    std::string output_path = argv[3];
    std::string format_str = argv[4];
    std::string log_file = argv[5];

    MatrixFormat* A_format_ptr;
    MatrixFormat* B_format_ptr;

    if (format_str == "ELL") {
        A_format_ptr = new EllFormat();
        B_format_ptr = new EllFormat();
    } else if (format_str == "HYB") {
        A_format_ptr = new HybFormat();
        B_format_ptr = new HybFormat();
    } else if (format_str == "CLASSIC") {
        A_format_ptr = new ClassicFormat();
        B_format_ptr = new ClassicFormat();
    } else if (format_str == "BSR") {
        A_format_ptr = new BsrFormat();
        B_format_ptr = new BsrFormat();
    } else {
        std::cerr << "Error: Unknown matrix format '" << format_str << "'. Supported formats are 'ELL' and 'HYB'.\n";
        return INPUT_ERROR;
    }

    std::cout << "Initializing matrices with " << format_str << " format...\n";
    A_format_ptr->initFromFile(matrixA_file);
    B_format_ptr->initFromFile(matrixB_file);

    std::cout << "\nPerforming multiplication...\n";
    auto start_total = std::chrono::high_resolution_clock::now(); 

    MatMult* C = MatMult::create(*A_format_ptr, *B_format_ptr);

    if (!C) {
        std::cerr << "Error: Failed to create MatMult object.\n";
        return MEMORY_ALLOCATION_ERROR;
    }

    auto start_alloc = std::chrono::high_resolution_clock::now();
    C->cudaMemoryAllocation();
    auto end_alloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_alloc = end_alloc - start_alloc;
    std::cout << "CUDA Memory Allocation took " << elapsed_alloc.count() << " ms\n";

    auto start_mult = std::chrono::high_resolution_clock::now();
    C->multiply();
    auto end_mult = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_mult = end_mult - start_mult;
    std::cout << "Matrix Multiplication took " << elapsed_mult.count() << " ms\n";

    auto start_free = std::chrono::high_resolution_clock::now();
    C->cudaMemoryFree();
    auto end_free = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_free = end_free - start_free;
    std::cout << "CUDA Memory Free took " << elapsed_free.count() << " ms\n";

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_total = end_total - start_total;

    std::cout << "\nRetrieving result matrix...\n";
    std::vector<std::vector<float>> resultMatrixC = C->getOutputDenseMatrix();

    delete C;
    C = nullptr;

    std::cout << "Writing result matrix to: " << output_path << "\n";
    writeMatrixToFile(output_path, resultMatrixC);

    std::ofstream log_stream(log_file, std::ios_base::app);
    if (!log_stream.is_open()) {
        std::cerr << "Error: Could not open log file " << log_file << "\n";
    } else {
        log_stream << "Matrix A: " << matrixA_file << ", Matrix B: " << matrixB_file
                   << ", Format: " << format_str
                   << ", Total Time: " << elapsed_total.count() << " ms"
                   << ", Alloc Time: " << elapsed_alloc.count() << " ms"
                   << ", Mult Time: " << elapsed_mult.count() << " ms"
                   << ", Free Time: " << elapsed_free.count() << " ms\n";
        log_stream.close();
        std::cout << "Timing logged to: " << log_file << "\n";
    }

    std::cout << "\nTotal Multiplication process completed in " << elapsed_total.count() << " ms\n";

    return 0;
}