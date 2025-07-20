#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include <iostream>
#include <cuda_runtime.h>

#include "../errors/errors_code.hpp"

inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << "Function: " << func << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(CUDA_ERROR);
    }
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

#endif