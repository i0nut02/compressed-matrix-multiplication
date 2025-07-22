#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void printMatrix(const std::vector<std::vector<float>>& matrix);

std::vector<std::vector<float>> readMatrixFromFile(const std::string& filename);

void writeMatrixToFile(const std::string& filename, const std::vector<std::vector<float>>& matrix);

#endif