# Matrix Multiplication Performance Analysis

This repository contains a performance evaluation of different sparse matrix formats for matrix multiplication operations.

## Overview

This project benchmarks four compressed matrix storage formats against dense matrix multiplication to evaluate their performance characteristics across different matrix patterns. The study focuses on sparse matrices with various structural properties to determine optimal format selection based on matrix characteristics.

## Matrix Formats Tested

- **CLASSIC** - Standard dense matrix format
- **ELL (ELLPACK)** - Row-based format with padding to uniform row length
- **HYB (Hybrid)** - Combination of ELL and COO formats for improved compression
- **BSR (Block Sparse Row)** - Block-based storage with configurable block sizes (4x4, 8x8, 16x16)

## Test Matrices

### Random Matrices
- Dimensions: 4096 x 4096
- Non-zero density: 0.5% and 1%
- Randomly distributed non-zero elements

### Banded Matrices
- Dimensions: 4096 x 4096
- Band widths: 8 and 16
- Non-zero density: 0.4% and 0.8%
- Non-zero elements concentrated in diagonal bands

### Block Matrices
- Dimensions: 4096 x 4096
- Block sizes: 8x8 and 16x16
- Non-zero density: 0.3% and 0.6%
- Non-zero elements grouped in dense blocks

## Performance Results

### Random Matrices (1% NNZ)
| Format  | Total (ms) | Allocation (ms) | Multiplication (ms) | Free (ms) |
|---------|------------|-----------------|-------------------|-----------|
| BSR 4   | 390.55     | 111.40          | 263.80            | 15.57     |
| BSR 8   | 431.37     | 121.07          | 292.59            | 17.11     |
| BSR 16  | 453.79     | 134.80          | 302.54            | 17.19     |
| CLASSIC | 386.73     | 137.80          | 178.91            | 70.33     |
| ELL     | 581.36     | 108.27          | 405.04            | 67.77     |
| HYB     | 15890.80   | 106.34          | 15764.20          | 14.27     |

### Banded Matrices
| Format  | Total (ms) | Allocation (ms) | Multiplication (ms) | Free (ms) |
|---------|------------|-----------------|-------------------|-----------|
| BSR 4   | 256.88     | 108.52          | 132.37            | 15.86     |
| BSR 8   | 252.79     | 105.72          | 131.81            | 15.03     |
| BSR 16  | 307.48     | 108.97          | 130.40            | 67.37     |
| CLASSIC | 385.35     | 137.27          | 177.85            | 70.31     |
| ELL     | 514.14     | 105.64          | 342.49            | 66.78     |
| HYB     | 1618.49    | 105.90          | 1479.66           | 14.76     |

### Block Matrices
| Format  | Total (ms) | Allocation (ms) | Multiplication (ms) | Free (ms) |
|---------|------------|-----------------|-------------------|-----------|
| BSR 4   | 261.81     | 105.13          | 138.65            | 14.73     |
| BSR 8   | 260.23     | 105.60          | 136.79            | 15.63     |
| BSR 16  | 315.67     | 107.67          | 141.52            | 67.23     |
| CLASSIC | 385.95     | 136.17          | 178.04            | 70.36     |
| ELL     | 584.64     | 108.38          | 394.36            | 67.50     |
| HYB     | 526.75     | 102.22          | 397.45            | 14.85     |

## Key Findings

**CLASSIC Format:**
- Consistent performance across all matrix types
- Simple memory access patterns
- High memory overhead due to storing zero elements

**BSR Format:**
- Best performance for structured matrices (banded and block patterns)
- Performance depends on block size selection relative to matrix structure
- Low branch divergence in GPU implementations

**ELL Format:**
- Moderate performance across all test cases
- Suffers from high branch divergence
- Lower cache efficiency compared to BSR

**HYB Format:**
- Excellent compression ratios
- Performance highly dependent on matrix structure
- Synchronization overhead for COO component affects performance

## Recommendations

- Use BSR format for matrices with block or banded structure
- Select BSR block size based on matrix characteristics
- CLASSIC format provides reliable baseline performance
- HYB format should be used cautiously due to variable performance

## Test Configuration

- 16 different matrix combinations tested
- 3 iterations per combination
- 48 data points collected per matrix format
- Performance metrics include allocation, multiplication, and deallocation times