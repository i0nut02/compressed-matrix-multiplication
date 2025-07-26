import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def read_sparse_matrix(file_path):
    with open(file_path, 'r') as f:
        # Read matrix dimensions
        first_line = f.readline()
        num_rows, num_cols = map(int, first_line.strip().split())

        # Read the remaining lines as triplets
        rows = []
        cols = []
        data = []
        for line in f:
            if line.strip() == "":
                continue
            r, c, v = line.strip().split()
            rows.append(int(r))
            cols.append(int(c))
            data.append(float(v))

    return coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

def plot_sparsity(matrix, title="Sparsity Pattern"):
    plt.figure(figsize=(6, 6))
    plt.spy(matrix, markersize=5)
    plt.title(title)
    plt.xlabel(f"nz = {matrix.nnz}")
    plt.show()

# Example usage:
file_path = 'matrix.txt'  # Replace with your file path
sparse_matrix = read_sparse_matrix(file_path)
plot_sparsity(sparse_matrix)
