from matGenerator import MatrixGenerator

import random
import os
from collections import defaultdict


class BlockMatrixGenerator(MatrixGenerator):
    """
    Generates a sparse matrix with a block-based structure.
    """
    def __init__(self):
        super().__init__()
        self.block_dim = 4
        self.filename = "block_matrix.txt"

    def setBlockDim(self, block_dim):
        """Sets the square dimension for the blocks."""
        assert(block_dim > 0)
        self.block_dim = block_dim
        return self

    def generate(self):
        """Generates and saves the block-based sparse matrix."""
        if self.seed is not None:
            random.seed(self.seed)

        if self.rows % self.block_dim != 0 or self.columns % self.block_dim != 0:
            raise ValueError("Matrix dimensions must be divisible by block_dim.")

        num_block_rows = self.rows // self.block_dim
        num_block_cols = self.columns // self.block_dim

        selected_blocks = []
        for block_row_idx in range(num_block_rows):
            block_col_idx = random.randint(0, num_block_cols - 1)
            selected_blocks.append((block_row_idx, block_col_idx))

        total_nnz = int(self.rows * self.columns * self.nnzPercentage)
        if total_nnz == 0 and self.nnzPercentage > 0: total_nnz = 1
        
        if not selected_blocks:
            nnz_per_block_list = []
        else:
            nnz_base = total_nnz // len(selected_blocks)
            nnz_rem = total_nnz % len(selected_blocks)
            nnz_per_block_list = [nnz_base + 1] * nnz_rem + [nnz_base] * (len(selected_blocks) - nnz_rem)
            random.shuffle(nnz_per_block_list)

        sparse_elements = {}
        for i, (block_row, block_col) in enumerate(selected_blocks):
            num_nnz_in_block = nnz_per_block_list[i]
            start_row, start_col = block_row * self.block_dim, block_col * self.block_dim
            placed_in_block = 0
            while placed_in_block < num_nnz_in_block:
                g_row, g_col = start_row + random.randint(0, self.block_dim - 1), start_col + random.randint(0, self.block_dim - 1)        
                sparse_elements[(g_row, g_col)] = random.uniform(self.minVal, self.maxVal)
                placed_in_block += 1
        
        full_path = os.path.join(self.path, self.filename)
        with open(full_path, 'w') as f:
            f.write(f"{self.rows} {self.columns} {len(sparse_elements)} {self.block_dim}\n")
            for (r, c), val in sorted(sparse_elements.items()):
                f.write(f"{r} {c} {val}\n")
        
        return full_path