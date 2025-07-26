from matGenerator import MatrixGenerator

import random
import os
from collections import defaultdict

class BandedMatrixGenerator(MatrixGenerator):
    """
    Generates a sparse, diagonally-dominant (banded) matrix.

    Non-zero elements are placed near the main diagonal. The matrix must be square.
    - `nnzPercentage`: The fraction of rows that will contain a non-zero element.
    - `band_width`: The maximum distance (in columns) an element can be from the diagonal.
    - `off_diagonal_prob`: The probability (0.0 to 1.0) that an element will be placed
      off-diagonal within the specified band.
    """
    def __init__(self):
        super().__init__()
        self.filename = "banded_matrix.txt"
        self.band_width = 1  # Default: can be on diagonal, or 1 column to left/right
        self.off_diagonal_prob = 0.5 # Default: 50% chance to be off-diagonal

    def setBandWidth(self, band_width):
        """Sets the maximum number of columns an element can be from the diagonal."""
        self.band_width = max(0, band_width)
        return self

    def setOffDiagonalProbability(self, probability):
        """Sets the probability (0.0 to 1.0) for an element to be off-diagonal."""
        assert 0.0 <= probability <= 1.0
        self.off_diagonal_prob = probability
        return self

    def generate(self):
        """Generates and saves the banded sparse matrix."""
        if self.seed is not None:
            random.seed(self.seed)

        if self.rows != self.columns:
            raise ValueError("A banded matrix must be square (rows == columns).")

        # Determine which rows will get a non-zero element
        num_potential_nnz = self.rows
        num_actual_nnz = int(num_potential_nnz * self.nnzPercentage)

        if num_actual_nnz == 0 and self.nnzPercentage > 0 and num_potential_nnz > 0:
            num_actual_nnz = 1

        # Randomly choose which rows will have a non-zero element
        rows_with_nnz = random.sample(range(self.rows), num_actual_nnz)

        sparse_elements = {}
        for r in rows_with_nnz:
            # Decide if this element should be off-diagonal
            if random.random() < self.off_diagonal_prob:
                # Place it off-diagonal, within the band
                offset = random.randint(-self.band_width, self.band_width)
                # Ensure the offset is not zero, otherwise it's on the diagonal
                if offset == 0:
                    offset = random.choice([-1, 1]) if self.band_width > 0 else 0
            else:
                # Place it on the diagonal
                offset = 0

            # Calculate the column, ensuring it stays within bounds
            c = r + offset
            c = max(0, min(self.columns - 1, c)) # Clamp to [0, cols-1]

            # Ensure the chosen spot is not already taken (unlikely but possible)
            # If it is, we just place it on the diagonal of that row as a fallback.
            if (r, c) in sparse_elements:
                c = r # Fallback to the diagonal

            val = random.uniform(self.minVal, self.maxVal)
            sparse_elements[(r, c)] = val

        # Write to file
        full_path = os.path.join(self.path, self.filename)
        with open(full_path, 'w') as f:
            # Header format: rows cols nnz band_width
            f.write(f"{self.rows} {self.columns} {len(sparse_elements)} {self.band_width}\n")
            for (row, col), val in sorted(sparse_elements.items()):
                f.write(f"{row} {col} {val}\n")

        return full_path