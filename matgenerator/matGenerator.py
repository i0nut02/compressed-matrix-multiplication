import random
import os
from collections import defaultdict

class MatrixGenerator():
    """Base class for generating sparse matrices."""
    def __init__(self):
        self.rows = 0
        self.columns = 0
        self.path = "."
        self.filename = "random_matrix.txt"
        self.nnzPercentage = 0.01
        self.minVal = -100
        self.maxVal = 100
        self.seed = None

    def setRows(self, rows):
        self.rows = rows
        return self

    def setColumns(self, columns):
        self.columns = columns
        return self

    def setPath(self, path):
        self.path = path
        return self

    def setFilename(self, filename):
        self.filename = filename
        return self

    def setNNZPercentage(self, nnzPercentage):
        assert(0 <= nnzPercentage <= 1)
        self.nnzPercentage = nnzPercentage
        return self

    def setValueRange(self, a, b):
        assert(a <= b)
        self.minVal = a
        self.maxVal = b
        return self

    def setRandomSeed(self, seed):
        self.seed = seed
        return self
        
    def generate(self):
        # This is the original generate method, which will be overridden.
        if self.seed is not None:
            random.seed(self.seed)

        total_elements = self.rows * self.columns
        num_nnz = int(total_elements * self.nnzPercentage)

        if num_nnz == 0 and self.nnzPercentage > 0 and total_elements > 0:
            num_nnz = 1
       
        sparse_elements = defaultdict(float)
        while len(sparse_elements) < num_nnz:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.columns - 1)
            val = random.uniform(self.minVal, self.maxVal)
            sparse_elements[(r, c)] = val

        full_path = os.path.join(self.path, self.filename)
        with open(full_path, 'w') as f:
            f.write(f"{self.rows} {self.columns}\n")
            for (r, c), val in sorted(sparse_elements.items()):
                f.write(f"{r} {c} {val}\n")
        return full_path