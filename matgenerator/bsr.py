import random
import os
from pathlib import Path

class SpecialMatrixGenerator:
    def __init__(self):
        self.rows = 0
        self.columns = 0
        self.path = "../matrices" # Default path
        self.filename = "special_matrix.inp"
        self.nnz_percentage = 0.01
        self.value_min = -100.0
        self.value_max = 100.0
        self.random_seed = None
        self.num_active_rows = 3 # Fixed as per requirement

    def set_dimensions(self, dim):
        """Sets both rows and columns to the same dimension for a square matrix."""
        self.rows = dim
        self.columns = dim
        return self

    def set_path(self, path):
        self.path = Path(path)
        return self

    def set_filename(self, filename):
        self.filename = filename
        return self

    def set_nnz_percentage(self, percentage):
        """Sets the overall non-zero percentage for the matrix."""
        if not (0 <= percentage <= 1):
            raise ValueError("NNZ percentage must be between 0 and 1.")
        self.nnz_percentage = percentage
        return self

    def set_value_range(self, min_val, max_val):
        """Sets the range for random non-zero values."""
        if min_val > max_val:
            raise ValueError("Min value cannot be greater than max value.")
        self.value_min = float(min_val)
        self.value_max = float(max_val)
        return self

    def set_random_seed(self, seed):
        """Sets the random seed for reproducibility."""
        self.random_seed = seed
        return self

    def generate(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)

        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Matrix dimensions must be positive.")
        
        if self.num_active_rows > self.rows:
            raise ValueError(f"Number of active rows ({self.num_active_rows}) cannot exceed total rows ({self.rows}).")

        total_elements = self.rows * self.columns
        total_nnz = int(total_elements * self.nnz_percentage)

        if total_nnz == 0 and self.nnz_percentage > 0: # Ensure at least one NNZ if percentage > 0
            total_nnz = 1
        elif total_nnz == 0: # If percentage is 0, no NNZ
            pass

        # Select the specific rows where NNZ will be concentrated
        # For simplicity, let's pick the first `num_active_rows` rows.
        # You could also randomize these rows if needed.
        active_rows = list(range(self.num_active_rows))

        # Distribute NNZ elements roughly equally among the active rows
        nnz_per_active_row = total_nnz // self.num_active_rows
        remaining_nnz = total_nnz % self.num_active_rows

        # Store non-zero elements as (row, col, value) tuples
        sparse_elements = set() # Use a set to avoid duplicate (row, col) entries

        for i, row_idx in enumerate(active_rows):
            current_row_nnz = nnz_per_active_row
            if i < remaining_nnz: # Distribute remainder
                current_row_nnz += 1
            
            # Ensure we don't try to add more NNZ than available columns in a row
            current_row_nnz = min(current_row_nnz, self.columns)

            added_count = 0
            while added_count < current_row_nnz:
                col_idx = random.randint(0, self.columns - 1)
                value = random.uniform(self.value_min, self.value_max)
                
                # Add to set if not already present
                if (row_idx, col_idx, value) not in sparse_elements:
                    sparse_elements.add((row_idx, col_idx, value))
                    added_count += 1
        
        # Convert set to list and sort for consistent output (optional, but good for debugging)
        sorted_elements = sorted(list(sparse_elements))

        full_path = self.path / self.filename
        
        # Create the directory if it doesn't exist
        self.path.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w') as f:
            f.write(f"{self.rows} {self.columns}\n")
            for r, c, val in sorted_elements:
                f.write(f"{r} {c} {val}\n")
        
        print(f"Generated matrix '{self.filename}' with {len(sorted_elements)} NNZ elements at {full_path}")
        return full_path

if __name__ == "__main__":
    generator = SpecialMatrixGenerator()

    # Example usage: Generate a 10k x 10k matrix with 1% NNZ in 3 rows
    generator.set_dimensions(10000) \
             .set_nnz_percentage(0.01) \
             .set_filename("10k_special.inp") \
             .set_path("../matrices/") \
             .set_random_seed(123) \
             .generate()

    # Example usage: Generate a 5k x 5k matrix with 1% NNZ in 3 rows
    generator.set_dimensions(5000) \
             .set_nnz_percentage(0.01) \
             .set_filename("5k_special.inp") \
             .set_path("../matrices/") \
             .set_random_seed(456) \
             .generate()

    # You can add more generations here for different dimensions or filenames
