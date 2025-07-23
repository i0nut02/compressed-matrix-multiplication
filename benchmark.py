import os
import subprocess
from pathlib import Path

MATRIX_DIR = Path("./matrices")
RESULTS_DIR = Path("./results")
EXECUTABLE_PATH = Path("./matmult/bin/main")
NUM_ITERATIONS = 3
MATRIX_FORMATS = ["ELL", "HYB", "BSR", "CLASSIC"]
DIMENSION_THRESHOLD_K = 6 # Represents 10,000 for dimension filtering

def get_matrix_dimensions(file_path):
    """Reads the first line of a matrix file to get its dimensions (rows, cols)."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            rows, cols = map(int, first_line.split())
            return rows, cols
    except FileNotFoundError:
        print(f"Error: Matrix file not found at {file_path}")
        return None, None
    except ValueError:
        print(f"Error: Invalid format in {file_path}. Expected 'rows cols' on the first line.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading dimensions from {file_path}: {e}")
        return None, None

def run_matrix_multiplication_automation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored under: {RESULTS_DIR.resolve()}")

    matrix_files = sorted([f for f in MATRIX_DIR.iterdir() if f.is_file() and f.suffix == ".inp"])

    if not matrix_files:
        print(f"Error: No matrix files found in '{MATRIX_DIR.resolve()}'.")
        print("Please ensure your matrix files (e.g., matrixA.inp) are placed in this directory.")
        return

    print(f"Found {len(matrix_files)} matrix files: {[f.name for f in matrix_files]}")

    # Pre-process to get dimensions for all matrices
    matrix_dimensions = {}
    for matrix_file in matrix_files:
        rows, cols = get_matrix_dimensions(matrix_file)
        if rows is not None and cols is not None:
            matrix_dimensions[matrix_file] = (rows, cols)
        else:
            print(f"Skipping {matrix_file.name} due to dimension reading error.")

    # Filter out files that failed dimension reading
    processable_matrix_files = [f for f in matrix_files if f in matrix_dimensions]
    if not processable_matrix_files:
        print("No processable matrix files after dimension check. Exiting.")
        return

    for i, matrix_a_path in enumerate(processable_matrix_files):
        for j, matrix_b_path in enumerate(processable_matrix_files):

            # Get dimensions for current pair
            rows_a, cols_a = matrix_dimensions[matrix_a_path]
            rows_b, cols_b = matrix_dimensions[matrix_b_path]

            # Check for matrix multiplication compatibility
            if cols_a != rows_b:
                print(f"  Skipping pair {matrix_a_path.name} and {matrix_b_path.name}: Incompatible dimensions for multiplication ({rows_a}x{cols_a} and {rows_b}x{cols_b}). Columns of A must equal Rows of B.")
                continue

            # Convert dimensions to 'k' (thousands) for comparison
            rows_a_k = rows_a // 1000
            cols_a_k = cols_a // 1000
            rows_b_k = rows_b // 1000
            cols_b_k = cols_b // 1000

            is_large_pair = (rows_a_k > DIMENSION_THRESHOLD_K or cols_a_k > DIMENSION_THRESHOLD_K or
                             rows_b_k > DIMENSION_THRESHOLD_K or cols_b_k > DIMENSION_THRESHOLD_K)
            is_small_pair = (rows_a_k <= DIMENSION_THRESHOLD_K and cols_a_k <= DIMENSION_THRESHOLD_K and
                             rows_b_k <= DIMENSION_THRESHOLD_K and cols_b_k <= DIMENSION_THRESHOLD_K)


            for matrix_format in MATRIX_FORMATS:
                if matrix_format in ["ELL", "HYB"]:
                    if not is_small_pair:
                        print(f"  Skipping {matrix_format} for {matrix_a_path.name}-{matrix_b_path.name}: Dimensions (in k) not <= {DIMENSION_THRESHOLD_K} in all dimensions.")
                        continue
                else:
                    print(f"  Warning: Unknown format '{matrix_format}'. Proceeding without dimension check.")


                print(f"\n--- Processing Pair: {matrix_a_path.name} and {matrix_b_path.name} (Format: {matrix_format}) ---")

                format_results_path = RESULTS_DIR / matrix_format
                format_results_path.mkdir(parents=True, exist_ok=True)

                pair_results_dir_name = f"{matrix_a_path.stem}-{matrix_b_path.stem}"
                pair_results_path = format_results_path / pair_results_dir_name
                pair_results_path.mkdir(parents=True, exist_ok=True)

                print(f"  Saving results for this pair in: {pair_results_path.resolve()}")

                for iteration in range(1, NUM_ITERATIONS + 1):
                    log_file_name = f"{iteration}.log"
                    output_file_name = f"{iteration}.out"

                    log_file_path = pair_results_path / log_file_name
                    output_file_path = pair_results_path / output_file_name

                    print(f"    Running iteration {iteration}/{NUM_ITERATIONS}...")
                    print(f"      Log file: {log_file_path.name}")
                    print(f"      Output file: {output_file_path.name}")

                    command = [
                        str(EXECUTABLE_PATH),
                        str(matrix_a_path),
                        str(matrix_b_path),
                        str(output_file_path),
                        matrix_format,
                        str(log_file_path)
                    ]

                    try:
                        result = subprocess.run(command, capture_output=True, text=True, check=True)

                        print(f"      Iteration {iteration} completed successfully.")
                        if result.stdout:
                            print("      C++ STDOUT:\n", result.stdout.strip())
                        if result.stderr:
                            print("      C++ STDERR:\n", result.stderr.strip())

                        # --- Explicitly check if files were created ---
                        if log_file_path.exists():
                            print(f"      Log file '{log_file_path.name}' created successfully.")
                        else:
                            print(f"      Warning: Log file '{log_file_path.name}' was NOT found after execution.")
                        
                        if output_file_path.exists():
                            print(f"      Output file '{output_file_path.name}' created successfully.")
                        else:
                            print(f"      Warning: Output file '{output_file_path.name}' was NOT found after execution.")
                        # --- End file creation check ---

                    except subprocess.CalledProcessError as e:
                        print(f"      Error running iteration {iteration} for {matrix_a_path.name}-{matrix_b_path.name} ({matrix_format}):")
                        print(f"      Command: {' '.join(e.cmd)}")
                        print(f"      Return Code: {e.returncode}")
                        print(f"      C++ STDOUT:\n{e.stdout.strip()}")
                        print(f"      C++ STDERR:\n{e.stderr.strip()}")
                    except FileNotFoundError:
                        print(f"Error: The executable was not found at '{EXECUTABLE_PATH.resolve()}'.")
                        print("Please ensure your C++ program is compiled and the path is correct.")
                        return
                    except Exception as e:
                        print(f"An unexpected error occurred during execution: {e}")
                        return

if __name__ == "__main__":
    print("Starting matrix multiplication automation script.")
    run_matrix_multiplication_automation()
    print("\nAutomation script finished.")
