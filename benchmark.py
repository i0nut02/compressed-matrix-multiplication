import os
import subprocess
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
MATRIX_DIR = Path("./matrices")
RESULTS_DIR = Path("./results")
EXECUTABLE_PATH = Path("./matmult/bin/main")
NUM_ITERATIONS = 3
# Define each BSR variation as a separate format for simpler logic
MATRIX_FORMATS = ["ELL", "HYB", "BSR16", "BSR8", "BSR4", "CLASSIC"]
DIMENSION_THRESHOLD_K = 6  # Represents 10,000 for dimension filtering

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
    except (ValueError, IndexError):
        print(f"Error: Invalid format in {file_path}. Expected 'rows cols' on the first line.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading dimensions from {file_path}: {e}")
        return None, None

def run_matrix_multiplication_automation():
    """
    Automates matrix multiplication tests, handling each format uniformly.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored under: {RESULTS_DIR.resolve()} ⚙️")

    all_matrix_files = [f for f in MATRIX_DIR.iterdir() if f.is_file() and f.suffix == ".inp"]

    if not all_matrix_files:
        print(f"Error: No matrix files found in '{MATRIX_DIR.resolve()}'.")
        return

    # --- Group matrices by their prefix ---
    grouped_matrices = defaultdict(list)
    for f in all_matrix_files:
        prefix = f.stem.split('_')[0]
        grouped_matrices[prefix].append(f)
    
    print("\n--- Found and Grouped Matrices ---")
    for prefix, files in grouped_matrices.items():
        print(f"  Prefix '{prefix}': {[f.name for f in files]}")
    print("------------------------------------\n")

    # --- Pre-process to get dimensions for all matrices ---
    matrix_dimensions = {}
    for matrix_file in all_matrix_files:
        rows, cols = get_matrix_dimensions(matrix_file)
        if rows is not None:
            matrix_dimensions[matrix_file] = (rows, cols)

    # --- Iterate through each group of matrices ---
    for prefix, matrix_list in grouped_matrices.items():
        print(f"--- Processing Group: {prefix} ---")
        
        processable_matrix_list = [f for f in matrix_list if f in matrix_dimensions]

        for matrix_a_path in processable_matrix_list:
            for matrix_b_path in processable_matrix_list:
                
                rows_a, cols_a = matrix_dimensions[matrix_a_path]
                rows_b, cols_b = matrix_dimensions[matrix_b_path]

                if cols_a != rows_b:
                    continue

                rows_a_k = rows_a // 1000
                cols_a_k = cols_a // 1000
                is_small_pair = (rows_a_k <= DIMENSION_THRESHOLD_K and cols_a_k <= DIMENSION_THRESHOLD_K)

                # --- Iterate through specified formats ---
                for matrix_format in MATRIX_FORMATS:
                    if matrix_format in ["ELL", "HYB"] and not is_small_pair:
                        print(f"  Skipping {matrix_format} for {matrix_a_path.name}-{matrix_b_path.name}: Dimensions too large.")
                        continue
                    
                    print(f"\n- Testing Pair: {matrix_a_path.name} & {matrix_b_path.name} (Format: {matrix_format})")

                    # --- Prepare arguments for the executable ---
                    format_arg = matrix_format
                    block_size_arg = None
                    if matrix_format.startswith("BSR"):
                        format_arg = "BSR"  # The executable needs the base format name
                        block_size_arg = matrix_format[3:] # The number part is the block size

                    # --- Create directory and filenames ---
                    # The full format name (e.g., BSR16) is used for the directory
                    output_dir = RESULTS_DIR / matrix_format / prefix
                    output_dir.mkdir(parents=True, exist_ok=True)

                    for iteration in range(1, NUM_ITERATIONS + 1):
                        # The full format name is used in the filename for clarity
                        base_filename = f"{matrix_a_path.stem}-{matrix_b_path.stem}-{matrix_format}-{iteration}"
                        log_file_path = output_dir / f"{base_filename}.log"
                        output_file_path = output_dir / f"{base_filename}.out"

                        print(f"    Running iteration {iteration}/{NUM_ITERATIONS} -> {output_dir.name}/{log_file_path.name}")

                        # --- Build the command list ---
                        command = [
                            str(EXECUTABLE_PATH), str(matrix_a_path), str(matrix_b_path),
                            str(output_file_path), format_arg, str(log_file_path)
                        ]
                        if block_size_arg:
                            command.append(block_size_arg)

                        # --- Execute the C++ program ---
                        try:
                            subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
                            print(f"      ✅ Iteration {iteration} completed successfully.")
                        except subprocess.CalledProcessError as e:
                            print(f"      ❌ Error running iteration {iteration}:")
                            print(f"      C++ STDERR:\n{e.stderr.strip()}")
                        except FileNotFoundError:
                            print(f"❌ Error: Executable not found at '{EXECUTABLE_PATH.resolve()}'.")
                            return
                        except Exception as e:
                            print(f"❌ An unexpected error occurred during execution: {e}")
                            return

if __name__ == "__main__":
    print("Starting matrix multiplication automation script.")
    run_matrix_multiplication_automation()
    print("\nAutomation script finished. 🎉")