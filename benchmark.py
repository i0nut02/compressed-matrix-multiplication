import os
import subprocess
from pathlib import Path

MATRIX_DIR = Path("./matrices")
RESULTS_DIR = Path("./results")
EXECUTABLE_PATH = Path("./mat_mult_app")
NUM_ITERATIONS = 3
MATRIX_FORMATS = ["ELL", "HYB", "BSR", "CLASSIC"]

def run_matrix_multiplication_automation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored under: {RESULTS_DIR.resolve()}")

    matrix_files = sorted([f for f in MATRIX_DIR.iterdir() if f.is_file() and f.suffix == ".txt"])

    if not matrix_files:
        print(f"Error: No matrix files found in '{MATRIX_DIR.resolve()}'.")
        print("Please ensure your matrix files (e.g., matrixA.txt) are placed in this directory.")
        return

    print(f"Found {len(matrix_files)} matrix files: {[f.name for f in matrix_files]}")

    for i, matrix_a_path in enumerate(matrix_files):
        for j, matrix_b_path in enumerate(matrix_files):
            if matrix_a_path == matrix_b_path:
                continue

            for matrix_format in MATRIX_FORMATS:
                print(f"\n--- Processing Pair: {matrix_a_path.name} and {matrix_b_path.name} (Format: {matrix_format}) ---")

                # Create the format-specific directory first
                format_results_path = RESULTS_DIR / matrix_format
                format_results_path.mkdir(parents=True, exist_ok=True)

                # Then create the matrix-pair specific directory inside the format directory
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