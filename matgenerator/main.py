from matGenerator import MatrixGenerator
from diag import BandedMatrixGenerator
from bsr import BlockMatrixGenerator

if __name__ == "__main__":
    MATRIX_DIM = 4096
    OUTPUT_DIR = "matrices"
    
    print(f"--- Starting Matrix Generation (Dimension: {MATRIX_DIM}x{MATRIX_DIM}) ---")
    print(f"--- Output directory: {OUTPUT_DIR}/\n")

    # --- 1. Generate Standard Random Matrices ---
    print(">>> Generating 4 'Random' matrices...")
    configs = [
        {"nnz": 0.01, "id": 1}, # Very Sparse
        {"nnz": 0.01,  "id": 2}, # Sparse
        {"nnz": 0.01,   "id": 3}, # Less Sparse
        {"nnz": 0.01,   "id": 4}  # Denser
    ]
    for config in configs:
        gen = MatrixGenerator()
        gen.setRows(MATRIX_DIM) \
           .setColumns(MATRIX_DIM) \
           .setNNZPercentage(config["nnz"]) \
           .setPath(OUTPUT_DIR) \
           .setFilename(f"random_matrix_{config['id']}.inp") \
           .generate()

    # --- 2. Generate Block Matrices ---
    print("\n>>> Generating 4 'Block' matrices...")
    configs = [
        {"block": 16,  "nnz": 0.01, "id": 1},
        {"block": 32,  "nnz": 0.01,  "id": 2},
        {"block": 8, "nnz": 0.01, "id": 3},
        {"block": 16, "nnz": 0.01,  "id": 4}
    ]
    for config in configs:
        gen = BlockMatrixGenerator()
        gen.setRows(MATRIX_DIM) \
           .setColumns(MATRIX_DIM) \
           .setBlockDim(config["block"]) \
           .setNNZPercentage(config["nnz"]) \
           .setPath(OUTPUT_DIR) \
           .setFilename(f"block_matrix_{config['id']}_b{config['block']}_nnz{config['nnz']}.txt") \
           .generate()

    # --- 3. Generate Banded (Diagonally-Dominant) Matrices ---
    print("\n>>> Generating 4 'Banded' matrices...")
    configs = [
        {"width": 3,   "prob": 0.4, "nnz": 0.05, "id": 1}, # Purely Diagonal
        {"width": 5,   "prob": 0.3, "nnz": 0.05, "id": 2}, # Narrow Band
        {"width": 30,  "prob": 0.4, "nnz": 0.05, "id": 3}, # Medium Band
        {"width": 50,  "prob": 0.4, "nnz": 0.05, "id": 4}  # Wide Band
    ]
    for config in configs:
        gen = BandedMatrixGenerator()
        gen.setRows(MATRIX_DIM) \
           .setColumns(MATRIX_DIM) \
           .setBandWidth(config["width"]) \
           .setOffDiagonalProbability(config["prob"]) \
           .setNNZPercentage(config["nnz"]) \
           .setPath(OUTPUT_DIR) \
           .setFilename(f"banded_matrix_{config['id']}_w{config['width']}_p{config['prob']:.1f}.txt") \
           .generate()
           
    print("\n--- All matrix generation complete. ---")