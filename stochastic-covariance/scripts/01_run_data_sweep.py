"""Master Stage 1 runner script with live progress reporting and instant flushing."""

from pathlib import Path
import time

import numpy as np
import scipy.linalg as sla

from stochastic_cov.builders import build_covariance_matrix
from stochastic_cov.cache import HDF5CacheManager
from stochastic_cov.grid import generate_grid

# Robust path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

H5_PATH = str(PROJECT_ROOT / "outputs" / "cache" / "cache_local_tier1.h5")
CSV_PATH = str(PROJECT_ROOT / "outputs" / "data" / "results_sweep.csv")

# Sweep Parameters
PROCESSES = ["rl-fbm", "std-fbm", "stn-fou"]
MATRIX_TYPES = ["path_value", "noise"]
H_RANGE = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_RANGE = np.arange(3, 454, 5)
GRID_TYPE = "uniform_0_1"
INCLUDE_Y0 = True


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def main():
    cache = HDF5CacheManager(H5_PATH)
    total_evals = (
        len(PROCESSES) * len(MATRIX_TYPES) * len(H_RANGE) * len(N_RANGE)
    )
    current_count = 0

    print("=" * 110)
    print(f"STARTING STAGE 1 SWEEP | Total Evaluations: {total_evals}")
    print(f"Initial HDF5 Size: {cache.get_file_size_str()}")
    print("=" * 110)

    for process in PROCESSES:
        for mtype in MATRIX_TYPES:
            for H in H_RANGE:
                for N in N_RANGE:
                    current_count += 1

                    # Set include_t0 to True ONLY for stn-fOU when INCLUDE_Y0 is enabled
                    include_t0 = (process == "stn-fou") and INCLUDE_Y0

                    # 0. Check cache using the computed process-specific boolean
                    if cache.exists(
                        process, mtype, GRID_TYPE, H, N, include_y0=include_t0
                    ):
                        print(
                            f"[{current_count}/{total_evals}] SKIPPED (Cached) | "
                            f"H5 Size: {cache.get_file_size_str()} -> "
                            f"{process} | {mtype} | H={H:.2f} | N={N} | include_t0={include_t0}"
                        )
                        continue

                    start_time = time.time()
                    
                    # 1. Generate grid & matrix
                    grid = generate_grid(N, GRID_TYPE)
                    covmat = build_covariance_matrix(
                        process,
                        mtype,
                        grid,
                        H,
                        include_t0=include_t0,
                    )

                    # 2. Eigendecomposition (Symmetric)
                    evals, evecs = sla.eigh(covmat)

                    compute_time = time.time() - start_time

                    # 3. Save & Flush to HDF5
                    res = cache.save_evaluation(
                        process,
                        mtype,
                        GRID_TYPE,
                        H,
                        N,
                        grid,
                        covmat,
                        evals,
                        evecs,
                        compute_time,
                        include_y0=include_t0,
                    )

                    # 4. Live Progress Report (lmin -> lmax -> frob -> file size)
                    print(
                        f"[{current_count}/{total_evals}] DONE ({format_time(compute_time)}) | "
                        f"Size: {res['file_size']} | "
                        f"Time: {res['timestamp']} | "
                        f"Proc: {process} | Type: {mtype} | H: {H:.2f} | N: {N} | "
                        f"lmin: {res['lambda_min']:.4e} | lmax: {res['lambda_max']:.4f} | "
                        f"Frob: {res['frobenius_norm']:.4f}"
                    )

    # cache.export_summary_csv(CSV_PATH)
    print("=" * 110)
    print(f"STAGE 1 SWEEP COMPLETE! Final HDF5 Size: {cache.get_file_size_str()}")
    print("=" * 110)


if __name__ == "__main__":
    main()