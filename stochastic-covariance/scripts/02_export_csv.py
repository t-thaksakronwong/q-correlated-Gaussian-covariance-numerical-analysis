"""Standalone script to export and sort summary CSV from HDF5 cache."""

import os
from pathlib import Path
import h5py
import pandas as pd

import sys

# Robust path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

H5_PATH = str(PROJECT_ROOT / "outputs" / "cache" / "cache_local_tier1.h5")
CSV_PATH = str(PROJECT_ROOT / "outputs" / "data" / "results_sweep.csv")


def export_hdf5_to_csv(h5_path: str, csv_path: str):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 cache file not found at: {h5_path}")

    rows = []
    with h5py.File(h5_path, "r") as f:

        def collect_attrs(name, obj):
            if isinstance(obj, h5py.Group) and "lambda_max" in obj.attrs:
                rows.append(dict(obj.attrs))

        f.visititems(collect_attrs)

    df = pd.DataFrame(rows)

    if df.empty:
        print("Warning: HDF5 file contains no evaluation groups.")
        return

    # -------------------------------------------------------------
    # 1. Custom Categorical Sorting:
    #    Process: rl-fbm > std-fbm > stn-fou
    #    Matrix Type: path_value > noise
    #    H: Increasing
    #    N: Increasing
    # -------------------------------------------------------------
    process_order = ["rl-fbm", "std-fbm", "stn-fou"]
    matrix_type_order = ["path_value", "noise"]

    df["process"] = pd.Categorical(
        df["process"], categories=process_order, ordered=True
    )
    df["matrix_type"] = pd.Categorical(
        df["matrix_type"], categories=matrix_type_order, ordered=True
    )

    df = df.sort_values(
        by=["process", "matrix_type", "H", "N"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # 2. Column Reordering
    # -------------------------------------------------------------
    first_cols = [
        "process",
        "matrix_type",
        "grid_type",
        "include_y0",
        "H",
        "N",
        "lambda_min",
        "lambda_max",
        "frobenius_norm",
        "compute_time_sec",
    ]
    
    # Filter for columns that actually exist in the DataFrame
    first_cols = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    
    df = df[first_cols + other_cols]

    # -------------------------------------------------------------
    # 3. Export
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        response = input(f"File '{csv_path}' already exists. Overwrite? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Export cancelled by user.")
            sys.exit(0)

    df.to_csv(csv_path, index=False)
    print(f" Successfully exported {len(df)} sorted records to: {csv_path}")


if __name__ == "__main__":
    export_hdf5_to_csv(H5_PATH, CSV_PATH)