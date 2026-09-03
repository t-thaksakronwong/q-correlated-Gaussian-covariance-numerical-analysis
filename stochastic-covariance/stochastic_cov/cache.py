"""HDF5 caching manager with instant disk flushing and CSV summary export."""

from datetime import datetime
import os
import h5py

import numpy as np
import pandas as pd
from typing import Optional

class HDF5CacheManager:

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    def get_file_size_str(self) -> str:
        """Returns human-readable size of the HDF5 file on disk."""
        if not os.path.exists(self.h5_path):
            return "0 B"
        size_bytes = os.path.getsize(self.h5_path)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def get_group_path(
        self,
        process: str,
        matrix_type: str,
        grid_type: str,
        H: float,
        N: int,
        include_y0: bool = True,
    ) -> str:
        tag = "" if (process != "stn-fou" or include_y0) else "_no_y0"
        return f"processes/{process}/{matrix_type}{tag}/{grid_type}/H_{H:.2f}/N_{N}"

    def exists(
        self,
        process: str,
        matrix_type: str,
        grid_type: str,
        H: float,
        N: int,
        include_y0: Optional[bool] = None,
    ) -> bool:
        if include_y0 is None:
            raise ValueError(
                f"include_y0 must be explicitly set to True or False for process '{process}'."
            )
        path = self.get_group_path(
            process, matrix_type, grid_type, H, N, include_y0
        )
        if not os.path.exists(self.h5_path):
            return False
        with h5py.File(self.h5_path, "a") as f:
            return path in f

    def save_evaluation(
        self,
        process: str,
        matrix_type: str,
        grid_type: str,
        H: float,
        N: int,
        grid: np.ndarray,
        cov_matrix: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        compute_time_sec: float,
        include_y0: Optional[bool] = None,
    ) -> dict:
        if include_y0 is None:
            raise ValueError(
                f"include_y0 must be explicitly set to True or False for process '{process}'."
            )
        path = self.get_group_path(
            process, matrix_type, grid_type, H, N, include_y0
        )
        lam_min = float(np.min(eigenvalues))
        lam_max = float(np.max(eigenvalues))
        frob = float(np.linalg.norm(cov_matrix, "fro"))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with h5py.File(self.h5_path, "a") as f:
            if path in f:
                del f[path]

            grp = f.create_group(path)
            grp.create_dataset("cov_matrix", data=cov_matrix, compression="gzip")
            grp.create_dataset(
                "eigenvalues", data=eigenvalues, compression="gzip"
            )
            grp.create_dataset(
                "eigenvectors", data=eigenvectors, compression="gzip"
            )
            grp.create_dataset("grid", data=grid)

            # Metadata attributes (Ordered: lmin -> lmax -> frob)
            grp.attrs["process"] = process
            grp.attrs["matrix_type"] = matrix_type
            grp.attrs["grid_type"] = grid_type
            grp.attrs["H"] = H
            grp.attrs["N"] = N
            grp.attrs["lambda_min"] = lam_min
            grp.attrs["lambda_max"] = lam_max
            grp.attrs["frobenius_norm"] = frob
            grp.attrs["compute_time_sec"] = compute_time_sec
            grp.attrs["timestamp"] = timestamp
            grp.attrs["include_y0"] = include_y0

            f.flush()

        return {
            "process": process,
            "matrix_type": matrix_type,
            "grid_type": grid_type,
            "H": H,
            "N": N,
            "lambda_min": lam_min,
            "lambda_max": lam_max,
            "frobenius_norm": frob,
            "compute_time_sec": compute_time_sec,
            "timestamp": timestamp,
            "include_y0": include_y0,
            "file_size": self.get_file_size_str(),
        }

    # helper method for outputting CSV (by default, use dedicated script instead)
    def export_summary_csv(self, csv_path: str):
        rows = []
        with h5py.File(self.h5_path, "r") as f:

            def collect_attrs(name, obj):
                if isinstance(obj, h5py.Group) and "lambda_max" in obj.attrs:
                    rows.append(dict(obj.attrs))

            f.visititems(collect_attrs)

        df = pd.DataFrame(rows)
        if not df.empty and "lambda_min" in df.columns:
            # Reorder columns explicitly
            first_cols = [
                "process",
                "matrix_type",
                "grid_type",
                "H",
                "N",
                "lambda_min",
                "lambda_max",
                "frobenius_norm",
            ]
            other_cols = [c for c in df.columns if c not in first_cols]
            df = df[first_cols + other_cols]

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Exported summary CSV with {len(df)} rows to {csv_path}")