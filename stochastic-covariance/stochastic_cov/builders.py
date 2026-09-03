"""Covariance matrix construction logic."""

from typing import Callable
import numpy as np
from stochastic_cov.processes import rl_fbm_cov, std_fbm_cov, stn_fou_cov

from typing import Optional


def build_covariance_matrix(
    process: str,
    matrix_type: str,
    grid: np.ndarray,
    H: float,
    include_t0: Optional[bool] = None,
    lam: float = 1.0,
    sgm: float = 1.0,
) -> np.ndarray:
    if include_t0 is None:
        raise ValueError(
            f"include_t0 must be explicitly set to True or False for process '{process}'."
        )
    """Constructs covariance matrix for path_value or noise configurations.

    Args:
        include_t0:
          - For path_value: Controls whether t_0=0 is included (True, size N+1)
            or omitted starting from t_1 (False, size N). Default is False.
          - For stn-fOU noise: Controls whether Y_0^H is included in row/col 0
            (True) or omitted for pure noise starting at Delta Y_t1 (False).
    """
    N = len(grid) - 1

    # Select kernel
    if process == "rl-fbm":
        kernel = lambda s, t: rl_fbm_cov(s, t, H)
    elif process == "std-fbm":
        kernel = lambda s, t: std_fbm_cov(s, t, H)
    elif process == "stn-fou":
        kernel = lambda s, t: stn_fou_cov(s, t, H, lam=lam, sgm=sgm)
    else:
        raise ValueError(f"Unknown process: {process}")

    if matrix_type == "path_value":
        if include_t0:
            # (N + 1) x (N + 1) matrix starting from t_0 = 0
            covmat = np.zeros((N + 1, N + 1))
            for i in range(N + 1):
                for j in range(i, N + 1):
                    val = kernel(grid[i], grid[j])
                    covmat[i, j] = val
                    covmat[j, i] = val
            return covmat
        else:
            # Pure path values: N x N matrix starting from t_1
            covmat = np.zeros((N, N))
            for i in range(1, N + 1):
                for j in range(i, N + 1):
                    val = kernel(grid[i], grid[j])
                    covmat[i - 1, j - 1] = val
                    covmat[j - 1, i - 1] = val
            return covmat

    elif matrix_type == "noise":
        if process == "stn-fou" and include_t0:
            # (N + 1) x (N + 1) matrix with Y_0^H
            covmat = np.zeros((N + 1, N + 1))
            for i in range(N + 1):
                for j in range(i, N + 1):
                    if i == 0 and j == 0:
                        val = kernel(grid[0], grid[0])
                    elif i == 0 and j > 0:
                        val = kernel(grid[0], grid[j]) - kernel(
                            grid[0], grid[j - 1]
                        )
                    else:
                        val = (
                            kernel(grid[i], grid[j])
                            + kernel(grid[i - 1], grid[j - 1])
                            - kernel(grid[i], grid[j - 1])
                            - kernel(grid[i - 1], grid[j])
                        )
                    covmat[i, j] = val
                    covmat[j, i] = val
            return covmat
        else:
            # Pure noise case: N x N matrix starting from increments Delta Y_{t_1}
            covmat = np.zeros((N, N))
            for i in range(1, N + 1):
                for j in range(i, N + 1):
                    val = (
                        kernel(grid[i], grid[j])
                        + kernel(grid[i - 1], grid[j - 1])
                        - kernel(grid[i], grid[j - 1])
                        - kernel(grid[i - 1], grid[j])
                    )
                    covmat[i - 1, j - 1] = val
                    covmat[j - 1, i - 1] = val
            return covmat
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")