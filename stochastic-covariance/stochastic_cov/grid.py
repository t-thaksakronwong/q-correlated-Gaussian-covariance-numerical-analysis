"""Time grid generation functions."""

import numpy as np


def generate_grid(N: int, grid_type: str = "uniform_0_1") -> np.ndarray:
    """Generates time grids t_0, t_1, ..., t_N based on specified grid_type."""
    if grid_type == "uniform_0_1":
        return np.linspace(0.0, 1.0, N + 1)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")