import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 0. LOGGING CONFIGURATION
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & USER SETTINGS
# ---------------------------------------------------------
PARAMS_CSV = "outputs/analysis_results/tables/fitted_parameters.csv"
OUTPUT_DIR = "outputs/analysis_results/figures/exponent_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# User Settings (Tuned for 3-across paper figure insertion)
PLOT_WIDTH = 5.5110337  # Inches
PLOT_HEIGHT = 5.0       # Inches
DPI = 600

CEX_MAIN = 14
CEX_LAB = 15
CEX_AXIS = 13
CEX_LEGEND = 10

# Distinct color palettes for Path Value (pv) vs Noise (ns)
CHAR_COLORS_BY_SUBTYPE = {
    "pv": {
        "lambda_min": "#0055FF",      # Deep Blue
        "lambda_max": "#00C853",      # Emerald Green
        "frobenius_norm": "#D50000"   # Deep Purple / Magenta
    },
    "ns": {
        "lambda_min": "#AA00FF",      # Crimson Red
        "lambda_max": "#FF6D00",      # Bright Orange
        "frobenius_norm": "#795548"   # Warm Brown / Ochre
    }
}

# Line styles: ensures overlapping curves remain distinguishable
CHAR_LINESTYLES = {
    "lambda_min": "--",           # Dashed
    "lambda_max": "--",          # Dashed
    "frobenius_norm": "-."       # Dash-dotted
}

# Distinct marker styles for empirical scatter points
CHAR_MARKERS = {
    "lambda_min": "s",           # Square
    "lambda_max": "o",           # Circle
    "frobenius_norm": "^"        # Triangle up
}

# Marker sizes
CHAR_MARKER_SIZES = {
    "lambda_min": 40,
    "lambda_max": 45,
    "frobenius_norm": 70
}

# Slight horizontal jitter for scatter points so exact overlapping markers remain distinct
CHAR_X_OFFSETS = {
    "lambda_min": 0.0, # -0.005,
    "lambda_max": 0.0,
    "frobenius_norm": 0.0 # 0.005
}

CHAR_LABELS = {
    "lambda_min": r"$\lambda_{\min}$",
    "lambda_max": r"$\lambda_{\max}$",
    "frobenius_norm": r"$\| \cdot \|_F$"
}

GROUP_LABEL_MAP = {"rl-fbm": "RL-fBM", "std-fbm": "Std-fBM", "stn-fou": "Stn-fOU"}

# ---------------------------------------------------------
# 2. CONJECTURED EXPONENT q(H) & UTILITIES
# ---------------------------------------------------------
def get_conjectured_q(matrix_type: str, process: str, characteristic: str, H: np.ndarray) -> np.ndarray:
    """Computes conjectured power-law exponent vector q(H) over an array of H values."""
    H = np.asarray(H, dtype=float)

    if matrix_type == "path_value":
        if characteristic == "lambda_min":
            return -2.0 * H
        elif characteristic in ["lambda_max", "frobenius_norm"]:
            return np.ones_like(H)

    elif matrix_type == "noise":
        if process in ["rl-fbm", "std-fbm"]:
            if characteristic == "lambda_min":
                return np.where(H < 0.5, -1.0, -2.0 * H)
            elif characteristic == "lambda_max":
                return np.where(H < 0.5, -2.0 * H, -1.0)
            elif characteristic == "frobenius_norm":
                return np.where(H < 0.75, 0.5 - 2.0 * H, -1.0)

        elif process == "stn-fou":
            if characteristic == "lambda_min":
                return np.where(H <= 0.5, -1.0, -2.0 * H)
            elif characteristic == "lambda_max":
                return np.zeros_like(H)
            elif characteristic == "frobenius_norm":
                return np.where(H < 0.25, 0.5 - 2.0 * H, 0.0)

    raise ValueError(f"Unknown configuration: matrix_type={matrix_type}, process={process}, characteristic={characteristic}")


def safe_save_figure(fig, filepath: str, dpi: int) -> bool:
    """Saves figure and reports whether an existing file was overwritten."""
    file_existed = os.path.exists(filepath)
    if file_existed:
        try:
            os.remove(filepath)
        except OSError as e:
            logging.warning(f"  [Warning] Could not remove existing file {filepath}: {e}")
            
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_existed


# ---------------------------------------------------------
# 3. DATA LOADING & EXECUTION
# ---------------------------------------------------------
logging.info(f"Loading fitted parameters from: {PARAMS_CSV}")
df_params = pd.read_csv(PARAMS_CSV)

matrix_types = ["path_value", "noise"]
processes = ["rl-fbm", "std-fbm", "stn-fou"]
characteristics = ["lambda_min", "lambda_max", "frobenius_norm"]

total_tasks = len(processes) * len(matrix_types)
current_task = 0

logging.info(f"Starting exponent comparison plot generation. Total tasks: {total_tasks}")

for proc in processes:
    for m_type in matrix_types:
        current_task += 1
        subtype = "pv" if m_type == "path_value" else "ns"
        outname = os.path.join(OUTPUT_DIR, f"exp-plot-{subtype}-{proc}.png")

        logging.info(f"[{current_task}/{total_tasks}] Generating plot for Process='{proc}', Type='{subtype}'")

        plt.close("all")
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)

        h_dense = np.linspace(0.05, 0.95, 400)
        active_colors = CHAR_COLORS_BY_SUBTYPE[subtype]

        for char in characteristics:
            sub = df_params[
                (df_params["process"] == proc) & 
                (df_params["matrix_type"] == m_type) & 
                (df_params["characteristic"] == char)
            ].sort_values("H")

            if sub.empty:
                logging.warning(f"  -> Missing parameter data for {proc}/{m_type}/{char}")
                continue

            h_vals = sub["H"].values
            p_hat = sub["beta_p"].values
            col = active_colors[char]
            ls = CHAR_LINESTYLES[char]
            marker = CHAR_MARKERS[char]
            msize = CHAR_MARKER_SIZES[char]
            x_offset = CHAR_X_OFFSETS[char]

            # 1. Conjectured line q(H)
            q_dense = get_conjectured_q(m_type, proc, char, h_dense)
            ax.plot(
                h_dense, q_dense, 
                color=col, 
                linestyle=ls, 
                linewidth=2.0, 
                alpha=0.85
            )

            # 2. Empirical points p_hat
            ax.scatter(
                h_vals + x_offset, p_hat, 
                color=col, 
                marker=marker,
                s=msize, 
                alpha=0.7, 
                edgecolor="black", 
                linewidth=0.5, 
                zorder=4
            )

        # ---------------------------------------------------------
        # Interleaved Custom Legend Handles (Estimate then Conjecture)
        # ---------------------------------------------------------
        legend_handles = []

        for char in characteristics:
            col = active_colors[char]
            marker = CHAR_MARKERS[char]
            msize = np.sqrt(CHAR_MARKER_SIZES[char])  # Convert area size to point size for Line2D
            ls = CHAR_LINESTYLES[char]

            # Pair 1: Characteristic Estimate
            estimate_handle = Line2D(
                [0], [0],
                color='none',
                marker=marker,
                markerfacecolor=col,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=msize,
                alpha=0.7,
                label=f"{CHAR_LABELS[char]} estimate"
            )
            legend_handles.append(estimate_handle)

            # Pair 2: Characteristic Conjecture
            conjecture_handle = Line2D(
                [0], [0],
                color=col,
                linestyle=ls,
                linewidth=2.0,
                alpha=0.85,
                label=f"{CHAR_LABELS[char]} conjecture"
            )
            legend_handles.append(conjecture_handle)

        # Set plot layout, labels, and titles
        ax.set_title(f"{GROUP_LABEL_MAP[proc]} ({subtype}): Estimated exponents", fontsize=CEX_MAIN)
        ax.set_xlabel("H", fontsize=CEX_LAB)
        ax.set_ylabel(r"Exponents (Estimate vs Conjecture)", fontsize=CEX_LAB)
        ax.tick_params(axis="both", labelsize=CEX_AXIS)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)

        # Legend position: bottom left for PV, top right for NS, center right (with a little adjustment) only for stn-fOU NS        
        if subtype == "pv":
            ax.legend(handles=legend_handles, loc="lower left", fontsize=CEX_LEGEND, framealpha=0.7)
        elif proc == "stn-fou" and subtype == "ns":
            # Shifts legend slightly upward above the exact middle-right position
            ax.legend(
                handles=legend_handles, 
                loc="center right", 
                bbox_to_anchor=(1.0, 0.58), 
                fontsize=CEX_LEGEND, 
                framealpha=0.7
            )
        else:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=CEX_LEGEND, framealpha=0.7)

        plt.tight_layout()
        overwritten = safe_save_figure(fig, outname, DPI)
        status_msg = "Overwrote existing figure" if overwritten else "Saved figure"
        logging.info(f"  -> {status_msg}: {os.path.basename(outname)}")

logging.info(f"Successfully finished generating all figures in: {OUTPUT_DIR}")