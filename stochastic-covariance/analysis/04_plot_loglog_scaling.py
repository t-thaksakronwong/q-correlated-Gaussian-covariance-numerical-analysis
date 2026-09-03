import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
SWEEP_CSV = "outputs/data/results_sweep.csv"
OUTPUT_DIR = "outputs/analysis_results/figures/loglog_scaling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid size threshold for log-log scaling plots
MAX_N_PLOT = 503
X_FLOOR = 1.0  # Extension limit for dotted conjecture line (N < 3)

# User Settings (Tuned for 3-across paper figure insertion)
PLOT_WIDTH = 5.5110337  # Inches
PLOT_HEIGHT = 5.0       # Inches
DPI = 600

CEX_MAIN = 14
CEX_LAB = 15
CEX_AXIS = 13
CEX_LEGEND = 12

LWD_SOLID = 2
LWD_DOTTED = 2

COLSET_PV = ["#CC00FF", "#0066FF", "#00FF66", "#FF9900", "#FF0000"]
COLSET_NS1 = ["#CC00FF", "orchid", "#0066FF", "darkturquoise", "#00FF66"]
COLSET_NS2 = ["#00FF66", "gold", "#FF9900", "deeppink", "#FF0000"]

# Numeric mapping matching exact output naming schema
GROUP_ID_MAP = {"rl-fbm": 1, "std-fbm": 2, "stn-fou": 3}
CHAR_ID_MAP = {"lambda_min": 1, "lambda_max": 2, "frobenius_norm": 3}
GROUP_LABEL_MAP = {"rl-fbm": "RL-fBM", "std-fbm": "Std-fBM", "stn-fou": "Stn-fOU"}

# ---------------------------------------------------------
# 2. CONJECTURED EXPONENT q(H) & LEGEND HELPER
# ---------------------------------------------------------
def get_exponent(group: str, subtype: str, char: str, H: float) -> float:
    if subtype == "pv":
        if char in ["lambda_max", "frobenius_norm"]:
            return 1.0
        if char == "lambda_min":
            return -2.0 * H

    if subtype == "ns" and group in ["rl-fbm", "std-fbm"]:
        if char == "lambda_min":
            return -1.0 if H < 0.5 else -2.0 * H
        if char == "lambda_max":
            return -2.0 * H if H < 0.5 else -1.0
        if char == "frobenius_norm":
            return (0.5 - 2.0 * H) if H < 0.75 else -1.0

    if subtype == "ns" and group == "stn-fou":
        if char == "lambda_min":
            return -1.0 if H <= 0.5 else -2.0 * H
        if char == "lambda_max":
            return 0.0
        if char == "frobenius_norm":
            return (0.5 - 2.0 * H) if H < 0.25 else 0.0

    raise ValueError(f"Unknown configuration: group={group}, subtype={subtype}, char={char}, H={H}")


def get_legend_position(group: str, subtype: str, char: str, ns_part: int = None) -> str:
    if subtype == "pv" and char in ["lambda_max", "frobenius_norm"]:
        return "upper left"
    if group == "stn-fou" and subtype == "ns" and ns_part == 2 and char == "frobenius_norm":
        return "upper left"
    if group == "stn-fou" and subtype == "ns" and ns_part == 2 and char == "lambda_max":
        return "upper right"
    return "lower left"


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
logging.info(f"Loading sweep data from: {SWEEP_CSV}")
df_raw = pd.read_csv(SWEEP_CSV)

matrix_types = ["path_value", "noise"]
processes = ["rl-fbm", "std-fbm", "stn-fou"]
characteristics = ["lambda_min", "lambda_max", "frobenius_norm"]

total_tasks = len(processes) * len(matrix_types) * len(characteristics)
current_task = 0

logging.info(f"Starting log-log plot generation (N <= {MAX_N_PLOT}). Total tasks: {total_tasks}")

for proc in processes:
    for m_type in matrix_types:
        subtype = "pv" if m_type == "path_value" else "ns"
        group_id = GROUP_ID_MAP[proc]
        
        for char in characteristics:
            current_task += 1
            char_id = CHAR_ID_MAP[char]
            
            logging.info(
                f"[{current_task}/{total_tasks}] Processing Process='{proc}' ({group_id}), "
                f"Type='{subtype}', Char='{char}' ({char_id})"
            )

            # Filter process, matrix_type, AND confine N <= 503
            sub = df_raw[
                (df_raw["process"] == proc) & 
                (df_raw["matrix_type"] == m_type) & 
                (df_raw["N"] <= MAX_N_PLOT)
            ].copy()
            
            if sub.empty:
                logging.warning(f"  -> No data found for {proc}/{m_type}/{char}. Skipping.")
                continue

            # PV CASE -> File name format: pv-plot-{group_id}-{char_id}.png
            if subtype == "pv":
                HH = [0.9, 0.7, 0.5, 0.3, 0.1]
                outname = os.path.join(OUTPUT_DIR, f"pv-plot-{group_id}-{char_id}.png")
                
                plt.close("all")  # Clear figure queue
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)
                
                data_x_min = sub["N"].min()
                data_x_max = sub["N"].max()

                # Track global y bounds for the empirical data in this panel
                data_y_min = float("inf")
                data_y_max = float("-inf")

                for idx, h_val in enumerate(HH):
                    h_sub = sub[np.isclose(sub["H"], h_val)].sort_values("N")
                    NN = h_sub["N"].values
                    YY = h_sub[char].values
                    col = COLSET_PV[idx]

                    if len(YY) > 0:
                        data_y_min = min(data_y_min, YY.min())
                        data_y_max = max(data_y_max, YY.max())

                    # 1. Solid empirical line (starts at NN.min(), e.g., N = 3)
                    ax.loglog(NN, YY, color=col, linewidth=LWD_SOLID, label=f"H={h_val:.1f}")

                    # 2. Dotted conjecture line extended to X_FLOOR (N = 1.0)
                    p = get_exponent(proc, subtype, char, h_val)
                    xx = np.geomspace(X_FLOOR, NN.max() * 1.3, 500)

                    # Baseline log(A) fitted on top 3 grid sizes <= 503
                    log_A = np.mean(np.log(YY[-3:]) - p * np.log(NN[-3:]))
                    A = np.exp(log_A)
                    
                    ax.loglog(xx, A * (xx ** p), color=col, linestyle=":", linewidth=LWD_DOTTED)

                # Set x and y limits explicitly based on solid empirical data bounds
                ax.set_xlim(0.8 * data_x_min, 1.2 * data_x_max)
                if data_y_min < float("inf") and data_y_max > float("-inf"):
                    ax.set_ylim(0.7 * data_y_min, 1.3 * data_y_max)

                ax.set_title(f"{GROUP_LABEL_MAP[proc]} (pv): {char} (log-log)", fontsize=CEX_MAIN)
                ax.set_xlabel("N", fontsize=CEX_LAB)
                ax.set_ylabel(char, fontsize=CEX_LAB)
                ax.tick_params(axis="both", labelsize=CEX_AXIS)
                ax.grid(True, which="both", linestyle=":", alpha=0.4)
                
                leg_loc = get_legend_position(proc, subtype, char)
                ax.legend(loc=leg_loc, fontsize=CEX_LEGEND, framealpha=0.6)
                
                plt.tight_layout()
                overwritten = safe_save_figure(fig, outname, DPI)
                status_msg = "Overwrote existing figure" if overwritten else "Saved figure"
                logging.info(f"  -> {status_msg}: {os.path.basename(outname)}")

            # NS CASE -> File name format: ns-plot-{group_id}-{char_id}-{k+1}.png
            elif subtype == "ns":
                HH_full = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                parts = [HH_full[0:5], HH_full[4:9]]
                colsets = [COLSET_NS1, COLSET_NS2]

                data_x_min = sub["N"].min()
                data_x_max = sub["N"].max()

                for k in range(2):
                    HH = parts[k]
                    col_set = colsets[k]
                    outname = os.path.join(OUTPUT_DIR, f"ns-plot-{group_id}-{char_id}-{k+1}.png")

                    plt.close("all")  # Clear figure queue
                    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)

                    # Track y bounds specific to the current panel subset of H values
                    data_y_min = float("inf")
                    data_y_max = float("-inf")

                    for idx, h_val in enumerate(HH):
                        h_sub = sub[np.isclose(sub["H"], h_val)].sort_values("N")
                        NN = h_sub["N"].values
                        YY = h_sub[char].values
                        col = col_set[idx]

                        if len(YY) > 0:
                            data_y_min = min(data_y_min, YY.min())
                            data_y_max = max(data_y_max, YY.max())

                        # 1. Solid empirical line (starts at NN.min(), e.g., N = 3)
                        ax.loglog(NN, YY, color=col, linewidth=LWD_SOLID, label=f"H={h_val:.1f}")

                        # 2. Dotted conjecture line extended to X_FLOOR (N = 1.0)
                        p = get_exponent(proc, subtype, char, h_val)
                        xx = np.geomspace(X_FLOOR, NN.max() * 1.3, 500)

                        # Frobenius Norm correction at H=0.75 for RL-fBM and Std-fBM (w/ sqrt log factor)
                        if char == "frobenius_norm" and proc in ["rl-fbm", "std-fbm"] and np.isclose(h_val, 0.75):
                            log_A = np.mean(np.log(YY[-3:]) + np.log(NN[-3:]) - 0.5 * np.log(np.log(NN[-3:])))
                            A = np.exp(log_A)
                            order_with_sqrt_log_factor = (xx ** -1.0) * np.sqrt(np.log(xx))
                            yy_conj = A * order_with_sqrt_log_factor
                        else:
                            log_A = np.mean(np.log(YY[-3:]) - p * np.log(NN[-3:]))
                            A = np.exp(log_A)
                            yy_conj = A * (xx ** p)

                        ax.loglog(xx, yy_conj, color=col, linestyle=":", linewidth=LWD_DOTTED)

                    # Set x and y limits explicitly based on solid empirical data bounds
                    ax.set_xlim(0.8 * data_x_min, 1.2 * data_x_max)
                    if data_y_min < float("inf") and data_y_max > float("-inf"):
                        ax.set_ylim(0.7 * data_y_min, 1.3 * data_y_max)

                    ax.set_title(f"{GROUP_LABEL_MAP[proc]} (ns): {char} (log-log)", fontsize=CEX_MAIN)
                    ax.set_xlabel("N", fontsize=CEX_LAB)
                    ax.set_ylabel(char, fontsize=CEX_LAB)
                    ax.tick_params(axis="both", labelsize=CEX_AXIS)
                    ax.grid(True, which="both", linestyle=":", alpha=0.4)

                    leg_loc = get_legend_position(proc, subtype, char, k + 1)
                    ax.legend(loc=leg_loc, fontsize=CEX_LEGEND, framealpha=0.6)

                    plt.tight_layout()
                    overwritten = safe_save_figure(fig, outname, DPI)
                    status_msg = "Overwrote existing figure" if overwritten else "Saved figure"
                    logging.info(f"  -> {status_msg} panel {k+1}/2: {os.path.basename(outname)}")

logging.info(f"Successfully finished generating all figures in: {OUTPUT_DIR}")