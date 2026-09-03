import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & USER SETTINGS
# ---------------------------------------------------------
OUTPUT_DIR = "outputs/analysis_results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# User Settings (Tuned for standalone figure display)
PLOT_WIDTH = 5.5110337  # Inches
PLOT_HEIGHT = 5.0       # Inches
DPI = 600

CEX_MAIN = 12
CEX_LAB = 12
CEX_AXIS = 11
CEX_LEGEND = 11

LWD_SOLID = 2.5
LWD_DOTTED = 2.5

# Colors matching peer-reviewed R script
PV_COL = "#00FF00"   # green1
NS_COL = "#FB9A99"   # fBM ns
NS_COL2 = "#FFD700"  # gold1 (fOU ns)

# ---------------------------------------------------------
# 2. DEFINE COST COMPLEXITY FUNCTIONS
# ---------------------------------------------------------
HH = np.linspace(0.0, 1.0, 500)

def pv_cost(H):
    return 1.5 + 3.0 * H

# --- fBM ---
def ns_fBM_k(H):
    return np.abs(1.0 - 2.0 * H)

def ns_fBM_F(H_arr):
    return np.array([min(max(0.0, 1.5 - 2.0 * h), 0.5) for h in H_arr])

def ns_fBM_cost1(H_arr):
    return 1.0 + 1.5 * ns_fBM_k(H_arr) + ns_fBM_F(H_arr)

# --- fOU ---
def ns_fOU_k(H_arr):
    return np.array([max(1.0, 2.0 * h) for h in H_arr])

def ns_fOU_F(H_arr):
    return np.array([max(0.0, 0.5 - 2.0 * h) for h in H_arr])

def ns_fOU_cost1(H_arr):
    return 1.0 + 1.5 * ns_fOU_k(H_arr) + ns_fOU_F(H_arr)

# ---------------------------------------------------------
# 3. GENERATE PLOT
# ---------------------------------------------------------
cost_pv = pv_cost(HH)
cost_fbm = ns_fBM_cost1(HH)
cost_fou = ns_fOU_cost1(HH)

ymin = min(cost_pv.min(), cost_fbm.min(), cost_fou.min())
ymax = max(cost_pv.max(), cost_fbm.max(), cost_fou.max())

fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)

# Main Cost Lines
ax.plot(HH, cost_fbm, color=NS_COL, linewidth=LWD_SOLID, label="fBM & fOU by pv")
ax.plot(HH, cost_pv, color=PV_COL, linewidth=LWD_SOLID, label="fBM by ns")
ax.plot(HH, cost_fou, color=NS_COL2, linewidth=LWD_SOLID, label="fOU by ns")

# Grid & Reference Dotted Lines
ax.grid(True, linestyle="-", alpha=0.3)
ax.axhline(y=2.0, color="black", linestyle=":", linewidth=LWD_DOTTED, alpha=0.4)
ax.axhline(y=3.0, color="black", linestyle=":", linewidth=LWD_DOTTED, alpha=0.4)

# Labels and Styling
ax.set_ylim(ymin - 0.1, ymax + 0.1)
ax.set_xlabel("H", fontsize=CEX_LAB)
ax.set_ylabel("power of N", fontsize=CEX_LAB)
ax.set_title("Overall cost dependence on N", fontsize=CEX_MAIN)
ax.tick_params(axis="both", labelsize=CEX_AXIS)

# Legend matching exact text from original code
ax.legend(
    handles=[
        plt.Line2D([0], [0], color=PV_COL, lw=LWD_SOLID, label="fBM & fOU by pv"),
        plt.Line2D([0], [0], color=NS_COL, lw=LWD_SOLID, label="fBM by ns"),
        plt.Line2D([0], [0], color=NS_COL2, lw=LWD_SOLID, label="fOU by ns")
    ],
    loc="upper left",
    fontsize=CEX_LEGEND,
    framealpha=0.6
)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "compare-plot-fBM-fOU.png")
plt.savefig(out_path)
plt.close()

print(f"Cost comparison plot successfully generated: {out_path}")