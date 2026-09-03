import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & LOAD DATA
# ---------------------------------------------------------
SWEEP_CSV = "outputs/data/results_sweep.csv"
PARAMS_CSV = "outputs/analysis_results/tables/fitted_parameters.csv"
OUTPUT_TABLE = "outputs/analysis_results/tables/out_of_sample_combined.csv"

os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)

df_sweep_raw = pd.read_csv(SWEEP_CSV)
df_params = pd.read_csv(PARAMS_CSV)

# Filter for out-of-sample grid sizes
df_oos = df_sweep_raw[df_sweep_raw["N"].isin([800, 1600])].copy()

df_oos_melt = pd.melt(
    df_oos,
    id_vars=["process", "matrix_type", "grid_type", "include_y0", "H", "N"],
    value_vars=["lambda_min", "lambda_max", "frobenius_norm"],
    var_name="characteristic",
    value_name="empirical_y_N"
)

matrix_order = ["path_value", "noise"]
process_order = ["rl-fbm", "std-fbm", "stn-fou"]
characteristic_order = ["lambda_min", "lambda_max", "frobenius_norm"]

df_merged = pd.merge(
    df_oos_melt,
    df_params[["process", "matrix_type", "characteristic", "H", "alpha", "beta_p", "gamma", "theta", "delta"]],
    on=["process", "matrix_type", "characteristic", "H"],
    how="inner"
)


# ---------------------------------------------------------
# 2. COMPUTATION & RELATIVE ERROR EVALUATION
# ---------------------------------------------------------
results_list = []
grouped = df_merged.groupby(["matrix_type", "process", "characteristic", "H"], sort=False, observed=False)

for (matrix_type, process, characteristic, h_val), group_df in grouped:
    if len(group_df) < 2:
        continue

    row_800 = group_df[group_df["N"] == 800].iloc[0]
    row_1600 = group_df[group_df["N"] == 1600].iloc[0]

    y_800_emp, y_1600_emp = row_800["empirical_y_N"], row_1600["empirical_y_N"]

    alpha, p, gamma, theta, delta = (
        row_800["alpha"], row_800["beta_p"], row_800["gamma"], row_800["theta"], row_800["delta"]
    )

    # Sub-leading exponent (p' = beta_p + delta)
    sub_lead_exp = p + delta

    # 1. Pointwise Predictions & Relative Errors
    hat_y_800 = np.exp(alpha + p * np.log(800.0) + gamma * (800.0 ** (-np.exp(theta))))
    hat_y_1600 = np.exp(alpha + p * np.log(1600.0) + gamma * (1600.0 ** (-np.exp(theta))))

    rel_err_800 = np.abs(y_800_emp - hat_y_800) / y_800_emp
    rel_err_1600 = np.abs(y_1600_emp - hat_y_1600) / y_1600_emp

    # 2. Empirical & Predicted Scaling Ratios on log2 Scale
    log2_R_emp = np.log2(y_1600_emp / y_800_emp)
    log2_R_pred = np.log2(hat_y_1600 / hat_y_800)
    exp_diff = np.abs(log2_R_emp - log2_R_pred)

    # Console readout
    print(
        f"[{process:<7} | {matrix_type:<10} | {characteristic:<14}] H={h_val:.2f}\n"
        f"  ├─ Exponents        : Fitted β_p={p:.6f} | Sub-leading={sub_lead_exp:.6f}\n"
        f"  ├─ Pointwise RelErr : RelErr(N=800)={rel_err_800:.6f} | RelErr(N=1600)={rel_err_1600:.6f}\n"
        f"  └─ log2 Scaling R   : Empirical={log2_R_emp:.6f} | Predicted={log2_R_pred:.6f} | Abs Diff={exp_diff:.6f}\n"
    )

    results_list.append({
        "process": process,
        "matrix_type": matrix_type,
        "characteristic": characteristic,
        "H": h_val,
        # Exponents
        "beta_p": p,
        "sub_lead_exp": sub_lead_exp,
        # Level metrics
        "y_800_emp": y_800_emp,
        "hat_y_800": hat_y_800,
        "rel_err_800": rel_err_800,
        "y_1600_emp": y_1600_emp,
        "hat_y_1600": hat_y_1600,
        "rel_err_1600": rel_err_1600,
        # Log2 Scaling Metrics
        "log2_R_emp": log2_R_emp,
        "log2_R_pred": log2_R_pred,
        "exp_diff": exp_diff
    })

# ---------------------------------------------------------
# 3. EXPORT COMBINED TABLE
# ---------------------------------------------------------
df_export = pd.DataFrame(results_list)

df_export["matrix_type"] = pd.Categorical(df_export["matrix_type"], categories=matrix_order, ordered=True)
df_export["process"] = pd.Categorical(df_export["process"], categories=process_order, ordered=True)
df_export["characteristic"] = pd.Categorical(df_export["characteristic"], categories=characteristic_order, ordered=True)

df_export = df_export.sort_values(by=["matrix_type", "process", "characteristic", "H"]).reset_index(drop=True)
df_export.to_csv(OUTPUT_TABLE, index=False)

print(f"\nSuccessfully written error analysis table to: {OUTPUT_TABLE}")