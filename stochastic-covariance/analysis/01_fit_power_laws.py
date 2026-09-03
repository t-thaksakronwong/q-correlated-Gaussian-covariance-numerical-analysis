import os
import sys
import numpy as np
import pandas as pd
import scipy.optimize as opt

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & LOAD DATA
# ---------------------------------------------------------
INPUT_CSV = "outputs/data/results_sweep.csv"
OUTPUT_TABLE = "outputs/analysis_results/tables/fitted_parameters.csv"

os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)

print(f"Loading master sweep data from: {INPUT_CSV}")
sys.stdout.flush()

df_raw = pd.read_csv(INPUT_CSV)

# Filter for in-sample asymptotic regime (103 <= N <= 503)
df_in_sample = df_raw[(df_raw["N"] >= 203) & (df_raw["N"] <= 503)].copy()

# Melt wide table to long format
df_fit = pd.melt(
    df_in_sample,
    id_vars=["process", "matrix_type", "grid_type", "include_y0", "H", "N"],
    value_vars=["lambda_min", "lambda_max", "frobenius_norm"],
    var_name="characteristic",
    value_name="value"
)

# ---------------------------------------------------------
# 2. DEFINE EXPLICIT ORDERING CATEGORIES
# ---------------------------------------------------------
matrix_order = ["path_value", "noise"]
process_order = ["rl-fbm", "std-fbm", "stn-fou"]
characteristic_order = ["lambda_min", "lambda_max", "frobenius_norm"]

df_fit["matrix_type"] = pd.Categorical(df_fit["matrix_type"], categories=matrix_order, ordered=True)
df_fit["process"] = pd.Categorical(df_fit["process"], categories=process_order, ordered=True)
df_fit["characteristic"] = pd.Categorical(df_fit["characteristic"], categories=characteristic_order, ordered=True)

# Sort working dataframe prior to grouping
df_fit = df_fit.sort_values(by=["matrix_type", "process", "characteristic", "H", "N"]).reset_index(drop=True)

# ---------------------------------------------------------
# 3. MODEL DEFINITION
# ---------------------------------------------------------
def transformed_model(N, alpha, p, gamma, theta):
    """
    Model: log y_N = alpha + p * log(N) + gamma * N^(-exp(theta))
    Note: delta = -exp(theta) guarantees negative decay exponent delta < 0
    """
    return alpha + p * np.log(N) + gamma * (N ** (-np.exp(theta)))


# ---------------------------------------------------------
# 4. FITTING PIPELINE WITH CUSTOM EXECUTION ORDER
# ---------------------------------------------------------
results_list = []

# Group using explicit sequence: matrix_type -> process -> characteristic
group_cols = ["matrix_type", "process", "characteristic"]
grouped = df_fit.groupby(group_cols, sort=False, observed=False)

total_groups = len(grouped)
print(f"Melted {len(df_fit)} evaluation records (103 <= N <= 503).")
print(f"Starting Levenberg-Marquardt fits across {total_groups} group combinations...\n")
sys.stdout.flush()

fit_counter = 0

for group_idx, ((matrix_type, process, characteristic), group_df) in enumerate(grouped, start=1):
    h_values = sorted(group_df["H"].unique())
    print(f"[{group_idx}/{total_groups}] Processing: {process} | {matrix_type} | {characteristic} ({len(h_values)} H values)")
    sys.stdout.flush()

    for h_val in h_values:
        fit_counter += 1
        sub_df = group_df[group_df["H"] == h_val].sort_values("N")
        
        N_data = sub_df["N"].to_numpy(dtype=float)
        y_data = sub_df["value"].to_numpy(dtype=float)
        log_y_data = np.log(y_data)
        
        # Initial guess strategy
        p_guess = 0.0
        initial_guess = [0.0, p_guess, 0.0, float(np.log(1.0))]
        initial_guess_str = f"[{initial_guess[0]:.2f}, {initial_guess[1]:.2f}, {initial_guess[2]:.2f}, {initial_guess[3]:.2f}]"
        
        try:
            params, _ = opt.curve_fit(
                transformed_model,
                N_data,
                log_y_data,
                p0=initial_guess,
                method='lm',
                maxfev=100000
            )
            
            alpha_fit, p_fit, gamma_fit, theta_fit = params
            delta_fit = -np.exp(theta_fit)
            
            # Residual & Metric Computations
            log_y_pred = transformed_model(N_data, alpha_fit, p_fit, gamma_fit, theta_fit)
            residuals = log_y_data - log_y_pred
            
            rss = float(np.sum(residuals ** 2))
            M = len(N_data)                    
            rmse = float(np.sqrt(rss / M))            
            max_err = float(np.max(np.abs(residuals)))
            
            # Variance metrics
            tss = np.sum((log_y_data - np.mean(log_y_data)) ** 2)
            
            if tss > 0:
                unexplained_var = float(rss / tss)  # 1 - R^2 (preserves high precision)
                # r2_log = 1.0 - unexplained_var
                
                # Adjusted R^2 for degrees of freedom (M observations, k parameters)
                # k = 4  # Number of fitted parameters: alpha, p, gamma, theta
                # if M > k + 1:
                #     r2_adj_log = 1.0 - ((1.0 - r2_log) * (M - 1) / (M - k - 1))
                # else:
                #     r2_adj_log = r2_log
            else:
                unexplained_var = 0.0
                # r2_log = 1.0
                # r2_adj_log = 1.0
            
            # res_mean = np.mean(residuals)
            # var_res = float(np.sum((residuals - res_mean) ** 2) / M)

            # Print with high precision in real-time logs including alpha and gamma
            print(
                f"  -> H={h_val:.2f} | alpha={alpha_fit:8.4f} | p={p_fit:8.4f} | "
                f"gamma={gamma_fit:8.4f} | delta={delta_fit:7.4f} | "
                f"RMSE={rmse:.3e} | MaxErr={max_err:.3e} | UnexplainedVar={unexplained_var:.3e}"
            )
            sys.stdout.flush()
            
            results_list.append({
                "process": process,
                "matrix_type": matrix_type,
                "characteristic": characteristic,
                "H": h_val,
                "initial_guess": initial_guess_str,
                "alpha": alpha_fit,
                "beta_p": p_fit,
                "gamma": gamma_fit,
                "delta": delta_fit,
                "theta": theta_fit,
                # "rss": rss,
                "rmse": rmse,
                # "var_res": var_res,
                "max_err": max_err,
                "unexplained_var": unexplained_var, # 1 - R2
                # "r2_log": r2_log,
                # "r2_adj_log": r2_adj_log
            })
            
        except Exception as e:
            print(f"  [ERROR] Fit failed for H={h_val:.2f}: {e}")
            sys.stdout.flush()

# ---------------------------------------------------------
# 5. EXPORT SORTED TABLE
# ---------------------------------------------------------
df_results = pd.DataFrame(results_list)

# Apply categories again to ensure exported CSV strictly matches your custom sequence
df_results["matrix_type"] = pd.Categorical(df_results["matrix_type"], categories=matrix_order, ordered=True)
df_results["process"] = pd.Categorical(df_results["process"], categories=process_order, ordered=True)
df_results["characteristic"] = pd.Categorical(df_results["characteristic"], categories=characteristic_order, ordered=True)

df_results = df_results.sort_values(
    by=["matrix_type", "process", "characteristic", "H"]
).reset_index(drop=True)

df_results.to_csv(OUTPUT_TABLE, index=False)

print(f"\n==================================================")
print(f"SUCCESS: Fitted parameters exported to {OUTPUT_TABLE}")
print(f"Total successful fits: {len(df_results)} / {fit_counter}")
print(f"==================================================")