# Numerical experiments on the power-law scaling of spectral characteristics of RL-fBMs, std-fBMs, and stn-fOU processes

This repository contains the source code for the covariance matrix generation, fitting analysis, and figure production for arXiv:2604.22463. 

---

## Folder Structure

```text
stochastic-covariance/
├── pyproject.toml           # Project configuration and dependencies
├── stochastic_cov/          # Core underlying computation package
│   ├── __init__.py
│   ├── processes.py
│   ├── grid.py
│   ├── builders.py
│   └── cache.py
├── scripts/                 # Data generation & exportation to CSV 
│   ├── 01_run_data_sweep.py #    - Creates the raw HDF5 dataset
│   └── 02_export_csv.py      #    - Converts HDF5 output to CSV format
├── analysis/                #  Statistical fitting & analysis
│   ├── 01_fit_power_laws.py
│   ├── 02_out_of_sample_ratio.py
│   ├── 03_plot_exponent_comparisons.py
│   ├── 04_plot_loglog_scaling.py
│   └── 05_plot_deduced_complexity.py
└── outputs/                 # Storage for data pipelines and final results
    ├── cache/               #    - Stores the generated HDF5 files
    ├── data/                #    - Stores the converted CSV files
    └── analysis_results/    #    - Storage for analysis results	
        ├── tables/          #    - Fit and error metric tables
        └── figures/         #    - Plots (Fit vs Conjecture / Asymptotic convergence visualisation)
```

---

## Installation & Setup

Run the following command at the root directory (`stochastic-covariance/`) to install the required packages and their dependencies:

```bash
pip install -e .
```

---

## Execution & Replication Workflow

To fully replicate the numerical experiments and findings in the paper, navigate to the root folder (`stochastic-covariance/`) and run the programs sequentially according to the stages below.

To modify simulation settings, see the "Advanced experimental settings" section below.

### Stage 1: Data Generation
First, generate the covariance matrix, perform eigendecomposition, and store the related data, which saves the HDF5 files into the `outputs/cache/` directory. The use of HDF5 allows the program to skip previously executed experiment settings. If new configurations contain duplicates, the program will only calculate values for cases without existing data.

[Optional]
If you want to save the covariance matrix itself and its eigenvectors for future analysis and reference, see the "Advanced experimental settings" section below. This will require approximately 14 GB of disk space under the default setting of N=3,8,13,...,503 plus the additional N=800,1600 for out-of-sample testing.

```bash
# 1. Run the massive data sweep to generate HDF5 data
python scripts/01_run_data_sweep.py
```

### Stage 2: Export to CSV for ease of lookup and analysis
Second, convert the generated HDF5 files into clean CSV format inside `outputs/data/`:

```bash
# 2. Export the HDF5 data into an accessible CSV format
python scripts/02_export_csv.py
```

### Stage 3: Statistical Analysis & Figure Generation
Once the CSV data is successfully exported, run the analysis scripts in chronological order to perform the non-linear regression, generate tables, and plots:

```bash
# 3. Fit the non-linear regression model to the generated data to analyse the power-law exponent
python analysis/01_fit_power_laws.py

# 4. Calculate out-of-sample error metrics
python analysis/02_out_of_sample_ratio.py

# 5. Generate plots as appearing in the paper
python analysis/03_plot_exponent_comparisons.py
python analysis/04_plot_loglog_scaling.py
python analysis/05_plot_deduced_complexity.py
```

All final statistical tables and visualisation plots will be generated inside the `outputs/analysis_results/` folders.

---

##  Advanced experimental settings

### 1. To save the covariance matrix and its eigenvectors for future analysis and reference
Open `stochastic_cov/cache.py`. Remove the '#' mark from line 92 (to save the covariance matrix), and lines 96-98 (to save the eigenvectors) as necessary.

### 2. Create custom grid settings.
Open `stochastic_cov/grid.py`. Add grid name and definition as desired. 
Then, change line 28 of `scripts/01_run_data_sweep.py` to that grid name before running all the program sequence.

### 3. Add/modify covariance structures.
First, define the new covariance structure or modify the existing one in `stochastic_cov/processes.py`. 
Second, if it is a new covariance structure, make sure to import it into `stochastic_cov/builders.py` at line 5, and include the covariance function name into lines 34-42 of the script.
Finally, change line 21 (definition of PROCESSES) of `scripts/01_run_data_sweep.py` to include the newly defined covariance function name before running all the program sequence.

### 4. Experiments on different ranges of N
Modify the range of N in lines 24-27 of `scripts/01_run_data_sweep.py` to generate data for your new grid setup. 
Ensure you update the corresponding range of N in the downstream CSV export and analysis scripts accordingly.
