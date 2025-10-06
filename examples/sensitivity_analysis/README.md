## Sensitivity Analysis Scripts

This folder contains scripts for analyzing SAW frequency sensitivity to model inputs, comparing predictions to experimental data, and visualizing results.

- `global_variance_sensitivity.py`
  - Purpose: Global variance-based sensitivity analysis using Sobol indices.
  - Highlights:
    - Uses `sawbench` `Material` and `SAWCalculator` as the forward model.
    - Loads experimental FFT signals from HDF5 (path configurable in the script) and extracts dominant peaks.
    - Metric is Wasserstein distance between predicted and experimental frequency distributions across measured in-plane angles.
    - Sampling with `SALib.sample.sobol`; parallel evaluation with `multiprocessing` and progress via `tqdm`.
  - Output: Prints a sorted DataFrame of `S1`, `S1_conf`, `ST`, `ST_conf`. Save it to CSV if needed.

- `derivative_sensitivity.py`
  - Purpose: Local (finite-difference) log-sensitivity of SAW frequency to elastic constants, density, Euler angles, and in-plane angle.
  - Highlights:
    - Uses `sawbench` to evaluate the forward model at baseline and perturbed states.
    - Parallelizes across angles with tqdm-based helpers.
  - Output: JSON of sensitivities and plots for median absolute sensitivities and angle/Euler slopes.

- `ebsd_vs_experiment_alignment.py`
  - Purpose: Compare per-angle predicted SAW distributions from EBSD to experimental distributions; identify best-aligned angle by KS or W1.
  - Highlights:
    - Loads EBSD via `sawbench.load_ebsd_map` and computes per-grain SAW frequencies.
    - Loads experimental FFT data and extracts dominant peaks.
  - Output: CSV table of distances vs angle and a small plot.

- `plot_sensitivity_bars.py`
  - Purpose: Plot horizontal bar chart from a Sobol results CSV.
  - Usage:
    - `python plot_sensitivity_bars.py results/sensitivity_analysis.csv --out results/sensitivity_analysis_bars.png`
  - Behavior:
    - Sorts parameters by `ST` descending.
    - Draws `ST` as a faint outline bar behind each parameter.
    - Draws `S1` as a solid bar with `± S1_conf` error caps.

### Notes
- Ensure `tqdm` and `SALib` are installed.
- For experimental FFT handling, adjust the HDF5 path and frequency window in scripts as needed.

