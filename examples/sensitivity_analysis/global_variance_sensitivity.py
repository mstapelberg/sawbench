# sobol_global.py
import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from scipy.stats import wasserstein_distance
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# sawbench imports: built-in materials, SAW calculator, and I/O helpers
from sawbench import (
    Material,
    SAWCalculator,
    load_fft_data_from_hdf5,
    extract_experimental_peak_parameters,
)

# ---- MEASURED ANGLES AND EXPERIMENTAL DATA (loaded from HDF5) ----
psis_measured_deg = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])

# Experimental FFT HDF5 (handled as in sensitivity_analysis_workflow)
EXPERIMENTAL_HDF5_PATH = \
    "/home/myless/Documents/saw_freq_analysis/fftData.h5"
EXPERIMENT_MIN_MHZ = 200.0
EXPERIMENT_MAX_MHZ = 400.0

def load_experimental_frequencies_mhz(hdf5_path: str) -> np.ndarray:
    """Return array of dominant experimental peak frequencies in MHz.

    Mirrors the extraction pattern used in examples/ebsd_analysis/sensitivity_analysis_workflow.py
    """
    try:
        fft = load_fft_data_from_hdf5(hdf5_path)
        if not fft:
            return np.array([])
        exp_freq_axis, _, _, exp_amp, (Ny, Nx) = fft
        dominant = []
        for iy in range(Ny):
            for ix in range(Nx):
                amp_trace = exp_amp[:, iy, ix]
                peak_params_all = extract_experimental_peak_parameters(
                    exp_freq_axis, amp_trace, num_peaks_to_extract=3
                )
                valid_mask = ~np.isnan(peak_params_all[:, 0])
                if np.any(valid_mask):
                    actual = peak_params_all[valid_mask, :]
                    strongest_idx = np.argmax(actual[:, 0])
                    mu = actual[strongest_idx, 1]
                    if np.isfinite(mu):
                        dominant.append(mu)
        dom = np.array(dominant)
        if dom.size == 0:
            return np.array([])
        min_hz = EXPERIMENT_MIN_MHZ * 1e6
        max_hz = EXPERIMENT_MAX_MHZ * 1e6
        dom = dom[(dom >= min_hz) & (dom <= max_hz)]
        return dom / 1e6 if dom.size else np.array([])
    except Exception:
        return np.array([])

def predict_saw_frequency(C11_GPa, C12_GPa, C44_GPa, rho_kg_m3, euler_deg, psi_deg, wavelength_um,
                          sampling: int = 400, psaw: int = 0) -> float:
    """Forward model via sawbench; returns predicted peak SAW frequency in MHz."""
    # Convert inputs to expected units
    GPA_TO_PA = 1e9
    C11 = float(C11_GPa) * GPA_TO_PA
    C12 = float(C12_GPa) * GPA_TO_PA
    C44 = float(C44_GPa) * GPA_TO_PA
    wavelength_m = float(wavelength_um) * 1e-6

    # Build material and calculator
    material = Material(
        formula='X', C11=C11, C12=C12, C44=C44, density=float(rho_kg_m3), crystal_class='cubic'
    )
    euler_rad = np.deg2rad(np.array(euler_deg, dtype=float))
    try:
        calc = SAWCalculator(material, euler_rad)
        v_mps, _, _ = calc.get_saw_speed(float(psi_deg), sampling=sampling, psaw=psaw)
        if v_mps is not None and len(v_mps) > 0 and np.isfinite(v_mps[0]):
            f_hz = float(v_mps[0]) / wavelength_m
            return f_hz / 1e6
    except Exception:
        pass
    return np.nan

# Baseline and wavelength
C11_0, C12_0, C44_0 = 231.3, 117.5, 51.7  # GPa
rho_0 = 6100.0  # kg/m^3
euler0_deg = (0.0, 0.0, 0.0)
wavelength_um = 8.80

# SAW calculation controls
SAW_SAMPLING = 4000
SAW_PSAW_FLAG = 0

# ---- INPUT DISTRIBUTIONS (edit to your uncertainties) ----
# Strategy: sample in (K, D=C11-C12, G=C44, rho, alpha, beta, gamma, psi_error)
# Then map to (C11, C12, C44) and enforce cubic stability.
K0  = (C11_0 + 2*C12_0)/3.0
D0  = (C11_0 - C12_0)
G0  = C44_0

# Bounds (uniforms) – replace with your 95% ranges or ±kσ
problem = {
    "num_vars": 8,
    "names": ["K", "D", "G", "rho", "alpha_deg", "beta_deg", "gamma_deg", "psi_err_deg"],
    "bounds": [
        [0.95*K0, 1.05*K0],      # K
        [0.80*D0,  1.20*D0],     # D (controls AZ)
        [0.90*G0,  1.10*G0],     # G = C44
        [0.98*rho_0, 1.02*rho_0],
        [-1.0, 1.0],             # alpha misorientation about baseline [deg]
        [-1.0, 1.0],             # beta
        [-1.0, 1.0],             # gamma
        [-0.5, 0.5],             # in-plane angle error at each psi [deg]
    ],
}

def KD_to_C11C12(K, D):
    C12 = K - D/3.0
    C11 = K + 2.0*D/3.0
    return C11, C12

def stable_cubic(C11, C12, C44):
    return (C11 - C12 > 0.0) and (C44 > 0.0) and (C11 + 2.0*C12 > 0.0)

# ---- OUTPUT METRICS ----
def model_frequency_singlepsi(sample, psi_deg):
    K, D, G, rho, a, b, g, psi_err = sample
    C11, C12 = KD_to_C11C12(K, D)
    if not stable_cubic(C11, C12, G):
        return np.nan
    euler = (euler0_deg[0]+a, euler0_deg[1]+b, euler0_deg[2]+g)
    return predict_saw_frequency(C11, C12, G, rho, euler, psi_deg+psi_err, wavelength_um,
                                 sampling=SAW_SAMPLING, psaw=SAW_PSAW_FLAG)

def model_wasserstein_vs_experiment(sample, experimental_mhz: np.ndarray):
    preds = [model_frequency_singlepsi(sample, psi) for psi in psis_measured_deg]
    preds = np.array([p for p in preds if np.isfinite(p)])
    if preds.size == 0 or experimental_mhz is None or experimental_mhz.size == 0:
        return np.nan
    # 1D distribution distance in MHz
    return float(wasserstein_distance(preds, experimental_mhz))

# ---- SAMPLING & ANALYSIS ----
N_base = 512  # increase to ~2k+ for stable indices
X = sobol_sample.sample(problem, N_base, calc_second_order=False)

# Load experimental distribution from HDF5
exp_freqs_mhz = load_experimental_frequencies_mhz(EXPERIMENTAL_HDF5_PATH)
if exp_freqs_mhz is None or exp_freqs_mhz.size == 0:
    print(f"Warning: No experimental data loaded from {EXPERIMENTAL_HDF5_PATH}. Sensitivity may be uninformative.")

# Choose metric: Wasserstein distance between predicted (across psi) and experimental distribution
# Parallel evaluation with tqdm progress bar
NUM_WORKERS = min(32, cpu_count())

def _eval_one(sample_row):
    return model_wasserstein_vs_experiment(sample_row, exp_freqs_mhz)

Y_list = []
if NUM_WORKERS > 1:
    chunksize = max(1, len(X) // (4 * NUM_WORKERS))
    with Pool(processes=NUM_WORKERS) as pool:
        pbar = tqdm(total=len(X), desc="Evaluating samples", unit="sample")
        for val in pool.imap(_eval_one, X, chunksize=chunksize):
            Y_list.append(val)
            pbar.update(1)
        pbar.close()
else:
    pbar = tqdm(total=len(X), desc="Evaluating samples", unit="sample")
    for row in X:
        Y_list.append(_eval_one(row))
        pbar.update(1)
    pbar.close()

Y = np.array(Y_list, dtype=float)
mask = ~np.isnan(Y)
Si = sobol_analyze.analyze(problem, Y[mask], calc_second_order=False, print_to_console=False)

# Results dataframe
df = pd.DataFrame({
    "parameter": problem["names"],
    "S1": Si["S1"],
    "S1_conf": Si["S1_conf"],
    "ST": Si["ST"],
    "ST_conf": Si["ST_conf"],
}).sort_values("ST", ascending=False)

print(df)
# You can now plot df['S1'] and df['ST'] as bars (Fig 2b).

# ---- OPTIONAL: PCE surrogate pattern (fill later if runtime is high) ----
"""
If the forward model is slow, fit a polynomial-chaos expansion (PCE)
on (K,D,G,rho,alpha,beta,gamma,psi_err) and replace the model calls above
with the surrogate. With PCE, Sobol’ indices follow directly from the
expansion coefficients (Sudret 2008). See, e.g., chaospy or UQ libraries.
"""
