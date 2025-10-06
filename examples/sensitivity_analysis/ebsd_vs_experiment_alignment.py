"""
Simple EBSD vs Experimental alignment analysis.

What it does (minimal):
- Loads experimental SAW peak frequencies from HDF5.
- Loads EBSD map and predicts per-grain SAW frequencies for each angle ψ.
- Computes KS and Wasserstein-1 distances between predicted and experimental distributions vs ψ.
- Prints best matching angle by KS and by W1, saves CSV and a small plot.

Adjust parameters at the top of this script. No CLI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance

# Set global font and font sizes - use Helvetica and increase all by 2 points
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Robust import of sawbench (installed package or local src)
try:
    from sawbench import (
        load_ebsd_map,
        calculate_saw_frequencies_for_ebsd_grains,
        extract_experimental_peak_parameters,
        load_fft_data_from_hdf5,
        Material,
    )
except ImportError:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    from sawbench import (
        load_ebsd_map,
        calculate_saw_frequencies_for_ebsd_grains,
        extract_experimental_peak_parameters,
        load_fft_data_from_hdf5,
        Material,
    )

# ---------- USER PARAMETERS (edit here) ----------
EBSD_DATA_PATH: str = "/home/myless/Packages/sawbench/examples/data/V-1_2Ti_EBSD_Map"
EBSD_DATA_TYPE: str = "OxfordText"
EBSD_BOUNDARY_DEF_DEG: float = 5.0
EBSD_MIN_GRAIN_PX: int = 10

# Experimental HDF5 path and optional freq window (MHz)
EXPERIMENTAL_HDF5_PATH: Optional[str] = \
    "/home/myless/Documents/saw_freq_analysis/fftData.h5"
EXP_MIN_MHZ: Optional[float] = 200.0
EXP_MAX_MHZ: Optional[float] = 400.0

# Material properties (GPa for elastic constants; density kg/m^3)
MATERIAL_PROPS_GPA: Dict[str, float] = {
    'formula': 'V',
    'C11': 229.0,
    'C12': 119.0,
    'C44': 43.0,
    'density': 6110.0,
    'crystal_class': 'cubic',
}

# Wavelength
WAVELENGTH_UM: float = 8.8

# Angle scan
ANGLES_TO_TEST_DEG = np.arange(0, 180, 2)  # 0..178

# SAW solver settings
SAW_SAMPLING: int = 400
SAW_PSAW: int = 0
NUM_SAW_WORKERS: int = 16

# Output
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


# ---------- HELPERS ----------
def load_experimental_freqs_mhz(hdf5_path: str,
                                min_mhz: Optional[float],
                                max_mhz: Optional[float]) -> np.ndarray:
    """Load experimental FFT data and extract dominant peak per pixel (MHz)."""
    if not hdf5_path:
        return np.array([])
    out = load_fft_data_from_hdf5(hdf5_path)
    if not out:
        return np.array([])
    exp_freq_axis, _, _, exp_amplitude, (Ny, Nx) = out
    dom = []
    for iy in range(Ny):
        for ix in range(Nx):
            amp_trace = exp_amplitude[:, iy, ix]
            peak_params = extract_experimental_peak_parameters(exp_freq_axis, amp_trace, num_peaks_to_extract=3)
            valid = ~np.isnan(peak_params[:, 0])
            if np.any(valid):
                actual = peak_params[valid, :]
                strongest = int(np.argmax(actual[:, 0]))
                mu_hz = actual[strongest, 1]
                if np.isfinite(mu_hz):
                    dom.append(mu_hz)
    if not dom:
        return np.array([])
    dom = np.array(dom)
    if min_mhz is not None and max_mhz is not None:
        mask = (dom >= min_mhz * 1e6) & (dom <= max_mhz * 1e6)
        dom = dom[mask]
    return dom / 1e6


def predict_freqs_mhz_for_angle(ebsd_map_obj, material_props_pa: Dict[str, float],
                                wavelength_m: float, angle_deg: float) -> np.ndarray:
    df = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map_obj,
        material=Material(**material_props_pa),
        wavelength=wavelength_m,
        saw_calc_angle_deg=float(angle_deg),
        saw_calc_sampling=SAW_SAMPLING,
        saw_calc_psaw=SAW_PSAW,
        num_workers=NUM_SAW_WORKERS
    )
    if df.empty or 'Peak SAW Frequency (Hz)' not in df.columns:
        return np.array([])
    arr_hz = df['Peak SAW Frequency (Hz)'].dropna().to_numpy()
    return arr_hz / 1e6


def compute_distances(exp_mhz: np.ndarray, pred_mhz: np.ndarray) -> Dict[str, float]:
    exp = exp_mhz[np.isfinite(exp_mhz)]
    pred = pred_mhz[np.isfinite(pred_mhz)]
    if exp.size == 0 or pred.size == 0:
        return {"ks": np.nan, "w1": np.nan}
    ks_stat, _ = ks_2samp(exp, pred)
    w1 = float(wasserstein_distance(exp, pred))
    return {"ks": float(ks_stat), "w1": w1}


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert material constants to Pa
    props_pa = MATERIAL_PROPS_GPA.copy()
    for k in ["C11", "C12", "C44"]:
        props_pa[k] = float(props_pa[k]) * 1e9

    wavelength_m = WAVELENGTH_UM * 1e-6

    # Load EBSD
    print("--- Loading EBSD map ---")
    ebsd_map = load_ebsd_map(EBSD_DATA_PATH, EBSD_DATA_TYPE, EBSD_BOUNDARY_DEF_DEG, EBSD_MIN_GRAIN_PX)
    if not ebsd_map:
        print("ERROR: Could not load EBSD map.")
        sys.exit(1)

    # Load experimental distribution
    print("--- Loading experimental FFT ---")
    exp_mhz = load_experimental_freqs_mhz(EXPERIMENTAL_HDF5_PATH, EXP_MIN_MHZ, EXP_MAX_MHZ)
    if exp_mhz.size == 0:
        print("WARNING: No experimental frequencies loaded; continuing to compute predicted-only scan.")

    # Scan angles
    rows = []
    print("--- Scanning angles vs experiment ---")
    for ang in tqdm(ANGLES_TO_TEST_DEG, desc="Angles"):
        pred_mhz = predict_freqs_mhz_for_angle(ebsd_map, props_pa, wavelength_m, float(ang))
        ks, w1 = np.nan, np.nan
        if exp_mhz.size > 0:
            d = compute_distances(exp_mhz, pred_mhz)
            ks, w1 = d["ks"], d["w1"]
        rows.append({"angle_deg": float(ang), "ks": ks, "w1": w1, "n_pred": int(pred_mhz.size), "n_exp": int(exp_mhz.size)})

    df = pd.DataFrame(rows).sort_values("angle_deg").reset_index(drop=True)
    out_csv = os.path.join(RESULTS_DIR, "angle_alignment_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print best angles
    if df['ks'].notna().any():
        i_ks = int(df['ks'].idxmin())
        print(f"Best angle by KS: {df.loc[i_ks, 'angle_deg']:.1f}° (KS={df.loc[i_ks, 'ks']:.4f})")
    if df['w1'].notna().any():
        i_w1 = int(df['w1'].idxmin())
        print(f"Best angle by W1: {df.loc[i_w1, 'angle_deg']:.1f}° (W1={df.loc[i_w1, 'w1']:.3f} MHz)")

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(df['angle_deg'], df['ks'], '-o', ms=3, color="#2A33C3")
    #axs[0].set_xlabel('Angle (deg)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('KS statistic', fontsize=12, fontweight='bold')
    axs[0].set_title('KS vs Angle', fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[1].plot(df['angle_deg'], df['w1'], '-o', ms=3, color="#A35D00")
    axs[1].set_xlabel('Angle (deg)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('W1 (MHz)', fontsize=12, fontweight='bold')
    axs[1].set_title('Wasserstein-1 vs Angle', fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "angle_alignment.png")
    fig.savefig(out_png, dpi=450)
    print(f"Saved: {out_png}")


