"""
Global variance-based sensitivity analysis with multiple representative grain orientations.

This script performs Sobol sensitivity analysis to understand how EBSD measurement
errors affect SAW frequency predictions across different grain orientations.

Key features:
- Uses 28 representative grain orientations (stratified by SAW frequency)
- Tests three EBSD error scales: ±1°, ±2.5°, and ±5°
- Evaluates sensitivity of Wasserstein distance (model-experiment fit)
- Parallel evaluation with progress tracking
"""

import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from scipy.stats import wasserstein_distance
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import json
from pathlib import Path

# sawbench imports
from sawbench import (
    Material,
    SAWCalculator,
    load_fft_data_from_hdf5,
    extract_experimental_peak_parameters,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Representative grain orientations (φ₁, Φ, φ₂) in radians
# Stratified by SAW frequency to ensure diverse orientation coverage
REPRESENTATIVE_ORIENTATIONS_RAD = [
    (2.866471, 0.138829, 1.592157),  # Grain 165, 288.42 MHz
    (5.309013, 0.374197, 0.046287),  # Grain 77, 288.84 MHz
    (5.395439, 0.231447, 0.123157),  # Grain 49, 289.05 MHz
    (2.126833, 0.259111, 0.239084),  # Grain 61, 289.26 MHz
    (3.299978, 0.197294, 1.007755),  # Grain 53, 289.47 MHz
    (3.170021, 0.159066, 0.806326),  # Grain 101, 289.89 MHz
    (6.053041, 0.411653, 0.126920),  # Grain 48, 290.10 MHz
    (2.614710, 0.554468, 1.081496),  # Grain 52, 290.52 MHz
    (1.674416, 0.316474, 0.654611),  # Grain 123, 290.95 MHz
    (1.996944, 0.727202, 1.314264),  # Grain 84, 291.38 MHz
    (0.728028, 0.224336, 1.519496),  # Grain 95, 291.80 MHz
    (0.521901, 0.295965, 0.641883),  # Grain 43, 292.02 MHz
    (0.639058, 0.322885, 1.114467),  # Grain 111, 292.66 MHz
    (6.166210, 0.631897, 1.539816),  # Grain 44, 293.09 MHz
    (2.953035, 0.658644, 0.111145),  # Grain 96, 293.31 MHz
    (1.445911, 0.601773, 0.457214),  # Grain 117, 293.96 MHz
    (1.498330, 0.520175, 1.586225),  # Grain 171, 294.61 MHz
    (1.844674, 0.793952, 1.216411),  # Grain 191, 295.49 MHz
    (3.968542, 0.450262, 0.846715),  # Grain 81, 295.93 MHz
    (0.747071, 0.383937, 1.457088),  # Grain 294, 296.37 MHz
    (1.698599, 0.713742, 1.253969),  # Grain 340, 297.03 MHz
    (0.460893, 0.648824, 0.778413),  # Grain 54, 297.92 MHz
    (1.748898, 0.855348, 1.182214),  # Grain 138, 298.82 MHz
    (0.705498, 0.569165, 0.427020),  # Grain 109, 299.72 MHz
    (4.116403, 0.725440, 1.287696),  # Grain 107, 300.40 MHz
    (3.988831, 0.573632, 1.418867),  # Grain 162, 301.31 MHz
    (3.542656, 0.575058, 0.226148),  # Grain 104, 302.22 MHz
    (0.421711, 0.723746, 0.198942),  # Grain 526, 305.01 MHz
]

#REPRESENTATIVE_ORIENTATIONS_RAD = REPRESENTATIVE_ORIENTATIONS_RAD[:5] # Only use first 5 orientations for testing

# Measured in-plane angles (degrees)
PSIS_MEASURED_DEG = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])

# Experimental FFT data
EXPERIMENTAL_HDF5_PATH = "/home/myless/Documents/saw_freq_analysis/fftData.h5"
EXPERIMENT_MIN_MHZ = 200.0
EXPERIMENT_MAX_MHZ = 400.0

# Material properties (baseline)
C11_0, C12_0, C44_0 = 229.0, 119.0, 43.0  # GPa
RHO_0 = 6100.0  # kg/m³
WAVELENGTH_UM = 8.80  # μm

# SAW calculation settings
SAW_SAMPLING = 4000  # Reduced from 4000 for speed (10x faster, ~1% accuracy loss)
SAW_PSAW_FLAG = 0

# Sensitivity analysis settings
N_SOBOL_SAMPLES = 256  # Reduced from 512 for initial run (double for production)
NUM_WORKERS = min(32, cpu_count())

# EBSD error scales to test
EULER_ERROR_SCALES_DEG = [1.0, 2.5, 5.0]  # Start with just middle scale (add [1.0, 5.0] later)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results" / "multi_orientation_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXPERIMENTAL DATA LOADING
# ============================================================================

def load_experimental_frequencies_mhz(hdf5_path: str) -> np.ndarray:
    """Extract dominant experimental peak frequencies in MHz."""
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
    except Exception as e:
        print(f"Warning: Could not load experimental data: {e}")
        return np.array([])

# ============================================================================
# FORWARD MODEL
# ============================================================================

def predict_saw_frequency(C11_GPa, C12_GPa, C44_GPa, rho_kg_m3, euler_rad, psi_deg, 
                         wavelength_um, sampling: int = 400, psaw: int = 0) -> float:
    """Forward model via sawbench; returns predicted SAW frequency in MHz."""
    GPA_TO_PA = 1e9
    C11 = float(C11_GPa) * GPA_TO_PA
    C12 = float(C12_GPa) * GPA_TO_PA
    C44 = float(C44_GPa) * GPA_TO_PA
    wavelength_m = float(wavelength_um) * 1e-6

    material = Material(
        formula='X', C11=C11, C12=C12, C44=C44, density=float(rho_kg_m3), 
        crystal_class='cubic'
    )
    try:
        calc = SAWCalculator(material, euler_rad)
        v_mps, _, _ = calc.get_saw_speed(float(psi_deg), sampling=sampling, psaw=psaw)
        if v_mps is not None and len(v_mps) > 0 and np.isfinite(v_mps[0]):
            f_hz = float(v_mps[0]) / wavelength_m
            return f_hz / 1e6
    except Exception:
        pass
    return np.nan

# ============================================================================
# PARAMETER MAPPING
# ============================================================================

def KD_to_C11C12(K, D):
    """Convert bulk modulus K and anisotropy measure D to C11, C12."""
    C12 = K - D/3.0
    C11 = K + 2.0*D/3.0
    return C11, C12

def stable_cubic(C11, C12, C44):
    """Check Born stability criteria for cubic crystal."""
    return (C11 - C12 > 0.0) and (C44 > 0.0) and (C11 + 2.0*C12 > 0.0)

# ============================================================================
# SENSITIVITY ANALYSIS FUNCTIONS
# ============================================================================

def model_frequency_singlepsi(sample, psi_deg, euler_baseline_rad):
    """Compute SAW frequency for a single psi angle with parameter perturbations."""
    K, D, G, rho, phi1_err_deg, Phi_err_deg, phi2_err_deg, psi_err = sample
    
    # Convert elastic parameters
    C11, C12 = KD_to_C11C12(K, D)
    if not stable_cubic(C11, C12, G):
        return np.nan
    
    # Apply Euler angle perturbations (convert deg to rad)
    phi1_err_rad = np.deg2rad(phi1_err_deg)
    Phi_err_rad = np.deg2rad(Phi_err_deg)
    phi2_err_rad = np.deg2rad(phi2_err_deg)
    
    euler_perturbed = (
        euler_baseline_rad[0] + phi1_err_rad,
        euler_baseline_rad[1] + Phi_err_rad,
        euler_baseline_rad[2] + phi2_err_rad
    )
    
    return predict_saw_frequency(
        C11, C12, G, rho, euler_perturbed, psi_deg + psi_err, 
        WAVELENGTH_UM, sampling=SAW_SAMPLING, psaw=SAW_PSAW_FLAG
    )

def model_wasserstein_vs_experiment(sample, experimental_mhz: np.ndarray, 
                                   euler_baseline_rad):
    """Compute Wasserstein distance between predicted and experimental frequencies."""
    preds = [model_frequency_singlepsi(sample, psi, euler_baseline_rad) 
             for psi in PSIS_MEASURED_DEG]
    preds = np.array([p for p in preds if np.isfinite(p)])
    
    if preds.size == 0 or experimental_mhz is None or experimental_mhz.size == 0:
        return np.nan
    
    return float(wasserstein_distance(preds, experimental_mhz))

def _eval_one_sample_wrapper(sample_row, exp_freqs, euler_baseline):
    """Wrapper function for parallel evaluation (must be at module level for pickling).
    
    Args are passed explicitly via functools.partial instead of globals.
    """
    try:
        result = model_wasserstein_vs_experiment(sample_row, exp_freqs, euler_baseline)
        return result
    except Exception as e:
        print(f"Error in worker: {e}")
        import traceback
        traceback.print_exc()
        return np.nan

# ============================================================================
# SOBOL ANALYSIS FOR SINGLE ORIENTATION
# ============================================================================

def run_sobol_for_orientation(euler_baseline_rad, euler_error_scale_deg, 
                              exp_freqs_mhz, orientation_idx):
    """Run Sobol sensitivity analysis for a single grain orientation."""
    
    # Setup parameter bounds
    K0 = (C11_0 + 2*C12_0) / 3.0
    D0 = C11_0 - C12_0
    G0 = C44_0
    
    problem = {
    "num_vars": 8,
    "names": ["K", "D", "G", "rho", "phi1_err_deg", "Phi_err_deg", 
              "phi2_err_deg", "psi_err_deg"],
    "bounds": [
        [1.00*K0, 1.17130620985*K0],      # K: from DFT/literature bounds
        [0.77272727273*D0, 1.27272727273*D0],  # D: anisotropy (MLIP-sensitive)
        [0.30232558140*G0, 1.00*G0],      # G: C44 bounds
        [0.95*RHO_0, 1.05*RHO_0],         # ρ: density ±5% (well-constrained)
        [-euler_error_scale_deg, euler_error_scale_deg],  # φ₁: EBSD fitting error
        [-euler_error_scale_deg, euler_error_scale_deg],  # Φ: EBSD fitting error
        [-euler_error_scale_deg, euler_error_scale_deg],  # φ₂: EBSD fitting error
        [-20.0, 20.0],                    # ψ: sample alignment error ±20°
                                          # (BCC: effective error reduced by 90° symmetry)
    ],
}

    
    # Generate Sobol samples
    X = sobol_sample.sample(problem, N_SOBOL_SAMPLES, calc_second_order=False)
    
    # Create partial function with fixed arguments for this orientation
    eval_func = partial(_eval_one_sample_wrapper, 
                       exp_freqs=exp_freqs_mhz, 
                       euler_baseline=euler_baseline_rad)
    
    # Evaluate model for all samples
    Y_list = []
    if NUM_WORKERS > 1:
        chunksize = max(1, len(X) // (4 * NUM_WORKERS))
        with Pool(processes=NUM_WORKERS) as pool:
            desc = f"Orientation {orientation_idx+1} (±{euler_error_scale_deg}°)"
            pbar = tqdm(total=len(X), desc=desc, unit="sample", position=0, leave=True)
            for val in pool.imap(eval_func, X, chunksize=chunksize):
                Y_list.append(val)
                pbar.update(1)
            pbar.close()
    else:
        desc = f"Orientation {orientation_idx+1} (±{euler_error_scale_deg}°)"
        pbar = tqdm(total=len(X), desc=desc, unit="sample")
        for row in X:
            Y_list.append(eval_func(row))
            pbar.update(1)
        pbar.close()
    
    # Analyze with Sobol
    Y = np.array(Y_list, dtype=float)
    mask = ~np.isnan(Y)
    n_valid = np.sum(mask)
    
    if n_valid < 0.5 * len(Y):
        print(f"Warning: Only {n_valid}/{len(Y)} valid samples for orientation {orientation_idx+1}")
        return None
    
    Si = sobol_analyze.analyze(problem, Y[mask], calc_second_order=False, 
                              print_to_console=False)
    
    # Package results (convert numpy types to native Python for JSON serialization)
    results = {
        "orientation_idx": int(orientation_idx),
        "euler_baseline_rad": [float(x) for x in euler_baseline_rad],
        "euler_baseline_deg": [float(x) for x in np.rad2deg(euler_baseline_rad)],
        "euler_error_scale_deg": float(euler_error_scale_deg),
        "n_samples": int(len(Y)),
        "n_valid": int(n_valid),
        "parameters": problem["names"],
        "S1": [float(x) for x in Si["S1"]],
        "S1_conf": [float(x) for x in Si["S1_conf"]],
        "ST": [float(x) for x in Si["ST"]],
        "ST_conf": [float(x) for x in Si["ST_conf"]],
    }
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("GLOBAL VARIANCE-BASED SENSITIVITY ANALYSIS")
    print("Multiple Representative Grain Orientations")
    print("="*70)
    
    # Load experimental data
    print(f"\nLoading experimental data from: {EXPERIMENTAL_HDF5_PATH}")
    exp_freqs_mhz = load_experimental_frequencies_mhz(EXPERIMENTAL_HDF5_PATH)
    
    if exp_freqs_mhz is None or exp_freqs_mhz.size == 0:
        print("WARNING: No experimental data loaded. Sensitivity may be uninformative.")
        exp_freqs_mhz = np.array([])
    else:
        print(f"Loaded {len(exp_freqs_mhz)} experimental frequency measurements")
        print(f"Frequency range: {exp_freqs_mhz.min():.2f} - {exp_freqs_mhz.max():.2f} MHz")
    
    # Calculate totals
    n_sobol_samples_actual = N_SOBOL_SAMPLES * (8 + 2)  # Sobol generates N*(D+2) samples for D parameters
    total_evaluations = len(REPRESENTATIVE_ORIENTATIONS_RAD) * len(EULER_ERROR_SCALES_DEG) * n_sobol_samples_actual
    total_saw_calcs = total_evaluations * len(PSIS_MEASURED_DEG)
    
    # Rough time estimate (depends on SAW_SAMPLING)
    est_time_per_saw_sec = 0.01 if SAW_SAMPLING <= 400 else 0.1  # Rough estimate
    est_total_hours = (total_saw_calcs * est_time_per_saw_sec / NUM_WORKERS) / 3600
    
    print(f"\nConfiguration:")
    print(f"  Representative orientations: {len(REPRESENTATIVE_ORIENTATIONS_RAD)}")
    print(f"  EBSD error scales: {EULER_ERROR_SCALES_DEG}")
    print(f"  Sobol samples per orientation: {n_sobol_samples_actual} (N={N_SOBOL_SAMPLES} × (D+2) where D=8)")
    print(f"  SAW sampling resolution: {SAW_SAMPLING}")
    print(f"  Parallel workers: {NUM_WORKERS}")
    print(f"  Total Sobol evaluations: {total_evaluations:,}")
    print(f"  Total SAW calculations: {total_saw_calcs:,}")
    print(f"  Estimated runtime: {est_total_hours:.1f} hours (rough estimate)")
    
    # Run analysis for each orientation and error scale
    all_results = []
    
    for error_scale in EULER_ERROR_SCALES_DEG:
        print(f"\n{'='*70}")
        print(f"EBSD ERROR SCALE: ±{error_scale}°")
        print(f"{'='*70}")
        
        for i, euler_baseline in enumerate(REPRESENTATIVE_ORIENTATIONS_RAD):
            print(f"\nOrientation {i+1}/{len(REPRESENTATIVE_ORIENTATIONS_RAD)}: "
                  f"φ₁={np.rad2deg(euler_baseline[0]):.1f}°, "
                  f"Φ={np.rad2deg(euler_baseline[1]):.1f}°, "
                  f"φ₂={np.rad2deg(euler_baseline[2]):.1f}°")
            
            result = run_sobol_for_orientation(
                euler_baseline, error_scale, exp_freqs_mhz, i
            )
            
            if result is not None:
                all_results.append(result)
                
                # Print summary for this orientation
                df = pd.DataFrame({
                    "parameter": result["parameters"],
                    "S1": result["S1"],
                    "ST": result["ST"],
                }).sort_values("ST", ascending=False)
                print(f"\nTop 3 sensitive parameters:")
                for idx, row in df.head(3).iterrows():
                    print(f"  {row['parameter']:15s}: ST={row['ST']:.3f}, S1={row['S1']:.3f}")
    
    # Save all results
    output_file = OUTPUT_DIR / "all_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Saved detailed results to: {output_file}")
    
    # Create summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for error_scale in EULER_ERROR_SCALES_DEG:
        results_at_scale = [r for r in all_results 
                           if r["euler_error_scale_deg"] == error_scale]
        
        if not results_at_scale:
            continue
            
        print(f"\nEBSD Error Scale: ±{error_scale}°")
        print(f"{'-'*70}")
        
        # Aggregate sensitivities across all orientations
        param_names = results_at_scale[0]["parameters"]
        S1_all = np.array([r["S1"] for r in results_at_scale])
        ST_all = np.array([r["ST"] for r in results_at_scale])
        
        summary_df = pd.DataFrame({
            "parameter": param_names,
            "S1_mean": S1_all.mean(axis=0),
            "S1_std": S1_all.std(axis=0),
            "ST_mean": ST_all.mean(axis=0),
            "ST_std": ST_all.std(axis=0),
        }).sort_values("ST_mean", ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = OUTPUT_DIR / f"summary_error_{error_scale}deg.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary to: {summary_file}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"1. Review summaries in: {OUTPUT_DIR}")
    print(f"2. Use plot_sensitivity_bars.py to visualize results")
    print(f"3. Compare sensitivity across error scales and orientations")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

