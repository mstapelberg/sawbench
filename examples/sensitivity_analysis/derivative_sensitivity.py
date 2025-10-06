# This script calculates the dimmensionless log-sensitivities within 1\% finite differences at the baseline

# S_theta = \frac{\partial long(f)}{\partial long(theta)} ~ \frac{log f(tehta * 1.01) - logf(theta)}{log(1.01)}
# where theta is the shear modulus, Zener anisotropy, density, Euler angles, or in-plane angle

# The script will calculate the sensitivities for the following parameters:
# - C11
# - C12
# - C44
# - Density
# - Euler angles
# - In-plane angle


# Figure 1b) The output of this script will be a bar chart of median |S_theta| for G, A_Z, density 
# Figure 1c) Another plot of IQR_{in-plane angle}(f) vs in-plane angle to show how much the in-place angle affects the frequency
# Figure 1d) A final plot will focus on the effects of Euler angles on the frequency 


# local_sensitivities.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import sys
from typing import Optional, Dict

# Progress and parallel tools
from tqdm import tqdm
try:
    from tqdm.contrib.concurrent import process_map as _process_map
except Exception:
    _process_map = None

# ---------- USER INPUTS ----------
# Baseline (from your ternary caption)
C11_0, C12_0, C44_0 = 229.0, 119.0, 43.0    # GPa
rho_0 = 6100.0                              # kg/m^3
euler0_deg = (0.0, 0.0, 0.0)                # (alpha, beta, gamma) in degrees
wavelength_um = 8.80                        # micrometers
psis_deg = np.arange(0, 180, 1)            # your measured in-plane angles

# Finite-difference steps
h_frac = 0.01                                # 1% multiplicative step for moduli and rho
h_angle_deg = 5.0                            # central step for Euler and psi slopes (deg)

# ---------- FORWARD MODEL ----------
def predict_saw_frequency(C11, C12, C44, rho, euler_deg, psi_deg, wavelength_um):
    """
    Use sawbench's SAWCalculator to compute SAW frequency for a given state.

    Args:
        C11, C12, C44: Elastic constants in GPa (cubic system).
        rho: Density in kg/m^3.
        euler_deg: Tuple/list of Euler angles (phi1, PHI, phi2) in degrees.
        psi_deg: In-plane propagation angle (degrees) on the surface.
        wavelength_um: Wavelength in micrometers.

    Returns:
        Frequency in MHz (float). np.nan if calculation fails.
    """
    # Robust import of sawbench from installed package or local src
    try:
        from sawbench import Material, SAWCalculator
    except ImportError:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        src_path = os.path.join(repo_root, "src")
        if src_path not in sys.path:
            sys.path.append(src_path)
        from sawbench import Material, SAWCalculator

    # Solver controls: balance accuracy vs speed
    SAW_SAMPLING = 4000  # increase to 4000 for higher accuracy
    SAW_PSAW = 0        # fundamental SAW only

    try:
        # Build material (sawbench expects Pa)
        material = Material(
            formula='X',
            C11=float(C11) * 1e9,
            C12=float(C12) * 1e9,
            C44=float(C44) * 1e9,
            density=float(rho),
            crystal_class='cubic'
        )

        # Euler angles in radians
        euler_rad = np.deg2rad(np.array(euler_deg, dtype=float))
        calculator = SAWCalculator(material, euler_rad)

        v_mps, _, _ = calculator.get_saw_speed(
            deg=float(psi_deg), sampling=SAW_SAMPLING, psaw=SAW_PSAW
        )

        # Take the primary SAW mode
        if v_mps is None or len(v_mps) == 0 or not np.isfinite(v_mps[0]):
            return np.nan

        lambda_m = float(wavelength_um) * 1.0e-6
        if lambda_m <= 0:
            return np.nan

        f_hz = v_mps[0] / lambda_m
        return f_hz / 1.0e6  # MHz
    except Exception:
        # Be robust to any numerical issues during sweeps
        return np.nan

# ---------- UTILITIES ----------
@dataclass
class ElasticParams:
    C11: float
    C12: float
    C44: float

def K_from_C(C11, C12):  # bulk-like combination
    return (C11 + 2.0*C12) / 3.0

def D_from_C(C11, C12):  # deviatoric difference
    return (C11 - C12)

def C_from_KD(K, D):
    C12 = K - D/3.0
    C11 = K + 2.0*D/3.0
    return C11, C12

def AZ_from_C(C11, C12, C44):
    return 2.0*C44 / (C11 - C12)

def f_at(params: ElasticParams, rho, euler_deg, psi_deg):
    return predict_saw_frequency(params.C11, params.C12, params.C44, rho, euler_deg, psi_deg, wavelength_um)

# Central log-sensitivity for multiplicative parameters
def S_log_param(eval_f, x0, h_frac):
    x_plus = x0 * (1.0 + h_frac)
    x_minus = x0 * (1.0 - h_frac)
    f_plus, f_minus = eval_f(x_plus), eval_f(x_minus)
    return (np.log(f_plus) - np.log(f_minus)) / (np.log(1.0 + h_frac) - np.log(1.0 - h_frac))

# Central slope for angles (MHz/deg)
def slope_angle(eval_f, x0_deg, h_deg):
    f_plus = eval_f(x0_deg + h_deg)
    f_minus = eval_f(x0_deg - h_deg)
    return (f_plus - f_minus) / (2.0*h_deg)

# ---------- (1) Compute sensitivities across measured psi ----------
def _compute_record_for_psi(psi: float) -> Dict[str, float]:
    C = ElasticParams(C11_0, C12_0, C44_0)
    # Retain for potential reporting/use; currently not used directly

    # Baseline frequency
    f0 = f_at(C, rho_0, euler0_deg, psi)

    # --- Elastic tensor sensitivities (log)
    def f_vs_C11(x):
        return f_at(ElasticParams(x, C.C12, C.C44), rho_0, euler0_deg, psi)
    S_C11 = S_log_param(f_vs_C11, C.C11, h_frac)

    def f_vs_C12(x):
        return f_at(ElasticParams(C.C11, x, C.C44), rho_0, euler0_deg, psi)
    S_C12 = S_log_param(f_vs_C12, C.C12, h_frac)

    def f_vs_C44(x):
        return f_at(ElasticParams(C.C11, C.C12, x), rho_0, euler0_deg, psi)
    S_C44 = S_log_param(f_vs_C44, C.C44, h_frac)

    # --- Density sensitivity (log)
    def f_vs_rho(x):
        return f_at(C, x, euler0_deg, psi)
    S_rho = S_log_param(f_vs_rho, rho_0, h_frac)

    # --- Euler angle slopes (MHz/deg)
    a0, b0, g0 = euler0_deg
    Sa = slope_angle(lambda a: f_at(C, rho_0, (a, b0, g0), psi), a0, h_angle_deg)
    Sb = slope_angle(lambda b: f_at(C, rho_0, (a0, b, g0), psi), b0, h_angle_deg)
    Sg = slope_angle(lambda g: f_at(C, rho_0, (a0, b0, g), psi), g0, h_angle_deg)

    # --- Rotation (psi) slope (MHz/deg)
    Spsi = slope_angle(lambda p: f_at(C, rho_0, euler0_deg, p), psi, h_angle_deg)

    return dict(psi=psi, f0=f0,
                S_C11=S_C11, S_C12=S_C12, S_C44=S_C44, S_rho=S_rho,
                dfdpsi=Spsi, dfdalpha=Sa, dfdBeta=Sb, dfdgamma=Sg)

def compute_sensitivities(psis_deg, n_workers: Optional[int] = None):
    psis_list = list(psis_deg)
    if _process_map is not None and (n_workers is None or n_workers != 1):
        records = _process_map(_compute_record_for_psi, psis_list, max_workers=n_workers, desc="Sensitivities ψ")
    else:
        records = [
            _compute_record_for_psi(psi)
            for psi in tqdm(psis_list, desc="Sensitivities ψ")
        ]
    return pd.DataFrame.from_records(records)

# (Removed IQR jitter analysis to keep the script focused on basic sensitivities)

# ---------- (3) Make the plots ----------
def make_plots(df_sens):
    # Panel 1: bar of median |log-sensitivities| for C11, C12, C44, ρ
    cols = ['S_C11','S_C12','S_C44','S_rho']
    med = df_sens[cols].abs().median().rename({'S_C11':'C11','S_C12':'C12','S_C44':'C44','S_rho':'density'})
    spread = df_sens[cols].abs().quantile([0.25,0.75])
    yerr = (spread.loc[0.75]-spread.loc[0.25]).values/2.0

    fig, axs = plt.subplots(1,3, figsize=(13,3.5))
    ax = axs[0]
    ax.bar(med.index, med.values, yerr=yerr, capsize=3)
    ax.set_ylabel('Median |∂log f / ∂log θ|')
    ax.set_title('Elastic and density sensitivities')

    # Panel 2: |df/dψ| vs ψ (rotation sensitivity)
    ax = axs[1]
    ax.plot(df_sens['psi'], df_sens['dfdpsi'].abs())
    ax.set_xlabel('in-plane angle ψ (deg)')
    ax.set_ylabel('|∂f/∂ψ| (MHz/deg)')
    ax.set_title('Rotation sensitivity')

    # Panel 3: Euler slopes (median with IQR whiskers)
    orient = df_sens[['dfdalpha','dfdBeta','dfdgamma']].abs()
    med_o = orient.median()
    iqr_o = orient.quantile([0.25,0.75])
    yerr_o = (iqr_o.loc[0.75]-iqr_o.loc[0.25]).values/2.0
    ax = axs[2]
    ax.bar(['α','β','γ'], med_o.values, yerr=yerr_o, capsize=3)
    ax.set_ylabel('|∂f/∂angle| (MHz/deg)')
    ax.set_title('Euler-angle sensitivity')

    fig.tight_layout()
    plt.show()

# ---------- (4) Save results as JSON ----------
def _json_default(o):
    try:
        import numpy as _np
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    try:
        return o.__dict__
    except Exception:
        return str(o)

def save_results_json(df_sens: pd.DataFrame, output_path: Optional[str] = None) -> str:
    import json
    import datetime as _dt
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(results_dir, 'derivative_sensitivity_results.json')

    payload = {
        'metadata': {
            'C11_0_GPa': float(C11_0),
            'C12_0_GPa': float(C12_0),
            'C44_0_GPa': float(C44_0),
            'rho_0_kg_m3': float(rho_0),
            'euler0_deg': tuple(float(x) for x in euler0_deg),
            'wavelength_um': float(wavelength_um),
            'psis_deg': [float(x) for x in psis_deg],
            'h_frac': float(h_frac),
            'h_angle_deg': float(h_angle_deg),
        },
        'sensitivities': df_sens.to_dict(orient='records'),
        'timestamp': _dt.datetime.now().isoformat(timespec='seconds')
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, default=_json_default)
    print(f"Saved results JSON to: {output_path}")
    return output_path

if __name__ == "__main__":
    df_sens = compute_sensitivities(psis_deg)
    save_results_json(df_sens)
    make_plots(df_sens)
