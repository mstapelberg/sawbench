#!/usr/bin/env python3
"""Ternary SAW frequency diagram for cubic elastic constants (C11, C12, C44).

This script generates a publication-quality ternary plot where each corner
represents emphasis on one of the three cubic elastic constants (C11, C12, C44).
The constants are varied ±50% around a specified base case, and the color map
indicates the predicted SAW frequency.

Interpretation and mapping:
- Let w = (w11, w12, w44) be barycentric weights that satisfy w11 + w12 + w44 = 1.
- We map these to elastic constants as:
    C11 = C11_base * (0.5 + w11)
    C12 = C12_base * (0.5 + w12)
    C44 = C44_base * (0.5 + w44)
  so each constant ranges from 0.5× to 1.5× its base value across the triangle.

Base case note:
- On a ternary diagram, the center corresponds to equal weights (w = 1/3 for each).
  Under the mapping above, the equal-weights point gives constants ≈ 0.833× base.
  We still annotate axis tick value 0.5 with the base constants to reflect your
  requested interpretation of axis values. If you prefer base to be exactly at the
  triangle center, we can change the mapping (at the expense of the exact ±50% range).

Dependencies: numpy, matplotlib, mpltern, sawbench (local).

Outputs: saves a high-resolution PNG in examples/elastic_tensor/results/ by default.
"""

from __future__ import annotations

import os
import argparse
import pickle
import json
import hashlib
import warnings
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import mpltern  # noqa: F401  # required for registering 'ternary' projection via side effect
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ks_2samp
from tqdm import tqdm

from sawbench.materials import Material
from sawbench.saw_calculator import SAWCalculator

# Suppress numpy linalg warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero encountered in det.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in det.*')

try:
    from sawbench import (
        load_ebsd_map,
        calculate_saw_frequencies_for_ebsd_grains,
        extract_experimental_peak_parameters,
        load_fft_data_from_hdf5,
    )
    EBSD_AVAILABLE = True
except ImportError:
    EBSD_AVAILABLE = False


class SAWCache:
    """Cache for SAW frequency calculations to avoid recomputation."""
    
    def __init__(self, cache_file: str = "saw_cache.pkl"):
        self.cache_file = cache_file
        self.cache: Dict[str, float] = {}
        self.load_cache()
    
    def _generate_key(self, C11: float, C12: float, C44: float, density: float, 
                     euler_angles: Tuple[float, float, float], deg_inplane: float,
                     sampling: int, wavelength: float) -> str:
        """Generate a unique hash key for the calculation parameters."""
        key_data = {
            'C11': round(C11, 6),
            'C12': round(C12, 6), 
            'C44': round(C44, 6),
            'density': round(density, 3),
            'euler': tuple(round(x, 6) for x in euler_angles),
            'deg_inplane': round(deg_inplane, 3),
            'sampling': sampling,
            'wavelength': round(wavelength, 10)
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, C11: float, C12: float, C44: float, density: float,
            euler_angles: Tuple[float, float, float], deg_inplane: float,
            sampling: int, wavelength: float) -> Optional[float]:
        """Get cached result if available."""
        key = self._generate_key(C11, C12, C44, density, euler_angles, deg_inplane, sampling, wavelength)
        return self.cache.get(key)
    
    def set(self, C11: float, C12: float, C44: float, density: float,
            euler_angles: Tuple[float, float, float], deg_inplane: float,
            sampling: int, wavelength: float, result: float) -> None:
        """Cache a result."""
        key = self._generate_key(C11, C12, C44, density, euler_angles, deg_inplane, sampling, wavelength)
        self.cache[key] = result
        self.save_cache()


class KSCache:
    """Cache for KS metric calculations: maps (K, D, G, EBSD, Exp) → KS metric.
    
    Computational parameters (sampling, wavelength, angle) are NOT included in the key,
    allowing you to identify best K/D/G combinations and recompute with different parameters.
    """
    
    def __init__(self, cache_file: str = "ks_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}  # key -> {KS, K, D, G, metadata}
        self.pending_writes: Dict[str, Dict[str, Any]] = {}
        self.load_cache()
    
    def _generate_key(self, C11: float, C12: float, C44: float, 
                     ebsd_hash: str, exp_hash: str) -> str:
        """Generate key from elastic constants and data hashes only.
        
        Excludes: sampling, wavelength, angle (computational parameters)
        Includes: C11, C12, C44 (material), EBSD hash, Experimental data hash
        """
        key_data = {
            'C11': round(C11, 6),
            'C12': round(C12, 6),
            'C44': round(C44, 6),
            'ebsd': ebsd_hash,
            'exp': exp_hash
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, C11: float, C12: float, C44: float,
            ebsd_hash: str, exp_hash: str) -> Optional[float]:
        """Get cached KS metric if available."""
        key = self._generate_key(C11, C12, C44, ebsd_hash, exp_hash)
        entry = self.cache.get(key)
        return entry['KS'] if entry else None
    
    def set(self, C11: float, C12: float, C44: float,
            ebsd_hash: str, exp_hash: str, ks_value: float,
            K: float, D: float, G: float) -> None:
        """Cache a KS metric result with K/D/G values."""
        key = self._generate_key(C11, C12, C44, ebsd_hash, exp_hash)
        
        # Store full entry with metadata
        entry = {
            'KS': ks_value,
            'K': round(K, 6),
            'D': round(D, 6),
            'G': round(G, 6),
            'C11': round(C11, 6),
            'C12': round(C12, 6),
            'C44': round(C44, 6),
        }
        
        self.cache[key] = entry
        self.pending_writes[key] = entry
        
        # Save every 10 entries to reduce I/O
        if len(self.pending_writes) >= 10:
            self.flush()
    
    def flush(self) -> None:
        """Flush pending writes to disk."""
        if self.pending_writes:
            self.save_cache()
            self.pending_writes = {}
    
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries as a list."""
        return list(self.cache.values())
    
    def load_cache(self) -> None:
        """Load cache from disk (JSON format)."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached KS calculations from {self.cache_file}")
            except Exception as e:
                print(f"Warning: Could not load cache file {self.cache_file}: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self) -> None:
        """Save cache to disk as JSON."""
        try:
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # Save as JSON for human-readability and stability
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.pending_writes = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Cache cleared.")
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        if total_entries > 0:
            ks_values = [e['KS'] for e in self.cache.values() if 'KS' in e]
            finite_entries = sum(1 for v in ks_values if np.isfinite(v))
            nan_entries = total_entries - finite_entries
        else:
            finite_entries = 0
            nan_entries = 0
        
        return {
            'total_entries': total_entries,
            'finite_entries': finite_entries,
            'nan_entries': nan_entries,
            'cache_file': self.cache_file
        }


# Global cache instances
_saw_cache = None
_ks_cache = None

def get_ks_cache(cache_file: str = "ks_cache.pkl") -> KSCache:
    """Get or create the global KS cache instance."""
    global _ks_cache
    if _ks_cache is None or _ks_cache.cache_file != cache_file:
        _ks_cache = KSCache(cache_file)
    return _ks_cache


def get_saw_cache(cache_file: str = "saw_cache.pkl") -> SAWCache:
    """Get or create the global SAW cache instance."""
    global _saw_cache
    if _saw_cache is None:
        _saw_cache = SAWCache(cache_file)
    return _saw_cache


def generate_ternary_grid(resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate all ternary coordinates (t, l, r) on a uniform triangular grid.

    resolution: number of segments per triangle edge; points per edge = resolution + 1.
    """
    t_list: List[float] = []
    l_list: List[float] = []
    r_list: List[float] = []

    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            t_list.append(i / resolution)
            l_list.append(j / resolution)
            r_list.append(k / resolution)

    return np.array(t_list), np.array(l_list), np.array(r_list)


def compute_saw_frequency_mhz(
    C11_GPa: float,
    C12_GPa: float,
    C44_GPa: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    deg_inplane: float = 0.0,
    sampling: int = 400,
    wavelength_m: float = 8.8e-6,
    use_cache: bool = True,
    cache_file: str = "saw_cache.pkl",
) -> float:
    """Compute predicted SAW frequency (MHz) for given elastic constants and density.

    Returns NaN on failure.
    """
    # Check cache first
    if use_cache:
        cache = get_saw_cache(cache_file)
        cached_result = cache.get(
            C11_GPa, C12_GPa, C44_GPa, density_kg_m3,
            euler_angles_rad, deg_inplane, sampling, wavelength_m
        )
        if cached_result is not None:
            return float(cached_result)
    
    # Compute result
    try:
        material = Material(
            formula="Cubic",
            C11=C11_GPa * 1e9,
            C12=C12_GPa * 1e9,
            C44=C44_GPa * 1e9,
            density=density_kg_m3,
            crystal_class="cubic",
        )
        calc = SAWCalculator(material=material, euler_angles=np.asarray(euler_angles_rad))
        # Try optimized path first
        try:
            v, _, _ = calc.get_saw_speed(
                deg_inplane,
                sampling=sampling,
                psaw=0,
                draw_plot=False,
                debug=False,
                use_optimized=True,
            )
        except Exception:
            v = np.array([])

        # Fallback to original implementation if necessary
        if v is None or len(v) == 0 or not np.isfinite(v[0]):
            try:
                v, _, _ = calc.get_saw_speed(
                    deg_inplane,
                    sampling=sampling,
                    psaw=0,
                    draw_plot=False,
                    debug=False,
                    use_optimized=False,
                )
            except Exception:
                result = float("nan")
                # Cache the NaN result to avoid recomputation
                if use_cache:
                    cache.set(C11_GPa, C12_GPa, C44_GPa, density_kg_m3,
                             euler_angles_rad, deg_inplane, sampling, wavelength_m, result)
                return result

        if v is None or len(v) == 0 or not np.isfinite(v[0]):
            result = float("nan")
        else:
            result = float(v[0] / wavelength_m / 1e6)  # MHz
        
        # Cache the result
        if use_cache:
            cache.set(C11_GPa, C12_GPa, C44_GPa, density_kg_m3,
                     euler_angles_rad, deg_inplane, sampling, wavelength_m, result)
        
        return result
    except Exception:
        result = float("nan")
        # Cache the NaN result to avoid recomputation
        if use_cache:
            cache = get_saw_cache(cache_file)
            cache.set(C11_GPa, C12_GPa, C44_GPa, density_kg_m3,
                     euler_angles_rad, deg_inplane, sampling, wavelength_m, result)
        return result


def format_gpa(value: float) -> str:
    return f"{value:.1f}"


def load_experimental_freqs_mhz(
    hdf5_path: str,
    min_mhz: Optional[float],
    max_mhz: Optional[float]
) -> np.ndarray:
    """Load experimental FFT data and extract dominant peak per pixel (MHz)."""
    if not hdf5_path or not EBSD_AVAILABLE:
        return np.array([])
    out = load_fft_data_from_hdf5(hdf5_path)
    if not out:
        return np.array([])
    exp_freq_axis, _, _, exp_amplitude, (Ny, Nx) = out
    dom = []
    for iy in range(Ny):
        for ix in range(Nx):
            amp_trace = exp_amplitude[:, iy, ix]
            peak_params = extract_experimental_peak_parameters(
                exp_freq_axis, amp_trace, num_peaks_to_extract=3
            )
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


def predict_freqs_mhz_for_material(
    ebsd_map_obj,
    material_props_pa: Dict[str, float],
    wavelength_m: float,
    angle_deg: float,
    saw_sampling: int,
    saw_psaw: int,
    num_workers: int
) -> np.ndarray:
    """Predict SAW frequencies for all grains in EBSD map with given material properties."""
    if not EBSD_AVAILABLE:
        return np.array([])
    df = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map_obj,
        material=Material(**material_props_pa),
        wavelength=wavelength_m,
        saw_calc_angle_deg=float(angle_deg),
        saw_calc_sampling=saw_sampling,
        saw_calc_psaw=saw_psaw,
        num_workers=num_workers
    )
    if df.empty or 'Peak SAW Frequency (Hz)' not in df.columns:
        return np.array([])
    arr_hz = df['Peak SAW Frequency (Hz)'].dropna().to_numpy()
    return arr_hz / 1e6


def compute_ks_metric(exp_mhz: np.ndarray, pred_mhz: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic between experimental and predicted distributions."""
    exp = exp_mhz[np.isfinite(exp_mhz)]
    pred = pred_mhz[np.isfinite(pred_mhz)]
    if exp.size == 0 or pred.size == 0:
        return float("nan")
    ks_stat, _ = ks_2samp(exp, pred)
    return float(ks_stat)


def generate_ebsd_hash(ebsd_path: str) -> str:
    """Generate a hash for EBSD data file."""
    # Use file path + file size + modification time for hash
    try:
        stat = os.stat(ebsd_path)
        hash_str = f"{ebsd_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    except:
        return hashlib.md5(ebsd_path.encode()).hexdigest()[:16]


def generate_exp_hash(exp_h5_path: str, min_mhz: float, max_mhz: float) -> str:
    """Generate a hash for experimental data."""
    try:
        stat = os.stat(exp_h5_path)
        hash_str = f"{exp_h5_path}_{stat.st_size}_{stat.st_mtime}_{min_mhz}_{max_mhz}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    except:
        hash_str = f"{exp_h5_path}_{min_mhz}_{max_mhz}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]


def auto_calibrate_relative_range(finite_values: np.ndarray) -> float:
    """Auto-calibrate the relative range for optimal visual sensitivity.
    
    This function analyzes the distribution of relative frequency changes and selects
    an optimal range that maximizes visual contrast while avoiding extreme outliers.
    
    Strategy:
    1. Calculate various percentile-based ranges
    2. Choose the range that provides good visual coverage
    3. Ensure the range is not too narrow (avoid pure white) or too wide (lose detail)
    """
    if finite_values.size == 0:
        return 0.1  # Default 10% range
    
    abs_values = np.abs(finite_values)
    
    # Calculate different percentile-based ranges
    p50 = np.percentile(abs_values, 50)  # Median absolute deviation
    p75 = np.percentile(abs_values, 75)  # 75th percentile
    p85 = np.percentile(abs_values, 85)  # 85th percentile  
    p90 = np.percentile(abs_values, 90)  # 90th percentile
    p95 = np.percentile(abs_values, 95)  # 95th percentile
    max_val = np.max(abs_values)
    
    # Strategy: Choose range that captures meaningful variation
    # without being dominated by outliers
    
    # Option 1: Use 90th percentile (captures most data, excludes extreme outliers)
    candidate_90 = p90
    
    # Option 2: Use 2x median (good for symmetric distributions)
    candidate_2x_median = 2.0 * p50
    
    # Option 3: Use 85th percentile (slightly more inclusive than 90th)
    candidate_85 = p85
    
    # Option 4: Adaptive based on data spread
    data_spread = p95 - p50  # Interquartile-like spread
    candidate_adaptive = p75 + 2.0 * data_spread
    
    # Choose the most appropriate range
    candidates = [
        candidate_90,
        candidate_2x_median, 
        candidate_85,
        candidate_adaptive
    ]
    
    # Filter out invalid candidates
    valid_candidates = [c for c in candidates if np.isfinite(c) and c > 0]
    
    if not valid_candidates:
        return 0.1  # Fallback to 10%
    
    # Choose the median of valid candidates (robust choice)
    chosen_range = np.median(valid_candidates)
    
    # Ensure reasonable bounds
    min_range = 0.001  # 0.1% minimum (avoid pure white)
    max_range = 0.50   # 50% maximum (avoid losing all detail)
    
    # Additional constraint: if max value is very small, don't make range too large
    if max_val > 0 and max_val < 0.02:  # If max change is <2%
        max_range = min(max_range, max_val * 3.0)  # Don't exceed 3x the max
    
    chosen_range = np.clip(chosen_range, min_range, max_range)
    
    return float(chosen_range)


def create_ternary_plot(
    t: np.ndarray,
    l: np.ndarray,
    r: np.ndarray,
    freq_mhz: np.ndarray,
    *,
    base_c11: float,
    base_c12: float,
    base_c44: float,
    title: str,
    cmap_name: str = "viridis",
    outfile: Optional[str] = None,
    focus_range_mhz: Optional[Tuple[float, float]] = (200.0, 500.0),
    scale: str = "log",
    power_gamma: float = 1.3,
    invalid_mask: Optional[np.ndarray] = None,
    legend_dx: float = 0.03,
    az_values: Optional[np.ndarray] = None,
    az_levels: Optional[List[float]] = None,
    relative_mode: bool = False,
    cbar_label: Optional[str] = None,
    plot_kdg: bool = False,
    base_k: Optional[float] = None,
    base_d: Optional[float] = None,
    base_g: Optional[float] = None,
    rel_max_percent: Optional[float] = None,
    show_baseline_marker: bool = False,
) -> None:
    """Create ternary plot using mpltern."""
    
    # Set up the figure with mpltern
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(111, projection='ternary')

    # Color normalization
    finite_values = freq_mhz[np.isfinite(freq_mhz)]
    norm = None
    vmin = None
    vmax = None
    if relative_mode:
        # Symmetric diverging normalization centered at 0
        if finite_values.size >= 1:
            if rel_max_percent is not None:
                # User-specified maximum percentage (convert to fraction)
                absmax = rel_max_percent / 100.0
            else:
                # Auto-calibrate range for optimal visual sensitivity
                absmax = auto_calibrate_relative_range(finite_values)
                print(f"Auto-calibrated relative range: ±{absmax*100:.1f}%")
            vmin, vmax = -absmax, absmax
            try:
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            except Exception:
                norm = None
    else:
        # Focused on a physically relevant band for absolute frequencies
        if finite_values.size >= 3 and np.min(finite_values) < np.max(finite_values):
            focus_low, focus_high = (focus_range_mhz or (150.0, 1000.0))
            focus_low = max(focus_low, float(np.min(finite_values)))
            focus_high = min(focus_high, float(np.max(finite_values)))
            if focus_low < focus_high:
                vmin, vmax = focus_low, focus_high
            else:
                vmin, vmax = np.percentile(finite_values, [5, 95])
                vmin = float(vmin)
                vmax = float(vmax)

            if scale == "log" and vmin > 0:
                norm = colors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
            elif scale == "power" and power_gamma > 0:
                norm = colors.PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax)

    # Create filled heat map using mpltern's tripcolor
    finite_mask = np.isfinite(freq_mhz)
    
    if np.sum(finite_mask) >= 3:
        # Use mpltern's tripcolor for filled heatmap
        contour = ax.tripcolor(
            t[finite_mask], l[finite_mask], r[finite_mask], 
            freq_mhz[finite_mask], 
            cmap=cmap_name, 
            norm=norm,
            shading='gouraud'  # Smooth shading for better appearance
        )
    else:
        # Not enough data for tripcolor, use scatter
        contour = ax.scatter(
            t[finite_mask], l[finite_mask], r[finite_mask], 
            c=freq_mhz[finite_mask], 
            s=16, 
            cmap=cmap_name, 
            norm=norm,
            edgecolors='none'
        )

    # Overlay invalid/stability-violating region in a distinct light red not in colorbar
    if invalid_mask is not None and np.any(invalid_mask):
        invalid_color = "#f6b0b0"  # light red
        try:
            ax.tripcolor(
                t[invalid_mask], l[invalid_mask], r[invalid_mask],
                np.zeros(int(np.sum(invalid_mask))),
                cmap=colors.ListedColormap([invalid_color]),
                shading='gouraud',
                alpha=0.35,
                zorder=3,
            )
        except Exception:
            ax.scatter(
                t[invalid_mask], l[invalid_mask], r[invalid_mask],
                c=[invalid_color], s=18, alpha=0.35, edgecolors='none', zorder=3,
            )
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=invalid_color, edgecolor='none', label='Mechanically unstable')],
            loc='lower left',
            frameon=True,
            bbox_to_anchor=(legend_dx, -0.01),  # move downward in y by 0.03
            bbox_transform=ax.transAxes,
        )

    # Set up ternary axes
    if plot_kdg:
        ax.set_tlabel('K (GPa)')
        ax.set_llabel('D (GPa)') 
        ax.set_rlabel('G (GPa)')
    else:
        ax.set_tlabel('C11 (GPa)')
        ax.set_llabel('C12 (GPa)') 
        ax.set_rlabel('C44 (GPa)')
    
    # Set tick positions and labels
    tick_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Calculate the actual GPa values for each axis
    if plot_kdg and base_k is not None and base_d is not None and base_g is not None:
        k_values = base_k * (0.5 + tick_positions)
        d_values = base_d * (0.5 + tick_positions)
        g_values = base_g * (0.5 + tick_positions)
    else:
        c11_values = base_c11 * (0.5 + tick_positions)
        c12_values = base_c12 * (0.5 + tick_positions)
        c44_values = base_c44 * (0.5 + tick_positions)
    
    # Set limits and ticks for each axis
    ax.set_tlim(0, 1)
    ax.set_llim(0, 1)
    ax.set_rlim(0, 1)
    
    # Set ticks and labels
    ax.taxis.set_ticks(tick_positions)
    ax.laxis.set_ticks(tick_positions)
    ax.raxis.set_ticks(tick_positions)

    if plot_kdg and base_k is not None and base_d is not None and base_g is not None:
        ax.taxis.set_ticklabels([format_gpa(val) for val in k_values])
        ax.laxis.set_ticklabels([format_gpa(val) for val in d_values])
        ax.raxis.set_ticklabels([format_gpa(val) for val in g_values])
    else:
        ax.taxis.set_ticklabels([format_gpa(val) for val in c11_values])
        ax.laxis.set_ticklabels([format_gpa(val) for val in c12_values])
        ax.raxis.set_ticklabels([format_gpa(val) for val in c44_values])

    # Add colorbar in an inset axis to the right to avoid overlap
    cax = inset_axes(
        ax,
        width="4.5%",
        height="80%",
        loc='center left',
        bbox_to_anchor=(1.04, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = fig.colorbar(contour, cax=cax)
    label_text = cbar_label or "Predicted SAW frequency (MHz)"
    cbar.set_label(label_text, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Improve colorbar tick formatting
    if vmin is not None and vmax is not None:
        # Ensure colorbar range focuses on the interesting band; clip outliers visually
        contour.set_clim(vmin, vmax)
        # Put extend markers so users see values outside the range
        try:
            cbar.extend = 'both'
        except Exception:
            pass

    # Improve colorbar tick formatting
    from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
    if scale == "log" and vmin is not None and vmin > 0:
        locator = LogLocator()
        cbar.locator = locator
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        cbar.locator = MaxNLocator(nbins=6, prune=None)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Overlay iso-A_Z lines with high contrast (black lines with white halo)
    # A_Z = 2*C44 / (C11 - C12)
    if az_values is not None:
        az_mask = np.isfinite(az_values)
        if np.sum(az_mask) >= 3:
            levels = az_levels or [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
            colors_list = ['k'] * len(levels)
            widths_list = [1.2] * len(levels)
            for i, lvl in enumerate(levels):
                if abs(lvl - 1.0) < 1e-12:
                    widths_list[i] = 2.2
            cs = ax.tricontour(
                t[az_mask], l[az_mask], r[az_mask], az_values[az_mask],
                levels=levels, colors=colors_list, linewidths=widths_list, zorder=5
            )
            # Add white halo around each contour for contrast across the colormap
            try:
                from matplotlib import patheffects as pe
                for coll, lw in zip(cs.collections, widths_list):
                    halo_width = lw + 2.0
                    coll.set_path_effects([
                        pe.Stroke(linewidth=halo_width, foreground='white'),
                        pe.Normal(),
                    ])
                    coll.set_solid_joinstyle('round')
                    coll.set_solid_capstyle('round')
            except Exception:
                pass

            # Label contours with halo for readability
            try:
                texts = ax.clabel(cs, fmt=lambda v: f"A_Z={v:g}", inline=True, fontsize=8, colors='k')
                try:
                    from matplotlib import patheffects as pe
                    for txt in texts:
                        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
                except Exception:
                    pass
            except Exception:
                pass

    # Add baseline marker if requested
    if show_baseline_marker:
        # Calculate baseline position in ternary coordinates
        if plot_kdg and base_k is not None and base_d is not None and base_g is not None:
            # For K/D/G plot, baseline is at (0.5, 0.5, 0.5) in normalized coordinates
            # which corresponds to the base values
            baseline_t = 0.5  # K axis
            baseline_l = 0.5  # D axis
            baseline_r = 0.5  # G axis
        else:
            # For C11/C12/C44 plot
            baseline_t = 0.5  # C11 axis
            baseline_l = 0.5  # C12 axis
            baseline_r = 0.5  # C44 axis
        
        # Plot baseline marker with white border for visibility
        ax.scatter(baseline_t, baseline_l, baseline_r, 
                  s=300, marker='*', 
                  c='gold', edgecolors='black', linewidths=2.5,
                  zorder=10, label='Baseline')
        
        # Add legend for baseline marker
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    # Set title if provided
    if title:
        ax.set_title(title, pad=20)

    # Use tight layout; no right adjustment needed with inset colorbar
    plt.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ternary diagram of SAW frequency vs C11/C12/C44 variations (±50%).")
    parser.add_argument("--c11_base_gpa", type=float, default=229.0, help="Base C11 in GPa (default: 229.0)")
    parser.add_argument("--c12_base_gpa", type=float, default=119.0, help="Base C12 in GPa (default: 119.0)")
    parser.add_argument("--c44_base_gpa", type=float, default=43.0, help="Base C44 in GPa (default: 43.0)")
    parser.add_argument("--density", type=float, default=6100.0, help="Density in kg/m^3 (default: 6100.0)")
    parser.add_argument("--wavelength_um", type=float, default=8.8, help="Acoustic wavelength in micrometers (default: 8.8)")
    parser.add_argument("--deg", type=float, default=135.0, help="In-plane propagation angle in degrees (default: 135.0)")
    parser.add_argument(
        "--euler_deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("alpha", "beta", "gamma"),
        help="Euler angles in degrees as alpha beta gamma (default: 0 0 0)",
    )
    parser.add_argument("--sampling", type=int, default=400, help="SAWCalculator sampling (400/4000/40000). Lower is faster (default: 400)")
    parser.add_argument("--res", type=int, default=30, help="Ternary grid resolution per edge (default: 30)")
    parser.add_argument("--cmap", type=str, default="custom", help="Matplotlib colormap (default: custom with #2A33C3 blue and #8F2D56 red)")
    parser.add_argument("--focus_low_mhz", type=float, default=200.0, help="Focus range lower bound in MHz (default: 200.0)")
    parser.add_argument("--focus_high_mhz", type=float, default=500.0, help="Focus range upper bound in MHz (default: 500.0)")
    parser.add_argument("--scale", type=str, default="log", choices=["linear", "log", "power"], help="Color normalization scale (default: log)")
    parser.add_argument("--power_gamma", type=float, default=1.3, help="Gamma for PowerNorm when --scale power (default: 1.3)")
    parser.add_argument("--relative", action="store_true", help="Plot normalized change relative to baseline instead of absolute frequency")
    parser.add_argument("--rel_max_percent", type=float, default=None, help="Maximum percentage change for relative mode colorbar (±X%). If not set, auto-calibrates optimal range for visual sensitivity.")
    parser.add_argument(
        "--outfile",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "ternary_saw_frequency.png"),
        help="Path to save the figure (PNG). Use '-' to show instead.",
    )
    parser.add_argument("--title", type=str, default="SAW Frequency vs C11/C12/C44 (±50%)", help="Figure title")
    parser.add_argument("--plot_kdg", action="store_true", help="Plot using K=(C11+2*C12)/3, D=C11-C12, G=C44 axes")
    
    # KS mode arguments
    parser.add_argument("--ks_mode", action="store_true", help="Plot normalized KS metric change vs baseline (requires EBSD and experimental data)")
    parser.add_argument("--ebsd_path", type=str, default=None, help="Path to EBSD CTF file for KS mode")
    parser.add_argument("--experimental_h5", type=str, default=None, help="Path to experimental HDF5 file for KS mode")
    parser.add_argument("--exp_min_mhz", type=float, default=200.0, help="Minimum experimental frequency in MHz (default: 200.0)")
    parser.add_argument("--exp_max_mhz", type=float, default=400.0, help="Maximum experimental frequency in MHz (default: 400.0)")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers for EBSD grain calculations (default: 32)")
    
    # Cache management arguments
    parser.add_argument("--cache_file", type=str, default="saw_cache.pkl", help="Path to cache file (default: saw_cache.pkl for SAW, .json for KS)")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--clear_cache", action="store_true", help="Clear the cache and exit")
    parser.add_argument("--cache_stats", action="store_true", help="Show cache statistics and exit")
    
    # Display options
    parser.add_argument("--show_title", action="store_true", help="Show plot title (default: no title)")
    parser.add_argument("--show_baseline", action="store_true", help="Show baseline marker on ternary plot")

    args = parser.parse_args()

    # Handle cache management commands
    if args.clear_cache:
        # Try to detect cache type from filename or clear both
        if 'ks' in args.cache_file.lower():
            cache = get_ks_cache(args.cache_file)
        else:
            cache = get_saw_cache(args.cache_file)
        cache.clear_cache()
        return
    
    if args.cache_stats:
        # Try to detect cache type from filename
        if 'ks' in args.cache_file.lower():
            cache = get_ks_cache(args.cache_file)
            cache_type = "KS"
        else:
            cache = get_saw_cache(args.cache_file)
            cache_type = "SAW"
        stats = cache.cache_stats()
        print(f"{cache_type} Cache Statistics:")
        print(f"  File: {stats['cache_file']}")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Finite results: {stats['finite_entries']}")
        print(f"  NaN results: {stats['nan_entries']}")
        if stats['total_entries'] > 0:
            print(f"  Success rate: {stats['finite_entries']/stats['total_entries']*100:.1f}%")
        return

    # Validate mutually exclusive modes
    if args.ks_mode and args.relative:
        parser.error("--ks_mode and --relative are mutually exclusive")
    
    # Validate KS mode requirements
    if args.ks_mode:
        if not EBSD_AVAILABLE:
            parser.error("--ks_mode requires EBSD functionality from sawbench")
        if not args.ebsd_path:
            parser.error("--ks_mode requires --ebsd_path")
        if not args.experimental_h5:
            parser.error("--ks_mode requires --experimental_h5")
        # Update default title and output file for KS mode if not explicitly set
        if args.title == "SAW Frequency vs C11/C12/C44 (±50%)":
            args.title = "KS Metric vs K/D/G (±50%)" if args.plot_kdg else "KS Metric vs C11/C12/C44 (±50%)"
        if args.outfile == os.path.join(os.path.dirname(__file__), "results", "ternary_saw_frequency.png"):
            args.outfile = os.path.join(os.path.dirname(__file__), "results", "ternary_ks_sensitivity.png")

    base_c11 = float(args.c11_base_gpa)
    base_c12 = float(args.c12_base_gpa)
    base_c44 = float(args.c44_base_gpa)
    plot_kdg = bool(args.plot_kdg)
    ks_mode = bool(args.ks_mode)
    density = float(args.density)
    wavelength_m = float(args.wavelength_um) * 1e-6
    deg_inplane = float(args.deg)
    euler_deg = tuple(float(x) for x in args.euler_deg)
    euler_angles_rad = tuple(np.radians(euler_deg))
    sampling = int(args.sampling)
    res = int(args.res)
    cmap_name = str(args.cmap)
    
    # Create custom colormap if requested
    if cmap_name == "custom":
        colors_list = ['#2A33C3', 'white', '#8F2D56']  # blue to white to red
        n_bins = 256
        cmap_name = LinearSegmentedColormap.from_list('custom_diverging', colors_list, N=n_bins)
    
    outfile = None if args.outfile == "-" else str(args.outfile)
    
    # For KS mode, use a focused range of 0-1 and linear scale by default
    if ks_mode:
        focus_range = (0.0, 1.0)
        scale = "linear"  # Linear scale for KS metric
    else:
        focus_range = (float(args.focus_low_mhz), float(args.focus_high_mhz))
        scale = str(args.scale)
    
    power_gamma = float(args.power_gamma)
    rel_max_percent = args.rel_max_percent

    # Build ternary grid
    t, l, r = generate_ternary_grid(res)

    # Map to constants (optionally via K/D/G axes)
    if plot_kdg:
        base_k = (base_c11 + 2.0 * base_c12) / 3.0
        base_d = (base_c11 - base_c12)
        base_g = base_c44

        K_arr = base_k * (0.5 + t)
        D_arr = base_d * (0.5 + l)
        G_arr = base_g * (0.5 + r)

        # Convert K, D, G back to C11, C12, C44 for physics and SAW calc
        C11_arr = (3.0 * K_arr + D_arr) / 2.0
        C12_arr = (3.0 * K_arr - D_arr) / 2.0
        C44_arr = G_arr
    else:
        base_k = base_d = base_g = None
        C11_arr = base_c11 * (0.5 + t)
        C12_arr = base_c12 * (0.5 + l)
        C44_arr = base_c44 * (0.5 + r)

    # Enforce cubic stability (Zener-related) conditions: C11 - C12 > 0, C44 > 0, C11 + 2*C12 > 0
    valid_mask = (C11_arr - C12_arr > 0.0) & (C44_arr > 0.0) & (C11_arr + 2.0 * C12_arr > 0.0)
    invalid_mask = ~valid_mask

    # Compute Zener anisotropy A_Z = 2*C44 / (C11 - C12) on valid points; set invalid to NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        az_values = np.full_like(t, np.nan, dtype=float)
        denom = (C11_arr - C12_arr)
        az_values[valid_mask] = 2.0 * C44_arr[valid_mask] / denom[valid_mask]

    # Main computation: either frequency-based or KS metric-based
    use_cache = not bool(args.no_cache)
    cache_file = str(args.cache_file)
    freq_mhz = np.full_like(t, np.nan, dtype=float)
    valid_indices = np.where(valid_mask)[0]
    
    if ks_mode:
        # KS mode: Load EBSD and experimental data, compute KS metrics
        print("--- KS Mode: Loading EBSD map ---")
        ebsd_map = load_ebsd_map(
            args.ebsd_path,
            "OxfordText",  # Hardcoded for CTF files
            5.0,  # boundary_def_deg
            10   # min_grain_px
        )
        if not ebsd_map:
            raise RuntimeError(f"Failed to load EBSD map from {args.ebsd_path}")
        
        print("--- KS Mode: Loading experimental data ---")
        exp_mhz = load_experimental_freqs_mhz(
            args.experimental_h5,
            args.exp_min_mhz,
            args.exp_max_mhz
        )
        if exp_mhz.size == 0:
            raise RuntimeError(f"No experimental frequencies loaded from {args.experimental_h5}")
        print(f"Loaded {exp_mhz.size} experimental frequency measurements")
        
        # Compute baseline KS metric
        print("--- KS Mode: Computing baseline KS metric ---")
        baseline_props_pa = {
            'formula': 'Cubic',
            'C11': base_c11 * 1e9,
            'C12': base_c12 * 1e9,
            'C44': base_c44 * 1e9,
            'density': density,
            'crystal_class': 'cubic',
        }
        baseline_pred_mhz = predict_freqs_mhz_for_material(
            ebsd_map, baseline_props_pa, wavelength_m, deg_inplane,
            sampling, 0, args.num_workers
        )
        baseline_ks = compute_ks_metric(exp_mhz, baseline_pred_mhz)
        baseline_str = f"{baseline_ks:.4f}" if np.isfinite(baseline_ks) else "N/A"
        print(f"Baseline KS metric: {baseline_str}")
        
        # Initialize KS cache
        if use_cache:
            ks_cache = get_ks_cache(cache_file)
            ebsd_hash = generate_ebsd_hash(args.ebsd_path)
            exp_hash = generate_exp_hash(args.experimental_h5, args.exp_min_mhz, args.exp_max_mhz)
            stats = ks_cache.cache_stats()
            print(f"KS cache loaded: {stats['total_entries']} entries from {cache_file}")
        
        # Compute KS metric for each valid K/D/G combination
        print(f"--- KS Mode: Computing KS metrics for {len(valid_indices)} grid points ---")
        cache_hits = 0
        for idx in tqdm(valid_indices, desc="KS metrics"):
            C11_gpa = float(C11_arr[idx])
            C12_gpa = float(C12_arr[idx])
            C44_gpa = float(C44_arr[idx])
            
            # Check cache first
            if use_cache:
                cached_ks = ks_cache.get(C11_gpa, C12_gpa, C44_gpa, ebsd_hash, exp_hash)
                if cached_ks is not None:
                    freq_mhz[idx] = cached_ks
                    cache_hits += 1
                    continue
            
            # Compute if not cached
            material_props_pa = {
                'formula': 'Cubic',
                'C11': C11_gpa * 1e9,
                'C12': C12_gpa * 1e9,
                'C44': C44_gpa * 1e9,
                'density': density,
                'crystal_class': 'cubic',
            }
            pred_mhz = predict_freqs_mhz_for_material(
                ebsd_map, material_props_pa, wavelength_m, deg_inplane,
                sampling, 0, args.num_workers
            )
            ks_value = compute_ks_metric(exp_mhz, pred_mhz)
            freq_mhz[idx] = ks_value
            
            # Cache the result with K/D/G values
            if use_cache:
                K, D, G = compute_kdg_from_c11c12c44(C11_gpa, C12_gpa, C44_gpa)
                ks_cache.set(C11_gpa, C12_gpa, C44_gpa, ebsd_hash, exp_hash, ks_value, K, D, G)
        
        if use_cache:
            print(f"Cache hits: {cache_hits}/{len(valid_indices)} ({cache_hits/len(valid_indices)*100:.1f}%)")
            # Flush any remaining pending writes
            ks_cache.flush()
            print(f"Cache flushed to {cache_file}")
        
        # Plot raw KS metric (ranges from 0 to 1)
        # Lower KS is better (distributions are more similar), so use reversed colormap
        cbar_label = "Kolmogorov-Smirnov metric"
        relative_mode = False  # Use absolute scale for KS metric
        
        # Use a reversed viridis colormap (purple=low/good, yellow=high/bad)
        if not isinstance(cmap_name, str):
            pass  # Keep custom colormap if already set
        elif cmap_name == "custom":
            # Use green (good) to red (bad) colormap for KS
            colors_list = ['#2A9D8F', '#E9C46A', '#E76F51']  # green to yellow to red
            n_bins = 256
            cmap_name = LinearSegmentedColormap.from_list('ks_colormap', colors_list, N=n_bins)
        else:
            cmap_name = cmap_name + "_r"  # Reverse the colormap
        
    else:
        # Original frequency-based mode
        if use_cache:
            cache = get_saw_cache(cache_file)
            stats = cache.cache_stats()
            print(f"--- Computing SAW frequencies for {len(valid_indices)} grid points ---")
            print(f"Using cache: {stats['total_entries']} entries loaded")
        else:
            print(f"--- Computing SAW frequencies for {len(valid_indices)} grid points (no cache) ---")
        
        for idx in tqdm(valid_indices, desc="SAW frequencies"):
            freq_mhz[idx] = compute_saw_frequency_mhz(
                float(C11_arr[idx]), float(C12_arr[idx]), float(C44_arr[idx]),
                density_kg_m3=density,
                euler_angles_rad=euler_angles_rad,
                deg_inplane=deg_inplane,
                sampling=sampling,
                wavelength_m=wavelength_m,
                use_cache=use_cache,
                cache_file=cache_file,
            )

        # Baseline frequency at the base constants
        baseline_freq = compute_saw_frequency_mhz(
            base_c11, base_c12, base_c44,
            density_kg_m3=density,
            euler_angles_rad=euler_angles_rad,
            deg_inplane=deg_inplane,
            sampling=sampling,
            wavelength_m=wavelength_m,
            use_cache=use_cache,
            cache_file=cache_file,
        )
        baseline_str = f"{baseline_freq:.1f} MHz" if np.isfinite(baseline_freq) else "N/A"

        # Optionally convert to relative normalized change vs baseline
        relative_mode = bool(args.relative)
        cbar_label = None
        if relative_mode and np.isfinite(baseline_freq) and baseline_freq != 0.0:
            # Use fractional change: (f - f0)/f0
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_rel = (freq_mhz - baseline_freq) / baseline_freq
            # If all NaN or zero spread, keep original to avoid degenerate plot
            if np.any(np.isfinite(freq_rel)):
                freq_mhz = freq_rel
                cbar_label = "Normalized change Δf/f₀"
                # Use custom diverging colormap with specified blue and red colors
                if not isinstance(cmap_name, str) or cmap_name != "custom":
                    colors_list = ['#2A33C3', 'white', '#8F2D56']  # blue to white to red
                    n_bins = 256
                    cmap_name = LinearSegmentedColormap.from_list('custom_diverging', colors_list, N=n_bins)

    # Build title if requested
    if args.show_title:
        if plot_kdg:
            if ks_mode:
                title = (
                    args.title
                    + f"\nBase: K={(base_k if base_k is not None else 0):.1f} GPa, D={(base_d if base_d is not None else 0):.1f} GPa, G={(base_g if base_g is not None else 0):.1f} GPa; density={density:.0f} kg/m³; λ={args.wavelength_um:.2f} µm"
                    + f"\nψ={deg_inplane:.1f}°; baseline KS₀={baseline_str}"
                )
            else:
                title = (
                    args.title
                    + f"\nBase: K={(base_k if base_k is not None else 0):.1f} GPa, D={(base_d if base_d is not None else 0):.1f} GPa, G={(base_g if base_g is not None else 0):.1f} GPa; density={density:.0f} kg/m³; λ={args.wavelength_um:.2f} µm"
                    + f"\nEuler=(α={euler_deg[0]:.1f}°, β={euler_deg[1]:.1f}°, γ={euler_deg[2]:.1f}°); baseline f₀={baseline_str}"
                )
        else:
            if ks_mode:
                title = (
                    args.title
                    + f"\nBase: C11={base_c11:.1f} GPa, C12={base_c12:.1f} GPa, C44={base_c44:.1f} GPa; density={density:.0f} kg/m³; λ={args.wavelength_um:.2f} µm"
                    + f"\nψ={deg_inplane:.1f}°; baseline KS₀={baseline_str}"
                )
            else:
                title = (
                    args.title
                    + f"\nBase: C11={base_c11:.1f} GPa, C12={base_c12:.1f} GPa, C44={base_c44:.1f} GPa; density={density:.0f} kg/m³; λ={args.wavelength_um:.2f} µm"
                    + f"\nEuler=(α={euler_deg[0]:.1f}°, β={euler_deg[1]:.1f}°, γ={euler_deg[2]:.1f}°); baseline f₀={baseline_str}"
                )
    else:
        title = ""  # No title

    create_ternary_plot(
        t, l, r, freq_mhz,
        base_c11=base_c11,
        base_c12=base_c12,
        base_c44=base_c44,
        title=title,
        cmap_name=cmap_name,
        outfile=outfile,
        focus_range_mhz=focus_range,
        scale=scale,
        power_gamma=power_gamma,
        invalid_mask=invalid_mask,
        az_values=az_values,
        az_levels=None,
        relative_mode=relative_mode,
        cbar_label=cbar_label,
        plot_kdg=plot_kdg,
        base_k=base_k,
        base_d=base_d,
        base_g=base_g,
        rel_max_percent=rel_max_percent,
        show_baseline_marker=args.show_baseline,
    )
    
    # Show final cache statistics
    if use_cache:
        if ks_mode:
            cache = get_ks_cache(cache_file)
        else:
            cache = get_saw_cache(cache_file)
        stats = cache.cache_stats()
        print(f"\nFinal cache statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Finite results: {stats['finite_entries']}")
        print(f"  NaN results: {stats['nan_entries']}")
        if stats['total_entries'] > 0:
            print(f"  Success rate: {stats['finite_entries']/stats['total_entries']*100:.1f}%")
        print(f"  Cache file: {stats['cache_file']}")


if __name__ == "__main__":
    main()