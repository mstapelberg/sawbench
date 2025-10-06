import os
import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.optimize as opt
from tqdm import tqdm
try:
    from tqdm.contrib.concurrent import process_map as _process_map
except Exception:  # pragma: no cover - optional dependency path
    _process_map = None
from typing import TYPE_CHECKING, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .materials import Material # Assuming Material class is in this relative path
from .saw_calculator import SAWCalculator # Assuming SAWCalculator is in this relative path

if TYPE_CHECKING:
    from defdap import ebsd # For type hinting ebsd.Map

# --- Functions to Keep ---

def _gauss(x: np.ndarray, A: float, mu: float, sig_val: float) -> np.ndarray:
    """
    Gaussian function for peak fitting.

    Args:
        x (np.ndarray): Input array (e.g., frequency).
        A (float): Amplitude of the Gaussian.
        mu (float): Mean (center) of the Gaussian.
        sig_val (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Gaussian curve evaluated at x.
    """
    return A * np.exp(-(x - mu)**2 / (2 * sig_val**2))

def extract_experimental_peak_parameters(
    freq_axis: np.ndarray,
    amp_trace: np.ndarray,
    num_peaks_to_extract: int = 3,
    num_candidates: int = 5,
    prominence: float = 0.04,
    min_peak_height_relative: float = 0.1
) -> np.ndarray:
    """
    Finds multiple peaks in an experimental spectrum, fits Gaussians,
    and returns parameters for the specified number of peaks, sorted by frequency.

    Args:
        freq_axis (np.ndarray): Array of frequencies.
        amp_trace (np.ndarray): Array of amplitudes for the spectrum.
        num_peaks_to_extract (int): Number of best peaks to return parameters for.
                                    Defaults to 3.
        num_candidates (int): Number of initial candidate peaks to consider (sorted by prominence).
                              Defaults to 5.
        prominence (float): Prominence for scipy.signal.find_peaks. Defaults to 0.04.
        min_peak_height_relative (float): Minimum peak height relative to the max amplitude.
                                         Defaults to 0.1.

    Returns:
        np.ndarray: An array of shape (num_peaks_to_extract, 3) for [Amplitude, Mu, Sigma].
                    Padded with NaNs if fewer peaks are found.
    """
    default_peak_params = np.full((num_peaks_to_extract, 3), np.nan)
    if amp_trace.size == 0 or np.all(amp_trace <= 1e-9): # Check for empty or effectively zero signal
        return default_peak_params
    
    max_amp = amp_trace.max()
    if max_amp <= 1e-9: # Check if max amplitude is negligible
        return default_peak_params

    min_abs_height = max_amp * min_peak_height_relative
    initial_idx, prop = sig.find_peaks(amp_trace, prominence=prominence, height=min_abs_height)

    if len(initial_idx) == 0:
        return default_peak_params

    # Sort candidates by prominence (descending) and take top N
    sorted_candidate_indices = initial_idx[np.argsort(prop["prominences"])[::-1]][:num_candidates]
    
    fitted_peaks = []
    for p_idx in sorted_candidate_indices:
        # Define a window around the peak for fitting
        sl_start = max(0, p_idx - 5) 
        sl_end = min(len(freq_axis), p_idx + 6)
        sl = slice(sl_start, sl_end)
        
        # Ensure there are enough points for curve_fit (at least p0 size + 1 for good practice)
        if sl_end - sl_start < 4:  # Need at least 3 parameters for _gauss, so >3 points
            continue

        try:
            A0, mu0 = amp_trace[p_idx], freq_axis[p_idx]
            # Robust sigma guess: at least a few data points wide, or a default if too narrow
            freq_window = freq_axis[sl]
            sigma_0_guess = (freq_window[-1] - freq_window[0]) / 6.0 if (freq_window[-1] - freq_window[0]) > 1e-9 else 5e4 # Avoid zero width
            sigma_0_guess = max(1e4, sigma_0_guess) # Ensure a minimum sensible sigma

            popt, _ = opt.curve_fit(_gauss, freq_window, amp_trace[sl], 
                                     p0=(A0, mu0, sigma_0_guess), maxfev=8000)
            
            # Basic sanity checks for fitted parameters
            if popt[0] > 0 and popt[2] > 0 and freq_axis.min() <= popt[1] <= freq_axis.max():
                 fitted_peaks.append({'A': popt[0], 'mu': popt[1], 'sigma': popt[2]})
        except (RuntimeError, ValueError):
            # Fitting failed for this candidate, skip it
            pass 

    if not fitted_peaks:
        return default_peak_params

    # Sort fitted peaks by Amplitude (descending) to select the strongest ones
    fitted_peaks.sort(key=lambda p: p['A'], reverse=True)
    # Select up to N_PEAKS_TO_EXTRACT
    selected_peaks = fitted_peaks[:num_peaks_to_extract]
    # Sort these selected peaks by frequency (Mu) for consistent output order
    selected_peaks.sort(key=lambda p: p['mu'])

    output_params = np.full((num_peaks_to_extract, 3), np.nan)
    for i, peak in enumerate(selected_peaks):
        if i < num_peaks_to_extract: # Should always be true due to slicing, but good check
            output_params[i, 0] = peak['A']
            output_params[i, 1] = peak['mu']
            output_params[i, 2] = peak['sigma']
            
    return output_params


def _compute_peak_saw_freq_for_euler(args_tuple: Tuple[Material, np.ndarray, float, float, int, int]) -> float:
    """Worker function to compute peak SAW frequency (Hz) for a given Euler orientation.

    Args tuple:
        (material, euler_angles_rad, wavelength, saw_calc_angle_deg, saw_calc_sampling, saw_calc_psaw)

    Returns:
        float: Peak SAW frequency (Hz), or NaN on failure
    """
    material, euler_angles_rad, wavelength, saw_calc_angle_deg, saw_calc_sampling, saw_calc_psaw = args_tuple
    try:
        calculator = SAWCalculator(material, euler_angles_rad)
        v_mps, _, _ = calculator.get_saw_speed(
            saw_calc_angle_deg,
            sampling=saw_calc_sampling,
            psaw=saw_calc_psaw
        )
        if v_mps and len(v_mps) > 0 and pd.notna(v_mps[0]):
            return float(v_mps[0] / wavelength)
    except Exception:
        pass
    return float('nan')


def calculate_saw_frequencies_for_ebsd_grains(
    ebsd_map_obj: "ebsd.Map",
    material: Material,
    wavelength: float,
    saw_calc_angle_deg: float = 0.0,
    saw_calc_sampling: int = 400,
    saw_calc_psaw: int = 0,
    num_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculates SAW frequencies for each grain in a processed EBSD map from defdap.

    Args:
        ebsd_map_obj (defdap.ebsd.Map): Processed EBSD map object from defdap,
                                     with grains identified (ebsd_map_obj.grainList populated).
        material (Material): Material object with elastic constants and density.
        wavelength (float): Wavelength (in meters) for SAW f = v/lambda calculation.
        saw_calc_angle_deg (float): Angle (in degrees) for SAW speed calculation. 
                                  The `deg` parameter in `SAWCalculator.get_saw_speed` expects degrees.
                                  Defaults to 0.0.
        saw_calc_sampling (int): Sampling parameter for SAWCalculator. Defaults to 400.
        saw_calc_psaw (int): Psaw parameter for SAWCalculator. Defaults to 0.
        num_workers (Optional[int]): Number of parallel worker processes to use.
            - If None, checks environment variable 'SAWBENCH_NUM_WORKERS'.
            - If neither is set, defaults to 1 (serial).

    Returns:
        pd.DataFrame: DataFrame containing:
            - "Grain ID": Unique identifier for the grain from defdap.
            - "Euler1 (rad)": Average phi1 Euler angle of the grain (radians).
            - "Euler2 (rad)": Average PHI Euler angle of the grain (radians).
            - "Euler3 (rad)": Average phi2 Euler angle of the grain (radians).
            - "Size (pixels)": Number of pixels in the grain.
            - "Size (um^2)": Area of the grain in square micrometers, based on map step size.
            - "Peak SAW Frequency (Hz)": Calculated peak SAW frequency for the grain (Hz).
                                         NaN if calculation failed.
    """
    grains_data_list = []
    if not ebsd_map_obj.grainList:
        print("Warning: No grains found in ebsd_map_obj.grainList. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "Grain ID", "Euler1 (rad)", "Euler2 (rad)", "Euler3 (rad)", 
            "Size (pixels)", "Size (um^2)", "Peak SAW Frequency (Hz)"
        ])

    def _normalize_bunge_euler_angles(euler_angles_rad: np.ndarray) -> np.ndarray:
        """Normalize Bunge Euler angles to canonical ranges.

        - phi1, phi2 in [0, 2π)
        - PHI in [0, π] with (phi1, PHI, phi2) ~ (phi1+π, 2π-PHI, phi2+π)
        """
        two_pi = 2 * np.pi
        phi1 = float(np.mod(euler_angles_rad[0], two_pi))
        PHI_raw = float(np.mod(euler_angles_rad[1], two_pi))
        phi2 = float(np.mod(euler_angles_rad[2], two_pi))

        if PHI_raw > np.pi:
            PHI = two_pi - PHI_raw
            phi1 = float(np.mod(phi1 + np.pi, two_pi))
            phi2 = float(np.mod(phi2 + np.pi, two_pi))
        else:
            PHI = PHI_raw

        return np.array([phi1, PHI, phi2], dtype=float)

    # Precompute Euler angles and grain meta to avoid pickling heavy EBSD objects
    grains_meta = []  # (grain_id, euler_angles_rad, size_px, size_um2)
    for grain in ebsd_map_obj.grainList:
        grain.calcAverageOri()
        euler_angles_raw = grain.refOri.eulerAngles()
        euler_angles_rad = _normalize_bunge_euler_angles(euler_angles_raw)
        size_px = len(grain.coordList)
        size_um2 = len(grain.coordList) * (ebsd_map_obj.stepSize**2)
        grains_meta.append((grain.grainID, euler_angles_rad, size_px, size_um2))

    # Determine parallelism: parameter > environment > default(1)
    if num_workers is None:
        num_workers_env = os.getenv("SAWBENCH_NUM_WORKERS")
        if num_workers_env is not None:
            try:
                num_workers = max(1, int(num_workers_env))
            except ValueError:
                num_workers = 1
        else:
            num_workers = 1
    else:
        try:
            num_workers = max(1, int(num_workers))
        except Exception:
            num_workers = 1

    if num_workers == 1:
        # Original serial path with progress bar
        for grain_id, euler_angles_rad, size_px, size_um2 in tqdm(grains_meta, desc="Calculating SAW for EBSD grains", unit="grain"):
            peak_saw_freq_hz = np.nan
            try:
                calculator = SAWCalculator(material, euler_angles_rad)
                v_mps, _, _ = calculator.get_saw_speed(
                    saw_calc_angle_deg,
                    sampling=saw_calc_sampling,
                    psaw=saw_calc_psaw
                )
                if v_mps and len(v_mps) > 0 and pd.notna(v_mps[0]):
                    peak_saw_freq_hz = v_mps[0] / wavelength
            except Exception as e:
                # Preserve prior behavior of reporting errors per grain
                print(f"Error calculating SAW for grain {grain_id} (Euler: {np.rad2deg(euler_angles_rad)}): {e}")

            grains_data_list.append({
                "Grain ID": grain_id,
                "Euler1 (rad)": euler_angles_rad[0],
                "Euler2 (rad)": euler_angles_rad[1],
                "Euler3 (rad)": euler_angles_rad[2],
                "Size (pixels)": size_px,
                "Size (um^2)": size_um2,
                "Peak SAW Frequency (Hz)": peak_saw_freq_hz
            })
    else:
        # Parallel path using tqdm's process_map if available; fallback to ProcessPoolExecutor
        args_list = [
            (material, euler_angles_rad, wavelength, saw_calc_angle_deg, saw_calc_sampling, saw_calc_psaw)
            for (_, euler_angles_rad, _, _) in grains_meta
        ]

        if _process_map is not None:
            peak_freqs = _process_map(
                _compute_peak_saw_freq_for_euler,
                args_list,
                max_workers=num_workers,
                chunksize=1,
                desc="Calculating SAW for EBSD grains"
            )
        else:
            peak_freqs = [np.nan] * len(grains_meta)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for idx, args_tuple in enumerate(args_list):
                    fut = executor.submit(_compute_peak_saw_freq_for_euler, args_tuple)
                    futures[fut] = idx

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Calculating SAW for EBSD grains", unit="grain"):
                    idx = futures[fut]
                    try:
                        peak_freqs[idx] = float(fut.result())
                    except Exception:
                        peak_freqs[idx] = float('nan')

        # Assemble output rows
        for (grain_id, euler_angles_rad, size_px, size_um2), f_hz in zip(grains_meta, peak_freqs):
            grains_data_list.append({
                "Grain ID": grain_id,
                "Euler1 (rad)": euler_angles_rad[0],
                "Euler2 (rad)": euler_angles_rad[1],
                "Euler3 (rad)": euler_angles_rad[2],
                "Size (pixels)": size_px,
                "Size (um^2)": size_um2,
                "Peak SAW Frequency (Hz)": f_hz
            })

    df_grains = pd.DataFrame(grains_data_list)
    return df_grains


def create_ebsd_saw_frequency_map(
    ebsd_map_obj: "ebsd.Map",
    grains_saw_data_df: pd.DataFrame
) -> np.ndarray | None:
    """
    Creates a 2D map of SAW frequencies based on EBSD grain data from defdap.

    The map dimensions are taken from `ebsd_map_obj.shape`.
    Each pixel in the output map is assigned the 'Peak SAW Frequency (Hz)'
    of the grain it belongs to. Pixels not belonging to any grain in
    `grains_saw_data_df` (or if the grain has a NaN frequency) will be NaN.

    Args:
        ebsd_map_obj (defdap.ebsd.Map): The EBSD map object from defdap.
                                     `findGrains()` must have been called to populate
                                     the grain ID map attribute (e.g., `ebsd_map_obj.grainIDMap`).
        grains_saw_data_df (pd.DataFrame): DataFrame returned by
                                        `calculate_saw_frequencies_for_ebsd_grains`,
                                        must contain 'Grain ID' and 'Peak SAW Frequency (Hz)' columns.

    Returns:
        np.ndarray | None: A 2D numpy array (yDim, xDim) representing the SAW frequency map (in Hz).
                           Pixels not belonging to a processed grain will be NaN.
                           Returns None if a suitable grain ID map attribute is not available or
                           if `grains_saw_data_df` is missing required columns.
    """
    grain_map_ids_array = None
    if hasattr(ebsd_map_obj, 'grainIDMap') and ebsd_map_obj.grainIDMap is not None:
        grain_map_ids_array = ebsd_map_obj.grainIDMap
        print("Using ebsd_map_obj.grainIDMap for frequency map.")
    elif hasattr(ebsd_map_obj, 'grains') and isinstance(getattr(ebsd_map_obj, 'grains'), np.ndarray) and getattr(ebsd_map_obj, 'grains').shape == ebsd_map_obj.shape:
        grain_map_ids_array = ebsd_map_obj.grains
        print("Using ebsd_map_obj.grains for frequency map (fallback).")
    
    if grain_map_ids_array is None:
        print("Error: Could not find a suitable grain ID map attribute (tried 'grainIDMap', 'grains') on the EBSD map object.")
        print("Ensure findGrains() was called and successfully populated the map.")
        return None
        
    if not isinstance(grains_saw_data_df, pd.DataFrame) or \
       'Grain ID' not in grains_saw_data_df.columns or \
       'Peak SAW Frequency (Hz)' not in grains_saw_data_df.columns:
        print("Error: grains_saw_data_df must be a DataFrame and contain 'Grain ID' and 'Peak SAW Frequency (Hz)' columns.")
        return None

    # Create a mapping from Grain ID to its SAW frequency
    # Ensure 'Grain ID' is the index for efficient lookup if not already
    if grains_saw_data_df.index.name != 'Grain ID':
        df_indexed = grains_saw_data_df.set_index('Grain ID')
    else:
        df_indexed = grains_saw_data_df
        
    # Use .reindex().values to map frequencies. This handles missing Grain IDs gracefully (assigns NaN).
    # grainIDMap contains the ID of the grain each pixel belongs to.
    # Values in grainIDMap that are not in df_indexed.index will result in NaN.
    # DefDAP grainIDMap uses 0 for non-indexed points / points not in any grain.
    # We should ensure that grain ID 0 (if it appears in grainIDMap) maps to NaN
    # unless it's explicitly in grains_saw_data_df (which is unlikely for a valid grain).
    
    # Get the unique grain IDs from the map (excluding 0, which is typically non-indexed)
    map_grain_ids = np.unique(grain_map_ids_array)
    map_grain_ids_valid = map_grain_ids[map_grain_ids > 0] # Consider only actual grain IDs

    # Create a series for mapping, ensuring all grain IDs from map are present
    # and get their corresponding frequency or NaN if not in the DataFrame
    freq_series = pd.Series(index=map_grain_ids_valid, dtype=float)
    
    # Populate with frequencies from the input DataFrame
    # Common IDs between the map's grains and the DataFrame's grains
    common_ids = df_indexed.index.intersection(map_grain_ids_valid)
    if not common_ids.empty:
        freq_series.loc[common_ids] = df_indexed.loc[common_ids, 'Peak SAW Frequency (Hz)']

    # Now create the output map
    # Initialize with NaNs
    saw_freq_map_array = np.full(grain_map_ids_array.shape, np.nan, dtype=float)

    # Map frequencies using the grainIDMap
    # Iterate through the valid grain IDs found on the map
    for gid in map_grain_ids_valid:
        saw_freq_map_array[grain_map_ids_array == gid] = freq_series.get(gid, np.nan)
                
    print(f"EBSD SAW frequency map created with shape: {saw_freq_map_array.shape}")
    return saw_freq_map_array

def create_experimental_peak_frequency_map(
    freq_axis: np.ndarray,
    amplitude_data: np.ndarray, 
    grid_shape: Tuple[int, int],
    num_peaks_to_extract: int = 3,
    filter_min_hz: Optional[float] = None,
    filter_max_hz: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D peak frequency map from raw FFT amplitude data.
    
    Args:
        freq_axis: 1D frequency array (Hz)
        amplitude_data: 3D array of shape (n_freq, ny, nx) 
        grid_shape: Tuple of (ny, nx) spatial dimensions
        num_peaks_to_extract: Number of peaks to fit per location
        filter_min_hz: Optional minimum frequency filter (Hz)
        filter_max_hz: Optional maximum frequency filter (Hz)
        
    Returns:
        peak_freq_map: 2D array of dominant peak frequencies (Hz)
        x_coords: 1D array of x coordinates (arbitrary units)
        y_coords: 1D array of y coordinates (arbitrary units)
    """
    ny, nx = grid_shape
    peak_freq_map = np.full((ny, nx), np.nan)
    
    print(f"Creating experimental peak frequency map for {ny}x{nx} grid...")
    
    for iy in tqdm(range(ny), desc="Processing experimental FFT grid"):
        for ix in range(nx):
            amp_trace = amplitude_data[:, iy, ix]
            
            # Extract peak parameters for this location
            peak_params = extract_experimental_peak_parameters(
                freq_axis, amp_trace, num_peaks_to_extract=num_peaks_to_extract
            )
            
            # Find valid peaks
            valid_mask = ~np.isnan(peak_params[:, 0])
            if np.any(valid_mask):
                valid_params = peak_params[valid_mask, :]
                # Sort by amplitude (column 0) and take the strongest
                strongest_idx = np.argmax(valid_params[:, 0])
                dominant_freq = valid_params[strongest_idx, 1]  # frequency is column 1
                
                # Apply frequency filter if specified
                if filter_min_hz is not None and filter_max_hz is not None:
                    if filter_min_hz <= dominant_freq <= filter_max_hz:
                        peak_freq_map[iy, ix] = dominant_freq
                else:
                    peak_freq_map[iy, ix] = dominant_freq
    
    # Create coordinate arrays (arbitrary units for now)
    x_coords = np.arange(nx)
    y_coords = np.arange(ny)
    
    return peak_freq_map, x_coords, y_coords