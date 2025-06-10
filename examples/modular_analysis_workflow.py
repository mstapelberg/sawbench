import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm # For experimental data processing loop
import os
from typing import Optional, Tuple # Import the necessary types

# Import from sawbench package
from sawbench import (
    # io
    load_fft_data_from_hdf5,
    load_ebsd_map,
    # grains
    extract_experimental_peak_parameters,
    calculate_saw_frequencies_for_ebsd_grains,
    create_ebsd_saw_frequency_map,
    create_experimental_peak_frequency_map,
    # materials
    Material,
    # statistics
    calculate_summary_statistics,
    perform_ks_test,
    # plotting
    plot_frequency_histogram,
    plot_frequency_cdfs,
    plot_cdf_difference,
    plot_ebsd_property_map,
    plot_experimental_heatmap
)

# --- Analysis Configuration ---

# Experimental Data Parameters
# use os to see if this path is defined


path1 = '/home/myless/Documents/saw_freq_analysis/fftData.h5' # USER: Please verify path
path2 = '/Users/myless/Dropbox (MIT)/Research/2025/Spring_2025/TGS-Mapping/processed_analysis/v-1_2ti/fftData.h5'

# Check which path exists and assign it to EXP_HDF5_PATH
if os.path.exists(path1):
    EXP_HDF5_PATH = path1
    print(f"Using experimental data path: {EXP_HDF5_PATH}")
elif os.path.exists(path2):
    EXP_HDF5_PATH = path2
    print(f"Using fallback experimental data path: {EXP_HDF5_PATH}")
else:
    # Handle the case where neither path exists
    EXP_HDF5_PATH = None
    print(f"Error: Neither experimental data path exists.")
    print(f"Tried: {path1}")
    print(f"Tried: {path2}")
    # You might want to exit or raise an error here depending on desired behavior
    # For now, we'll let the processing function handle the None case.

N_PEAKS_TO_EXTRACT_EXP = 3
N_EXP_PEAKS_FOR_HIST = 1
FILTER_EXP_MIN_MHZ = 200.0 
FILTER_EXP_MAX_MHZ = 400.0 

# EBSD Data Parameters
EBSD_DATA_PATH = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected" # USER: Please verify path
EBSD_DATA_TYPE = "OxfordText" 
EBSD_BOUNDARY_DEF_ANGLE = 5.0
EBSD_MIN_GRAIN_PX_SIZE = 10

# Material and SAW Calculation Parameters
WAVELENGTH_M = 8.8e-6 
MATERIAL_PROPS = {
    'formula': 'V', 'C11': 229e9, 'C12': 119e9, 
    'C44': 43e9, 'density': 6110, 'crystal_class': 'cubic'
}
SAW_CALC_ANGLE_DEG = 135 # Degrees for SAW calculator (previously 180 radians)

# Plotting Parameters
HIST_XMIN_MHZ_PLOT = 250.0 
HIST_XMAX_MHZ_PLOT = 350.0
COMMON_XLIM_MHZ = (HIST_XMIN_MHZ_PLOT, HIST_XMAX_MHZ_PLOT) if HIST_XMIN_MHZ_PLOT is not None and HIST_XMAX_MHZ_PLOT is not None else None


def process_experimental_data(
    hdf5_path: str, 
    num_peaks_to_extract: int,
    num_top_peaks_for_dist: int,
    filter_min_mhz: Optional[float],
    filter_max_mhz: Optional[float]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads and processes experimental FFT data to extract dominant frequencies and create peak map."""
    print("--- Processing Experimental FFT Data ---")
    fft_data_tuple = load_fft_data_from_hdf5(hdf5_path)
    if not fft_data_tuple:
        print("Failed to load experimental FFT data. Returning empty arrays.")
        return np.array([]), None, None, None

    exp_freq_axis, _, _, exp_amplitude_data, (Ny_exp, Nx_exp) = fft_data_tuple
    print(f"Experimental data grid: {Ny_exp} (Y) x {Nx_exp} (X)")
    
    # Create experimental peak frequency map
    filter_min_hz = filter_min_mhz * 1e6 if filter_min_mhz is not None else None
    filter_max_hz = filter_max_mhz * 1e6 if filter_max_mhz is not None else None
    
    exp_peak_map_hz, x_coords, y_coords = create_experimental_peak_frequency_map(
        exp_freq_axis, exp_amplitude_data, (Ny_exp, Nx_exp),
        num_peaks_to_extract=num_peaks_to_extract,
        filter_min_hz=filter_min_hz,
        filter_max_hz=filter_max_hz
    )
    
    # Extract dominant frequencies for statistical analysis (existing code)
    dominant_freq_list = []
    print("Extracting dominant frequencies for statistical analysis...")
    for iy in tqdm(range(Ny_exp), desc="Processing exp data Y rows"):
        for ix in range(Nx_exp):
            amp_trace = exp_amplitude_data[:, iy, ix]
            peak_params_all = extract_experimental_peak_parameters(
                exp_freq_axis, amp_trace, 
                num_peaks_to_extract=num_peaks_to_extract
            )
            valid_peaks_mask = ~np.isnan(peak_params_all[:, 0])
            if np.any(valid_peaks_mask):
                actual_valid_params = peak_params_all[valid_peaks_mask, :]
                sorted_amp_indices = np.argsort(actual_valid_params[:, 0])[::-1]
                num_to_add = min(num_top_peaks_for_dist, len(sorted_amp_indices))
                for i in range(num_to_add):
                    mu_to_add = actual_valid_params[sorted_amp_indices[i], 1]
                    if pd.notna(mu_to_add):
                        dominant_freq_list.append(mu_to_add)
    
    dominant_freq_array_hz = np.array(dominant_freq_list)
    print(f"Extracted {len(dominant_freq_array_hz)} experimental peak frequencies (raw).")

    if dominant_freq_array_hz.size > 0 and filter_min_mhz is not None and filter_max_mhz is not None:
        min_f_hz, max_f_hz = filter_min_mhz * 1e6, filter_max_mhz * 1e6
        original_count = len(dominant_freq_array_hz)
        dominant_freq_array_hz = dominant_freq_array_hz[
            (dominant_freq_array_hz >= min_f_hz) & (dominant_freq_array_hz <= max_f_hz)
        ]
        print(f"Filtered exp. frequencies to [{filter_min_mhz:.1f} - {filter_max_mhz:.1f}] MHz. Kept {len(dominant_freq_array_hz)}/{original_count}.")
    
    return dominant_freq_array_hz, exp_peak_map_hz, x_coords, y_coords


def process_ebsd_data(
    ebsd_path: str, 
    data_type: str, 
    boundary_def: float, 
    min_grain_size: int,
    material_properties: dict, 
    lambda_saw_m: float,
    saw_angle_deg: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional["defdap.ebsd.Map"]]:
    """Loads EBSD, calculates SAW frequencies for grains, and creates a frequency map."""
    print("\n--- Processing EBSD Data ---")
    ebsd_map = load_ebsd_map(ebsd_path, data_type, boundary_def, min_grain_size)
    if not ebsd_map:
        print("Failed to load EBSD data. Returning Nones.")
        return None, None, None

    print(f"EBSD map loaded. Grains: {len(ebsd_map.grainList)}")
    material = Material(**material_properties)
    
    df_ebsd_grains = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=material,
        wavelength=lambda_saw_m,
        saw_calc_angle_deg=saw_angle_deg
    )
    
    predicted_freq_array_hz = np.array([])
    if not df_ebsd_grains.empty and 'Peak SAW Frequency (Hz)' in df_ebsd_grains.columns:
        predicted_freq_array_hz = df_ebsd_grains['Peak SAW Frequency (Hz)'].dropna().to_numpy()
        print(f"Calculated {len(predicted_freq_array_hz)} predicted EBSD frequencies.")

    saw_freq_map_hz = None
    if not df_ebsd_grains.empty:
        saw_freq_map_hz = create_ebsd_saw_frequency_map(ebsd_map, df_ebsd_grains)
        if saw_freq_map_hz is not None:
            print(f"EBSD SAW frequency map generated with shape: {saw_freq_map_hz.shape}")
        else:
            print("Failed to generate EBSD SAW frequency map image.")
            
    return predicted_freq_array_hz, saw_freq_map_hz, ebsd_map


if __name__ == "__main__":
    # 1. Process Experimental Data
    exp_dominant_freqs_hz, exp_peak_map_hz, exp_x_coords, exp_y_coords = process_experimental_data(
        EXP_HDF5_PATH, 
        N_PEAKS_TO_EXTRACT_EXP, 
        N_EXP_PEAKS_FOR_HIST,
        FILTER_EXP_MIN_MHZ,
        FILTER_EXP_MAX_MHZ
    )
    exp_dominant_freqs_mhz = exp_dominant_freqs_hz / 1e6
    exp_peak_map_mhz = exp_peak_map_hz / 1e6 if exp_peak_map_hz is not None else None

    # 2. Process EBSD Data
    pred_ebsd_freqs_hz, ebsd_saw_freq_map_hz, ebsd_map_obj = process_ebsd_data(
        EBSD_DATA_PATH, EBSD_DATA_TYPE, EBSD_BOUNDARY_DEF_ANGLE, EBSD_MIN_GRAIN_PX_SIZE,
        MATERIAL_PROPS, WAVELENGTH_M, SAW_CALC_ANGLE_DEG
    )
    pred_ebsd_freqs_mhz = pred_ebsd_freqs_hz / 1e6 if pred_ebsd_freqs_hz is not None else np.array([])
    ebsd_saw_freq_map_mhz = ebsd_saw_freq_map_hz / 1e6 if ebsd_saw_freq_map_hz is not None else None
    
    # 3. Perform Statistical Analysis
    print("\n--- Statistical Analysis ---")
    exp_stats = calculate_summary_statistics(exp_dominant_freqs_mhz, "Experimental SAW Frequencies")
    pred_stats = calculate_summary_statistics(pred_ebsd_freqs_mhz, "Predicted EBSD SAW Frequencies")
    
    if exp_dominant_freqs_mhz.size > 0 and pred_ebsd_freqs_mhz.size > 0:
        perform_ks_test(exp_dominant_freqs_mhz, pred_ebsd_freqs_mhz, "Experimental", "Predicted EBSD")

    # 4. Plotting
    print("\n--- Generating Plots ---")
    
    num_main_plots = 0
    if exp_dominant_freqs_mhz.size > 0: num_main_plots +=1
    if pred_ebsd_freqs_mhz.size > 0: num_main_plots +=1
    if exp_dominant_freqs_mhz.size > 0 and pred_ebsd_freqs_mhz.size > 0 : num_main_plots +=2 # CDF and CDF diff
    if ebsd_saw_freq_map_mhz is not None: num_main_plots +=1
    if exp_peak_map_mhz is not None: num_main_plots += 1  # Add experimental heatmap
        
    if num_main_plots == 0:
        print("No data to plot. Exiting.")
        exit()

    # Determine figure layout (example: try to fit in 2 rows)
    ncols = 3 if num_main_plots > 3 else num_main_plots 
    nrows = (num_main_plots + ncols - 1) // ncols 
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
    axs_flat = axs.flatten()
    plot_idx = 0

    # Plot Experimental Histogram
    if exp_dominant_freqs_mhz.size > 0 and plot_idx < len(axs_flat):
        plot_frequency_histogram(
            axs_flat[plot_idx], exp_dominant_freqs_mhz, 
            'Experimental SAW Frequencies', color='blue', bins=50, xlim=COMMON_XLIM_MHZ
        )
        plot_idx += 1

    # Plot Predicted EBSD Histogram
    if pred_ebsd_freqs_mhz.size > 0 and plot_idx < len(axs_flat):
        plot_frequency_histogram(
            axs_flat[plot_idx], pred_ebsd_freqs_mhz, 
            'Predicted EBSD SAW Frequencies', color='green', bins=20, xlim=COMMON_XLIM_MHZ
        )
        plot_idx += 1

    # Plot CDFs
    datasets_for_cdf = []
    labels_for_cdf = []
    colors_for_cdf = []
    exp_sorted_mhz, pred_sorted_mhz = None, None

    if exp_dominant_freqs_mhz.size > 0:
        exp_sorted_mhz = np.sort(exp_dominant_freqs_mhz)
        datasets_for_cdf.append(exp_sorted_mhz)
        labels_for_cdf.append("Experimental")
        colors_for_cdf.append("blue")
    if pred_ebsd_freqs_mhz.size > 0:
        pred_sorted_mhz = np.sort(pred_ebsd_freqs_mhz)
        datasets_for_cdf.append(pred_sorted_mhz)
        labels_for_cdf.append("Predicted EBSD")
        colors_for_cdf.append("green")

    if datasets_for_cdf and plot_idx < len(axs_flat):
        plot_frequency_cdfs(
            axs_flat[plot_idx], datasets_for_cdf, labels_for_cdf, colors_for_cdf, xlim=COMMON_XLIM_MHZ
        )
        plot_idx += 1
        
    # Plot CDF Difference
    if exp_sorted_mhz is not None and pred_sorted_mhz is not None and plot_idx < len(axs_flat):
        plot_cdf_difference(
            axs_flat[plot_idx], exp_sorted_mhz, pred_sorted_mhz, 
            label1="Exp", label2="Pred", xlim=COMMON_XLIM_MHZ
        )
        plot_idx += 1
        
    # Plot EBSD SAW Frequency Map
    if ebsd_saw_freq_map_mhz is not None and ebsd_map_obj is not None and plot_idx < len(axs_flat):
        plot_ebsd_property_map(
            axs_flat[plot_idx], ebsd_saw_freq_map_mhz, 
            'EBSD Predicted SAW Frequency Map (MHz)', 
            cbar_label='Frequency (MHz)',
            step_size_um=ebsd_map_obj.stepSize
        )
        plot_idx += 1

    # Plot Experimental Peak Frequency Heatmap
    if exp_peak_map_mhz is not None and plot_idx < len(axs_flat):
        plot_experimental_heatmap(
            axs_flat[plot_idx], exp_peak_map_mhz, exp_x_coords, exp_y_coords,
            title='Experimental TGS Peak Frequency Map (MHz)',
            cbar_label='Frequency (MHz)',
            vmin=FILTER_EXP_MIN_MHZ if FILTER_EXP_MIN_MHZ else None,
            vmax=FILTER_EXP_MAX_MHZ if FILTER_EXP_MAX_MHZ else None,
            levels=50
        )
        plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(axs_flat)):
        fig.delaxes(axs_flat[i])

    plt.tight_layout()
    plt.show()

    print("\nModular analysis workflow finished.") 