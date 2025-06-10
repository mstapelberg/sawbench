import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import scipy.signal as sig
import scipy.optimize as opt
from defdap import ebsd
import pandas as pd
from scipy.stats import ks_2samp, kurtosis

# Assuming these are in the same parent directory or installed
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator

# Import from sawbench package
from sawbench import (
    load_fft_data_from_hdf5,
    load_ebsd_map,
    extract_experimental_peak_parameters,
    calculate_saw_frequencies_for_ebsd_grains,
    create_ebsd_saw_frequency_map,
    Material,
)

# --- Constants ---
N_PEAKS_TO_EXTRACT = 3  # Number of peaks to extract parameters for from experimental data
HIST_XMIN_MHZ = 250 # Set to a float value (e.g., 200.0) or None for auto
HIST_XMAX_MHZ = 350 # Set to a float value (e.g., 500.0) or None for auto
N_EXP_PEAKS_FOR_HISTOGRAM = 2 # Number of top experimental peaks (by amplitude) to include in analysis (e.g., 1 or 2)
FILTER_EXP_MIN_MHZ = 200.0 # Min frequency in MHz for filtering experimental data, set to None to disable
FILTER_EXP_MAX_MHZ = 400.0 # Max frequency in MHz for filtering experimental data, set to None to disable
WAVELENGTH_M = 8.8e-6  # Assuming same wavelength as before

# --- Helper Functions ---
def _gauss(x, A, mu, sig_val):
    return A * np.exp(-(x - mu)**2 / (2 * sig_val**2))

def refine_exp_peaks(freq, amp, N_candidates=5, prom=0.04, min_peak_height_rel=0.1):
    """
    Finds multiple peaks in the spectrum, fits Gaussians, and returns parameters
    for N_PEAKS_TO_EXTRACT peaks, sorted by frequency (mu), padded with NaNs.
    Returns an array of shape (N_PEAKS_TO_EXTRACT, 3) for [A, mu, sigma].
    """
    default_peak_params = np.full((N_PEAKS_TO_EXTRACT, 3), np.nan)
    if amp.size == 0 or amp.max() <= 1e-9:
        return default_peak_params

    min_abs_height = amp.max() * min_peak_height_rel
    initial_idx, prop = sig.find_peaks(amp, prominence=prom, height=min_abs_height)

    if len(initial_idx) == 0:
        return default_peak_params

    sorted_candidate_indices = initial_idx[np.argsort(prop["prominences"])[::-1]][:N_candidates]
    
    fitted_peaks = []
    for p_idx in sorted_candidate_indices:
        sl = slice(max(0, p_idx - 5), min(len(freq), p_idx + 6))
        try:
            A0, mu0 = amp[p_idx], freq[p_idx]
            sigma_0_guess = max(5e4, (freq[sl][-1] - freq[sl][0]) / 6) 
            popt, pcov = opt.curve_fit(_gauss, freq[sl], amp[sl], p0=(A0, mu0, sigma_0_guess), maxfev=8000)
            if popt[0] > 0 and popt[2] > 0 and freq.min() <= popt[1] <= freq.max():
                 fitted_peaks.append({'A': popt[0], 'mu': popt[1], 'sigma': popt[2]})
        except (RuntimeError, ValueError):
            pass

    if not fitted_peaks:
        return default_peak_params

    fitted_peaks.sort(key=lambda p: p['A'], reverse=True)
    selected_peaks = fitted_peaks[:N_PEAKS_TO_EXTRACT]
    selected_peaks.sort(key=lambda p: p['mu'])

    output_params = np.full((N_PEAKS_TO_EXTRACT, 3), np.nan)
    for i, peak in enumerate(selected_peaks):
        if i < N_PEAKS_TO_EXTRACT:
            output_params[i, 0] = peak['A']
            output_params[i, 1] = peak['mu']
            output_params[i, 2] = peak['sigma']
            
    return output_params

def calculate_predicted_saw_frequency(euler_angles, material):
    try:
        calculator = SAWCalculator(material, euler_angles)
        v, _, _ = calculator.get_saw_speed(0.0, sampling=400, psaw=0)
        wavelength = 8.8e-6  # Assuming same wavelength as before
        return v[0] / wavelength if len(v) > 0 else np.nan
    except Exception:
        return np.nan

# --- Main Script Logic ---
if __name__ == "__main__":
    # 1. Load Experimental FFT Data and Get Fits
    print("--- Loading Experimental FFT Data ---")
    exp_hdf5_path = '/home/myless/Documents/saw_freq_analysis/fftData.h5' # USER: Please verify path
    dominant_exp_freq_list = []
    multi_peak_spectra_count = 0  # Diagnostic counter
    spectra_with_multiple_printed = 0 # Diagnostic print limit

    fft_data_tuple = load_fft_data_from_hdf5(exp_hdf5_path)
    
    if fft_data_tuple:
        exp_freq_axis, _, _, exp_amplitude_data, (Ny_exp, Nx_exp) = fft_data_tuple
        print(f"Experimental data grid: {Ny_exp} (Y) x {Nx_exp} (X)")
        print("Fitting peaks to experimental SAW FFT data...")

        for iy in tqdm(range(Ny_exp), desc="Processing exp data Y rows"):
            for ix in range(Nx_exp):
                amp_trace = exp_amplitude_data[:, iy, ix]
                peak_params_all = extract_experimental_peak_parameters(
                    exp_freq_axis, amp_trace, 
                    num_peaks_to_extract=N_PEAKS_TO_EXTRACT
                )
                
                valid_peaks_mask = ~np.isnan(peak_params_all[:, 0]) 
                if np.any(valid_peaks_mask):
                    actual_valid_params = peak_params_all[valid_peaks_mask, :]
                    
                    if actual_valid_params.shape[0] > 1:
                        multi_peak_spectra_count += 1
                        if spectra_with_multiple_printed < 5:
                            print(f"\nSpectrum (iy={iy}, ix={ix}) found multiple valid peaks ({actual_valid_params.shape[0]}):\n{actual_valid_params}")
                            spectra_with_multiple_printed += 1

                    sorted_amp_indices = np.argsort(actual_valid_params[:, 0])[::-1]
                    num_peaks_to_add = min(N_EXP_PEAKS_FOR_HISTOGRAM, len(sorted_amp_indices))
                    
                    for i in range(num_peaks_to_add):
                        mu_to_add = actual_valid_params[sorted_amp_indices[i], 1] 
                        if pd.notna(mu_to_add):
                            dominant_exp_freq_list.append(mu_to_add)
        
        dominant_exp_freq_array = np.array(dominant_exp_freq_list)
        print(f"\nTotal spectra where extract_experimental_peak_parameters found >1 valid peak: {multi_peak_spectra_count}") 
        print(f"Processed {len(dominant_exp_freq_array)} experimental peak frequencies (using top {N_EXP_PEAKS_FOR_HISTOGRAM} per spectrum).")
        if len(dominant_exp_freq_array) > 0:
            print(f"Min/Max dominant exp freq (raw): {np.nanmin(dominant_exp_freq_array):.2e} / {np.nanmax(dominant_exp_freq_array):.2e} Hz")
        else:
            print("No dominant experimental frequencies extracted (raw).")

        if len(dominant_exp_freq_array) > 0 and FILTER_EXP_MIN_MHZ is not None and FILTER_EXP_MAX_MHZ is not None:
            min_freq_hz = FILTER_EXP_MIN_MHZ * 1e6
            max_freq_hz = FILTER_EXP_MAX_MHZ * 1e6
            original_count = len(dominant_exp_freq_array)
            dominant_exp_freq_array = dominant_exp_freq_array[
                (dominant_exp_freq_array >= min_freq_hz) & (dominant_exp_freq_array <= max_freq_hz)
            ]
            print(f"Filtered experimental frequencies to range [{FILTER_EXP_MIN_MHZ:.1f} MHz - {FILTER_EXP_MAX_MHZ:.1f} MHz].")
            print(f"Retained {len(dominant_exp_freq_array)} peaks out of {original_count}.")
            if len(dominant_exp_freq_array) > 0:
                 print(f"Min/Max dominant exp freq (filtered): {np.nanmin(dominant_exp_freq_array):.2e} / {np.nanmax(dominant_exp_freq_array):.2e} Hz")
            else:
                print("No dominant experimental frequencies remaining after filtering.")

        if len(dominant_exp_freq_array) > 0:
            print(f"DIAGNOSTIC: After filtering, dominant_exp_freq_array has len: {len(dominant_exp_freq_array)}, sum: {np.sum(dominant_exp_freq_array):.2e}")
        else:
            print("DIAGNOSTIC: After filtering, dominant_exp_freq_array is empty.")
    else:
        print("Failed to load experimental FFT data. Skipping experimental analysis.")
        dominant_exp_freq_array = np.array([]) 

    # 2. Load EBSD Data and Calculate Predicted SAW Frequencies
    print("\n--- Loading and Processing EBSD Data ---")
    ebsd_data_path = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected" # USER: Please verify path
    ebsd_data_type = "OxfordText" 
    ebsd_boundary_def_angle = 5.0
    ebsd_min_grain_px_size = 10
    
    predicted_ebsd_freq_array = np.array([]) # Initialize
    ebsd_saw_freq_map = None # Initialize
    df_ebsd_grains = pd.DataFrame() # Initialize

    ebsd_map_object = load_ebsd_map(
        ebsd_data_path, 
        data_type=ebsd_data_type,
        boundary_def=ebsd_boundary_def_angle,
        min_grain_size=ebsd_min_grain_px_size
    )

    if ebsd_map_object:
        print(f"EBSD map loaded successfully. Shape: {ebsd_map_object.shape}, Step: {ebsd_map_object.stepSize}")
        vanadium = Material(formula='V', C11=229e9, C12=119e9, C44=43e9, density=6110, crystal_class='cubic')
        
        print("\n--- Calculating Predicted SAW Frequencies from EBSD Grains ---")
        df_ebsd_grains = calculate_saw_frequencies_for_ebsd_grains(
            ebsd_map_obj=ebsd_map_object,
            material=vanadium,
            wavelength=WAVELENGTH_M
            # saw_calc_angle, saw_calc_sampling, saw_calc_psaw can be added if defaults need changing
        )
        
        if not df_ebsd_grains.empty and 'Peak SAW Frequency (Hz)' in df_ebsd_grains.columns:
            predicted_ebsd_freq_list = df_ebsd_grains['Peak SAW Frequency (Hz)'].dropna().tolist()
            predicted_ebsd_freq_array = np.array(predicted_ebsd_freq_list)
            print(f"Calculated {len(predicted_ebsd_freq_array)} predicted EBSD frequencies from grains.")
            if len(predicted_ebsd_freq_array) > 0:
                print(f"Min/Max predicted EBSD freq: {np.nanmin(predicted_ebsd_freq_array):.2e} / {np.nanmax(predicted_ebsd_freq_array):.2e} Hz")
            else:
                print("No valid EBSD frequencies calculated.")

            # 3. Create EBSD SAW Frequency Map for Visualization
            print("\n--- Creating EBSD SAW Frequency Map ---")
            ebsd_saw_freq_map = create_ebsd_saw_frequency_map(
                ebsd_map_obj=ebsd_map_object,
                grains_saw_data_df=df_ebsd_grains
            )
            if ebsd_saw_freq_map is not None:
                print(f"EBSD SAW frequency map generated with shape: {ebsd_saw_freq_map.shape}")
            else:
                print("Failed to generate EBSD SAW frequency map.")

        else:
            print("No EBSD grain data or frequencies calculated to create map.")
            predicted_ebsd_freq_array = np.array([])
    else:
        print("Failed to load EBSD data. Skipping EBSD-based analysis.")
        predicted_ebsd_freq_array = np.array([])

    # Perform KS test if both datasets have data
    if len(dominant_exp_freq_array) > 0 and len(predicted_ebsd_freq_array) > 0:
        print("\n--- Kolmogorov-Smirnov Test ---")
        ks_statistic, p_value = ks_2samp(dominant_exp_freq_array, predicted_ebsd_freq_array)
        print(f"KS Statistic: {ks_statistic:.4f}")
        print(f"P-value: {p_value:.4g}")
        if p_value < 0.05:
            print("Distributions are statistically different.")
        else:
            print("No statistically significant difference between distributions.")
    else:
        print("\nSkipping KS test as one or both datasets are empty.")

    # Calculate and print summary statistics
    print("\n--- Summary Statistics (MHz) ---")
    if len(dominant_exp_freq_array) > 0:
        exp_freq_mhz = dominant_exp_freq_array / 1e6
        print("Experimental Frequencies:")
        print(f"  Mean: {np.nanmean(exp_freq_mhz):.2f}, Median: {np.nanmedian(exp_freq_mhz):.2f}, Std Dev: {np.nanstd(exp_freq_mhz):.2f}, Kurtosis: {kurtosis(exp_freq_mhz, nan_policy='omit'):.2f}")
    else:
        print("Experimental Frequencies: No data")

    if len(predicted_ebsd_freq_array) > 0:
        pred_freq_mhz = predicted_ebsd_freq_array / 1e6
        print("Predicted EBSD Frequencies:")
        print(f"  Mean: {np.nanmean(pred_freq_mhz):.2f}, Median: {np.nanmedian(pred_freq_mhz):.2f}, Std Dev: {np.nanstd(pred_freq_mhz):.2f}, Kurtosis: {kurtosis(pred_freq_mhz, nan_policy='omit'):.2f}")
    else:
        print("Predicted EBSD Frequencies: No data")

    # 4. Plotting
    print("\n--- Plotting Results ---")
    num_plots = 3 # Start with 3 (2 histograms, 1 CDF)
    if ebsd_saw_freq_map is not None:
        num_plots += 1 # Add EBSD map
    if len(dominant_exp_freq_array) > 0 and len(predicted_ebsd_freq_array) > 0:
        num_plots +=1 # Add CDF diff

    if num_plots <= 3:
        fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    elif num_plots == 4: # 2x2 layout
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    elif num_plots == 5: # Try 2 rows, 3 then 2
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)
        axs = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]), 
               fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]
    else: # Default to a single row if too many, or adjust as needed
        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
    axs_flat = np.array(axs).flatten()
    plot_idx = 0

    common_xlim = []
    if HIST_XMIN_MHZ is not None: common_xlim.append(HIST_XMIN_MHZ)
    if HIST_XMAX_MHZ is not None: common_xlim.append(HIST_XMAX_MHZ)

    # Plot 1: Histogram of Experimental Frequencies
    ax_exp_hist = axs_flat[plot_idx]
    plot_idx +=1
    if len(dominant_exp_freq_array) > 0:
        ax_exp_hist.hist(dominant_exp_freq_array / 1e6, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax_exp_hist.set_title('Experimental SAW Frequencies')
        ax_exp_hist.set_xlabel('Frequency (MHz)')
        ax_exp_hist.set_ylabel('Counts')
        ax_exp_hist.grid(True, alpha=0.3)
        if len(common_xlim) == 2: ax_exp_hist.set_xlim(common_xlim)
    else:
        ax_exp_hist.text(0.5, 0.5, "No experimental data", ha='center', va='center', transform=ax_exp_hist.transAxes)
        ax_exp_hist.set_title('Experimental SAW Frequencies')

    # Plot 2: Histogram of Predicted EBSD Frequencies
    ax_pred_hist = axs_flat[plot_idx]
    plot_idx += 1
    if len(predicted_ebsd_freq_array) > 0:
        ax_pred_hist.hist(predicted_ebsd_freq_array / 1e6, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax_pred_hist.set_title('Predicted EBSD SAW Frequencies')
        ax_pred_hist.set_xlabel('Frequency (MHz)')
        ax_pred_hist.set_ylabel('Counts')
        ax_pred_hist.grid(True, alpha=0.3)
        if len(common_xlim) == 2: ax_pred_hist.set_xlim(common_xlim)
    else:
        ax_pred_hist.text(0.5, 0.5, "No predicted EBSD data", ha='center', va='center', transform=ax_pred_hist.transAxes)
        ax_pred_hist.set_title('Predicted EBSD SAW Frequencies')

    # Plot 3: CDF Plot
    ax_cdf = axs_flat[plot_idx]
    plot_idx += 1
    exp_freq_sorted_mhz, pred_freq_sorted_mhz = None, None
    plotted_cdf = False
    if len(dominant_exp_freq_array) > 0:
        exp_freq_sorted_mhz = np.sort(dominant_exp_freq_array / 1e6)
        exp_cdf = np.arange(1, len(exp_freq_sorted_mhz) + 1) / len(exp_freq_sorted_mhz)
        ax_cdf.plot(exp_freq_sorted_mhz, exp_cdf, color='blue', label='Experimental')
        plotted_cdf = True
    if len(predicted_ebsd_freq_array) > 0:
        pred_freq_sorted_mhz = np.sort(predicted_ebsd_freq_array / 1e6)
        pred_cdf = np.arange(1, len(pred_freq_sorted_mhz) + 1) / len(pred_freq_sorted_mhz)
        ax_cdf.plot(pred_freq_sorted_mhz, pred_cdf, color='green', label='Predicted EBSD')
        plotted_cdf = True
    
    ax_cdf.set_title('Cumulative Distribution Functions (CDFs)')
    ax_cdf.set_xlabel('Frequency (MHz)')
    ax_cdf.set_ylabel('Cumulative Probability')
    ax_cdf.grid(True, alpha=0.3)
    if plotted_cdf: ax_cdf.legend(loc='best')
    if len(common_xlim) == 2: ax_cdf.set_xlim(common_xlim)
    if not plotted_cdf:
        ax_cdf.text(0.5, 0.5, "No data for CDF plot", ha='center', va='center', transform=ax_cdf.transAxes)

    # Plot 4: EBSD SAW Frequency Map (if available)
    if ebsd_saw_freq_map is not None and plot_idx < len(axs_flat):
        ax_ebsd_map = axs_flat[plot_idx]
        plot_idx += 1
        im = ax_ebsd_map.imshow(ebsd_saw_freq_map / 1e6, cmap='viridis', origin='lower', aspect='equal')
        ax_ebsd_map.set_title('EBSD Predicted SAW Frequency Map')
        ax_ebsd_map.set_xlabel('X pixel')
        ax_ebsd_map.set_ylabel('Y pixel')
        cbar = fig.colorbar(im, ax=ax_ebsd_map, label='Frequency (MHz)', fraction=0.046, pad=0.04)
    elif ebsd_saw_freq_map is None and plot_idx < len(axs_flat): # Placeholder if map couldn't be made but slot exists
        ax_ebsd_map = axs_flat[plot_idx]
        plot_idx += 1
        ax_ebsd_map.text(0.5, 0.5, "EBSD map not generated", ha='center', va='center', transform=ax_ebsd_map.transAxes)
        ax_ebsd_map.set_title('EBSD Predicted SAW Frequency Map')

    # Plot 5: CDF Difference Plot (if data available)
    if exp_freq_sorted_mhz is not None and pred_freq_sorted_mhz is not None and plot_idx < len(axs_flat):
        ax_cdf_diff = axs_flat[plot_idx]
        plot_idx += 1
        all_freqs_mhz = np.sort(np.unique(np.concatenate((exp_freq_sorted_mhz, pred_freq_sorted_mhz))))
        
        exp_cdf_full = np.arange(1, len(exp_freq_sorted_mhz) + 1) / len(exp_freq_sorted_mhz)
        pred_cdf_full = np.arange(1, len(pred_freq_sorted_mhz) + 1) / len(pred_freq_sorted_mhz)

        interp_exp_freqs = np.concatenate(([all_freqs_mhz.min() -1e-9], exp_freq_sorted_mhz, [all_freqs_mhz.max() + 1e-9])) # Extend ends
        interp_exp_cdf_vals = np.concatenate(([0], exp_cdf_full, [1]))
        unique_exp_freqs, unique_exp_indices = np.unique(interp_exp_freqs, return_index=True)
        interp_exp_cdf_on_common = np.interp(all_freqs_mhz, unique_exp_freqs, interp_exp_cdf_vals[unique_exp_indices], left=0, right=1)

        interp_pred_freqs = np.concatenate(([all_freqs_mhz.min() - 1e-9], pred_freq_sorted_mhz, [all_freqs_mhz.max() + 1e-9])) # Extend ends
        interp_pred_cdf_vals = np.concatenate(([0], pred_cdf_full, [1]))
        unique_pred_freqs, unique_pred_indices = np.unique(interp_pred_freqs, return_index=True)
        interp_pred_cdf_on_common = np.interp(all_freqs_mhz, unique_pred_freqs, interp_pred_cdf_vals[unique_pred_indices], left=0, right=1)

        cdf_difference = interp_exp_cdf_on_common - interp_pred_cdf_on_common
        
        ax_cdf_diff.plot(all_freqs_mhz, cdf_difference, color='purple', label='CDF Diff (Exp - Pred)')
        ax_cdf_diff.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax_cdf_diff.set_title('CDF Difference')
        ax_cdf_diff.set_xlabel('Frequency (MHz)')
        ax_cdf_diff.set_ylabel('Difference in Cum. Prob.')
        ax_cdf_diff.grid(True, alpha=0.3)
        ax_cdf_diff.legend(loc='best')
        if len(common_xlim) == 2: ax_cdf_diff.set_xlim(common_xlim)
    elif plot_idx < len(axs_flat): # Placeholder if data not available but slot exists
        ax_cdf_diff = axs_flat[plot_idx]
        plot_idx += 1
        ax_cdf_diff.text(0.5, 0.5, "Not enough data for CDF difference", ha='center', va='center', transform=ax_cdf_diff.transAxes)
        ax_cdf_diff.set_title('CDF Difference')

    # Remove any unused subplots if gs was used for a specific layout (like 2x3 with 5 plots)
    for i in range(plot_idx, len(axs_flat)):
        fig.delaxes(axs_flat[i])

    plt.tight_layout()
    plt.show()

    print("\nBasic analysis script finished.") 