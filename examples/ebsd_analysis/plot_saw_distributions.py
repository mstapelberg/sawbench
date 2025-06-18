import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd

# Import from sawbench package
from sawbench import (
    plot_frequency_histogram,
    plot_frequency_cdfs,
    plot_cdf_difference,
    load_fft_data_from_hdf5,
    extract_experimental_peak_parameters,
)

# --- Configuration ---
# Assumes the script is run from the 'examples/ebsd_analysis/' directory.
RESULTS_DIR = Path("results")
REFERENCE_MODEL_KEY = 'exp' # Use experimental data as the reference

# Experimental Data Config
path1 = '/home/myless/Documents/saw_freq_analysis/fftData.h5'
path2 = '/Users/myless/Dropbox (MIT)/Research/2025/Spring_2025/TGS-Mapping/processed_analysis/v-1_2ti/fftData.h5'
EXP_HDF5_PATH = path1 if os.path.exists(path1) else path2
N_EXP_PEAKS_FOR_HIST = 1
FILTER_EXP_MIN_MHZ = 200.0 
FILTER_EXP_MAX_MHZ = 400.0 

# Plotting Parameters
HIST_BINS = 50
COMMON_XLIM_MHZ = (250, 350)
OUTPUT_FILENAME = RESULTS_DIR / "model_saw_comparison.png"


def load_model_results(results_dir: Path) -> list:
    """Loads all SAW frequency model result files from the specified directory."""
    if not results_dir.exists():
        print(f"Warning: Model results directory '{results_dir}' not found.")
        return []

    all_results = []
    for filepath in results_dir.glob("saw_frequencies_*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            data['saw_frequencies_mhz'] = np.array(data['saw_frequencies_hz']) / 1e6
            all_results.append(data)
            print(f"Loaded {len(data['saw_frequencies_mhz'])} results from {filepath.name}")
    return all_results

def load_experimental_results(hdf5_path: str) -> dict:
    """Loads and processes experimental data to get a frequency distribution."""
    print("\n--- Processing Experimental FFT Data ---")
    if not os.path.exists(hdf5_path):
        print(f"Error: Experimental data file not found at {hdf5_path}")
        return {}

    fft_data_tuple = load_fft_data_from_hdf5(hdf5_path)
    if not fft_data_tuple:
        print("Failed to load experimental FFT data.")
        return {}
    
    exp_freq_axis, _, _, exp_amplitude_data, (Ny_exp, Nx_exp) = fft_data_tuple
    
    dominant_freq_list = []
    print("Extracting dominant experimental frequencies...")
    for iy in tqdm(range(Ny_exp), desc="Processing exp data"):
        for ix in range(Nx_exp):
            amp_trace = exp_amplitude_data[:, iy, ix]
            peak_params = extract_experimental_peak_parameters(exp_freq_axis, amp_trace, num_peaks_to_extract=3)
            if pd.notna(peak_params[0, 1]):
                dominant_freq_list.append(peak_params[0, 1])

    dominant_freqs_hz = np.array(dominant_freq_list)
    
    # Filter frequencies
    min_f_hz = FILTER_EXP_MIN_MHZ * 1e6
    max_f_hz = FILTER_EXP_MAX_MHZ * 1e6
    filtered_freqs_hz = dominant_freqs_hz[(dominant_freqs_hz >= min_f_hz) & (dominant_freqs_hz <= max_f_hz)]
    
    print(f"Extracted and filtered {len(filtered_freqs_hz)} experimental peak frequencies.")
    
    return {
        'model_name': 'Experimental',
        'model_key': 'exp',
        'saw_frequencies_mhz': filtered_freqs_hz / 1e6
    }

def main():
    """
    Main function to load SAW frequency data and generate comparison plots.
    """
    # Load all model results
    all_results = load_model_results(RESULTS_DIR)
    
    # Load experimental results and add them to the list
    exp_results = load_experimental_results(EXP_HDF5_PATH)
    if exp_results:
        all_results.append(exp_results)

    if len(all_results) < 2:
        print("Not enough data to create comparison plots. Need at least one model and experimental data.")
        return

    # Sort results to have a consistent plotting order, with reference first
    all_results.sort(key=lambda x: x['model_key'] != REFERENCE_MODEL_KEY)

    # --- Plotting Setup ---
    num_plots = 3  # Histogram, CDF, CDF Difference
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 6, 5))
    
    # Prepare data for plotting functions
    datasets_mhz = [res['saw_frequencies_mhz'] for res in all_results]
    labels = [res['model_name'] for res in all_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets_mhz)))
    
    # --- 1. Plot Combined Histogram ---
    ax = axs[0]
    for i, (data, label) in enumerate(zip(datasets_mhz, labels)):
        # We need to call the histogram plot for each dataset individually.
        plot_frequency_histogram(
            ax, data, 
            title='SAW Frequency Distributions',
            label=label,
            color=colors[i],
            bins=HIST_BINS, 
            xlim=COMMON_XLIM_MHZ, 
            alpha=0.6,
            density='frequency'
        )
    ax.legend() # Add legend to the histogram plot

    # --- 2. Plot Combined CDFs ---
    # Sort the datasets before plotting the CDF
    sorted_datasets_mhz = [np.sort(d) for d in datasets_mhz]
    plot_frequency_cdfs(
        axs[1], sorted_datasets_mhz, labels=labels, colors=colors,
        title='Cumulative Distribution Functions (CDFs)',
        xlim=COMMON_XLIM_MHZ
    )

    # --- 3. Plot CDF Differences ---
    reference_data = None
    other_data = []
    other_labels = []

    for res in all_results:
        sorted_data = np.sort(res['saw_frequencies_mhz'])
        if res['model_key'] == REFERENCE_MODEL_KEY:
            reference_data = sorted_data
            reference_label = res['model_name']
        else:
            other_data.append(sorted_data)
            other_labels.append(res['model_name'])
            
    if reference_data is not None and other_data:
        plot_cdf_difference(
            axs[2], 
            reference_data,
            other_data,
            label1=f"Reference: {reference_label}",
            labels2=other_labels,
            title='CDF Difference from Reference',
            xlim=COMMON_XLIM_MHZ
        )
    else:
        axs[2].text(0.5, 0.5, "Reference or other models not found\nfor CDF difference plot.", 
                    ha='center', va='center', transform=axs[2].transAxes)
        axs[2].set_title('CDF Difference from Reference')


    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"\nSaved comparison plot to: {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    main() 