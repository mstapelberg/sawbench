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
DFT_MODEL_KEY = 'dft' # Use DFT data as a secondary reference

# Experimental Data Config
path1 = '/home/myless/Documents/saw_freq_analysis/fftData.h5'
path2 = '/Users/myless/Dropbox (MIT)/Research/2025/Spring_2025/TGS-Mapping/processed_analysis/v-1_2ti/fftData.h5'
EXP_HDF5_PATH = path1 if os.path.exists(path1) else path2
N_EXP_PEAKS_FOR_HIST = 1
FILTER_EXP_MIN_MHZ = 200.0 
FILTER_EXP_MAX_MHZ = 500.0 

# Plotting Parameters
HIST_BINS = 50
COMMON_XLIM_MHZ = (250, 450)
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

def plot_cdf_differences_to_reference(ax, all_results, reference_key, xlim, colors):
    """Helper function to plot CDF differences against a specific reference."""
    reference_data = None
    reference_label = ""
    other_data = []
    other_labels = []
    other_colors = []

    # Find the reference dataset
    for res in all_results:
        if res['model_key'] == reference_key:
            reference_data = np.sort(res['saw_frequencies_mhz'])
            reference_label = res['model_name']
            break

    if reference_data is None:
        ax.text(0.5, 0.5, f"Reference data '{reference_key}' not found.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'CDF Difference from {reference_key.upper()}')
        return

    # Collect other datasets and their corresponding colors
    color_map = {res['model_key']: color for res, color in zip(all_results, colors)}

    for res in all_results:
        if res['model_key'] != reference_key:
            other_data.append(np.sort(res['saw_frequencies_mhz']))
            other_labels.append(res['model_name'])
            # Default to black if a color isn't found, though it shouldn't happen
            other_colors.append(color_map.get(res['model_key'], 'black'))

    if other_data:
        plot_cdf_difference(
            ax,
            reference_data,
            other_data,
            label1=f"Reference: {reference_label}",
            labels2=other_labels,
            colors=other_colors,
            title=f'CDF Difference from {reference_label}',
            xlim=xlim
        )
    else:
        ax.text(0.5, 0.5, "No other models to compare.",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'CDF Difference from {reference_label}')


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
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data for plotting functions
    datasets_mhz = [res['saw_frequencies_mhz'] for res in all_results]
    labels = [res['model_name'] for res in all_results]
    
    # Use a color-blind friendly and distinct color palette
    # Assign colors to each model based on the sorted list of results
    model_keys = [res['model_key'] for res in all_results]
    palette = plt.cm.get_cmap('tab10').colors
    color_map = {key: palette[i % len(palette)] for i, key in enumerate(model_keys)}
    colors = [color_map[key] for key in model_keys]
    
    # --- 1. Plot Combined Histogram ---
    ax = axs[0, 0]
    for i, (data, label) in enumerate(zip(datasets_mhz, labels)):
        plot_frequency_histogram(
            ax, data, 
            title='SAW Frequency Distributions',
            label=label,
            color=colors[i],
            bins=HIST_BINS, 
            xlim=COMMON_XLIM_MHZ, 
            alpha=0.7,
            density='frequency'
        )
    ax.legend()

    # --- 2. Plot Combined CDFs ---
    ax = axs[0, 1]
    sorted_datasets_mhz = [np.sort(d) for d in datasets_mhz]
    plot_frequency_cdfs(
        ax, sorted_datasets_mhz, labels=labels, 
        colors=colors,
        title='Cumulative Distribution Functions (CDFs)',
        xlim=COMMON_XLIM_MHZ
    )

    # --- 3. Plot CDF Differences vs. Experimental ---
    plot_cdf_differences_to_reference(axs[1, 0], all_results, REFERENCE_MODEL_KEY, COMMON_XLIM_MHZ, colors)

    # --- 4. Plot CDF Differences vs. DFT ---
    plot_cdf_differences_to_reference(axs[1, 1], all_results, DFT_MODEL_KEY, COMMON_XLIM_MHZ, colors)


    plt.tight_layout(pad=3.0)
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"\nSaved comparison plot to: {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    main() 