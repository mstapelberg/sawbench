import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json # Added for JSON output
from typing import Dict, List, Tuple, Optional, Any # Added Any for recursive helper

# Import from sawbench package
from sawbench import (
    load_ebsd_map,
    calculate_saw_frequencies_for_ebsd_grains,
    Material,
    # plot_frequency_histogram, # If needed directly and compatible
)
# from defdap.ebsd import Map # For type hinting if ebsd_map_obj type is strictly needed

# --- Configuration ---
# EBSD Data Parameters (similar to modular_analysis_workflow.py)
# USER: Please verify this path. It should point to your EBSD data file/folder.
EBSD_DATA_PATH = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected" 
EBSD_DATA_TYPE = "OxfordText" 
EBSD_BOUNDARY_DEF_ANGLE = 5.0
EBSD_MIN_GRAIN_PX_SIZE = 10

# Material and SAW Calculation Parameters (Vanadium, from modular_analysis_workflow.py)
NOMINAL_MATERIAL_PROPS = {
    'formula': 'V', 'C11': 229e9, 'C12': 119e9, 
    'C44': 43e9, 'density': 6110, 'crystal_class': 'cubic'
}
WAVELENGTH_M = 8.8e-6 
DEFAULT_SAW_CALC_ANGLE_RAD = 0.0 # Radians, used as baseline for elasticity sensitivity

# Sensitivity Analysis Parameters
ANGLES_TO_TEST_DEG = np.arange(0, 181, 15) # Test angles from 0 to 180 degrees, step 5
ELASTIC_CONSTANT_VARIATIONS_PERCENT = np.array([-50, -25, -10, -5, -2, 2, 5, 10, 25, 50]) / 100.0 # e.g., -0.10 for -10%
#ANGLES_TO_TEST_DEG = np.arange(0, 31, 3) # Test angles from 0 to 30 degrees, step 5
#ELASTIC_CONSTANT_VARIATIONS_PERCENT = np.array([-5, -2, 2, 5]) / 100.0 # e.g., -0.10 for -10%

# Plotting Parameters
HIST_BINS = 50
PLOT_FONT_SIZE = 12
TITLE_FONT_SIZE = 14

# --- Script Configuration ---
# Set to True to load data from JSON and replot. Set to False to run full analysis.
REPLOT_FROM_JSON_FILE = True
JSON_FILENAME = "sensitivity_analysis_results.json"

# --- Helper Functions ---

def load_ebsd(ebsd_path: str, data_type: str, boundary_def: float, min_grain_size: int): # -> Optional[Map]
    """Loads EBSD map using sawbench utility."""
    print(f"--- Loading EBSD Data ---")
    print(f"Attempting to load from: {os.path.abspath(ebsd_path)}")
    ebsd_map = load_ebsd_map(ebsd_path, data_type, boundary_def, min_grain_size)
    if not ebsd_map:
        print(f"Error: Failed to load EBSD data from '{ebsd_path}'. Please check the path and data format.")
        return None
    print(f"EBSD map loaded successfully. Number of grains: {len(ebsd_map.grainList)}")
    return ebsd_map

def calculate_saw_frequencies(
    ebsd_map_obj, #: Map, 
    material_properties: dict, 
    wavelength_m: float, 
    saw_calc_angle_rad: float
) -> np.ndarray:
    """
    Calculates SAW frequencies for all grains in an EBSD map for a given material and angle.
    Returns frequencies in MHz.
    """
    material = Material(**material_properties)
    
    df_ebsd_grains = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map_obj,
        material=material,
        wavelength=wavelength_m,
        saw_calc_angle=saw_calc_angle_rad 
    )
    
    if df_ebsd_grains.empty or 'Peak SAW Frequency (Hz)' not in df_ebsd_grains.columns:
        print(f"Warning: No SAW frequencies calculated or 'Peak SAW Frequency (Hz)' column missing for angle {np.rad2deg(saw_calc_angle_rad):.1f} deg, material: {material_properties.get('formula', 'N/A')}.")
        return np.array([])
        
    predicted_freq_array_hz = df_ebsd_grains['Peak SAW Frequency (Hz)'].dropna().to_numpy()
    
    if predicted_freq_array_hz.size == 0:
        print(f"Warning: Resulting frequency array is empty after dropna for angle {np.rad2deg(saw_calc_angle_rad):.1f} deg.")

    return predicted_freq_array_hz / 1e6 # Convert to MHz

def plot_frequency_distributions_hist(
    ax: plt.Axes, 
    distributions: Dict[str, np.ndarray], 
    title: str,
    xlabel: str = "Frequency (MHz)", 
    ylabel: str = "Density", 
    bins: int = HIST_BINS,
    xlim: Optional[Tuple[float, float]] = None,
    alpha: float = 0.6,
    colormap_name: Optional[str] = 'viridis',
    is_main_title_disabled: bool = False,
    subplot_annotation_text: Optional[str] = None,
    fixed_y_max: Optional[float] = None # New parameter for fixed Y upper limit
):
    """Plots multiple frequency distributions as overlapping histograms on the same axes."""
    if not distributions:
        print(f"No distributions to plot for '{title}'.")
        if not is_main_title_disabled:
            ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
        return

    has_data = False
    num_distributions = len(distributions)
    colors = None
    if colormap_name and num_distributions > 0:
        try:
            cmap = plt.get_cmap(colormap_name)
            colors = [cmap(i/num_distributions) for i in range(num_distributions)]
        except ValueError:
            print(f"Warning: Colormap '{colormap_name}' not found. Using default colors.")
            colors = [None] * num_distributions # Fallback to default color cycling
    elif num_distributions > 0:
        colors = [None] * num_distributions # Fallback for default color cycling

    for i, (label, data) in enumerate(distributions.items()):
        if data is not None and data.size > 0:
            color_to_use = colors[i] if colors else None
            ax.hist(data, bins=bins, density=True, alpha=alpha, label=str(label), color=color_to_use)
            has_data = True
        else:
            print(f"Info: No data or empty data for label '{label}' in '{title}'. Skipping histogram.")
            
    if not has_data:
        ax.text(0.5, 0.5, "All datasets empty", ha='center', va='center', transform=ax.transAxes)

    if not is_main_title_disabled:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    elif subplot_annotation_text: # Ensure title space isn't totally empty if annotation is also main title
        ax.set_title(" ", fontsize=TITLE_FONT_SIZE) # Minimal title to maintain layout if needed

    ax.set_xlabel(xlabel, fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE)
    if xlim and has_data: # Only set xlim if there's data and xlim is provided
        ax.set_xlim(xlim)
    if fixed_y_max is not None and has_data: # Apply fixed Y upper limit if provided
        ax.set_ylim(0, fixed_y_max)
    if has_data: # Only add legend if there's something to label
      ax.legend(fontsize=PLOT_FONT_SIZE-2)
    ax.grid(True, linestyle='--', alpha=0.7)

    if subplot_annotation_text:
        ax.text(0.98, 0.02, subplot_annotation_text, 
                transform=ax.transAxes, ha='right', va='bottom', 
                fontsize=PLOT_FONT_SIZE, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# --- New CDF Plotting Function ---
def plot_frequency_distributions_cdf(
    ax: plt.Axes,
    distributions: Dict[str, np.ndarray],
    title: str,
    xlabel: str = "Frequency (MHz)",  # Ensuring xlabel is defined once with a default
    ylabel: str = "Cumulative Probability",
    xlim: Optional[Tuple[float, float]] = None,
    alpha: float = 0.8, 
    colormap_name: Optional[str] = 'viridis',
    is_main_title_disabled: bool = False,
    subplot_annotation_text: Optional[str] = None
):
    """Plots multiple frequency distributions as CDFs on the same axes."""
    if not distributions:
        print(f"No distributions to plot CDF for '{title}'.")
        if not is_main_title_disabled:
            ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.text(0.5, 0.5, "No data for CDF", ha='center', va='center', transform=ax.transAxes)
        return

    has_data = False
    num_distributions = len(distributions)
    colors = None
    if colormap_name and num_distributions > 0:
        try:
            cmap = plt.get_cmap(colormap_name)
            colors = [cmap(i/num_distributions) for i in range(num_distributions)]
        except ValueError:
            print(f"Warning: Colormap '{colormap_name}' not found. Using default colors.")
            colors = [None] * num_distributions
    elif num_distributions > 0:
        colors = [None] * num_distributions

    for i, (item_label, data) in enumerate(distributions.items()):
        if data is not None and data.size > 0:
            sorted_data = np.sort(data)
            yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            color_to_use = colors[i] if colors else None
            
            current_linestyle = '--' if str(item_label) == "Nominal" else '-'
            current_marker = '.' if str(item_label) != "Nominal" else None # No markers for nominal to make it cleaner
            line_alpha = alpha
            line_zorder = 10 if str(item_label) == "Nominal" else 5 # Ensure nominal is plotted on top if overlapping

            ax.plot(sorted_data, yvals, 
                    marker=current_marker, 
                    linestyle=current_linestyle, 
                    label=str(item_label), 
                    alpha=line_alpha, 
                    color=color_to_use, 
                    markersize=2,
                    zorder=line_zorder)
            has_data = True
        else:
            print(f"Info: No data or empty data for label '{item_label}' in CDF '{title}'. Skipping plot.")

    if not has_data:
        ax.text(0.5, 0.5, "All datasets for CDF empty", ha='center', va='center', transform=ax.transAxes)

    if not is_main_title_disabled:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    elif subplot_annotation_text: 
        ax.set_title(" ", fontsize=TITLE_FONT_SIZE) 

    ax.set_xlabel(xlabel, fontsize=PLOT_FONT_SIZE) 
    ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE)
    if xlim and has_data:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 1.05) 
    if has_data:
      ax.legend(fontsize=PLOT_FONT_SIZE-2)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if subplot_annotation_text:
        ax.text(0.98, 0.02, subplot_annotation_text, 
                transform=ax.transAxes, ha='right', va='bottom', 
                fontsize=PLOT_FONT_SIZE, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# --- Utility for JSON Saving & Loading ---

def convert_numpy_to_lists(item: Any) -> Any:
    """Recursively converts numpy arrays in a data structure to lists for JSON serialization."""
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, dict):
        return {k: convert_numpy_to_lists(v) for k, v in item.items()}
    if isinstance(item, list):
        return [convert_numpy_to_lists(i) for i in item]
    return item

def convert_lists_to_numpy(item: Any) -> Any:
    """Recursively converts lists (that were numpy arrays) back to numpy arrays."""
    if isinstance(item, dict):
        return {k: convert_lists_to_numpy(v) for k, v in item.items()}
    # Check if it's a list of numbers (potential numpy array) or list of lists (nested)
    # This is a simple check; more robust checks might be needed for complex structures
    if isinstance(item, list):
        if item and all(isinstance(x, (int, float)) for x in item):
            return np.array(item)
        elif item and all(isinstance(x, list) for x in item): # For nested arrays if any
             return np.array([convert_lists_to_numpy(i) for i in item], dtype=object) # dtype=object for ragged arrays
        return [convert_lists_to_numpy(i) for i in item] # For lists of other things (e.g. dicts)
    return item

def save_results_to_json(data: Dict[str, Any], filename: str):
    """Saves the provided data dictionary to a JSON file."""
    print(f"--- Saving Results to JSON ---")
    try:
        serializable_data = convert_numpy_to_lists(data) # Changed helper name
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"Results successfully saved to {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

def load_results_from_json(filename: str) -> Optional[Dict[str, Any]]:
    """Loads analysis results from a JSON file and converts lists back to numpy arrays."""
    print(f"--- Loading Results from JSON: {filename} ---")
    if not os.path.exists(filename):
        print(f"Error: JSON file '{filename}' not found. Cannot replot.")
        return None
    try:
        with open(filename, 'r') as f:
            loaded_data_raw = json.load(f)
        
        # Convert frequency data back to numpy arrays
        if 'angle_sensitivity_results' in loaded_data_raw:
            loaded_data_raw['angle_sensitivity_results'] = {
                k: np.array(v) if isinstance(v, list) else v 
                for k, v in loaded_data_raw['angle_sensitivity_results'].items()
            }
        if 'elastic_constant_sensitivity_results' in loaded_data_raw:
            loaded_data_raw['elastic_constant_sensitivity_results'] = {
                const: {k: np.array(v) if isinstance(v, list) else v 
                        for k, v in data.items()}
                for const, data in loaded_data_raw['elastic_constant_sensitivity_results'].items()
            }
        # Convert parameter arrays back if needed (ANGLES_TO_TEST_DEG, ELASTIC_CONSTANT_VARIATIONS_PERCENT)
        if 'parameters' in loaded_data_raw and 'ANGLES_TESTED_DEG_FOR_ANGLE_STUDY' in loaded_data_raw['parameters']:
            loaded_data_raw['parameters']['ANGLES_TESTED_DEG_FOR_ANGLE_STUDY'] = np.array(loaded_data_raw['parameters']['ANGLES_TESTED_DEG_FOR_ANGLE_STUDY'])
        if 'parameters' in loaded_data_raw and 'ELASTIC_CONSTANT_VARIATIONS_PERCENT' in loaded_data_raw['parameters']:
            loaded_data_raw['parameters']['ELASTIC_CONSTANT_VARIATIONS_PERCENT'] = np.array(loaded_data_raw['parameters']['ELASTIC_CONSTANT_VARIATIONS_PERCENT'])

        print(f"Results loaded and processed from {filename}")
        return loaded_data_raw
    except Exception as e:
        print(f"Error loading or processing results from JSON: {e}")
        return None

# --- Main Analysis Functions ---

def perform_angle_sensitivity_analysis(
    ebsd_map_obj, # Map, 
    base_material_props: dict, 
    wavelength_m: float, 
    angles_deg: np.ndarray
) -> Dict[str, np.ndarray]:
    """Performs sensitivity analysis by varying the SAW calculation angle."""
    print("--- Performing Angle Sensitivity Analysis ---")
    results_mhz = {}
    angles_rad = np.deg2rad(angles_deg)
    
    for angle_deg, angle_rad in zip(angles_deg, angles_rad):
        print(f"Calculating for angle: {angle_deg:.1f} degrees")
        freqs_mhz = calculate_saw_frequencies(
            ebsd_map_obj, base_material_props, wavelength_m, angle_rad
        )
        results_mhz[f"{angle_deg:.0f} deg"] = freqs_mhz
    return results_mhz

def perform_elastic_constant_sensitivity_analysis(
    ebsd_map_obj, #: Map, 
    nominal_material_props: dict, 
    wavelength_m: float, 
    saw_calc_angle_rad: float, 
    variations_percent: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """Performs sensitivity analysis by varying individual elastic constants."""
    print("--- Performing Elastic Constant Sensitivity Analysis ---")
    # Structure: {'C11': {'Nominal': array, 'C11 +5%': array, ...}, 'C12': {...}, ...}
    all_results_mhz: Dict[str, Dict[str, np.ndarray]] = {}
    
    constants_to_vary = ['C11', 'C12', 'C44']
    
    # First, calculate the absolute nominal case once
    print(f"Calculating absolute nominal case for elasticity study (angle: {np.rad2deg(saw_calc_angle_rad):.1f} deg)...")
    freqs_nominal_mhz = calculate_saw_frequencies(
        ebsd_map_obj, nominal_material_props, wavelength_m, saw_calc_angle_rad
    )
    if freqs_nominal_mhz.size == 0:
        print("Warning: Absolute nominal calculation for elasticity study yielded no frequencies. Sensitivity results may be affected.")

    for const_name in constants_to_vary:
        print(f"-- Varying {const_name} --")
        results_for_const_mhz: Dict[str, np.ndarray] = {}
        
        # Add the pre-calculated nominal to this constant's dict for comparison in its plot
        results_for_const_mhz["Nominal"] = freqs_nominal_mhz
        
        nominal_val = nominal_material_props[const_name]
        
        for var_percent in variations_percent:
            if var_percent == 0: # Should not happen if variations_percent doesn't include 0
                continue

            current_props = nominal_material_props.copy()
            current_props[const_name] = nominal_val * (1 + var_percent)
            
            label = f"{const_name} {var_percent*100:+.0f}%"
            print(f"Calculating for {label} (angle: {np.rad2deg(saw_calc_angle_rad):.1f} deg)")
            
            freqs_mhz = calculate_saw_frequencies(
                ebsd_map_obj, current_props, wavelength_m, saw_calc_angle_rad
            )
            results_for_const_mhz[label] = freqs_mhz
            
        all_results_mhz[const_name] = results_for_const_mhz
        
    return all_results_mhz

# --- Main Execution ---
if __name__ == "__main__":
    angle_sensitivity_results = None
    elastic_sensitivity_results = None
    analysis_parameters = None

    if REPLOT_FROM_JSON_FILE:
        print("Attempting to replot from JSON...")
        loaded_data = load_results_from_json(JSON_FILENAME)
        if loaded_data:
            angle_sensitivity_results = loaded_data.get('angle_sensitivity_results')
            elastic_sensitivity_results = loaded_data.get('elastic_constant_sensitivity_results')
            analysis_parameters = loaded_data.get('parameters')
            # Potentially override global config with loaded parameters for consistency in plotting
            # For example, if ANGLES_TO_TEST_DEG is used in plotting titles/logic
            if analysis_parameters and 'ANGLES_TESTED_DEG_FOR_ANGLE_STUDY' in analysis_parameters:
                ANGLES_TO_TEST_DEG = analysis_parameters['ANGLES_TESTED_DEG_FOR_ANGLE_STUDY']
            if analysis_parameters and 'ELASTIC_CONSTANT_VARIATIONS_PERCENT' in analysis_parameters:
                ELASTIC_CONSTANT_VARIATIONS_PERCENT = analysis_parameters['ELASTIC_CONSTANT_VARIATIONS_PERCENT']
        else:
            print("Failed to load data for replotting. Exiting.")
            exit()
    else:
        print("Running full sensitivity analysis...")
        # 1. Load EBSD Data
        ebsd_map = load_ebsd(
            EBSD_DATA_PATH, EBSD_DATA_TYPE, 
            EBSD_BOUNDARY_DEF_ANGLE, EBSD_MIN_GRAIN_PX_SIZE
        )

        if not ebsd_map:
            print("Critical Error: EBSD data could not be loaded. Exiting analysis.")
            exit()

        # 2. Perform Angle Sensitivity Analysis
        angle_sensitivity_results = perform_angle_sensitivity_analysis(
            ebsd_map, NOMINAL_MATERIAL_PROPS, WAVELENGTH_M, ANGLES_TO_TEST_DEG
        )

        # 3. Perform Elastic Constant Sensitivity Analysis
        elastic_sensitivity_results = perform_elastic_constant_sensitivity_analysis(
            ebsd_map, NOMINAL_MATERIAL_PROPS, WAVELENGTH_M, 
            DEFAULT_SAW_CALC_ANGLE_RAD, ELASTIC_CONSTANT_VARIATIONS_PERCENT
        )

        # 4. Combine and Save Results to JSON
        all_results_for_saving = {
            "angle_sensitivity_results": angle_sensitivity_results,
            "elastic_constant_sensitivity_results": elastic_sensitivity_results,
            "parameters": {
                "EBSD_DATA_PATH": EBSD_DATA_PATH,
                "EBSD_DATA_TYPE": EBSD_DATA_TYPE,
                "EBSD_BOUNDARY_DEF_ANGLE": EBSD_BOUNDARY_DEF_ANGLE,
                "EBSD_MIN_GRAIN_PX_SIZE": EBSD_MIN_GRAIN_PX_SIZE,
                "NOMINAL_MATERIAL_PROPS": NOMINAL_MATERIAL_PROPS,
                "WAVELENGTH_M": WAVELENGTH_M,
                "DEFAULT_SAW_CALC_ANGLE_RAD_FOR_ELASTICITY_STUDY": DEFAULT_SAW_CALC_ANGLE_RAD,
                "ANGLES_TESTED_DEG_FOR_ANGLE_STUDY": ANGLES_TO_TEST_DEG, # np array directly used by convert_numpy_to_lists
                "ELASTIC_CONSTANT_VARIATIONS_PERCENT": ELASTIC_CONSTANT_VARIATIONS_PERCENT # np array directly used by convert_numpy_to_lists
            }
        }
        save_results_to_json(all_results_for_saving, JSON_FILENAME)

    # 5. Plotting Results
    if angle_sensitivity_results is None and elastic_sensitivity_results is None:
        print("No data available for plotting. Exiting.")
        exit()

    print("--- Plotting Results ---")
    
    # Determine common x-axis limits for ALL plots (primarily for elasticity comparison)
    all_freq_data_for_overall_xlim = []
    if angle_sensitivity_results:
        for freqs_array in angle_sensitivity_results.values():
            if freqs_array is not None and freqs_array.size > 0:
                all_freq_data_for_overall_xlim.append(freqs_array)
    if elastic_sensitivity_results:
        for const_results in elastic_sensitivity_results.values():
            for freqs_array in const_results.values():
                if freqs_array is not None and freqs_array.size > 0:
                    all_freq_data_for_overall_xlim.append(freqs_array)
    
    common_xlim = None
    if all_freq_data_for_overall_xlim:
        try:
            full_dataset = np.concatenate([arr for arr in all_freq_data_for_overall_xlim if arr.size > 0])
            if full_dataset.size > 0:
                min_freq_overall = np.percentile(full_dataset, 0.5)
                max_freq_overall = np.percentile(full_dataset, 99.5)
                padding_overall = (max_freq_overall - min_freq_overall) * 0.05 
                common_xlim = (min_freq_overall - padding_overall, max_freq_overall + padding_overall)
                print(f"Determined common x-axis limits for elasticity plots: ({common_xlim[0]:.2f} - {common_xlim[1]:.2f}) MHz")
            else:
                print("No data to determine common x-axis limits for elasticity plots.")
        except ValueError as e:
            print(f"Could not determine common xlim for elasticity due to: {e}. Elasticity plots will use auto-scaling or fallback.")

    # Determine specific x-axis limits for Angle Sensitivity plots
    angle_plots_xlim = None
    if angle_sensitivity_results:
        angle_freq_data_for_xlim = []
        for freqs_array in angle_sensitivity_results.values():
            if freqs_array is not None and freqs_array.size > 0:
                angle_freq_data_for_xlim.append(freqs_array)
        
        if angle_freq_data_for_xlim:
            try:
                angle_full_dataset = np.concatenate([arr for arr in angle_freq_data_for_xlim if arr.size > 0])
                if angle_full_dataset.size > 0:
                    min_freq_angle = np.min(angle_full_dataset)
                    max_freq_angle = np.max(angle_full_dataset)
                    angle_plots_xlim = (min_freq_angle - 10, max_freq_angle + 10)
                    print(f"Determined x-axis limits for angle plots: ({angle_plots_xlim[0]:.2f} - {angle_plots_xlim[1]:.2f}) MHz")
                else:
                    print("No data in angle_sensitivity_results to determine specific x-axis limits.")
            except ValueError as e:
                print(f"Could not determine specific xlim for angle plots due to: {e}. Angle plots may use auto-scaling or common_xlim if specific fails.")
        else:
            print("No frequency data found in angle_sensitivity_results for xlim calculation.")
    
    # Fallback for angle_plots_xlim if not determinable, could use common_xlim or let matplotlib auto-scale
    # For now, if angle_plots_xlim is None, the plotting functions will handle it (likely auto-scale by matplotlib if xlim=None)

    # Plot Angle Sensitivity (Histogram)
    fig_angle_hist, ax_angle_hist = plt.subplots(1, 1, figsize=(12, 7))
    plot_frequency_distributions_hist(
        ax_angle_hist, angle_sensitivity_results, 
        "Angle Sensitivity: SAW Frequency Distribution (Vanadium) - Histograms",
        xlim=angle_plots_xlim, # Use specific xlim for angle plots
        alpha=0.5, 
        colormap_name='viridis',
        bins=max(10, HIST_BINS // len(ANGLES_TO_TEST_DEG) if len(ANGLES_TO_TEST_DEG) > 0 else HIST_BINS)
    )
    fig_angle_hist.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot Angle Sensitivity (CDF)
    fig_angle_cdf, ax_angle_cdf = plt.subplots(1, 1, figsize=(12, 7))
    plot_frequency_distributions_cdf(
        ax_angle_cdf, angle_sensitivity_results,
        "Angle Sensitivity: SAW Frequency Distribution (Vanadium) - CDFs",
        xlim=angle_plots_xlim, # Use specific xlim for angle plots
        alpha=0.7,
        colormap_name='viridis'
    )
    fig_angle_cdf.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot Elastic Constant Sensitivity (CDFs)
    num_elastic_plots = len(elastic_sensitivity_results)
    if num_elastic_plots > 0:
        fig_elastic_cdf, axs_elastic_cdf = plt.subplots(
            nrows=num_elastic_plots, ncols=1, 
            figsize=(12, 5 * num_elastic_plots), # Adjusted height slightly per plot
            squeeze=False,
            sharex=True  # Enable shared X-axis
        )
        axs_elastic_cdf_flat = axs_elastic_cdf.flatten()
        
        plot_idx = 0
        for const_name, results_for_const in elastic_sensitivity_results.items():
            if plot_idx < len(axs_elastic_cdf_flat):
                current_ax = axs_elastic_cdf_flat[plot_idx]
                is_last_plot_in_col = (plot_idx == num_elastic_plots - 1)
                current_xlabel = "Frequency (MHz)" if is_last_plot_in_col else ""
                
                plot_frequency_distributions_cdf(
                    current_ax, results_for_const,
                    f"Elasticity Sensitivity ({const_name}): SAW Frequency Distribution (Vanadium) - CDFs", 
                    xlim=common_xlim, # Elasticity plots continue to use common_xlim
                    alpha=0.7,
                    colormap_name='plasma',
                    is_main_title_disabled=True, 
                    subplot_annotation_text=const_name,
                    xlabel=current_xlabel
                )
                plot_idx += 1
        
        fig_elastic_cdf.suptitle("Elasticity Sensitivity: SAW Frequency Distributions (CDFs)", fontsize=TITLE_FONT_SIZE + 2, y=0.99)
        fig_elastic_cdf.tight_layout(rect=[0, 0.03, 1, 0.97]) 

    # Plot Elastic Constant Sensitivity (Histograms) - New section
    if num_elastic_plots > 0 and elastic_sensitivity_results: 
        # Pre-calculate a common Y-axis upper limit for elasticity histograms (e.g., 99th percentile of densities)
        all_hist_densities = []
        current_bins_for_calc = max(10, HIST_BINS // (len(ELASTIC_CONSTANT_VARIATIONS_PERCENT) + 1) if len(ELASTIC_CONSTANT_VARIATIONS_PERCENT) > 0 else HIST_BINS)
        for const_name_calc, results_for_const_calc in elastic_sensitivity_results.items():
            for label_calc, data_calc in results_for_const_calc.items():
                if data_calc is not None and data_calc.size > 0:
                    # Ensure xlim for density calculation is consistent if possible, or use data range
                    # Using common_xlim if available, else data range for hist calculation
                    hist_range = None
                    if common_xlim and common_xlim[0] is not None and common_xlim[1] is not None:
                        # Filter data to be within common_xlim for density calculation, if data exceeds it, to avoid issues with range
                        data_for_hist_calc = data_calc[(data_calc >= common_xlim[0]) & (data_calc <= common_xlim[1])]
                        if data_for_hist_calc.size > 0:
                             hist_range = (common_xlim[0], common_xlim[1]) 
                    else: # Fallback if common_xlim is not fully defined
                        data_for_hist_calc = data_calc
                    
                    if data_for_hist_calc.size > 0:
                        densities, _ = np.histogram(data_for_hist_calc, bins=current_bins_for_calc, density=True, range=hist_range)
                        all_hist_densities.extend(densities)
        
        elasticity_hist_y_max = None
        if all_hist_densities:
            elasticity_hist_y_max = np.percentile(all_hist_densities, 99) # Cap at 99th percentile
            print(f"Determined common Y-axis upper limit for elasticity histograms: {elasticity_hist_y_max:.2f}")
        else:
            print("Could not determine a common Y-max for elasticity histograms. Auto-scaling will be used.")

        fig_elastic_hist, axs_elastic_hist = plt.subplots(
            nrows=num_elastic_plots, ncols=1,
            figsize=(12, 5 * num_elastic_plots), 
            squeeze=False,
            sharex=True, 
            sharey=True # Enable shared Y-axis
        )
        axs_elastic_hist_flat = axs_elastic_hist.flatten()
        
        plot_idx = 0
        for const_name, results_for_const in elastic_sensitivity_results.items():
            if plot_idx < len(axs_elastic_hist_flat):
                current_ax = axs_elastic_hist_flat[plot_idx]
                is_last_plot_in_col = (plot_idx == num_elastic_plots - 1)
                current_xlabel_for_plot = "Frequency (MHz)" if is_last_plot_in_col else ""
                current_ylabel_for_plot = "Density" if plot_idx == 0 else "" # Y-label only for the first plot
                
                plot_frequency_distributions_hist(
                    current_ax, results_for_const,
                    f"Elasticity Sensitivity ({const_name}): SAW Frequency Distribution (Vanadium) - Histograms", 
                    xlim=common_xlim, 
                    alpha=0.5, 
                    colormap_name='plasma',
                    is_main_title_disabled=True,
                    subplot_annotation_text=const_name,
                    xlabel=current_xlabel_for_plot,
                    ylabel=current_ylabel_for_plot, # Pass conditional ylabel
                    bins=current_bins_for_calc, # Use pre-calculated bins
                    fixed_y_max=elasticity_hist_y_max # Pass the calculated common Y-max
                )
                plot_idx += 1
        
        fig_elastic_hist.suptitle("Elasticity Sensitivity: SAW Frequency Distributions (Histograms)", fontsize=TITLE_FONT_SIZE + 2, y=0.99)
        fig_elastic_hist.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect

    if not plt.get_fignums():
        print("No figures were generated, plt.show() will not be called.")
    else:
        plt.show()

    print("Sensitivity analysis workflow finished.") 