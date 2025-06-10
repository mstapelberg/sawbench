# from defdap import ebsd # No longer directly needed for loading
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import tqdm # No longer directly needed for grain loop here

# Import from sawbench package
from sawbench.materials import Material
# from sawbench.saw_calculator import SAWCalculator # Used internally by calculate_saw_frequencies_for_ebsd_grains
from sawbench.io import load_ebsd_map
from sawbench.grains import calculate_saw_frequencies_for_ebsd_grains, create_ebsd_saw_frequency_map

# Define constants
DATA_PATH = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected"
EBSD_DATA_TYPE = "OxfordText" # or "ङ्Oxford rýchlosť" if that's the correct defdap identifier
BOUNDARY_DEF_ANGLE = 5.0
MIN_GRAIN_PX_SIZE = 10
WAVELENGTH_M = 8.8e-6 # meters, for f = v/lambda

if __name__ == "__main__":
    print(f"--- Loading EBSD Data using sawbench.io ---")
    ebsd_map = load_ebsd_map(
        file_path=DATA_PATH,
        data_type=EBSD_DATA_TYPE,
        boundary_def=BOUNDARY_DEF_ANGLE,
        min_grain_size=MIN_GRAIN_PX_SIZE
    )

    if ebsd_map is None:
        print(f"Failed to load EBSD map from {DATA_PATH}. Exiting.")
        exit()

    print(f"EBSD Map loaded. Shape: {ebsd_map.shape}, Step Size: {ebsd_map.stepSize} um (assuming um)")
    print(f"Phases: {[phase.name for phase in ebsd_map.phases]}")
    print(f"Identified {len(ebsd_map.grainList)} grains by defdap.")

    # Create Vanadium material object
    vanadium = Material(
        formula='V',
        C11=229e9,  # Pa
        C12=119e9,  # Pa 
        C44=43e9,   # Pa
        density=6110,  # kg/m^3
        crystal_class='cubic'
    )

    print("\n--- Calculating SAW Frequencies for Grains using sawbench.grains ---")
    df_grains_saw = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=vanadium,
        wavelength=WAVELENGTH_M
    )

    if df_grains_saw.empty:
        print("No SAW frequency data calculated for grains. Exiting.")
        exit()
    
    print(df_grains_saw.head())
    
    # Remove grains with NaN frequencies for plotting histogram
    df_valid_for_hist = df_grains_saw.dropna(subset=['Peak SAW Frequency (Hz)'])
    print(f"Valid grains for histogram: {len(df_valid_for_hist)} out of {len(df_grains_saw)}")

    print("\n--- Creating EBSD SAW Frequency Map using sawbench.grains ---")
    # --- DEBUGGING ---
    print(f"DEBUG: Before create_ebsd_saw_frequency_map:")
    print(f"DEBUG: type(ebsd_map): {type(ebsd_map)}")
    print(f"DEBUG: hasattr(ebsd_map, 'grainIDMap'): {hasattr(ebsd_map, 'grainIDMap')}")
    if hasattr(ebsd_map, 'grainIDMap'):
        print(f"DEBUG: ebsd_map.grainIDMap is None: {ebsd_map.grainIDMap is None}")
        if ebsd_map.grainIDMap is not None:
            print(f"DEBUG: ebsd_map.grainIDMap.shape: {ebsd_map.grainIDMap.shape}")
    # --- END DEBUGGING ---
    saw_freq_map_image = create_ebsd_saw_frequency_map(
        ebsd_map_obj=ebsd_map,
        grains_saw_data_df=df_grains_saw
    )

    if saw_freq_map_image is None:
        print("Failed to create SAW frequency map image.")
    else:
        print(f"SAW frequency map image generated with shape: {saw_freq_map_image.shape}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # Adjusted figsize
    
    # Plot 1: SAW frequency map
    if saw_freq_map_image is not None:
        im1 = ax1.imshow(saw_freq_map_image / 1e6, cmap='viridis', origin='lower', aspect='equal') # Convert Hz to MHz
        ax1.set_title('EBSD Map Colored by Peak SAW Frequency (MHz)')
        ax1.set_xlabel(f'X Pixel (step: {ebsd_map.stepSize:.2f} $\\mu$m)')
        ax1.set_ylabel(f'Y Pixel (step: {ebsd_map.stepSize:.2f} $\\mu$m)')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Peak SAW Frequency (MHz)')
        # Optionally, plot grain boundaries from defdap
        # ebsd_map.plotGrainBoundaries(ax=ax1, c='black', linewidth=0.5) # defdap plotting
    else:
        ax1.text(0.5, 0.5, "SAW Frequency Map not available", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('EBSD Map - SAW Frequency')

    # Plot 2: Histogram of SAW frequencies
    if not df_valid_for_hist.empty:
        freq_values_hz = df_valid_for_hist["Peak SAW Frequency (Hz)"].values
        freq_values_mhz = freq_values_hz / 1e6 # Convert to MHz for plotting

        ax2.hist(freq_values_mhz, bins=20, alpha=0.7, color='green', edgecolor='black', linewidth=0.5) # Adjusted bins
        ax2.set_xlabel('Peak SAW Frequency (MHz)')
        ax2.set_ylabel('Number of Grains')
        ax2.set_title('Distribution of Peak SAW Frequencies')
        ax2.grid(True, alpha=0.3)
        # ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0)) # May not be needed if in MHz
    else:
        ax2.text(0.5, 0.5, "No valid frequency data for histogram", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Distribution of Peak SAW Frequencies')

    plt.tight_layout()
    plt.show()

    # Print some statistics (from MHz values)
    if not df_valid_for_hist.empty:
        freq_values_mhz_stats = df_valid_for_hist["Peak SAW Frequency (Hz)"].values / 1e6
        print(f"\nSAW Frequency Statistics (MHz):")
        print(f"  Count: {len(freq_values_mhz_stats)}")
        print(f"  Mean: {np.nanmean(freq_values_mhz_stats):.2f} MHz")
        print(f"  Std:  {np.nanstd(freq_values_mhz_stats):.2f} MHz")
        print(f"  Min:  {np.nanmin(freq_values_mhz_stats):.2f} MHz")
        print(f"  Max:  {np.nanmax(freq_values_mhz_stats):.2f} MHz")
    else:
        print("\nNo valid SAW frequency data for statistics.")

    # Save the dataframe for further analysis
    output_csv_path = 'grains_data_with_saw_frequency_refactored.csv'
    df_grains_saw.to_csv(output_csv_path, index=False)
    print(f"\nSaved grain data with SAW frequencies to '{output_csv_path}'")

    print("\nRefactored old_working_analysis.py finished.")