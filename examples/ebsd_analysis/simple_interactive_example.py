"""
Simple example demonstrating the streamlined interactive EBSD grain analysis.

This example shows the minimal code needed to perform interactive subregion
selection and SAW frequency analysis.
"""

from sawbench import Material, interactive_grain_analysis
import os

# Define your material
material = Material(
    formula='V',
    C11=229e9,
    C12=119e9,
    C44=43e9,
    density=6110,
    crystal_class='cubic'
)

# Path to your CTF file
ctf_path = "../data/V-1_2Ti_EBSD_Map.ctf"

# Check if file exists
if not os.path.exists(ctf_path):
    print(f"CTF file not found: {ctf_path}")
    print("Please update the ctf_path variable to point to your CTF file.")
    exit(1)

# Run the interactive analysis
# This will:
# 1. Display the EBSD map
# 2. Let you draw a polygon to select a region
# 3. Calculate SAW frequencies for all grains
# 4. Show overlaid histogram and spatial frequency map
# 5. Show Euler angle distribution analysis (if enabled)
# 6. Save results if output_dir is specified

grains_all, grains_roi, metadata = interactive_grain_analysis(
    ctf_path=ctf_path,
    material=material,
    wavelength=8.8e-6,  # 8.8 microns
    saw_calc_angle_deg=135.0,
    num_workers=32,
    output_dir="./results/interactive_analysis_output_4000",  # Comment out to skip saving
    saw_calc_sampling=4000,  # Default value - use 400, 4000, or 40000
    saw_calc_psaw=0,  # 0 = SAW only (recommended), 1 = include PSAW (experimental)
    boundary_def=5.0,
    min_grain_size=10,
    show_euler_analysis=True,  # Set to False to disable Euler angle analysis
    find_top_n_orientations=5,  # Find top 5 orientations (set to None to disable)
    clustering_threshold_deg=5.0  # Clustering threshold in degrees
)

# Access results
print("\n" + "="*60)
print("Results Summary")
print("="*60)
print(f"Total grains analyzed: {len(grains_all)}")
print(f"Grains in selected region: {len(grains_roi)}")

# Check if valid frequencies were calculated
if 'freq_range_roi_mhz' in metadata:
    print(f"\nFrequency range in ROI: {metadata['freq_range_roi_mhz']}")
    print(f"Number of grains in ROI: {metadata['num_grains_roi']}")
else:
    print("\n⚠️  WARNING: No valid SAW frequencies were calculated.")
    print("This can happen when:")
    print("  - psaw=1 but not enough distinct peaks are found")
    print("  - sampling parameter is too high or too low")
    print("  - peak detection threshold is not met")
    print("\nTry:")
    print("  - Setting psaw=0 (standard SAW mode)")
    print("  - Using default sampling=400")
    print("  - Checking your material parameters")

# You can now work with the DataFrames
if not grains_roi.empty and grains_roi['Peak SAW Frequency (Hz)'].notna().any():
    print("\nFirst 5 grains in ROI:")
    print(grains_roi[['Grain ID', 'Peak SAW Frequency (Hz)', 'Size (um^2)']].head())
else:
    print("\nNo valid grain data to display.")

# Access top orientations if clustering was performed
if 'top_orientations' in metadata and metadata['top_orientations'] is not None:
    print("\n" + "="*60)
    print("Top Orientations for Sensitivity Analysis")
    print("="*60)
    top_df = metadata['top_orientations']
    for idx, row in top_df.iterrows():
        print(f"\nOrientation #{row['Rank']}:")
        print(f"  Euler angles (rad): ({row['Mean_phi1_rad']:.4f}, {row['Mean_Phi_rad']:.4f}, {row['Mean_phi2_rad']:.4f})")
        print(f"  Euler angles (deg): ({row['Mean_phi1_deg']:.1f}°, {row['Mean_Phi_deg']:.1f}°, {row['Mean_phi2_deg']:.1f}°)")
        print(f"  Represents {row['Fraction_by_area']*100:.1f}% of sample area ({row['N_Grains']} grains)")
        print("  → Use these angles in your SAW calculator for sensitivity analysis!")

