"""
Extract representative grain orientations for sensitivity analysis.

This script stratifies grains by SAW frequency and selects one representative
grain from each bin to ensure diverse orientation coverage.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Load grain data from EBSD analysis
results_dir = Path("../ebsd_analysis/results/interactive_analysis_output_4000")
grains_roi = pd.read_csv(results_dir / "grains_roi.csv")

print("="*70)
print("EXTRACTING REPRESENTATIVE GRAIN ORIENTATIONS")
print("="*70)

# Filter valid grains (have SAW frequency calculated)
valid_grains = grains_roi[grains_roi['Peak SAW Frequency (Hz)'].notna()].copy()
print(f"\nTotal valid grains: {len(valid_grains)}")

# Determine number of bins (aim for ~29 bins as you mentioned)
n_bins = min(29, len(valid_grains))
print(f"Creating {n_bins} frequency bins for stratified sampling")

# Create frequency bins
try:
    valid_grains['Freq_Bin'] = pd.qcut(
        valid_grains['Peak SAW Frequency (Hz)'], 
        q=n_bins, 
        labels=False,
        duplicates='drop'
    )
    actual_n_bins = valid_grains['Freq_Bin'].nunique()
    print(f"Actual bins created: {actual_n_bins} (some may be merged due to duplicates)")
except Exception as e:
    print(f"Warning: Could not create {n_bins} bins, using fewer: {e}")
    # Fall back to fewer bins if there are issues
    n_bins = 20
    valid_grains['Freq_Bin'] = pd.qcut(
        valid_grains['Peak SAW Frequency (Hz)'], 
        q=n_bins, 
        labels=False,
        duplicates='drop'
    )
    actual_n_bins = valid_grains['Freq_Bin'].nunique()
    print(f"Fallback: Created {actual_n_bins} bins")

# Select one representative grain from each bin
# Strategy: choose the grain closest to the median frequency in each bin
representative_grains = []

for bin_id in sorted(valid_grains['Freq_Bin'].unique()):
    bin_grains = valid_grains[valid_grains['Freq_Bin'] == bin_id]
    
    # Find grain closest to median frequency
    median_freq = bin_grains['Peak SAW Frequency (Hz)'].median()
    distances = (bin_grains['Peak SAW Frequency (Hz)'] - median_freq).abs()
    representative_idx = distances.idxmin()
    
    representative_grains.append(bin_grains.loc[representative_idx])

# Create dataframe of representative grains
rep_df = pd.DataFrame(representative_grains)
rep_df = rep_df.sort_values('Peak SAW Frequency (Hz)').reset_index(drop=True)

print(f"\n" + "="*70)
print(f"SELECTED {len(rep_df)} REPRESENTATIVE GRAIN ORIENTATIONS")
print("="*70)

# Display summary
print(f"\nFrequency range: {rep_df['Peak SAW Frequency (Hz)'].min()/1e6:.2f} - {rep_df['Peak SAW Frequency (Hz)'].max()/1e6:.2f} MHz")
print(f"Frequency spacing: ~{(rep_df['Peak SAW Frequency (Hz)'].max() - rep_df['Peak SAW Frequency (Hz)'].min())/(len(rep_df)-1)/1e6:.2f} MHz between bins")

# Show first and last few grains
print(f"\nFirst 5 grains:")
for i in range(min(5, len(rep_df))):
    row = rep_df.iloc[i]
    print(f"  Bin {i}: Grain {row['Grain ID']:3.0f}, "
          f"φ₁={np.rad2deg(row['Euler1 (rad)']):6.1f}°, "
          f"Φ={np.rad2deg(row['Euler2 (rad)']):5.1f}°, "
          f"φ₂={np.rad2deg(row['Euler3 (rad)']):6.1f}°, "
          f"f={row['Peak SAW Frequency (Hz)']/1e6:.2f} MHz")

print(f"\nLast 5 grains:")
for i in range(max(0, len(rep_df)-5), len(rep_df)):
    row = rep_df.iloc[i]
    print(f"  Bin {i}: Grain {row['Grain ID']:3.0f}, "
          f"φ₁={np.rad2deg(row['Euler1 (rad)']):6.1f}°, "
          f"Φ={np.rad2deg(row['Euler2 (rad)']):5.1f}°, "
          f"φ₂={np.rad2deg(row['Euler3 (rad)']):6.1f}°, "
          f"f={row['Peak SAW Frequency (Hz)']/1e6:.2f} MHz")

# Save to CSV
output_file = Path(__file__).parent / "representative_grain_orientations.csv"
rep_df[['Grain ID', 'Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)', 
        'Peak SAW Frequency (Hz)', 'Size (um^2)']].to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Generate Python list for easy copy-paste into sensitivity script
print(f"\n" + "="*70)
print("READY-TO-USE FORMAT FOR SENSITIVITY ANALYSIS")
print("="*70)
print(f"\n# Representative grain orientations (φ₁, Φ, φ₂) in radians")
print(f"# Stratified by SAW frequency across {len(rep_df)} bins")
print("representative_orientations_rad = [")
for i, row in rep_df.iterrows():
    freq_mhz = row['Peak SAW Frequency (Hz)'] / 1e6
    print(f"    ({row['Euler1 (rad)']:.6f}, {row['Euler2 (rad)']:.6f}, {row['Euler3 (rad)']:.6f}),  "
          f"# Grain {row['Grain ID']:.0f}, {freq_mhz:.2f} MHz")
print("]")

print(f"\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Copy the 'representative_orientations_rad' list above")
print("2. Update global_variance_sensitivity.py to:")
print("   - Use Bunge convention (φ₁, Φ, φ₂) instead of (α, β, γ)")
print("   - Loop over each representative orientation")
print("   - Test perturbations of ±1°, ±2.5°, and ±5°")
print("="*70)

