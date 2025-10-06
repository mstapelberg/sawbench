"""
Analyze texture strength and recommend grain selection strategy for sensitivity analysis.

This script helps you decide:
1. Is your sample textured or random?
2. What clustering threshold should you use?
3. Which grains to use for sensitivity analysis?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load your results
results_dir = Path("./results/interactive_analysis_output_4000")

# Load grain data
grains_roi = pd.read_csv(results_dir / "grains_roi.csv")
euler_angles_roi = pd.read_csv(results_dir / "euler_angles_roi.csv")
top_orientations = pd.read_csv(results_dir / "top_orientations.csv")

print("="*70)
print("TEXTURE ANALYSIS AND GRAIN SELECTION STRATEGY")
print("="*70)

# 1. Basic statistics
n_grains = len(grains_roi)
print(f"\n1. SAMPLE STATISTICS")
print(f"   Total grains in ROI: {n_grains}")
print(f"   Total area: {grains_roi['Size (um^2)'].sum():.0f} μm²")

# 2. Clustering results
print(f"\n2. CLUSTERING RESULTS (5° threshold)")
print(f"   Number of clusters found: {len(top_orientations)}")
print(f"   Largest cluster: {top_orientations.iloc[0]['N_Grains']:.0f} grains ({top_orientations.iloc[0]['Fraction_by_area']*100:.1f}% area)")
print(f"   Top 5 clusters combined: {top_orientations['Fraction_by_area'].sum()*100:.1f}% of area")

# Assess texture strength
top5_fraction = top_orientations['Fraction_by_area'].sum()
if top5_fraction < 0.15:
    texture_strength = "WEAK/RANDOM"
    recommendation = "random sampling"
elif top5_fraction < 0.40:
    texture_strength = "MODERATE"
    recommendation = "cluster-based + random sampling"
else:
    texture_strength = "STRONG"
    recommendation = "cluster-based sampling"

print(f"\n3. TEXTURE ASSESSMENT")
print(f"   Texture strength: {texture_strength}")
print(f"   Recommended strategy: {recommendation}")

# 4. Suggest different clustering thresholds
print(f"\n4. CLUSTERING THRESHOLD RECOMMENDATIONS")
print(f"   Current threshold: 5.0°")
print(f"   Try these thresholds to see if you get better clustering:")
print(f"   - 7.5° (moderate grouping)")
print(f"   - 10.0° (loose grouping)")
print(f"   - 15.0° (very loose grouping)")

# 5. Grain selection strategies
print(f"\n5. GRAIN SELECTION STRATEGIES FOR SENSITIVITY ANALYSIS")
print(f"   " + "="*65)

if texture_strength == "WEAK/RANDOM":
    print(f"\n   Strategy A: RANDOM SAMPLING (Recommended for weak texture)")
    print(f"   " + "-"*65)
    print(f"   Since your sample is weakly textured, randomly sample grains")
    print(f"   across the full orientation space.")
    print(f"")
    print(f"   Suggested approach:")
    print(f"   1. Randomly select 10-20 grains from your ROI")
    print(f"   2. Use their orientations for sensitivity analysis")
    print(f"   3. This gives you good coverage of orientation space")
    print(f"")
    
    # Generate random sample
    np.random.seed(42)  # For reproducibility
    n_random = min(10, n_grains)
    random_indices = np.random.choice(n_grains, size=n_random, replace=False)
    random_grains = grains_roi.iloc[random_indices]
    
    print(f"   Random sample of {n_random} grains:")
    print(f"   " + "-"*65)
    for i, (idx, grain) in enumerate(random_grains.iterrows(), 1):
        euler = euler_angles_roi[euler_angles_roi['Grain_ID'] == grain['Grain ID']].iloc[0]
        print(f"   Grain {i}: ID={grain['Grain ID']:.0f}, "
              f"Euler=({euler['phi1_deg']:.1f}°, {euler['PHI_deg']:.1f}°, {euler['phi2_deg']:.1f}°), "
              f"Area={grain['Size (um^2)']:.0f} μm²")
    
    print(f"\n   Strategy B: STRATIFIED SAMPLING BY SAW FREQUENCY")
    print(f"   " + "-"*65)
    print(f"   Select grains from different SAW frequency bins to ensure")
    print(f"   you test orientations that give different velocities.")
    
    # Bin by frequency
    valid_freqs = grains_roi['Peak SAW Frequency (Hz)'].dropna()
    if len(valid_freqs) > 0:
        freq_bins = pd.qcut(valid_freqs, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        grains_with_bins = grains_roi[grains_roi['Peak SAW Frequency (Hz)'].notna()].copy()
        grains_with_bins['Freq_Bin'] = freq_bins
        
        print(f"")
        print(f"   Frequency distribution:")
        for bin_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            if bin_name in grains_with_bins['Freq_Bin'].values:
                bin_grains = grains_with_bins[grains_with_bins['Freq_Bin'] == bin_name]
                freq_range = (bin_grains['Peak SAW Frequency (Hz)'].min()/1e6, 
                             bin_grains['Peak SAW Frequency (Hz)'].max()/1e6)
                print(f"   {bin_name:12s}: {len(bin_grains):3d} grains, "
                      f"{freq_range[0]:.1f}-{freq_range[1]:.1f} MHz")
        
        print(f"")
        print(f"   Suggested: Pick 2 grains from each frequency bin")
        print(f"   This ensures you test the full range of SAW velocities")

elif texture_strength == "MODERATE":
    print(f"\n   Strategy: HYBRID APPROACH")
    print(f"   " + "-"*65)
    print(f"   1. Use the top 3-5 cluster centers (covers ~{top5_fraction*100:.0f}% of sample)")
    print(f"   2. Add 5-10 random grains from unclustered regions")
    print(f"   3. This balances common orientations with rare ones")

else:  # STRONG texture
    print(f"\n   Strategy: CLUSTER-BASED SAMPLING")
    print(f"   " + "-"*65)
    print(f"   Your sample is textured! Use cluster centers.")
    print(f"   The top 5 clusters represent {top5_fraction*100:.0f}% of your sample.")

# 6. Specific recommendations
print(f"\n6. SPECIFIC RECOMMENDATIONS FOR YOUR SAMPLE")
print(f"   " + "="*65)

if texture_strength == "WEAK/RANDOM":
    print(f"\n   ✓ Your sample appears to be weakly textured or random")
    print(f"   ✓ This is actually GOOD for sensitivity analysis!")
    print(f"   ✓ You naturally have grains with many different orientations")
    print(f"")
    print(f"   RECOMMENDED WORKFLOW:")
    print(f"   1. Try re-running with clustering_threshold_deg=10.0")
    print(f"      (might find slightly larger clusters)")
    print(f"   2. Use Strategy A (random sampling) above")
    print(f"   3. Select 10-15 random grains for sensitivity analysis")
    print(f"   4. This gives you good orientation coverage")
    print(f"")
    print(f"   WHY THIS IS GOOD:")
    print(f"   - You'll test many different orientations")
    print(f"   - Your sensitivity analysis will be more comprehensive")
    print(f"   - You're not missing any dominant texture")

else:
    print(f"\n   ✓ Use the cluster centers from top_orientations.csv")
    print(f"   ✓ Consider also testing a few random outliers")

# 7. Generate output for easy copy-paste
print(f"\n7. READY-TO-USE EULER ANGLES FOR SENSITIVITY ANALYSIS")
print(f"   " + "="*65)

if texture_strength == "WEAK/RANDOM":
    print(f"\n   Random sample (copy these into your sensitivity script):")
    print(f"   " + "-"*65)
    print(f"   orientations_to_test = [")
    for i, (idx, grain) in enumerate(random_grains.iterrows(), 1):
        euler = euler_angles_roi[euler_angles_roi['Grain_ID'] == grain['Grain ID']].iloc[0]
        print(f"       ({euler['phi1_rad']:.4f}, {euler['PHI_rad']:.4f}, {euler['phi2_rad']:.4f}),  # Grain {i}")
    print(f"   ]")
else:
    print(f"\n   Cluster centers (copy these into your sensitivity script):")
    print(f"   " + "-"*65)
    print(f"   orientations_to_test = [")
    for i, row in top_orientations.iterrows():
        print(f"       ({row['Mean_phi1_rad']:.4f}, {row['Mean_Phi_rad']:.4f}, {row['Mean_phi2_rad']:.4f}),  # Cluster {i+1}")
    print(f"   ]")

print(f"\n" + "="*70)
print(f"SUMMARY")
print(f"="*70)
print(f"Texture: {texture_strength}")
print(f"Strategy: {recommendation.upper()}")
if texture_strength == "WEAK/RANDOM":
    print(f"Action: Use random sampling (10-15 grains)")
    print(f"Benefit: Comprehensive orientation coverage")
else:
    print(f"Action: Use cluster centers + some random grains")
    print(f"Benefit: Focus on common orientations")
print(f"="*70)
