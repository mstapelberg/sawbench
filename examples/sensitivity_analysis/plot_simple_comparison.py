"""
Simple side-by-side bar chart comparing sensitivity across error scales.

This creates a clean publication-ready figure showing how parameter importance
changes with EBSD measurement error magnitude.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent / "results" / "multi_orientation_sensitivity"

# Parameter formatting
PARAM_LABELS = {
    "K": r"$K$",
    "D": r"$D$",
    "G": r"$G$",
    "rho": r"$\rho$",
    "phi1_err_deg": r"$\varphi_1$",
    "Phi_err_deg": r"$\Phi$",
    "phi2_err_deg": r"$\varphi_2$",
    "psi_err_deg": r"$\psi$",
}

# Semantic colors
PARAM_COLORS = {
    "K": "#2A33C3",      # Blue - elastic
    "D": "#4056C7",      # Blue - elastic
    "G": "#5B7ACB",      # Blue - elastic
    "rho": "#0B7285",    # Teal - density
    "phi1_err_deg": "#A35D00",  # Orange - EBSD
    "Phi_err_deg": "#B97200",   # Orange - EBSD
    "phi2_err_deg": "#CF8700",  # Orange - EBSD
    "psi_err_deg": "#8F2D56",   # Magenta - sample alignment
}

# Load results
results_file = RESULTS_DIR / "all_results.json"
if not results_file.exists():
    print(f"Error: {results_file} not found. Run the sensitivity analysis first!")
    exit(1)

with open(results_file, 'r') as f:
    all_results = json.load(f)

error_scales = sorted(set(r["euler_error_scale_deg"] for r in all_results))

print(f"Creating comparison plot for error scales: {error_scales}")

# Create figure with side-by-side subplots
fig, axes = plt.subplots(1, len(error_scales), 
                        figsize=(5 + 3*len(error_scales), 7), 
                        sharey=True)
if len(error_scales) == 1:
    axes = [axes]

for i, error_scale in enumerate(error_scales):
    ax = axes[i]
    
    # Get results for this error scale
    results_at_scale = [r for r in all_results 
                       if r["euler_error_scale_deg"] == error_scale]
    
    # Aggregate across orientations
    param_names = results_at_scale[0]["parameters"]
    S1_all = np.array([r["S1"] for r in results_at_scale])
    ST_all = np.array([r["ST"] for r in results_at_scale])
    
    S1_mean = S1_all.mean(axis=0)
    S1_std = S1_all.std(axis=0)
    ST_mean = ST_all.mean(axis=0)
    ST_std = ST_all.std(axis=0)
    
    # Sort by ST_mean descending
    sort_idx = np.argsort(ST_mean)[::-1]
    param_names_sorted = [param_names[j] for j in sort_idx]
    S1_mean_sorted = S1_mean[sort_idx]
    S1_std_sorted = S1_std[sort_idx]
    ST_mean_sorted = ST_mean[sort_idx]
    ST_std_sorted = ST_std[sort_idx]
    
    # Get labels and colors
    labels = [PARAM_LABELS.get(p, p) for p in param_names_sorted]
    colors = [PARAM_COLORS.get(p, '#666666') for p in param_names_sorted]
    y_pos = np.arange(len(labels))
    
    # Plot ST (total) as gray outline bars
    ax.barh(y_pos, ST_mean_sorted, 
           color='none', edgecolor='gray', linewidth=2.5, height=0.7, 
           label='$S_T$' if i == 0 else None)
    
    # Plot S1 (first-order) as colored filled bars with error bars
    for j, (y, s1, err, color) in enumerate(zip(y_pos, S1_mean_sorted, S1_std_sorted, colors)):
        ax.barh(y, s1, color=color, alpha=0.85, height=0.5)
        # Add error bar
        ax.errorbar(s1, y, xerr=err, fmt='none', 
                   ecolor=color, elinewidth=2, capsize=4, capthick=2)
    
    # Add S1 label to legend only for first subplot
    if i == 0:
        ax.barh([-1], [0], color='gray', alpha=0.85, height=0.5, label='$S_1$')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Sensitivity Index', fontsize=13, fontweight='bold')
    ax.set_title(f'±{error_scale}°', fontsize=15, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, min(1.0, max(ST_mean_sorted.max() * 1.15, 0.5)))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add vertical line at 0.1 (10% threshold)
    ax.axvline(0.1, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Legend only on first panel
    if i == 0:
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95, 
                 edgecolor='gray', fancybox=False)

# Overall title
fig.suptitle('Sensitivity to EBSD Measurement Error\n(Averaged Across 28 Grain Orientations)', 
             fontsize=16, fontweight='bold', y=0.98)

# Add footnote explaining colors
fig.text(0.5, 0.01, 
         'Colors: Blue = Elastic constants | Teal = Density | Orange = EBSD orientation errors | Magenta = Sample alignment error',
         ha='center', fontsize=10, style='italic', color='dimgray')

fig.tight_layout(rect=[0, 0.03, 1, 0.96])

# Save
output_file = RESULTS_DIR / "plots" / "simple_comparison.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {output_file}")

# Also save as PDF for publication
output_pdf = output_file.with_suffix('.pdf')
fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_pdf}")

plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
for error_scale in error_scales:
    results_at_scale = [r for r in all_results 
                       if r["euler_error_scale_deg"] == error_scale]
    param_names = results_at_scale[0]["parameters"]
    ST_all = np.array([r["ST"] for r in results_at_scale])
    ST_mean = ST_all.mean(axis=0)
    
    print(f"\n±{error_scale}° Error Scale:")
    print("-" * 70)
    
    # Sort by ST
    sort_idx = np.argsort(ST_mean)[::-1]
    for rank, idx in enumerate(sort_idx, 1):
        param = param_names[idx]
        st = ST_mean[idx]
        label = PARAM_LABELS.get(param, param)
        print(f"  {rank}. {label:10s} ({param:15s}): ST = {st:.3f}")

