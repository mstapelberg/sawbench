"""
Visualize sensitivity analysis results across multiple orientations and error scales.

This script creates comprehensive plots showing:
1. Average sensitivity indices across all orientations for each error scale
2. Sensitivity variation across different grain orientations
3. Comparison of sensitivity between error scales
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent / "results" / "multi_orientation_sensitivity"
OUTPUT_DIR = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter name formatting
PARAM_LABELS = {
    "K": r"$K$ (bulk modulus)",
    "D": r"$D$ (C$_{11}$-C$_{12}$)",
    "G": r"$G$ (C$_{44}$)",
    "rho": r"$\rho$ (density)",
    "phi1_err_deg": r"$\varphi_1$ error (EBSD)",
    "Phi_err_deg": r"$\Phi$ error (EBSD)",
    "phi2_err_deg": r"$\varphi_2$ error (EBSD)",
    "psi_err_deg": r"$\psi$ error (sample alignment)",
}

# Color scheme: group by parameter type
PARAM_COLORS = {
    # Elastic constants (blues)
    "K": "#2A33C3",
    "D": "#4056C7", 
    "G": "#5B7ACB",
    # Density (teal)
    "rho": "#0B7285",
    # Euler angles (oranges/browns)
    "phi1_err_deg": "#A35D00",
    "Phi_err_deg": "#B97200",
    "phi2_err_deg": "#CF8700",
    # Sample angle (magenta - systematic error)
    "psi_err_deg": "#8F2D56",
}

def format_param_label(param_name: str) -> str:
    """Format parameter name for plotting."""
    return PARAM_LABELS.get(param_name, param_name)

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("VISUALIZING MULTI-ORIENTATION SENSITIVITY RESULTS")
print("="*70)

# Load all results
results_file = RESULTS_DIR / "all_results.json"
if not results_file.exists():
    print(f"\nError: Results file not found: {results_file}")
    print("Please run global_variance_sensitivity_multi_orientation.py first!")
    exit(1)

with open(results_file, 'r') as f:
    all_results = json.load(f)

print(f"\nLoaded {len(all_results)} sensitivity analysis results")

# Get unique error scales
error_scales = sorted(set(r["euler_error_scale_deg"] for r in all_results))
print(f"Error scales tested: {error_scales}")

# ============================================================================
# FIGURE 1: Average sensitivity by error scale (comparison)
# ============================================================================

print("\nCreating Figure 1: Average sensitivity comparison across error scales...")

fig1, axes = plt.subplots(1, len(error_scales), figsize=(6*len(error_scales), 6), 
                          sharey=True)
if len(error_scales) == 1:
    axes = [axes]

for i, error_scale in enumerate(error_scales):
    ax = axes[i]
    
    # Get results for this error scale
    results_at_scale = [r for r in all_results 
                       if r["euler_error_scale_deg"] == error_scale]
    
    if not results_at_scale:
        continue
    
    # Aggregate across orientations
    param_names = results_at_scale[0]["parameters"]
    S1_all = np.array([r["S1"] for r in results_at_scale])
    ST_all = np.array([r["ST"] for r in results_at_scale])
    
    S1_mean = S1_all.mean(axis=0)
    S1_std = S1_all.std(axis=0)
    ST_mean = ST_all.mean(axis=0)
    ST_std = ST_all.std(axis=0)
    
    # Sort by ST_mean
    sort_idx = np.argsort(ST_mean)[::-1]
    param_names_sorted = [param_names[j] for j in sort_idx]
    S1_mean_sorted = S1_mean[sort_idx]
    S1_std_sorted = S1_std[sort_idx]
    ST_mean_sorted = ST_mean[sort_idx]
    ST_std_sorted = ST_std[sort_idx]
    
    # Format labels and get semantic colors
    labels = [format_param_label(p) for p in param_names_sorted]
    colors = [PARAM_COLORS.get(p, '#666666') for p in param_names_sorted]
    y_pos = np.arange(len(labels))
    
    # Plot ST as outline
    ax.barh(y_pos, ST_mean_sorted, xerr=ST_std_sorted, 
           color='none', edgecolor='0.5', linewidth=2, height=0.7, 
           label='$S_T$ (total)', capsize=3, error_kw={'elinewidth': 1})
    
    # Plot S1 as filled bars with semantic colors
    bars = ax.barh(y_pos, S1_mean_sorted, xerr=S1_std_sorted,
                   color=colors, alpha=0.85, height=0.5, 
                   label='$S_1$ (first-order)', capsize=3,
                   error_kw={'elinewidth': 1.5, 'ecolor': colors})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Sensitivity Index', fontsize=12, fontweight='bold')
    ax.set_title(f'EBSD Error: ±{error_scale}°', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(1.0, ST_mean_sorted.max() * 1.1))
    
    if i == 0:
        # Add legend for first panel
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        # Add color coding explanation as text
        ax.text(0.02, 0.98, 
                'Colors:\nBlue = Elastic\nTeal = Density\nOrange = EBSD\nMagenta = Sample',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

fig1.suptitle('Average Sobol Sensitivity Indices\nAcross All Grain Orientations', 
             fontsize=14, fontweight='bold', y=0.98)
fig1.tight_layout(rect=[0, 0, 1, 0.96])

output_file = OUTPUT_DIR / "fig1_average_sensitivity_by_error_scale.png"
fig1.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# ============================================================================
# FIGURE 2: Sensitivity variation across orientations
# ============================================================================

print("\nCreating Figure 2: Sensitivity variation across orientations...")

# Focus on one error scale (the middle one)
mid_error_scale = error_scales[len(error_scales)//2]
results_mid = [r for r in all_results if r["euler_error_scale_deg"] == mid_error_scale]

if results_mid:
    param_names = results_mid[0]["parameters"]
    n_params = len(param_names)
    
    # Extract S1 and ST for each orientation
    orientations = [r["orientation_idx"] for r in results_mid]
    S1_matrix = np.array([r["S1"] for r in results_mid])  # shape: (n_orientations, n_params)
    ST_matrix = np.array([r["ST"] for r in results_mid])
    
    # Create subplots for each parameter
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Plot S1 and ST across orientations
        ax.plot(orientations, ST_matrix[:, i], 'o-', color='#2A33C3', 
               linewidth=2, markersize=6, label='$S_T$', alpha=0.7)
        ax.plot(orientations, S1_matrix[:, i], 's-', color='#A35D00', 
               linewidth=2, markersize=5, label='$S_1$', alpha=0.7)
        
        # Add mean line
        ST_mean = ST_matrix[:, i].mean()
        ax.axhline(ST_mean, color='#2A33C3', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Orientation Index', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sensitivity Index', fontsize=10, fontweight='bold')
        ax.set_title(format_param_label(param), fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(1.0, ST_matrix[:, i].max() * 1.1))
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
    
    fig2.suptitle(f'Sensitivity Variation Across Grain Orientations\n(EBSD Error: ±{mid_error_scale}°)', 
                 fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = OUTPUT_DIR / f"fig2_orientation_variation_{mid_error_scale}deg.png"
    fig2.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

# ============================================================================
# FIGURE 3: Error scale comparison (how sensitivity changes with error magnitude)
# ============================================================================

print("\nCreating Figure 3: Sensitivity vs error scale...")

if len(error_scales) > 1:
    # For each parameter, show how average sensitivity changes with error scale
    param_names = all_results[0]["parameters"]
    
    # Collect data
    data_by_scale = {}
    for scale in error_scales:
        results_at_scale = [r for r in all_results if r["euler_error_scale_deg"] == scale]
        S1_all = np.array([r["S1"] for r in results_at_scale])
        ST_all = np.array([r["ST"] for r in results_at_scale])
        data_by_scale[scale] = {
            'S1_mean': S1_all.mean(axis=0),
            'S1_std': S1_all.std(axis=0),
            'ST_mean': ST_all.mean(axis=0),
            'ST_std': ST_all.std(axis=0),
        }
    
    fig3, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        ST_means = [data_by_scale[s]['ST_mean'][i] for s in error_scales]
        ST_stds = [data_by_scale[s]['ST_std'][i] for s in error_scales]
        S1_means = [data_by_scale[s]['S1_mean'][i] for s in error_scales]
        S1_stds = [data_by_scale[s]['S1_std'][i] for s in error_scales]
        
        ax.errorbar(error_scales, ST_means, yerr=ST_stds, 
                   marker='o', linewidth=2, markersize=8, capsize=5,
                   color='#2A33C3', label='$S_T$', alpha=0.8)
        ax.errorbar(error_scales, S1_means, yerr=S1_stds, 
                   marker='s', linewidth=2, markersize=7, capsize=5,
                   color='#A35D00', label='$S_1$', alpha=0.8)
        
        ax.set_xlabel('EBSD Error Scale (degrees)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sensitivity Index', fontsize=10, fontweight='bold')
        ax.set_title(format_param_label(param), fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(1.0, max(ST_means) * 1.15))
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper left')
    
    fig3.suptitle('How Sensitivity Changes with EBSD Measurement Error Scale\n(Averaged Across All Orientations)', 
                 fontsize=14, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = OUTPUT_DIR / "fig3_sensitivity_vs_error_scale.png"
    fig3.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

# ============================================================================
# FIGURE 4: Heatmap of sensitivities
# ============================================================================

print("\nCreating Figure 4: Sensitivity heatmaps...")

fig4, axes = plt.subplots(2, len(error_scales), figsize=(6*len(error_scales), 10))
if len(error_scales) == 1:
    axes = axes.reshape(-1, 1)

for col, error_scale in enumerate(error_scales):
    results_at_scale = [r for r in all_results 
                       if r["euler_error_scale_deg"] == error_scale]
    
    if not results_at_scale:
        continue
    
    param_names = results_at_scale[0]["parameters"]
    orientations = [r["orientation_idx"] for r in results_at_scale]
    
    S1_matrix = np.array([r["S1"] for r in results_at_scale]).T  # shape: (n_params, n_orientations)
    ST_matrix = np.array([r["ST"] for r in results_at_scale]).T
    
    # Format parameter labels
    param_labels = [format_param_label(p) for p in param_names]
    
    # S1 heatmap
    ax_s1 = axes[0, col]
    im1 = ax_s1.imshow(S1_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax_s1.set_yticks(range(len(param_labels)))
    ax_s1.set_yticklabels(param_labels, fontsize=10)
    ax_s1.set_xlabel('Orientation Index', fontsize=11, fontweight='bold')
    ax_s1.set_title(f'$S_1$ (First-Order)\n±{error_scale}°', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax_s1, fraction=0.046, pad=0.04)
    
    # ST heatmap
    ax_st = axes[1, col]
    im2 = ax_st.imshow(ST_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax_st.set_yticks(range(len(param_labels)))
    ax_st.set_yticklabels(param_labels, fontsize=10)
    ax_st.set_xlabel('Orientation Index', fontsize=11, fontweight='bold')
    ax_st.set_title(f'$S_T$ (Total)\n±{error_scale}°', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax_st, fraction=0.046, pad=0.04)

fig4.suptitle('Sensitivity Heatmaps: Parameters × Orientations', 
             fontsize=14, fontweight='bold', y=0.995)
fig4.tight_layout(rect=[0, 0, 1, 0.99])

output_file = OUTPUT_DIR / "fig4_sensitivity_heatmaps.png"
fig4.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nGenerated plots in: {OUTPUT_DIR}")
print("\nFiles created:")
print("  1. fig1_average_sensitivity_by_error_scale.png")
print("  2. fig2_orientation_variation_<scale>deg.png")
if len(error_scales) > 1:
    print("  3. fig3_sensitivity_vs_error_scale.png")
print("  4. fig4_sensitivity_heatmaps.png")
print("\nInterpretation guide:")
print("  - S1 (first-order): Direct effect of parameter alone")
print("  - ST (total): Includes parameter + all its interactions")
print("  - ST - S1: Importance of parameter interactions")
print("  - High variance across orientations: Sensitivity depends on grain orientation")
print("="*70)

