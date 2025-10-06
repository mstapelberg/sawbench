"""
Interactive EBSD grain analysis with subregion selection.

This module provides a streamlined interface for:
1. Loading EBSD data from CTF files
2. Interactively selecting a region of interest
3. Calculating SAW frequencies for grains in the selected region
4. Visualizing results with overlaid histograms and spatial frequency maps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .materials import Material
from .subregion_selector import SubregionSelector
from .grains import calculate_saw_frequencies_for_ebsd_grains
from .orientation_clustering import find_top_orientations


def interactive_grain_analysis(
    ctf_path: str,
    material: Material,
    wavelength: float,
    saw_calc_angle_deg: float = 135.0,
    num_workers: Optional[int] = 16,
    output_dir: Optional[str] = None,
    saw_calc_sampling: int = 400,
    saw_calc_psaw: int = 0,
    boundary_def: float = 5.0,
    min_grain_size: int = 10,
    show_euler_analysis: bool = True,
    find_top_n_orientations: Optional[int] = None,
    clustering_threshold_deg: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Interactive workflow for EBSD grain analysis with subregion selection.
    
    This function provides a complete workflow:
    1. Loads EBSD data and displays the grain map
    2. Allows interactive polygon selection of a region of interest
    3. Calculates SAW frequencies for all grains
    4. Generates visualizations comparing the selected region to the full dataset
    
    Args:
        ctf_path: Path to the CTF file
        material: Material object with elastic constants and density
        wavelength: SAW wavelength in meters (e.g., 8.8e-6)
        saw_calc_angle_deg: Angle for SAW calculation in degrees (default: 135.0)
        num_workers: Number of parallel workers for SAW calculation (default: 16)
        output_dir: Optional directory to save outputs (plots, CTF, CSV). If None, only displays interactively.
        saw_calc_sampling: Sampling parameter for SAW calculator (default: 400)
        saw_calc_psaw: PSAW flag for SAW calculator (default: 0)
        boundary_def: Misorientation threshold in degrees for grain boundaries (default: 5.0)
        min_grain_size: Minimum grain size in pixels (default: 10)
        show_euler_analysis: Whether to show Euler angle distribution analysis (default: True)
        find_top_n_orientations: If set, find the top N most prevalent orientations using 
            symmetry-aware clustering (default: None, no clustering)
        clustering_threshold_deg: Maximum misorientation angle (degrees) for clustering (default: 5.0)
    
    Returns:
        Tuple containing:
        - grains_df_all: DataFrame with SAW frequencies for all grains
        - grains_df_roi: DataFrame with SAW frequencies for grains in selected region
        - metadata: Dictionary with selection info and statistics
    
    Examples:
        >>> from sawbench import Material
        >>> material = Material(
        ...     formula='V',
        ...     C11=229e9, C12=119e9, C44=43e9,
        ...     density=6110,
        ...     crystal_class='cubic'
        ... )
        >>> grains_all, grains_roi, info = interactive_grain_analysis(
        ...     ctf_path='data/sample.ctf',
        ...     material=material,
        ...     wavelength=8.8e-6,
        ...     output_dir='./analysis_results'
        ... )
        >>> print(f"Analyzed {len(grains_roi)} grains in selected region")
    """
    print("=" * 60)
    print("Interactive EBSD Grain Analysis")
    print("=" * 60)
    
    # Step 1: Load EBSD data and create selector
    print(f"\nLoading EBSD data from: {ctf_path}")
    selector = SubregionSelector(
        ctf_path,
        data_type="OxfordText",
        boundary_def=boundary_def,
        min_grain_size=min_grain_size
    )
    
    ebsd_map = selector.ebsd_map
    print(f"Loaded EBSD map with {len(ebsd_map.grainList)} total grains")
    print(f"Map dimensions: {selector._nx} x {selector._ny} pixels")
    print(f"Step size: {selector._step_um:.3f} μm")
    
    # Step 2: Interactive region selection
    print("\n" + "=" * 60)
    print("Interactive Region Selection")
    print("=" * 60)
    print("Instructions:")
    print("  - Click on the map to add polygon vertices")
    print("  - Press Enter when done to finalize selection")
    print("  - Close the window to cancel")
    print()
    
    subregion_df = selector.interactive_region_selection()
    
    if subregion_df is None or subregion_df.empty or selector.selected_region is None:
        print("\nNo region was selected. Exiting.")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    print(f"\n✓ Selected region with {len(subregion_df)} pixels")
    
    # Step 3: Get grain IDs in the selected polygon
    if selector.selected_region.get('type') != 'polygon':
        print("ERROR: Selection must be a polygon. Exiting.")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    vertices = selector.selected_region.get('coords', [])
    grain_ids_in_roi = selector.get_grain_ids_in_polygon(vertices)
    print(f"✓ Found {len(grain_ids_in_roi)} grains in the selected region")
    
    if len(grain_ids_in_roi) == 0:
        print("\nNo grains found in selection. Exiting.")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Step 4: Calculate SAW frequencies for all grains
    print("\n" + "=" * 60)
    print("Calculating SAW Frequencies")
    print("=" * 60)
    print(f"Material: {material.formula}")
    print(f"Wavelength: {wavelength*1e6:.2f} μm")
    print(f"SAW calculation angle: {saw_calc_angle_deg}°")
    print(f"Parallel workers: {num_workers}")
    print()
    
    grains_df_all = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=material,
        wavelength=wavelength,
        saw_calc_angle_deg=saw_calc_angle_deg,
        saw_calc_sampling=saw_calc_sampling,
        saw_calc_psaw=saw_calc_psaw,
        num_workers=num_workers
    )
    
    # Filter to ROI grains
    grains_df_roi = grains_df_all[grains_df_all['Grain ID'].isin(grain_ids_in_roi)].copy()
    
    print("\n✓ Calculated SAW frequencies:")
    print(f"  - Total grains: {len(grains_df_all)}")
    print(f"  - Grains in ROI: {len(grains_df_roi)}")
    
    # Get valid frequencies
    valid_freqs_all = grains_df_all['Peak SAW Frequency (Hz)'].dropna()
    valid_freqs_roi = grains_df_roi['Peak SAW Frequency (Hz)'].dropna()
    
    if valid_freqs_roi.empty:
        print("\nWARNING: No valid SAW frequencies calculated for selected region.")
        return grains_df_all, grains_df_roi, {'vertices': vertices}
    
    print(f"  - Valid frequencies (All): {len(valid_freqs_all)}")
    print(f"  - Valid frequencies (ROI): {len(valid_freqs_roi)}")
    print("\nFrequency ranges:")
    print(f"  - All grains: {valid_freqs_all.min()/1e6:.2f} - {valid_freqs_all.max()/1e6:.2f} MHz")
    print(f"  - ROI grains: {valid_freqs_roi.min()/1e6:.2f} - {valid_freqs_roi.max()/1e6:.2f} MHz")
    
    # Convert radians to degrees for display (only if Euler analysis is enabled)
    if show_euler_analysis:
        euler_all_deg = grains_df_all[['Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)']].values * 180 / np.pi
        euler_roi_deg = grains_df_roi[['Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)']].values * 180 / np.pi
        
        print("\nEuler angle ranges (degrees):")
        print(f"  - All grains φ₁: {euler_all_deg[:, 0].min():.1f} - {euler_all_deg[:, 0].max():.1f}")
        print(f"  - All grains Φ:  {euler_all_deg[:, 1].min():.1f} - {euler_all_deg[:, 1].max():.1f}")
        print(f"  - All grains φ₂: {euler_all_deg[:, 2].min():.1f} - {euler_all_deg[:, 2].max():.1f}")
        print(f"  - ROI grains φ₁: {euler_roi_deg[:, 0].min():.1f} - {euler_roi_deg[:, 0].max():.1f}")
        print(f"  - ROI grains Φ:  {euler_roi_deg[:, 1].min():.1f} - {euler_roi_deg[:, 1].max():.1f}")
        print(f"  - ROI grains φ₂: {euler_roi_deg[:, 2].min():.1f} - {euler_roi_deg[:, 2].max():.1f}")
    
    # Step 5: Create visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Main SAW frequency analysis figure (original 2-panel layout)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Overlaid histogram
    freq_min = min(valid_freqs_all.min(), valid_freqs_roi.min()) / 1e6
    freq_max = max(valid_freqs_all.max(), valid_freqs_roi.max()) / 1e6
    bins = np.linspace(freq_min, freq_max, 30)
    
    ax1.hist(valid_freqs_all/1e6, bins=bins, alpha=0.5, edgecolor='black', 
            color='lightgray', label=f'All grains (n={len(valid_freqs_all)})')
    ax1.hist(valid_freqs_roi/1e6, bins=bins, alpha=0.7, edgecolor='darkred', 
            color='red', label=f'Selected region (n={len(valid_freqs_roi)})')
    ax1.set_xlabel('SAW Frequency (MHz)', fontsize=12)
    ax1.set_ylabel('Number of Grains', fontsize=12)
    ax1.set_title('SAW Frequency Distribution', fontsize=14)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spatial map colored by SAW frequency with selected polygon
    # Create a 2D frequency map using the grains array
    grain_map = ebsd_map.grains  # Shape: (ny, nx)
    freq_map = np.full(grain_map.shape, np.nan)
    
    # Map grain IDs to their frequencies
    grain_id_to_freq = dict(zip(
        grains_df_all['Grain ID'], 
        grains_df_all['Peak SAW Frequency (Hz)']
    ))
    
    for grain_id, freq_hz in grain_id_to_freq.items():
        if not np.isnan(freq_hz):
            freq_map[grain_map == grain_id] = freq_hz / 1e6  # Convert to MHz
    
    # Plot the frequency map
    extent = [0.0, selector._nx * selector._step_um, 0.0, selector._ny * selector._step_um]
    im = ax2.imshow(freq_map, origin='lower', cmap='viridis', 
                   interpolation='nearest', extent=extent, aspect='equal')
    
    # Overlay the selected polygon
    poly_patch = Polygon(vertices, linewidth=3, edgecolor='red', 
                        facecolor='none', label='Selected region')
    ax2.add_patch(poly_patch)
    
    ax2.set_xlabel('X (μm)', fontsize=12)
    ax2.set_ylabel('Y (μm)', fontsize=12)
    ax2.set_title(f'SAW Frequency Map - {material.formula}', fontsize=14)
    ax2.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, label='SAW Frequency (MHz)')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Show main figure
    plt.show()
    
    # Step 5b: Create separate Euler angle analysis figure (if requested)
    if show_euler_analysis:
        print("\n" + "=" * 60)
        print("Generating Euler Angle Analysis")
        print("=" * 60)
        
        fig_euler, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(18, 6))
        
        # phi1 histogram (0 to 360 degrees)
        ax4.hist(euler_all_deg[:, 0], bins=30, alpha=0.5, edgecolor='black', 
                 color='lightgray', label=f'All grains (n={len(euler_all_deg)})', range=(0, 360))
        ax4.hist(euler_roi_deg[:, 0], bins=30, alpha=0.7, edgecolor='darkred', 
                 color='red', label=f'Selected region (n={len(euler_roi_deg)})', range=(0, 360))
        ax4.set_xlabel('φ₁ (degrees)', fontsize=12)
        ax4.set_ylabel('Number of Grains', fontsize=12)
        ax4.set_title('φ₁ Distribution', fontsize=14)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 360)
        
        # PHI histogram (0 to 180 degrees)
        ax5.hist(euler_all_deg[:, 1], bins=30, alpha=0.5, edgecolor='black', 
                 color='lightgray', label=f'All grains (n={len(euler_all_deg)})', range=(0, 180))
        ax5.hist(euler_roi_deg[:, 1], bins=30, alpha=0.7, edgecolor='darkred', 
                 color='red', label=f'Selected region (n={len(euler_roi_deg)})', range=(0, 180))
        ax5.set_xlabel('Φ (degrees)', fontsize=12)
        ax5.set_ylabel('Number of Grains', fontsize=12)
        ax5.set_title('Φ Distribution', fontsize=14)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 180)
        
        # phi2 histogram (0 to 360 degrees)
        ax6.hist(euler_all_deg[:, 2], bins=30, alpha=0.5, edgecolor='black', 
                 color='lightgray', label=f'All grains (n={len(euler_all_deg)})', range=(0, 360))
        ax6.hist(euler_roi_deg[:, 2], bins=30, alpha=0.7, edgecolor='darkred', 
                 color='red', label=f'Selected region (n={len(euler_roi_deg)})', range=(0, 360))
        ax6.set_xlabel('φ₂ (degrees)', fontsize=12)
        ax6.set_ylabel('Number of Grains', fontsize=12)
        ax6.set_title('φ₂ Distribution', fontsize=14)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, 360)
        
        plt.tight_layout()
        plt.show()
    
    # Step 5c: Find top orientations if requested
    top_orientations_df = None
    cluster_labels = None
    if find_top_n_orientations is not None and find_top_n_orientations > 0:
        print("\n" + "=" * 60)
        print(f"Finding Top {find_top_n_orientations} Most Prevalent Orientations")
        print("=" * 60)
        print(f"Using symmetry-aware clustering with threshold: {clustering_threshold_deg}°")
        
        # Perform clustering on ROI grains
        top_orientations_df, cluster_labels, cluster_info = find_top_orientations(
            grains_df_roi,
            n_top=find_top_n_orientations,
            eps_degrees=clustering_threshold_deg,
            min_samples=2,
            weight_by_size=True
        )
        
        print(f"\nTop {len(top_orientations_df)} Orientations Found:")
        print("-" * 60)
        for idx, row in top_orientations_df.iterrows():
            print(f"\nRank {row['Rank']}:")
            print(f"  Number of grains: {row['N_Grains']}")
            print(f"  Total area: {row['Total_Area_um2']:.1f} μm²")
            print(f"  Fraction (by count): {row['Fraction_by_count']*100:.1f}%")
            print(f"  Fraction (by area): {row['Fraction_by_area']*100:.1f}%")
            print(f"  Mean orientation (deg): φ₁={row['Mean_phi1_deg']:.1f}°, "
                  f"Φ={row['Mean_Phi_deg']:.1f}°, φ₂={row['Mean_phi2_deg']:.1f}°")
            print(f"  Spread: {row['Spread_deg']:.2f}° (max: {row['Max_Spread_deg']:.2f}°)")
    
    # Step 6: Save outputs if directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving outputs to: {output_dir}")
        
        # Save main SAW frequency figure
        fig_path = output_dir / "saw_frequency_analysis.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved SAW frequency figure: {fig_path}")
        
        # Save Euler angle figure if analysis was performed
        if show_euler_analysis:
            fig_euler_path = output_dir / "euler_angle_distributions.png"
            fig_euler.savefig(fig_euler_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved Euler angle figure: {fig_euler_path}")
        
        # Save grain data CSVs
        csv_all_path = output_dir / "grains_all.csv"
        grains_df_all.to_csv(csv_all_path, index=False)
        print(f"  ✓ Saved all grains data: {csv_all_path}")
        
        csv_roi_path = output_dir / "grains_roi.csv"
        grains_df_roi.to_csv(csv_roi_path, index=False)
        print(f"  ✓ Saved ROI grains data: {csv_roi_path}")
        
        # Save Euler angles as separate CSV files (if analysis was performed)
        if show_euler_analysis:
            euler_all_path = output_dir / "euler_angles_all.csv"
            euler_all_df = pd.DataFrame({
                'Grain_ID': grains_df_all['Grain ID'],
                'phi1_deg': grains_df_all['Euler1 (rad)'] * 180 / np.pi,
                'PHI_deg': grains_df_all['Euler2 (rad)'] * 180 / np.pi,
                'phi2_deg': grains_df_all['Euler3 (rad)'] * 180 / np.pi,
                'phi1_rad': grains_df_all['Euler1 (rad)'],
                'PHI_rad': grains_df_all['Euler2 (rad)'],
                'phi2_rad': grains_df_all['Euler3 (rad)']
            })
            euler_all_df.to_csv(euler_all_path, index=False)
            print(f"  ✓ Saved all Euler angles: {euler_all_path}")
            
            euler_roi_path = output_dir / "euler_angles_roi.csv"
            euler_roi_df = pd.DataFrame({
                'Grain_ID': grains_df_roi['Grain ID'],
                'phi1_deg': grains_df_roi['Euler1 (rad)'] * 180 / np.pi,
                'PHI_deg': grains_df_roi['Euler2 (rad)'] * 180 / np.pi,
                'phi2_deg': grains_df_roi['Euler3 (rad)'] * 180 / np.pi,
                'phi1_rad': grains_df_roi['Euler1 (rad)'],
                'PHI_rad': grains_df_roi['Euler2 (rad)'],
                'phi2_rad': grains_df_roi['Euler3 (rad)']
            })
            euler_roi_df.to_csv(euler_roi_path, index=False)
            print(f"  ✓ Saved ROI Euler angles: {euler_roi_path}")
        
        # Save top orientations if clustering was performed
        if find_top_n_orientations is not None and top_orientations_df is not None:
            top_orient_path = output_dir / "top_orientations.csv"
            top_orientations_df.to_csv(top_orient_path, index=False)
            print(f"  ✓ Saved top {len(top_orientations_df)} orientations: {top_orient_path}")
            
            # Save cluster assignments
            cluster_assign_path = output_dir / "grain_cluster_assignments.csv"
            cluster_df = pd.DataFrame({
                'Grain_ID': grains_df_roi['Grain ID'].values,
                'Cluster_ID': cluster_labels
            })
            cluster_df.to_csv(cluster_assign_path, index=False)
            print(f"  ✓ Saved cluster assignments: {cluster_assign_path}")
        
        # Save cropped CTF if we have valid CTF data
        if not subregion_df.empty and 'Phase' in subregion_df.columns:
            ctf_path_out = output_dir / "cropped_region.ctf"
            try:
                selector.save_cropped_ctf(str(ctf_path_out), subregion_df)
                print(f"  ✓ Saved cropped CTF: {ctf_path_out}")
            except Exception as e:
                print(f"  ⚠ Could not save cropped CTF: {e}")
        
        # Save metadata
        metadata_path = output_dir / "analysis_metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write("EBSD Grain Analysis Metadata\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"CTF File: {ctf_path}\n")
            f.write(f"Material: {material.formula}\n")
            f.write(f"Wavelength: {wavelength*1e6:.3f} μm\n")
            f.write(f"SAW Angle: {saw_calc_angle_deg}°\n")
            f.write(f"\nTotal Grains: {len(grains_df_all)}\n")
            f.write(f"Grains in ROI: {len(grains_df_roi)}\n")
            f.write(f"Valid Frequencies (All): {len(valid_freqs_all)}\n")
            f.write(f"Valid Frequencies (ROI): {len(valid_freqs_roi)}\n")
            f.write(f"\nFrequency Range (All): {valid_freqs_all.min()/1e6:.2f} - {valid_freqs_all.max()/1e6:.2f} MHz\n")
            f.write(f"Frequency Range (ROI): {valid_freqs_roi.min()/1e6:.2f} - {valid_freqs_roi.max()/1e6:.2f} MHz\n")
            
            # Add Euler angle statistics (if analysis was performed)
            if show_euler_analysis:
                f.write("\nEuler Angle Statistics (degrees):\n")
                f.write("All Grains:\n")
                f.write(f"  φ₁: {euler_all_deg[:, 0].min():.1f} - {euler_all_deg[:, 0].max():.1f} (mean: {euler_all_deg[:, 0].mean():.1f})\n")
                f.write(f"  Φ:  {euler_all_deg[:, 1].min():.1f} - {euler_all_deg[:, 1].max():.1f} (mean: {euler_all_deg[:, 1].mean():.1f})\n")
                f.write(f"  φ₂: {euler_all_deg[:, 2].min():.1f} - {euler_all_deg[:, 2].max():.1f} (mean: {euler_all_deg[:, 2].mean():.1f})\n")
                f.write("ROI Grains:\n")
                f.write(f"  φ₁: {euler_roi_deg[:, 0].min():.1f} - {euler_roi_deg[:, 0].max():.1f} (mean: {euler_roi_deg[:, 0].mean():.1f})\n")
                f.write(f"  Φ:  {euler_roi_deg[:, 1].min():.1f} - {euler_roi_deg[:, 1].max():.1f} (mean: {euler_roi_deg[:, 1].mean():.1f})\n")
                f.write(f"  φ₂: {euler_roi_deg[:, 2].min():.1f} - {euler_roi_deg[:, 2].max():.1f} (mean: {euler_roi_deg[:, 2].mean():.1f})\n")
            
            f.write("\nSelected Region Vertices:\n")
            for i, (x, y) in enumerate(vertices):
                f.write(f"  {i+1}: ({x:.2f}, {y:.2f}) μm\n")
        print(f"  ✓ Saved metadata: {metadata_path}")
    
    # Show plots interactively
    plt.show()
    
    # Prepare metadata dictionary
    metadata = {
        'vertices': vertices,
        'num_grains_total': len(grains_df_all),
        'num_grains_roi': len(grains_df_roi),
        'freq_range_all_mhz': (valid_freqs_all.min()/1e6, valid_freqs_all.max()/1e6),
        'freq_range_roi_mhz': (valid_freqs_roi.min()/1e6, valid_freqs_roi.max()/1e6),
    }
    
    # Add top orientations to metadata if clustering was performed
    if find_top_n_orientations is not None and top_orientations_df is not None:
        metadata['top_orientations'] = top_orientations_df
        metadata['cluster_labels'] = cluster_labels
    
    # Add Euler angle statistics to metadata if analysis was performed
    if show_euler_analysis:
        metadata.update({
            'euler_stats_all': {
                'phi1_range': (euler_all_deg[:, 0].min(), euler_all_deg[:, 0].max()),
                'phi1_mean': euler_all_deg[:, 0].mean(),
                'PHI_range': (euler_all_deg[:, 1].min(), euler_all_deg[:, 1].max()),
                'PHI_mean': euler_all_deg[:, 1].mean(),
                'phi2_range': (euler_all_deg[:, 2].min(), euler_all_deg[:, 2].max()),
                'phi2_mean': euler_all_deg[:, 2].mean(),
            },
            'euler_stats_roi': {
                'phi1_range': (euler_roi_deg[:, 0].min(), euler_roi_deg[:, 0].max()),
                'phi1_mean': euler_roi_deg[:, 0].mean(),
                'PHI_range': (euler_roi_deg[:, 1].min(), euler_roi_deg[:, 1].max()),
                'PHI_mean': euler_roi_deg[:, 1].mean(),
                'phi2_range': (euler_roi_deg[:, 2].min(), euler_roi_deg[:, 2].max()),
                'phi2_mean': euler_roi_deg[:, 2].mean(),
            }
        })
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return grains_df_all, grains_df_roi, metadata


if __name__ == "__main__":
    # Example usage
    print("Example: Interactive EBSD Grain Analysis")
    print("=" * 60)
    print("\nThis module provides interactive_grain_analysis() function.")
    print("\nUsage:")
    print("  from sawbench import Material")
    print("  from sawbench.interactive_ebsd_analysis import interactive_grain_analysis")
    print()
    print("  material = Material(")
    print("      formula='V',")
    print("      C11=229e9, C12=119e9, C44=43e9,")
    print("      density=6110,")
    print("      crystal_class='cubic'")
    print("  )")
    print()
    print("  grains_all, grains_roi, info = interactive_grain_analysis(")
    print("      ctf_path='path/to/your/file.ctf',")
    print("      material=material,")
    print("      wavelength=8.8e-6,")
    print("      output_dir='./results'  # Optional")
    print("  )")

