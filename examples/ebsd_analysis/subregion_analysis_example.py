"""
Example script demonstrating subregion selection and analysis.

This script shows how to:
1. Load a CTF file and examine its contents
2. Select rectangular or polygonal subregions
3. Crop the data to selected regions
4. Process the subregion with sawbench grain analysis
5. Visualize results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Import sawbench modules
from sawbench.subregion_selector import SubregionSelector
from sawbench import Material, calculate_saw_frequencies_for_ebsd_grains


def find_ctf_files(search_dirs=None):
    """Find available CTF files in common locations."""
    if search_dirs is None:
        search_dirs = [
            "../data/",
            "./data/",
            "../../data/",
            ".",
            ".."
        ]
    
    ctf_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # Search for .ctf files
            pattern = os.path.join(search_dir, "**", "*.ctf")
            found_files = glob.glob(pattern, recursive=True)
            ctf_files.extend(found_files)
    
    return list(set(ctf_files))  # Remove duplicates


def create_synthetic_ctf(output_path="synthetic_example.ctf"):
    """Create a minimal synthetic CTF file for testing purposes."""
    print(f"Creating synthetic CTF file: {output_path}")
    
    # Create a small grid of synthetic EBSD data
    nx, ny = 50, 50
    step_size = 1.0  # 1 micron step size
    
    with open(output_path, 'w') as f:
        # Write CTF header
        f.write("Channel Text File\n")
        f.write("Prj\tSynthetic EBSD Data\n")
        f.write("Author\tSawbench Synthetic Generator\n")
        f.write("JobMode\tGrid\n")
        f.write("XCells\t50\n")
        f.write("YCells\t50\n")
        f.write("XStep\t1.0\n")
        f.write("YStep\t1.0\n")
        f.write("AcqE1\t0\n")
        f.write("AcqE2\t0\n")
        f.write("AcqE3\t0\n")
        f.write("Euler angles refer to Sample Coordinate system (CS0)!\n")
        f.write("Mag\t10000\n")
        f.write("Coverage\t100\n")
        f.write("Device\t0\n")
        f.write("KV\t20\n")
        f.write("TiltAngle\t70\n")
        f.write("TiltAxis\t0\n")
        f.write("DetectorOrientationE1\t0\n")
        f.write("DetectorOrientationE2\t0\n")
        f.write("DetectorOrientationE3\t0\n")
        f.write("WorkingDistance\t15\n")
        f.write("InsertionDelay\t0\n")
        f.write("InsertionDelay2\t0\n")
        f.write("InsertionDelay3\t0\n")
        f.write("X\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n")
        
        # Generate synthetic data
        for j in range(ny):
            for i in range(nx):
                x = i * step_size
                y = j * step_size
                
                # Create some grain-like structure with varying orientations
                grain_id = (i // 10) * 5 + (j // 10)
                
                # Simple Euler angles that vary by grain
                euler1 = (grain_id * 30) % 360
                euler2 = (grain_id * 45) % 180
                euler3 = (grain_id * 60) % 360
                
                # Write data line
                f.write(f"{x:.6f}\t{y:.6f}\t7\t0\t{euler1:.6f}\t{euler2:.6f}\t{euler3:.6f}\t1.5\t50\t50\n")
    
    print(f"Synthetic CTF file created with {nx}x{ny} points")
    return output_path


def get_example_ctf_path():
    """Get a valid CTF file path for examples."""
    # Prefer the specific file the user wants
    preferred_file = "../data/V-1_2Ti_EBSD_Map.ctf"
    if os.path.exists(preferred_file):
        print(f"Using preferred CTF file: {preferred_file}")
        return preferred_file
    
    # Try to find other CTF files
    ctf_files = find_ctf_files()
    
    if ctf_files:
        # Use the first available CTF file
        ctf_path = ctf_files[0]
        print(f"Using existing CTF file: {ctf_path}")
        return ctf_path
    
    # If no CTF files found, create a synthetic one
    print("No CTF files found. Creating synthetic CTF for demonstration...")
    synthetic_path = create_synthetic_ctf()
    return synthetic_path


def debug_ctf_loading(ctf_path):
    """Debug CTF file loading to identify issues."""
    print(f"\n--- Debugging CTF file: {ctf_path} ---")
    
    # Show current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check file existence and size
    if not os.path.exists(ctf_path):
        print(f"ERROR: File does not exist: {ctf_path}")
        
        # Check if it exists as absolute path
        abs_path = os.path.abspath(ctf_path)
        print(f"Trying absolute path: {abs_path}")
        if os.path.exists(abs_path):
            print(f"File exists at absolute path: {abs_path}")
            return abs_path
        else:
            print("File does not exist at absolute path either")
            return False
    
    file_size = os.path.getsize(ctf_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("ERROR: File is empty")
        return False
    
    # Try to read the first few lines
    try:
        with open(ctf_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:20]
        print(f"File has {len(lines)} lines (showing first 20):")
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1}: {repr(line[:80])}")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return False
    
    # Try to load with defdap directly
    try:
        from defdap import ebsd
        print("\nTrying to load with defdap directly...")
        
        # Try with relative path first
        try:
            ebsd_map = ebsd.Map(ctf_path, dataType="OxfordText")
            print(f"Success with relative path! Map shape: {getattr(ebsd_map, 'shape', 'unknown')}")
            return ctf_path
        except FileNotFoundError:
            print("Relative path failed, trying absolute path...")
            abs_path = os.path.abspath(ctf_path)
            ebsd_map = ebsd.Map(abs_path, dataType="OxfordText")
            print(f"Success with absolute path! Map shape: {getattr(ebsd_map, 'shape', 'unknown')}")
            return abs_path
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR loading with defdap: {error_msg}")
        
        # Check if it's the double .ctf extension issue
        if ".ctf.ctf" in error_msg and ctf_path.endswith('.ctf'):
            print("Detected double .ctf extension issue. Trying without .ctf extension...")
            try:
                # Try without .ctf extension
                path_without_ext = ctf_path[:-4] if ctf_path.endswith('.ctf') else ctf_path
                abs_path_no_ext = os.path.abspath(path_without_ext)
                print(f"Trying: {abs_path_no_ext}")
                ebsd_map = ebsd.Map(abs_path_no_ext, dataType="OxfordText")
                print(f"Success without extension! Map shape: {getattr(ebsd_map, 'shape', 'unknown')}")
                return abs_path_no_ext
            except Exception as e2:
                print(f"ERROR loading without extension: {e2}")
        
        return False


def example_rectangular_selection():
    """Example of selecting a rectangular subregion."""
    print("=== Rectangular Subregion Selection Example ===")
    
    # Get a valid CTF file path
    ctf_path = get_example_ctf_path()
    if ctf_path is None:
        print("No CTF files available for this example.")
        return
    
    # Debug the CTF file first
    debug_result = debug_ctf_loading(ctf_path)
    if not debug_result:
        print("CTF file has issues, cannot proceed.")
        return
    
    # Use the corrected path if debug found one
    if debug_result != ctf_path:
        print(f"Using corrected path: {debug_result}")
        ctf_path = debug_result
    
    # Initialize selector
    selector = SubregionSelector(ctf_path)
    
    # Get data bounds
    bounds = selector.get_data_bounds()
    print(f"Data bounds: {bounds}")
    
    # Select a rectangular region (adjust coordinates as needed)
    x_min = bounds['x_min'] + bounds['width'] * 0.2
    y_min = bounds['y_min'] + bounds['height'] * 0.2
    x_max = bounds['x_min'] + bounds['width'] * 0.8
    y_max = bounds['y_min'] + bounds['height'] * 0.8
    
    print(f"Selecting rectangle: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")
    
    subregion_df = selector.select_rectangular_region(x_min, y_min, x_max, y_max)
    print(f"Selected {len(subregion_df)} points")
    
    # Visualize the selection (title shows number of grains in ROI)
    _fig = selector.visualize_region(subregion_df)
    plt.show()
    
    # Get statistics
    stats = selector.get_region_statistics(subregion_df)
    print("Region statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Save cropped CTF (optional; may be skipped if ROI lacks full CTF rows)
    output_path = "cropped_rectangle.ctf"
    selector.save_cropped_ctf(output_path, subregion_df)
    
    return subregion_df, selector


def example_polygonal_selection():
    """Example of selecting a polygonal subregion."""
    print("\n=== Polygonal Subregion Selection Example ===")
    
    # Get a valid CTF file path
    ctf_path = get_example_ctf_path()
    if ctf_path is None:
        print("No CTF files available for this example.")
        return
    
    # Initialize selector
    selector = SubregionSelector(ctf_path)
    
    # Define a polygon (triangle in this example)
    bounds = selector.get_data_bounds()
    center_x = bounds['x_min'] + bounds['width'] * 0.5
    center_y = bounds['y_min'] + bounds['height'] * 0.5
    size = min(bounds['width'], bounds['height']) * 0.3
    
    vertices = [
        (center_x, center_y + size),
        (center_x - size, center_y - size),
        (center_x + size, center_y - size)
    ]
    
    print(f"Selecting polygon with vertices: {vertices}")
    
    subregion_df = selector.select_polygonal_region(vertices)
    print(f"Selected {len(subregion_df)} points")
    
    # Visualize the selection
    _fig = selector.visualize_region(subregion_df)
    plt.show()
    
    return subregion_df, selector


def example_interactive_selection():
    """Example using the new simplified interactive_grain_analysis API."""
    print("\n=== Interactive Selection with Simplified API ===")
    
    # Get a valid CTF file path
    ctf_path = get_example_ctf_path()
    if ctf_path is None:
        print("No CTF files available for this example.")
        return
    
    # Import the new simplified function
    from sawbench import interactive_grain_analysis
    
    # Define material properties
    material = Material(
        formula='V',
        C11=229e9,
        C12=119e9,
        C44=43e9,
        density=6110,
        crystal_class='cubic'
    )
    
    # Run the complete interactive workflow with a single function call
    grains_all, grains_roi, metadata = interactive_grain_analysis(
        ctf_path=ctf_path,
        material=material,
        wavelength=8.8e-6,
        saw_calc_angle_deg=135.0,
        num_workers=16,
        output_dir="./interactive_subregion_output"
    )
    
    # Results are automatically displayed and saved
    print("\n✓ Analysis complete!")
    print(f"  - Total grains: {metadata['num_grains_total']}")
    print(f"  - Grains in ROI: {metadata['num_grains_roi']}")
    
    return grains_all, grains_roi, metadata


def example_grain_analysis_subregion():
    """Example of performing grain analysis on a subregion."""
    print("\n=== Grain Analysis on Subregion Example ===")
    
    # Get a valid CTF file path
    ctf_path = get_example_ctf_path()
    if ctf_path is None:
        print("No CTF files available for this example.")
        return
    
    # Step 1: Select subregion
    selector = SubregionSelector(ctf_path)
    bounds = selector.get_data_bounds()
    
    # Select center 50% of the data
    x_min = bounds['x_min'] + bounds['width'] * 0.25
    y_min = bounds['y_min'] + bounds['height'] * 0.25
    x_max = bounds['x_min'] + bounds['width'] * 0.75
    y_max = bounds['y_min'] + bounds['height'] * 0.75
    
    _ = selector.select_rectangular_region(x_min, y_min, x_max, y_max)
    
    # Step 2: Get grain IDs in the selected region and analyze them
    try:
        # Get grain IDs in the selection
        grain_ids_in_roi = selector.get_grain_ids_in_selection()
        print(f"Found {len(grain_ids_in_roi)} grains in the selected region")
        
        if len(grain_ids_in_roi) > 0:
            # Use the original EBSD map but filter to only grains in ROI
            ebsd_map = selector.ebsd_map
            
            # Define material properties (example for vanadium)
            material = Material(
                formula='V',
                C11=229e9,
                C12=119e9,
                C44=43e9,
                density=6110,
                crystal_class='cubic'
            )
            
            # Calculate SAW frequencies for ALL grains first
            wavelength = 8.8e-6  # meters
            print(f"Calculating SAW frequencies for all {len(ebsd_map.grainList)} grains...")
            grains_df = calculate_saw_frequencies_for_ebsd_grains(
                ebsd_map_obj=ebsd_map,
                material=material,
                wavelength=wavelength,
                saw_calc_angle_deg=135.0,
                num_workers=16
            )
            
            # Filter to only grains in the ROI
            grains_df_roi = grains_df[grains_df['Grain ID'].isin(grain_ids_in_roi)]
            
            print(f"Calculated SAW frequencies for {len(grains_df_roi)} grains in ROI (out of {len(grains_df)} total)")
            
            # Get valid frequencies for both all grains and ROI grains
            valid_freqs_all = grains_df['Peak SAW Frequency (Hz)'].dropna()
            valid_freqs_roi = grains_df_roi['Peak SAW Frequency (Hz)'].dropna()
            
            if not valid_freqs_roi.empty:
                print(f"Valid frequencies in ROI: {len(valid_freqs_roi)}")
                print(f"Frequency range (ROI): {valid_freqs_roi.min()/1e6:.1f} - {valid_freqs_roi.max()/1e6:.1f} MHz")
                print(f"Frequency range (All): {valid_freqs_all.min()/1e6:.1f} - {valid_freqs_all.max()/1e6:.1f} MHz")
                
                # Plot overlaid histogram: all grains (gray) + ROI grains (red)
                plt.figure(figsize=(12, 7))
                
                # Determine common bins for both histograms
                freq_min = min(valid_freqs_all.min(), valid_freqs_roi.min()) / 1e6
                freq_max = max(valid_freqs_all.max(), valid_freqs_roi.max()) / 1e6
                bins = np.linspace(freq_min, freq_max, 30)
                
                # Plot all grains in gray/blue (background)
                plt.hist(valid_freqs_all/1e6, bins=bins, alpha=0.5, edgecolor='black', 
                        color='lightgray', label=f'All grains (n={len(valid_freqs_all)})')
                
                # Overlay ROI grains in red
                plt.hist(valid_freqs_roi/1e6, bins=bins, alpha=0.7, edgecolor='darkred', 
                        color='red', label=f'Selected region (n={len(valid_freqs_roi)})')
                
                plt.xlabel('SAW Frequency (MHz)', fontsize=12)
                plt.ylabel('Number of Grains', fontsize=12)
                plt.title('SAW Frequency Distribution: All Grains vs Selected Subregion', fontsize=14)
                plt.legend(loc='upper right', fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
        else:
            print("No grains found in the selected region")
            
    except Exception as e:
        print(f"Error during grain analysis: {e}")
    
    # No cleanup needed since we're not creating temporary files


def example_multiple_subregions():
    """Example of analyzing multiple subregions."""
    print("\n=== Multiple Subregions Analysis Example ===")
    
    # Get a valid CTF file path
    ctf_path = get_example_ctf_path()
    if ctf_path is None:
        print("No CTF files available for this example.")
        return
    
    selector = SubregionSelector(ctf_path)
    bounds = selector.get_data_bounds()
    
    # Define multiple rectangular regions
    regions = [
        {"name": "top_left", "coords": (bounds['x_min'], bounds['y_min'] + bounds['height']*0.5, 
                                       bounds['x_min'] + bounds['width']*0.5, bounds['y_max'])},
        {"name": "top_right", "coords": (bounds['x_min'] + bounds['width']*0.5, bounds['y_min'] + bounds['height']*0.5,
                                        bounds['x_max'], bounds['y_max'])},
        {"name": "bottom_left", "coords": (bounds['x_min'], bounds['y_min'],
                                          bounds['x_min'] + bounds['width']*0.5, bounds['y_min'] + bounds['height']*0.5)},
        {"name": "bottom_right", "coords": (bounds['x_min'] + bounds['width']*0.5, bounds['y_min'],
                                           bounds['x_max'], bounds['y_min'] + bounds['height']*0.5)}
    ]
    
    # Analyze each region
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, region in enumerate(regions):
        x_min, y_min, x_max, y_max = region["coords"]
        subregion_df = selector.select_rectangular_region(x_min, y_min, x_max, y_max)
        
        # Store results
        stats = selector.get_region_statistics(subregion_df)
        results[region["name"]] = {
            "data": subregion_df,
            "stats": stats
        }
        
        # Plot - use X_um/Y_um if available (from grainIDMap fallback)
        ax = axes[i]
        if 'X_um' in subregion_df.columns and 'Y_um' in subregion_df.columns:
            ax.scatter(subregion_df['X_um'], subregion_df['Y_um'], 
                      c='red', s=1, alpha=0.8)
        elif 'X' in subregion_df.columns and 'Y' in subregion_df.columns:
            ax.scatter(subregion_df['X'], subregion_df['Y'], 
                      c=subregion_df['Euler1'] % 360, cmap='hsv', s=1, alpha=0.8)
        ax.set_title(f"{region['name']} ({len(subregion_df)} points)")
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_aspect('equal')
        
        print(f"{region['name']}: {len(subregion_df)} points, "
              f"center=({stats['x_center']:.1f}, {stats['y_center']:.1f})")
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    print("Subregion Analysis Examples")
    print("=" * 50)
    
    # Check for available CTF files
    ctf_files = find_ctf_files()
    if not ctf_files:
        print("No CTF files found in common locations.")
        print("Please ensure you have CTF files available or update the search paths.")
        print("\nSearch locations:")
        for search_dir in ["../data/", "./data/", "../../data/", ".", ".."]:
            status = "✓" if os.path.exists(search_dir) else "✗"
            print(f"  {status} {search_dir}")
        exit(1)
    
    print(f"Found {len(ctf_files)} CTF file(s):")
    for i, ctf_file in enumerate(ctf_files):
        print(f"  {i+1}: {ctf_file}")
    print()
    
    # Uncomment the examples you want to run:
    
    example_rectangular_selection()
    #example_polygonal_selection()
    example_interactive_selection()
    #example_grain_analysis_subregion()
    #example_multiple_subregions()
    
    print("Examples completed. Check the function implementations above.")
