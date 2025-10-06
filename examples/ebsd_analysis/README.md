# EBSD Analysis Examples

This directory contains examples for analyzing EBSD (Electron Backscatter Diffraction) data with SAW (Surface Acoustic Wave) frequency calculations.

## Quick Start: Interactive Analysis

The simplest way to analyze a subregion of your EBSD data:

```python
from sawbench import Material, interactive_grain_analysis

# Define your material
material = Material(
    formula='V',
    C11=229e9, C12=119e9, C44=43e9,
    density=6110,
    crystal_class='cubic'
)

# Run interactive analysis
grains_all, grains_roi, metadata = interactive_grain_analysis(
    ctf_path='path/to/your/file.ctf',
    material=material,
    wavelength=8.8e-6,
    output_dir='./results'  # Optional: saves plots and data
)
```

See `simple_interactive_example.py` for a complete working example.

## What It Does

1. **Loads your EBSD data** from a CTF file
2. **Interactive selection**: Click to draw a polygon around your region of interest
3. **Calculates SAW frequencies** for all grains using parallel processing
4. **Generates two plots**:
   - Overlaid histogram comparing selected region vs. full dataset
   - Spatial frequency map with your selected region highlighted
5. **Optionally saves** plots, grain data CSVs, and metadata

## Files

- `simple_interactive_example.py` - **Start here!** Minimal example using the new API
- `subregion_analysis_example.py` - Advanced examples showing lower-level control
- `modular_analysis_workflow.py` - Complete workflow examples
- `mapping_example.py` - Frequency map generation examples

## Requirements

Your CTF file should be from Oxford Instruments EBSD systems. The module automatically handles:
- Grain segmentation using defdap
- Coordinate normalization
- Parallel SAW calculations
- Publication-quality visualization

## Output Files (when `output_dir` is specified)

- `saw_frequency_analysis.png` - Combined histogram and spatial map figure
- `grains_all.csv` - SAW frequencies for all grains in the EBSD map
- `grains_roi.csv` - SAW frequencies for grains in your selected region
- `analysis_metadata.txt` - Summary of analysis parameters and results
- `cropped_region.ctf` - Cropped CTF file (if applicable)
