# Examples Directory

This directory contains examples demonstrating different aspects of the `sawbench` package.

## Directory Structure

### `elastic_tensor/`
Examples for calculating and analyzing elastic tensors:
- `elastic_tensor_comparison.py` - Compare DFT vs MLIP vs experimental elastic tensors
- `mlip_calculator_setup.py` - Guide for setting up different MLIP calculators
- `ts-elastic.py` - Reference implementation from torch-sim
- `validate_elastic_tensor.py` - Validation tools for elastic tensor calculations
- `investigate_c44_issue.py` - Analysis of C44 underestimation in DFT
- `verify_strain_generation.py` - Verification of strain generation methods
- Documentation and result files

### `saw_analysis/`
Examples for Surface Acoustic Wave calculations:
- `example_ni3al.py` - Basic SAW calculation example
- `generate_saw_euler_surface.py` - Generate SAW velocity surfaces
- `plot_3d_saw_velocity_vs_euler.py` - 3D visualization of SAW velocities

### `ebsd_analysis/`
Examples for EBSD (Electron Backscatter Diffraction) analysis:
- `basic_analysis.py` - Basic EBSD data analysis
- `heatmap_analysis.py` - Heatmap visualization of EBSD data
- `mapping_example.py` - EBSD mapping examples
- `modular_analysis_workflow.py` - Complete EBSD analysis workflow
- `sensitivity_analysis_workflow.py` - Sensitivity analysis for EBSD parameters
- Sample EBSD data files

### `utilities/`
Utility scripts and helper functions (currently empty)

### `archived/`
Old or deprecated examples (currently empty)

### `data/`
Data files used by examples:
- `PBE_Manual_Elastic/` - VASP elastic tensor calculation data
- Other data files as needed

## Getting Started

1. **For elastic tensor calculations**: Start with `elastic_tensor/elastic_tensor_comparison.py`
2. **For SAW analysis**: Start with `saw_analysis/example_ni3al.py`
3. **For EBSD analysis**: Start with `ebsd_analysis/basic_analysis.py`

## Installation Requirements

Different examples may require different optional dependencies. See the main `INSTALLATION.md` for details on installing with specific MLIP support.

## Common Issues

- **Elastic tensor calculations**: Ensure VASP data is properly formatted and strain magnitudes are correctly detected
- **MLIP compatibility**: Use separate environments for MACE vs NequIP due to conflicting dependencies
- **EBSD data**: Ensure proper file paths and data format compatibility 