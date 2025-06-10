"""
sawbench Package

A package for calculating Surface Acoustic Wave (SAW) velocities 
in anisotropic elastic materials, potentially using EBSD data for orientation.
"""

# Define what is available directly from `import sawbench`
from .materials import Material
from .saw_calculator import SAWCalculator
from .euler_transformations import EulerAngles, euler2matrix, C_modifi # Expose core functions
from .orientations import Orientations
from .io import load_fft_data_from_hdf5, load_ebsd_map
from .grains import (
    _gauss, # Typically helper, not exposed unless needed by users
    extract_experimental_peak_parameters,
    calculate_saw_frequencies_for_ebsd_grains,
    create_ebsd_saw_frequency_map,
    create_experimental_peak_frequency_map,
)
from .statistics import (
    calculate_summary_statistics,
    perform_ks_test
)
from .plotting import (
    plot_frequency_histogram,
    plot_frequency_cdfs,
    plot_cdf_difference,
    plot_ebsd_property_map,
    plot_experimental_heatmap,
)

# Optionally include EBSDAnalyzer if it's considered core functionality
# from .centroid_finder import EBSDAnalyzer 

# Define package version (consider using importlib.metadata for installed package)
# If using setuptools_scm, version might be injected automatically.
# Otherwise, define it here or read from a VERSION file.
__version__ = "0.1.0" # Keep in sync with pyproject.toml

# Define __all__ for `from sawbench import *` behavior (optional but good practice)
__all__ = [
    # From io.py
    "load_fft_data_from_hdf5",
    "load_ebsd_map",
    # From grains.py
    "extract_experimental_peak_parameters",
    "calculate_saw_frequencies_for_ebsd_grains",
    "create_ebsd_saw_frequency_map",
    # From materials.py
    "Material", 
    # From saw_calculator.py
    "SAWCalculator", 
    'EulerAngles', 
    'euler2matrix',
    'C_modifi', 
    'Orientations',
    # From statistics.py
    "calculate_summary_statistics",
    "perform_ks_test",
    # From plotting.py
    "plot_frequency_histogram",
    "plot_frequency_cdfs",
    "plot_cdf_difference",
    "plot_ebsd_property_map",
    "plot_experimental_heatmap",
    # 'EBSDAnalyzer' # Uncomment if EBSDAnalyzer is included
] 