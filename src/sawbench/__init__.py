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

# Optionally include EBSDAnalyzer if it's considered core functionality
# from .centroid_finder import EBSDAnalyzer 

# Define package version (consider using importlib.metadata for installed package)
# If using setuptools_scm, version might be injected automatically.
# Otherwise, define it here or read from a VERSION file.
__version__ = "0.1.0" # Keep in sync with pyproject.toml

# Define __all__ for `from sawbench import *` behavior (optional but good practice)
__all__ = [
    'Material', 
    'SAWCalculator', 
    'EulerAngles', 
    'euler2matrix',
    'C_modifi', 
    'Orientations',
    # 'EBSDAnalyzer' # Uncomment if EBSDAnalyzer is included
] 