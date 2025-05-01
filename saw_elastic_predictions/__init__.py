"""
SAW Elastic Predictions

A package for calculating Surface Acoustic Wave (SAW) velocities in elastic materials.
"""

from .materials import Material
from .saw_calculator import SAWCalculator
from .euler_transformations import EulerAngles

__version__ = "0.1.0"
__all__ = ['Material', 'SAWCalculator', 'EulerAngles'] 