# sawbench/src/sawbench/materials.py
"""Module for defining material properties and elastic constants.

This module provides the Material class for representing elastic materials
with their crystal structure and elastic constants.
"""
import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

class Material:
    """Represents an elastic material with its crystal structure and elastic constants.
    
    Supports cubic crystal systems with plans to extend to other crystal classes.
    Used throughout sawbench for SAW calculations.
    
    Attributes:
        formula (str): Chemical formula of the material.
        C11 (float): Primary elastic constant C11 in Pa.
        C12 (float): Primary elastic constant C12 in Pa.
        C44 (float): Primary elastic constant C44 in Pa.
        density (float): Material density in kg/m³.
        crystal_class (str): Crystal system ('cubic', 'hexagonal', etc.).
        C13 (Optional[float]): Elastic constant C13 for non-cubic systems.
        C33 (Optional[float]): Elastic constant C33 for non-cubic systems.
        C66 (Optional[float]): Elastic constant C66 for non-cubic systems.
    
    Examples:
        >>> # Create a cubic material (Vanadium)
        >>> vanadium = Material(
        ...     formula='V',
        ...     C11=229e9,
        ...     C12=119e9,
        ...     C44=43e9,
        ...     density=6110,
        ...     crystal_class='cubic'
        ... )
        >>> print(vanadium)
        Material(V, Class: cubic, Density: 6.11e+03 kg/m^3)
        
        >>> # Get the full elastic tensor
        >>> C_tensor = vanadium.get_cijkl()
        >>> print(C_tensor.shape)
        (3, 3, 3, 3)
    
    Notes:
        Currently only cubic crystal systems are fully implemented.
        Future versions will support hexagonal and other crystal classes.
    """
    
    def __init__(
        self,
        formula: str,
        C11: float,
        C12: float,
        C44: float,
        density: float,
        crystal_class: str,
        **kwargs: Any
    ) -> None:
        """Initialize material with elastic constants and properties.
        
        Args:
            formula: Chemical formula of the material (e.g., 'V', 'Ti', 'Al2O3').
            C11: Primary elastic constant C11 in Pa.
            C12: Primary elastic constant C12 in Pa.
            C44: Primary elastic constant C44 in Pa.
            density: Material density in kg/m³.
            crystal_class: Crystal system ('cubic', 'hexagonal', etc.).
            **kwargs: Additional elastic constants for non-cubic systems
                (e.g., C13, C33, C66 for hexagonal).
        
        Examples:
            >>> # Cubic material
            >>> material = Material('V', 229e9, 119e9, 43e9, 6110, 'cubic')
            
            >>> # Hexagonal material (future implementation)
            >>> material = Material(
            ...     'Ti', 162e9, 92e9, 46.7e9, 4507, 'hexagonal',
            ...     C13=69e9, C33=180.7e9, C66=35e9
            ... )
        """
        self.formula = formula
        self.C11 = C11
        self.C12 = C12
        self.C44 = C44
        self.density = density
        self.crystal_class = crystal_class.lower()
        
        # Additional constants for non-cubic systems
        self.C13 = kwargs.get('C13', None)
        self.C33 = kwargs.get('C33', None)
        self.C66 = kwargs.get('C66', None)

    def get_cijkl(self) -> np.ndarray:
        """Convert from Voigt notation to full tensor (C_ijkl).
        
        Converts the material's elastic constants from Voigt notation
        to the full 4th-rank elastic tensor representation.
        
        Returns:
            np.ndarray: 4th-rank elastic tensor of shape (3, 3, 3, 3) in Pa.
        
        Raises:
            NotImplementedError: If crystal class is not 'cubic'.
        
        Examples:
            >>> material = Material('V', 229e9, 119e9, 43e9, 6110, 'cubic')
            >>> C_tensor = material.get_cijkl()
            >>> # Check a specific component (should equal C11)
            >>> print(C_tensor[0, 0, 0, 0] == material.C11)
            True
        
        Notes:
            For cubic symmetry, the tensor has specific symmetries:
            - C_iiii = C11 (diagonal terms)
            - C_iijj = C12 (off-diagonal terms, i≠j)
            - C_ijij = C_ijji = C44 (shear terms, i≠j)
        """
        if self.crystal_class == 'cubic':
            C11 = self.C11
            C12 = self.C12
            C44 = self.C44
            
            C = np.zeros((3, 3, 3, 3))
            
            # Fill in the tensor components for cubic symmetry
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            if i == j and k == l:  # Diagonal terms C_iiii
                                C[i,j,k,l] = C12
                                if i == k:
                                    C[i,j,k,l] = C11
                            elif (i == k and j == l) or (i == l and j == k):  # Shear terms C_ijij or C_ijji
                                C[i,j,k,l] = C44
            return C
        else:
            # TODO: Implement for other crystal classes (e.g., hexagonal)
            raise NotImplementedError(f"get_cijkl not implemented for crystal class '{self.crystal_class}'")

    def get_density(self) -> float:
        """Get material density.
        
        Returns:
            float: Material density in kg/m³.
        
        Examples:
            >>> material = Material('V', 229e9, 119e9, 43e9, 6110, 'cubic')
            >>> density = material.get_density()
            >>> print(f"Density: {density} kg/m³")
            Density: 6110 kg/m³
        """
        return self.density

    def __str__(self) -> str:
        """Return string representation of the material.
        
        Returns:
            str: String description of the material.
        """
        return f"Material({self.formula}, Class: {self.crystal_class}, Density: {self.density:.2e} kg/m^3)" 