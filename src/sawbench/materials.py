# sawbench/src/sawbench/materials.py
import numpy as np
import matplotlib.pyplot as plt

class Material:
    def __init__(self, formula, C11, C12, C44, density, crystal_class, **kwargs):
        """
        Initialize material with elastic constants and properties.
        
        Args:
            formula: Chemical formula of the material
            C11, C12, C44: Primary elastic constants (in Pa)
            density: Material density (in kg/m^3)
            crystal_class: Crystal system ('cubic', 'hexagonal', etc.)
            **kwargs: Additional elastic constants for non-cubic systems
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

    def get_cijkl(self):
        """Convert from Voigt notation to full tensor (C_ijkl)."""
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

    def get_density(self):
        """Get material density in kg/m^3"""
        return self.density

    def __str__(self):
        return f"Material({self.formula}, Class: {self.crystal_class}, Density: {self.density:.2e} kg/m^3)" 