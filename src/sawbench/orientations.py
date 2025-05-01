"""Database of common crystallographic orientations and their Euler angles."""

import numpy as np
from typing import Tuple, Dict

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def cross_product_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Create orientation matrix from two vectors defining a coordinate system.

    The resulting matrix transforms from the sample coordinate system 
    (where z is the surface normal and y is the propagation direction)
    to the crystal coordinate system.
    
    Following the convention where:
    - Column 1 (new x') is parallel to `v1 x v2` (propagation normal direction on surface)
    - Column 2 (new y') is parallel to `v1` (propagation direction on surface)
    - Column 3 (new z') is parallel to `v2` (surface normal)
    
    Args:
        v1: Vector defining the new y' direction (propagation direction) in crystal coordinates.
        v2: Vector defining the new z' direction (surface normal) in crystal coordinates.

    Returns:
        3x3 orientation matrix.
    """
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    v3 = np.cross(v1_norm, v2_norm) # Should be np.cross(v2_norm, v1_norm) for right-handed system? Let's test.
    # Test: If v1=[0,1,0] (Y) and v2=[0,0,1] (Z), we expect v3=[1,0,0] (X)
    # np.cross([0,1,0], [0,0,1]) = [1, 0, 0] -> Correct.
    v3_norm = normalize(v3)
    
    # Ensure orthogonality (important if input vectors aren't perfectly orthogonal)
    # Recalculate v1 to be orthogonal to v2 and v3
    v1_recalc = np.cross(v2_norm, v3_norm)
    v1_recalc_norm = normalize(v1_recalc)

    # Note: The order might depend on the specific definition of Euler angles 
    # used later. This order matches the `matrix_to_euler` convention.
    return np.column_stack([v3_norm, v1_recalc_norm, v2_norm])

def matrix_to_euler(M: np.ndarray, tol=1e-8) -> Tuple[float, float, float]:
    """Convert orientation matrix to Euler angles (in radians).
    
    Following the Bunge convention (z-x'-z'') used in many EBSD/texture analysis software:
    phi1: First rotation around Z
    Phi: Rotation around the new X' axis
    phi2: Rotation around the final Z'' axis
    """
    # Check for Gimbal lock cases
    if abs(M[2, 2]) >= 1 - tol:
        if M[2, 2] > 0:  # Phi = 0
            phi1 = np.arctan2(M[0, 1], M[0, 0])
            Phi = 0.0
            phi2 = 0.0
        else:  # Phi = pi
            # Ambiguity: phi1 + phi2 = atan2(-M[0,1], -M[0,0])
            # Set phi2 = 0 arbitrarily
            phi1 = np.arctan2(-M[0, 1], M[0, 0]) # Note: -M[0,1] used here vs MATLAB original
            Phi = np.pi
            phi2 = 0.0
    else:
        phi1 = np.arctan2(M[0, 2], -M[1, 2]) # Changed from M[2,0], M[2,1] for Bunge
        Phi = np.arccos(M[2, 2])
        phi2 = np.arctan2(M[2, 0], M[2, 1]) # Changed from M[0,2], M[1,2] for Bunge
    
    # Ensure angles are in [0, 2*pi)
    # return np.mod(phi1, 2*np.pi), np.mod(Phi, np.pi), np.mod(phi2, 2*np.pi)
    # Or keep in standard range for atan2/acos
    return phi1, Phi, phi2

class Orientations:
    """Standard crystallographic orientations database.
    Provides Euler angles (Bunge z-x'-z' convention, radians) for named orientations.
    Orientations are defined by (Surface Plane Normal){hkl}<uvw>(Propagation Direction).
    """
    
    _orientations = {
        # {hkl}<uvw> : (surface_normal, propagation_direction)
        '100_001': (np.array([1, 0, 0]), np.array([0, 0, 1])),
        '100_011': (np.array([1, 0, 0]), np.array([0, 1, 1])),
        '110_001': (np.array([1, 1, 0]), np.array([0, 0, 1])),
        '110_1m10': (np.array([1, 1, 0]), np.array([1, -1, 0])),
        '110_111': (np.array([1, 1, 0]), np.array([1, 1, 1])), # Often written as (110)[1-11]? Check this
        '111_1m10': (np.array([1, 1, 1]), np.array([1, -1, 0])),
        '111_11m2': (np.array([1, 1, 1]), np.array([1, 1, -2])),
        '112_111': (np.array([1, 1, 2]), np.array([1, 1, 1])), # Check this definition
        # Add more orientations as needed...
    }

    @classmethod
    def get_euler_angles(cls, name: str) -> np.ndarray:
        """Get Euler angles for a standard orientation name.
        
        Args:
            name: Orientation name (e.g., '100_001', '110_111')
                Format: 'hkl_uvw' where {hkl} is the surface plane normal 
                and <uvw> is the propagation direction ON the surface.
            
        Returns:
            numpy array of [phi1, Phi, phi2] in radians (Bunge z-x'-z' convention).
        
        Raises:
            ValueError: If orientation name is not recognized.
        """
        if name not in cls._orientations:
            valid_names = list(cls._orientations.keys())
            raise ValueError(f"Unknown orientation '{name}'. Valid options are: {valid_names}")
            
        surface_normal, prop_direction = cls._orientations[name]
        
        # Create orientation matrix: transforms sample coords (z=normal, y=prop) to crystal coords
        matrix = cross_product_matrix(prop_direction, surface_normal)
        
        # Convert matrix to Euler angles
        phi1, Phi, phi2 = matrix_to_euler(matrix)
        return np.array([phi1, Phi, phi2])
    
    @classmethod
    def list_orientations(cls) -> Dict[str, str]:
        """List available orientations with descriptions."""
        # Basic descriptions, can be expanded
        descriptions = {
            '100_001': '(100)[001] Cube: Normal (100), Propagation [001]',
            '100_011': '(100)[011] Normal (100), Propagation [011]',
            '110_001': '(110)[001] Normal (110), Propagation [001]',
            '110_1m10': '(110)[1-10] Normal (110), Propagation [1-10]',
            '110_111': '(110)[111] Normal (110), Propagation [111]',
            '111_1m10': '(111)[1-10] Normal (111), Propagation [1-10]',
            '111_11m2': '(111)[11-2] Normal (111), Propagation [11-2]',
            '112_111': '(112)[111] Normal (112), Propagation [111]',
        }
        # Return descriptions only for orientations present in _orientations
        return {k: descriptions.get(k, "No description") for k in cls._orientations}

# Example Usage:
# euler = Orientations.get_euler_angles('110_111')
# print(euler)
# print(Orientations.list_orientations()) 