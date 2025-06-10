# sawbench/src/sawbench/euler_transformations.py
"""Module for Euler angle conversions and tensor transformations.

This module provides functions to convert Euler angles (Bunge Z-X-Z' convention)
to rotation matrices and to transform 4th-rank tensors (like the elastic
stiffness tensor) using these rotation matrices. These operations are fundamental
for calculating orientation-dependent properties in materials science.
"""
import numpy as np
from typing import Tuple

def euler2matrix(phi1: float, Phi: float, phi2: float) -> np.ndarray:
    """Convert Bunge Euler angles (Z-X-Z') to a rotation matrix.

    This function implements the conversion from Bunge convention Euler angles
    (phi1, Phi, phi2) to a 3x3 passive rotation matrix. The rotation sequence
    is:
    1. phi1 around the original Z-axis.
    2. Phi around the new X-axis (X').
    3. phi2 around the new Z-axis (Z'').

    The resulting matrix transforms coordinates from the sample/reference frame
    to the crystal frame. This convention is common in texture analysis and EBSD.

    Args:
        phi1 (float): First Euler angle (rotation around Z) in radians.
        Phi (float): Second Euler angle (rotation around X') in radians.
        phi2 (float): Third Euler angle (rotation around Z'') in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix representing the orientation.
                    This matrix transforms vectors from the sample/reference
                    coordinate system to the crystal coordinate system.

    Examples:
        >>> import numpy as np
        >>> # No rotation (identity matrix)
        >>> r_matrix_identity = euler2matrix(0, 0, 0)
        >>> print(np.allclose(r_matrix_identity, np.eye(3)))
        True

        >>> # 90-degree rotation around Z-axis
        >>> r_matrix_z90 = euler2matrix(np.pi/2, 0, 0)
        >>> # Expected: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        >>> print(r_matrix_z90)
        [[ 0.0000000e+00  1.0000000e+00  0.0000000e+00]
         [-1.0000000e+00  0.0000000e+00  0.0000000e+00]
         [ 0.0000000e+00  0.0000000e+00  1.0000000e+00]]

        >>> # A more complex rotation
        >>> r_matrix_complex = euler2matrix(np.pi/4, np.pi/3, np.pi/6)
        >>> # This matrix can be used to rotate vectors or tensors.

    Notes:
        The rotation order Z-X-Z' (Bunge convention) is followed.
        The angles are assumed to be in radians.
        This implementation is designed to match the behavior of
        `Euler2matrix.m` from common MATLAB toolboxes for materials science.
    """
    # First rotation around z (Rza)
    cos_phi1, sin_phi1 = np.cos(phi1), np.sin(phi1)
    Rza = np.array([
        [cos_phi1, sin_phi1, 0],
        [-sin_phi1, cos_phi1, 0],
        [0, 0, 1]
    ])
    
    # Second rotation around x' (Rxb)
    cos_Phi, sin_Phi = np.cos(Phi), np.sin(Phi)
    Rxb = np.array([
        [1, 0, 0],
        [0, cos_Phi, sin_Phi],
        [0, -sin_Phi, cos_Phi]
    ])
    
    # Third rotation around z'' (Rzr)
    cos_phi2, sin_phi2 = np.cos(phi2), np.sin(phi2)
    Rzr = np.array([
        [cos_phi2, sin_phi2, 0],
        [-sin_phi2, cos_phi2, 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: M = Rzr @ Rxb @ Rza
    # This order means Rza is applied first, then Rxb in the new frame, then Rzr in the newest frame.
    # The resulting matrix 'a' transforms from sample to crystal coordinates: X_crystal = a @ X_sample
    return Rzr @ Rxb @ Rza

def c_modifi(C: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Transform a 4th-rank elastic tensor C using a rotation matrix a.

    This function applies the standard transformation rule for a 4th-rank
    tensor: C'_ijkl = a_im * a_jn * a_ko * a_lp * C_mnop.
    It is used to rotate the elastic stiffness tensor (C_ijkl) from one
    coordinate system to another (e.g., from a crystal's principal axes
    to a sample reference frame).

    Args:
        C (np.ndarray): The 3x3x3x3 elastic stiffness tensor in the original
                        coordinate system (e.g., crystal frame). Units are
                        typically Pascals (Pa).
        a (np.ndarray): The 3x3 rotation matrix that transforms coordinates
                        from the NEW coordinate system to the ORIGINAL
                        coordinate system of C.
                        For example, if C is in crystal coordinates and you want
                        C' in sample coordinates, 'a' should be the matrix
                        that transforms from sample to crystal coordinates
                        (i.e., X_crystal = a @ X_sample).

    Returns:
        np.ndarray: The transformed 3x3x3x3 elastic tensor (C') in the new
                    coordinate system. Values close to zero due to numerical
                    precision are thresholded.

    Examples:
        >>> # Assume C_crystal is the elastic tensor in crystal coordinates
        >>> # and rot_matrix transforms from sample to crystal coordinates
        >>> # C_crystal = material.get_cijkl() # From sawbench.materials
        >>> # rot_matrix = euler2matrix(phi1, Phi, phi2)
        >>> # C_sample = c_modifi(C_crystal, rot_matrix)
        >>> # C_sample now represents the elastic properties in the sample frame.

    Notes:
        The transformation matrix 'a' should represent the rotation from the
        target coordinate system of the transformed tensor (newC) to the
        original coordinate system of the input tensor (C).
        A small threshold (1e-8 relative to max(abs(newC))) is applied to
        zero out numerical noise, mimicking behavior in some MATLAB scripts.
        The Einstein summation convention is implied in the transformation rule.
    """
    # Initialize output tensor
    newC = np.zeros_like(C)
    
    # Transform tensor: C'_ip,jp,kp,lp = a_ip,i * a_jp,j * a_kp,k * a_lp,l * C_i,j,k,l
    # Summation over i, j, k, l from 0 to 2
    for ip in range(3):
        for jp in range(3):
            for kp in range(3):
                for lp in range(3):
                    sum_val = 0.0
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for l in range(3):
                                    sum_val += a[ip,i] * a[jp,j] * a[kp,k] * a[lp,l] * C[i,j,k,l]
                    newC[ip,jp,kp,lp] = sum_val
    
    # Clean up numerical noise
    # This step helps in making comparisons and ensuring symmetry for certain rotations.
    max_abs_C = np.max(np.abs(newC))
    if max_abs_C > 0: # Avoid division by zero if newC is all zeros
        newC[np.abs(newC) < 1e-8 * max_abs_C] = 0.0
    
    return newC

# For backward compatibility and convenience
class EulerAngles:
    """Represents Bunge Euler angles (phi1, Phi, phi2) in Z-X-Z' convention.

    This class provides a simple container for Euler angles and a method
    to convert them to a rotation matrix using the `euler2matrix` function.

    Attributes:
        phi1 (float): First Euler angle (rotation about Z-axis) in radians.
        Phi (float): Second Euler angle (rotation about new X'-axis) in radians.
        phi2 (float): Third Euler angle (rotation about new Z''-axis) in radians.

    Examples:
        >>> angles = EulerAngles(np.pi/4, np.pi/3, np.pi/6)
        >>> rotation_m = angles.to_matrix()
        >>> print(rotation_m.shape)
        (3, 3)
    """
    def __init__(self, phi1: float, Phi: float, phi2: float):
        """Initialize EulerAngles object.

        Args:
            phi1 (float): First Euler angle (rotation about Z) in radians.
            Phi (float): Second Euler angle (rotation about X') in radians.
            phi2 (float): Third Euler angle (rotation about Z'') in radians.
        """
        self.phi1 = phi1
        self.Phi = Phi
        self.phi2 = phi2

    def to_matrix(self) -> np.ndarray:
        """Convert the stored Euler angles to a rotation matrix.

        Uses the `euler2matrix` function for the conversion.

        Returns:
            np.ndarray: The 3x3 rotation matrix.
        """
        return euler2matrix(self.phi1, self.Phi, self.phi2)

# Alias for consistency or if c_modifi is preferred in other parts of the code
C_modifi = c_modifi 