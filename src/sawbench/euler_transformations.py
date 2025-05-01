# sawbench/src/sawbench/euler_transformations.py
import numpy as np 

def euler2matrix(phi1, Phi, phi2):
    """Convert Euler angles to rotation matrix following MATLAB's Euler2matrix.m exactly.
    
    Args:
        phi1: First rotation around z axis (in radians)
        Phi: Second rotation around x' axis (in radians)
        phi2: Third rotation around z'' axis (in radians)
    
    Returns:
        3x3 rotation matrix that brings sample coordinates into crystal coordinates
    """
    # First rotation around z (Rza)
    Rza = np.array([
        [np.cos(phi1), np.sin(phi1), 0],
        [-np.sin(phi1), np.cos(phi1), 0],
        [0, 0, 1]
    ])
    
    # Second rotation around x' (Rxb)
    Rxb = np.array([
        [1, 0, 0],
        [0, np.cos(Phi), np.sin(Phi)],
        [0, -np.sin(Phi), np.cos(Phi)]
    ])
    
    # Third rotation around z'' (Rzr)
    Rzr = np.array([
        [np.cos(phi2), np.sin(phi2), 0],
        [-np.sin(phi2), np.cos(phi2), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation exactly as MATLAB: M = Rzr*Rxb*Rza
    return Rzr @ Rxb @ Rza

def c_modifi(C, a):
    """Transform elastic constant tensor C using transformation matrix a.
    
    Args:
        C: 3x3x3x3 elastic tensor
        a: 3x3 transformation matrix
    
    Returns:
        Transformed elastic tensor
    """
    # Initialize output tensor
    newC = np.zeros_like(C)
    
    # Transform tensor exactly as MATLAB
    for ip in range(3):
        for jp in range(3):
            for kp in range(3):
                for lp in range(3):
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for l in range(3):
                                    newC[ip,jp,kp,lp] += a[ip,i] * a[jp,j] * a[kp,k] * a[lp,l] * C[i,j,k,l]
    
    # Clean up numerical noise exactly as MATLAB
    maxC = np.max(np.abs(newC))
    newC[np.abs(newC) < 1e-8 * maxC] = 0
    
    return newC

# For backward compatibility
class EulerAngles:
    def __init__(self, phi1, Phi, phi2):
        """Initialize with Euler angles in Bunge convention (z-x'-z'')
        Args:
            phi1: First rotation around z (phi1)
            Phi: Second rotation around x' (Phi)
            phi2: Third rotation around z'' (phi2)
        """
        self.phi1 = phi1  # First rotation (z)
        self.Phi = Phi    # Second rotation (x')
        self.phi2 = phi2  # Third rotation (z'')

    def to_matrix(self):
        return euler2matrix(self.phi1, self.Phi, self.phi2)

# Rename for consistency with MATLAB
C_modifi = c_modifi 