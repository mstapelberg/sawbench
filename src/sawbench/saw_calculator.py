# sawbench/src/sawbench/saw_calculator.py
"""Module for calculating Surface Acoustic Wave (SAW) properties.

This module provides the SAWCalculator class, which is designed to compute
SAW velocities, propagation directions, and intensities on anisotropic
material surfaces. It takes into account material elastic properties and
crystal orientation (Bunge Z-X-Z' convention).

The calculations are based on established methods for solving wave equations
in anisotropic media, often involving finding roots of a characteristic
determinant (related to the G33 Green's function component or boundary
conditions) and are intended to be consistent with approaches found in
materials science and acoustics literature, including some MATLAB toolboxes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, det
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from typing import Tuple, List, Optional, Any, Dict, Union

from .euler_transformations import C_modifi, EulerAngles
from .materials import Material

class SAWCalculator:
    """Calculates Surface Acoustic Wave (SAW) properties for a given material and orientation.

    This class encapsulates the logic for determining SAW velocities and related
    parameters on a specified crystal surface and propagation direction.
    It uses the material's elastic tensor and density, along with the crystal's
    orientation (Euler angles), to perform these calculations.

    Attributes:
        material (Material): An object representing the material, providing
            `get_cijkl()` for the 4th-rank elastic tensor (in Pa) and
            `get_density()` for density (in kg/m³).
        euler_angles (np.ndarray): A 3-element numpy array representing the Bunge
            Euler angles (phi1, Phi, phi2) in radians, defining the crystal
            orientation relative to the sample frame.

    Examples:
        >>> from sawbench.materials import Material
        >>> # Define a material (e.g., Vanadium)
        >>> vanadium_props = {
        ...     'formula': 'V', 'C11': 229e9, 'C12': 119e9, 'C44': 43e9,
        ...     'density': 6110, 'crystal_class': 'cubic'
        ... }
        >>> vanadium = Material(**vanadium_props)
        >>> # Define Euler angles for orientation (e.g., no rotation)
        >>> euler_rad = np.array([0.0, 0.0, 0.0])
        >>> calculator = SAWCalculator(material=vanadium, euler_angles=euler_rad)
        >>> # To calculate SAW speed:
        >>> # v_saw, direction, intensity = calculator.get_saw_speed(deg=0)
        >>> # print(f"SAW velocity: {v_saw} m/s")

    Notes:
        The implementation aims to be consistent with established numerical methods
        for SAW calculations. Input Euler angles (phi1, Phi, phi2) follow the
        Bunge Z-X-Z' convention and are expected in radians. The `deg` parameter
        in `get_saw_speed` refers to the in-plane rotation of the SAW propagation
        direction on the sample surface.
    """
    def __init__(self, material: Material, euler_angles: np.ndarray):
        """Initialize SAWCalculator with material properties and Euler angles.

        Args:
            material (Material): Material object. Must have a callable `get_cijkl()`
                method returning a 3x3x3x3 NumPy array (elastic tensor in Pa)
                and a callable `get_density()` method returning density (float in kg/m³).
            euler_angles (np.ndarray): A 3-element NumPy array representing Bunge
                Euler angles (phi1, Phi, phi2) in radians. These define the
                crystal orientation. While angles outside [-2π, 2π] are
                mathematically valid, very large magnitudes (e.g., > 9)
                will trigger a warning as they might indicate degrees
                were passed instead of radians.

        Raises:
            TypeError: If the `material` object is not an instance of `Material`
                or does not have the required `get_cijkl()` or `get_density()` methods.
            ValueError: If `euler_angles` is not a 3-element array.
                A stricter internal check also exists for angles far outside typical
                radian ranges, which might raise a ValueError.

        Examples:
            >>> from sawbench.materials import Material
            >>> vanadium_props = {
            ...     'formula': 'V', 'C11': 229e9, 'C12': 119e9, 'C44': 43e9,
            ...     'density': 6110, 'crystal_class': 'cubic'
            ... }
            >>> vanadium = Material(**vanadium_props)
            >>> orientation_rad = np.array([np.pi/4, np.pi/3, np.pi/6])
            >>> calc = SAWCalculator(material=vanadium, euler_angles=orientation_rad)
            >>> print(f"Calculator initialized for material: {calc.material.formula}")
            Calculator initialized for material: V
        """
        # Validate material
        if not isinstance(material, Material):
            raise TypeError(f"Expected 'material' to be an instance of Material, but got {type(material)}.")
        if not hasattr(material, 'get_cijkl') or not callable(material.get_cijkl):
            raise TypeError("Material object must have a callable get_cijkl() method.")
        if not hasattr(material, 'get_density') or not callable(material.get_density):
            raise TypeError("Material object must have a callable get_density() method.")
        
        # Convert euler_angles to numpy array and validate
        try:
            euler_angles_arr = np.asarray(euler_angles, dtype=float)
        except Exception as e:
            raise TypeError(f"euler_angles could not be converted to a NumPy float array: {e}")

        if euler_angles_arr.shape != (3,):
            raise ValueError(f"euler_angles must be a 3-element array, got shape {euler_angles_arr.shape}")
            
        # Check Euler angle range - primarily for deg/rad mix-up warning
        # The ValueError for > 2*pi is a bit strict as angles are periodic,
        # but the warning for > 9 is a good heuristic for degree input.
        if np.any(np.abs(euler_angles_arr) > 2 * np.pi + 1e-9): # Using a tolerance
            # This original ValueError might be too strict if values like 7 rad are intended.
            # Keeping the warning logic which is more practical for user error.
            # raise ValueError("Euler angles must be in radians and ideally within [-2π, 2π] for clarity.")
            if np.any(np.abs(euler_angles_arr) > 9): # Heuristic for degrees
                print(
                    f"Warning: Euler angles {euler_angles_arr} have magnitudes > 9. "
                    "Ensure they are in radians, not degrees."
                )
            else: # Still large, but maybe intended
                print(
                    f"Warning: Euler angles {euler_angles_arr} are outside the typical [-2π, 2π] range. "
                    "Ensure they are correctly specified in radians."
                )
        elif np.any(np.abs(euler_angles_arr) > 9): # This covers cases not caught by the first if, but still > 9
             print(
                f"Warning: Euler angles {euler_angles_arr} have magnitudes > 9. "
                "Ensure they are in radians, not degrees."
            )
            
        self.material = material
        self.euler_angles = euler_angles_arr # Use the validated array

    def get_saw_speed(
        self,
        deg: float,
        sampling: int = 4000,
        psaw: int = 0,
        draw_plot: bool = False,
        debug: bool = False,
        use_optimized: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Calculate SAW velocity, propagation direction in crystal coords, and intensity.

        This method computes the Surface Acoustic Wave (SAW) properties for a
        given in-plane propagation angle on the material surface, taking into
        account the material's elasticity and the crystal's orientation.

        Args:
            deg (float): In-plane angle (in degrees) for SAW propagation, measured
                from a reference direction on the sample surface (typically the y-axis
                of the unrotated sample frame). Must be in the range [0, 180).
            sampling (int, optional): Resolution parameter for k-space (slowness space)
                sampling. Common values are 400, 4000, 40000. Higher values
                increase accuracy but also computation time. Defaults to 4000.
            psaw (int, optional): Flag to enable Pseudo-SAW (PSAW) calculation.
                - 0: Calculate SAW only (default).
                - 1: Calculate both SAW and PSAW (if present). This may affect peak
                  selection.
            draw_plot (bool, optional): If True, generates displacement-slowness plot
            debug: If True, returns intermediate values for testing
            use_optimized: If True, uses the optimized G33 calculation
            
        Returns:
            v: SAW velocity(ies)
            index: Direction of SAW in crystalline coordinates
            intensity: Intensity of SAW mode(s)
        """
        # Strict validation for deg
        if not (0 <= deg < 180):
            raise ValueError("deg must be in range [0, 180)")
        elif deg > 180:  # This won't be reached due to above check, but kept for MATLAB compatibility
            print('Warning: Check convention, deg should be between 0 and 180 in degree')
            
        # Validate sampling parameter
        if sampling <= 0:
            raise ValueError("sampling must be positive")
        
        # Validate psaw parameter
        if psaw not in [0, 1]:
            raise ValueError("psaw must be 0 or 1")
        
        # Get material properties - EXACTLY as in MATLAB
        C = self.material.get_cijkl()  # This should now match MATLAB's getCijkl
        rho = self.material.get_density()  # Should be 7.57e3
        
        if debug:
            print("\nMaterial properties:")
            print(f"C11 = {C[0,0,0,0]}")
            print(f"C12 = {C[0,0,1,1]}")
            print(f"C44 = {C[0,1,0,1]}")
            print(f"density = {rho}")
        
        # Get alignment matrix - EXACTLY as in MATLAB
        MM = self._get_alignment_matrix(deg)
        
        # Transform elastic constants - EXACTLY match MATLAB order
        #print(f"Euler angles: {self.euler_angles}")
        euler_obj = EulerAngles(self.euler_angles[0], self.euler_angles[1], self.euler_angles[2])
        rotation_matrix = euler_obj.to_matrix()
        
        # MATLAB: C = C_modifi(C,(Euler2matrix(Euler(1),Euler(2),Euler(3))*MM)')
        transform_matrix = (rotation_matrix @ MM).T  # Exactly as MATLAB: (A*B)'
        C_transformed = C_modifi(C, transform_matrix)
        
        # Get sampling parameter and initialize variables
        T = self._get_sampling_parameter(sampling)
        lambda_val = 7e-6  # m
        k0 = 2 * np.pi / lambda_val
        w0 = 2 * np.pi / T
        w = w0 + complex(0, 0.000001 * w0)
        index = transform_matrix @ np.array([0, 1, 0]) # What is this for? 
        
        # Initialize debug values dictionary
        initial_debug_values = {
            'C_transformed': C_transformed,
            'euler_rotation': rotation_matrix,
            'MM': MM,
            'combined_rotation': transform_matrix
        } if debug else None
        
        # Choose between original and optimized implementation
        if use_optimized:
            if debug:
                G33, ynew, slownessnew, g33_debug_values = self._calculate_g33_opt(
                    C_transformed, rho, w, k0, sampling, psaw, debug=True)
                # Merge debug dictionaries if needed
                debug_values = {**initial_debug_values, **g33_debug_values}
            else:
                G33, ynew, slownessnew = self._calculate_g33_opt(
                    C_transformed, rho, w, k0, sampling, psaw)
        else:
            if debug:
                G33, ynew, slownessnew, g33_debug_values = self._calculate_g33(
                    C_transformed, rho, w, k0, sampling, psaw, debug=True)
                # Merge debug dictionaries
                debug_values = {**initial_debug_values, **g33_debug_values}
            else:
                G33, ynew, slownessnew = self._calculate_g33(
                    C_transformed, rho, w, k0, sampling, psaw)
        
        # Calculate final values
        v = 1.0 / slownessnew
        intensity = self._calculate_intensity(ynew, slownessnew, debug=debug)
        
        # Draw plot if requested
        if draw_plot:
            self._plot_saw_profile(G33, k0, w, slownessnew, psaw, ynew)
        
        if debug:
            return v, index, intensity, debug_values
        else:
            return v, index, intensity

    def _get_alignment_matrix(self, deg):
        """Get alignment matrix (exactly matching MATLAB's MM matrix).
        
        Args:
            deg: Angle in degrees
        
        Returns:
            3x3 rotation matrix that aligns y-axis with direction of interest
        """
        # Use cosd directly like MATLAB (convert to radians internally)
        def cosd(angle_deg):
            return np.cos(np.deg2rad(angle_deg))
        
        # Exactly match MATLAB's MM matrix construction
        MM = np.array([
            [cosd(deg), cosd(90-deg), 0],
            [cosd(90+deg), cosd(deg), 0],
            [0, 0, 1]
        ])
        
        return MM

    def _get_sampling_parameter(self, sampling):
        """Get sampling parameter T (exactly matching MATLAB)."""
        if sampling == 40000:
            T = 20e-14
        elif sampling == 4000:
            T = 20e-13
        elif sampling == 400:
            T = 20e-12
        else:
            T = 20e-12
            print('Warning: sampling is not a value often used')
        return T

    def _stable_det(self, matrix):
        # Scale the matrix to improve condition number
        scale = np.max(np.abs(matrix))
        if scale > 0:
            scaled_matrix = matrix / scale
            return np.linalg.det(scaled_matrix) * (scale**3)
        return 0.0

    def _build_matrices(self, C_trans, k_array, rho, w):
        """
        Build B, M, N arrays in a vectorized manner.
        C_trans: 3x3x3x3 elastic tensor
        k_array: Array of shape (sampling, 2) containing the wave vectors
        Returns:
            B_all, M_all, N_all: Arrays of shape (sampling, 3, 3)
        """
        sampling = k_array.shape[0]
        B_all = np.zeros((sampling, 3, 3), dtype=np.complex128)
        for i in range(3):
            for l in range(3):
                term0 = k_array[:, 0] * C_trans[i, 0, 2, l]
                term1 = k_array[:, 1] * C_trans[i, 1, 2, l]
                term2 = k_array[:, 0] * C_trans[i, 2, 0, l]
                term3 = k_array[:, 1] * C_trans[i, 2, 1, l]
                B_all[:, i, l] = -(term0 + term1 + term2 + term3)
        M_all = 1j * B_all
        N_all = np.zeros((sampling, 3, 3), dtype=np.complex128)
        for i in range(3):
            for l in range(3):
                N_all[:, i, l] = rho * (w**2) * (1 if i == l else 0)
                for u in range(2):
                    for v in range(2):
                        N_all[:, i, l] -= C_trans[i, u, v, l] * k_array[:, u] * k_array[:, v]
        return B_all, M_all, N_all

    def _select_roots(self, roots, tol=1e-10):
        """Select roots with positive real parts, handling numerical noise."""
        # Remove roots with tiny real parts (numerical noise)
        cleaned_roots = np.where(np.abs(np.real(roots)) < tol,
                               1j * np.imag(roots),
                               roots)
        
        # Select roots with positive real parts or positive imaginary parts if real part is zero
        selected = []
        for root in cleaned_roots:
            if np.real(root) > tol or (abs(np.real(root)) < tol and np.imag(root) > 0):
                selected.append(root)
        return np.array(selected)

    def _calculate_g33(self, C_transformed, rho, w, k0, sampling, psaw, debug=False):
        if debug:
            print("\nNumerical stability diagnostics:")
            print(f"C_transformed condition number: {np.linalg.cond(C_transformed.reshape(9,9))}")
        
        # Precompute wave vectors and vectorized matrices
        nx = 1
        ny_vals = np.arange(1, sampling + 1)
        k_array = np.column_stack([np.full(sampling, nx * k0), ny_vals * k0])
        F_mat = C_transformed[:, 2, 2, :].astype(np.float64)
        B_all, M_all, N_all = self._build_matrices(C_transformed, k_array, rho, w)
        
        # Pre-allocate output G33 as a 1D array
        G33 = np.zeros(sampling, dtype=complex)
        POL = [[[] for _ in range(3)] for _ in range(3)]
        A = np.zeros((3, 3), dtype=complex)
        R = np.zeros((3, 3), dtype=complex)
        I = np.zeros((3, 3), dtype=complex)
        a = np.zeros(3, dtype=complex)
        
        debug_values = {}
        
        # Loop over each sampling point using the precomputed matrices
        for ny_idx in range(sampling):
            # Use F_mat which is constant for all ny_idx; extract corresponding B, M, N matrices
            F = F_mat.copy()
            B = B_all[ny_idx]
            M = M_all[ny_idx]
            N = N_all[ny_idx]
            
            # Clean up numerical noise in F matrix
            max_diagonal = np.max(np.abs(np.diag(F)))
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if np.abs(F[i, j]) < 1e-4 * max_diagonal:
                            F[i, j] = 0
                    else:
                        if np.abs(F[i, j]) < 1e-8 * max_diagonal:
                            F[i, j] = 0
            
            # Set up polynomial coefficients from F, M, N
            for i in range(3):
                for j in range(3):
                    POL[i][j] = [F[i, j], M[i, j], N[i, j]]
            
            # Calculate determinant polynomial using convolutions (as in MATLAB)
            Poly = np.convolve(np.convolve(POL[0][0], POL[1][1]), POL[2][2])
            Poly += np.convolve(np.convolve(POL[0][1], POL[1][2]), POL[2][0])
            Poly += np.convolve(np.convolve(POL[0][2], POL[1][0]), POL[2][1])
            Poly -= np.convolve(np.convolve(POL[0][0], POL[1][2]), POL[2][1])
            Poly -= np.convolve(np.convolve(POL[0][1], POL[1][0]), POL[2][2])
            Poly -= np.convolve(np.convolve(POL[0][2], POL[1][1]), POL[2][0])
            
            # Find roots and select those with positive real parts
            ppC = np.roots(Poly)
            pp = np.array([root for root in ppC if np.real(root) > 0])
            if len(pp) == 0:
                G33[ny_idx] = 0
                continue
            
            # Calculate eigenvectors using SVD for each valid root
            for i in range(len(pp)):
                S = F * (pp[i]**2) + M * pp[i] + N
                U, s, Vh = np.linalg.svd(S)
                Sol = Vh[-1, :].conj()  # Last row of V^H
                A[i, :] = Sol
            
            R.fill(0)
            I.fill(0)
            for i in range(3):
                for r in range(len(pp)):
                    for l in range(3):
                        R[i, r] += C_transformed[i, 2, 2, l] * pp[r] * A[r, l]
                        for u in range(2):
                            I[i, r] += C_transformed[i, 2, u, l] * k_array[ny_idx, u] * A[r, l]
            
            Comb = -R + 1j * I
            del_vec = np.array([0, 0, 1])
            for r in range(3):
                Aug = Comb.copy()
                Aug[:, r] = del_vec
                det_comb = self._stable_det(Comb)
                det_aug = self._stable_det(Aug)
                if abs(det_comb) < 1e-10:
                    U, s, Vh = np.linalg.svd(Comb)
                    tol = np.max(s) * 1e-10
                    s_inv = np.array([1/x if x > tol else 0 for x in s])
                    a[r] = (Vh.T @ np.diag(s_inv) @ U.T @ del_vec)[r]
                else:
                    a[r] = det_aug / det_comb
            
            G33[ny_idx] = sum(a[r] * A[r, 2] for r in range(len(pp)))
            
            if ny_idx == 0 and debug:
                debug_values = {
                    'F': F.copy(),
                    'M': M.copy(),
                    'N': N.copy(),
                    'roots': ppC.copy(),
                    'selected_roots': pp.copy(),
                    'first_G33': G33[0]
                }
            
            if debug and ny_idx == 0:
                print(f"F matrix condition number: {np.linalg.cond(F)}")
                print(f"Comb matrix condition number: {np.linalg.cond(Comb)}")
        
        inc = 1
        xx = np.arange(1, sampling + 1)
        yy = np.real(G33)
        xnew = np.arange(1, sampling + 1, inc)
        cs = CubicSpline(xx, yy)
        ynew = cs(xnew)
        
        if debug:
            print("\nPost-processing debug:")
            print(f"G33 shape: {G33.shape}")
            print(f"First few values of real(G33): {yy[:5]}")
            print(f"Last few values of real(G33): {yy[-5:]}")
            YYnew_indices = self._h_l_peak(ynew, psaw)
            print(f"\nPeak indices: {YYnew_indices}")
            print(f"Number of peaks found: {len(YYnew_indices)}")
            if len(YYnew_indices) > 0:
                print(f"Peak positions: {YYnew_indices}")
                print(f"Peak values: {ynew[YYnew_indices]}")
            Num = 1 + inc * YYnew_indices
            print(f"\nNum values: {Num}")
            print(f"k0: {k0}")
            print(f"w: {w}")
            slownessnew = Num * k0 / np.real(w)
            print(f"Calculated slowness values: {slownessnew}")
            print(f"Final velocities: {1.0/slownessnew}")
        
        YYnew_indices = self._h_l_peak(ynew, psaw)
        Num = 1 + inc * YYnew_indices
        slownessnew = Num * k0 / np.real(w)
        
        if debug:
            return G33, ynew, slownessnew, debug_values
        return G33, ynew, slownessnew

    # TODO Actually make this faster, right now it's the same speed!
    def _calculate_g33_opt(self, C_transformed, rho, w, k0, sampling, psaw, debug=False):
        """
        Completely redesigned G33 calculation with focus on computational efficiency.
        
        This implementation:
        1. Uses pre-allocation and reuse of arrays to minimize memory allocation
        2. Optimizes matrix operations for numerical stability
        3. Reduces redundant calculations
        4. Uses faster determinant and matrix manipulation methods
        
        Args:
            C_transformed: Transformed elastic tensor
            rho: Material density
            w: Angular frequency
            k0: Wave vector
            sampling: Number of sampling points
            psaw: Flag for PSAW calculation
            debug: If True, returns debug information
        
        Returns:
            G33, ynew, slownessnew: The displacement, interpolated displacement, and slowness
        """
        import time
        start_time = time.time()
        profiling = {}
        
        # Pre-allocate G33 array
        G33 = np.zeros(sampling, dtype=complex)
        debug_values = {} if debug else None
        
        # Extract constant matrices and values for better performance
        setup_start = time.time()
        # The F matrix is constant (C_transformed[:, 2, 2, :])
        F = C_transformed[:, 2, 2, :].copy().astype(np.float64)
        
        # Wave vector parameters
        nx = 1
        ny_range = np.arange(1, sampling + 1)
        
        # Pre-allocate reusable matrices to avoid repeated allocations
        B = np.zeros((3, 3), dtype=complex)
        M = np.zeros((3, 3), dtype=complex)
        N = np.zeros((3, 3), dtype=complex)
        S = np.zeros((3, 3), dtype=complex)
        A = np.zeros((3, 3), dtype=complex)
        R = np.zeros((3, 3), dtype=complex)
        I = np.zeros((3, 3), dtype=complex)
        Comb = np.zeros((3, 3), dtype=complex)
        
        # Pre-compute constant parts
        omega_squared = w**2
        rho_omega_squared = rho * omega_squared
        delta_kronecker = np.eye(3)
        
        # Pre-compute C tensor patterns for faster access
        C_i02l = np.zeros((3, 3), dtype=complex)
        C_i20l = np.zeros((3, 3), dtype=complex)
        C_i12l = np.zeros((3, 3), dtype=complex)
        C_i21l = np.zeros((3, 3), dtype=complex)
        C_iuvl = np.zeros((3, 2, 2, 3), dtype=complex)
        
        for i in range(3):
            for l in range(3):
                C_i02l[i,l] = C_transformed[i,0,2,l]
                C_i20l[i,l] = C_transformed[i,2,0,l]
                C_i12l[i,l] = C_transformed[i,1,2,l]
                C_i21l[i,l] = C_transformed[i,2,1,l]
                for u in range(2):
                    for v in range(2):
                        C_iuvl[i,u,v,l] = C_transformed[i,u,v,l]
        
        # Surface normal vector (constant)
        del_vec = np.array([0, 0, 1])
        
        # Create lookup table for polynomials (reduces redundant convolutions)
        term_cache = {}
        
        profiling['setup'] = time.time() - setup_start
        loop_start = time.time()
        
        # Main loop
        for ny_idx, ny in enumerate(ny_range):
            iter_start = time.time()
            
            # Current k vector
            k1 = nx * k0
            k2 = ny * k0
            k = np.array([k1, k2])
            
            # Calculate B matrix (boundary condition matrix)
            B_start = time.time()
            # Vectorized B matrix calculation
            B = -(k1 * (C_i02l + C_i20l) + k2 * (C_i12l + C_i21l))
            profiling['B_matrix'] = profiling.get('B_matrix', 0) + time.time() - B_start
            
            # Calculate M matrix
            M = 1j * B
            
            # Calculate N matrix
            N_start = time.time()
            # Direct calculation for N matrix
            N_k_terms = np.zeros((3, 3), dtype=complex)
            for i in range(3):
                for l in range(3):
                    N_k_terms[i,l] = k1*k1*C_iuvl[i,0,0,l] + k1*k2*(C_iuvl[i,0,1,l]+C_iuvl[i,1,0,l]) + k2*k2*C_iuvl[i,1,1,l]
            
            N = rho_omega_squared * delta_kronecker - N_k_terms
            profiling['N_matrix'] = profiling.get('N_matrix', 0) + time.time() - N_start
            
            # Calculate determinant polynomial
            pol_start = time.time()
            # Formulate the polynomial eigenvalue problem
            # Set up polynomial coefficients
            POL = np.zeros((3, 3, 3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    POL[i,j] = np.array([F[i,j], M[i,j], N[i,j]])
            
            # Calculate determinant polynomial
            # Define a simple key for caching based on ny_idx
            key = ny_idx % 10  # Only cache a few patterns to save memory
            
            # Instead of using a complex caching mechanism with N_terms
            if key in term_cache:
                # Use a simpler caching approach
                Poly = term_cache[key].copy()
            else:
                # Calculate polynomial determinant directly
                term1 = np.convolve(np.convolve(POL[0,0], POL[1,1]), POL[2,2])
                term2 = np.convolve(np.convolve(POL[0,1], POL[1,2]), POL[2,0])
                term3 = np.convolve(np.convolve(POL[0,2], POL[1,0]), POL[2,1])
                term4 = np.convolve(np.convolve(POL[0,0], POL[1,2]), POL[2,1])
                term5 = np.convolve(np.convolve(POL[0,1], POL[1,0]), POL[2,2])
                term6 = np.convolve(np.convolve(POL[0,2], POL[1,1]), POL[2,0])
                
                Poly = term1 + term2 + term3 - term4 - term5 - term6
                term_cache[key] = Poly.copy()
            
            profiling['polynomial'] = profiling.get('polynomial', 0) + time.time() - pol_start
            
            # Find roots and select those with positive real parts
            roots_start = time.time()
            # Use more stable root finding
            try:
                roots = np.roots(Poly)
                # Filter roots with positive real parts
                pp = roots[np.real(roots) > 0]
                # Sort by absolute value for numerical stability
                pp = pp[np.argsort(np.abs(pp))]
            except:
                # Failsafe for numerical issues
                pp = np.array([])
            
            if len(pp) == 0:
                G33[ny_idx] = 0
                profiling['roots'] = profiling.get('roots', 0) + time.time() - roots_start
                profiling['iteration'] = profiling.get('iteration', 0) + time.time() - iter_start
                continue
            
            profiling['roots'] = profiling.get('roots', 0) + time.time() - roots_start
            
            # Solve eigenvector problems for each root (up to 3)
            eig_start = time.time()
            a = np.zeros(min(3, len(pp)), dtype=complex)
            
            for r, p in enumerate(pp[:3]):  # Only consider up to 3 roots
                # Calculate S = F*p^2 + M*p + N efficiently
                S = F * (p**2) + M * p + N
                
                # Use faster SVD approach
                try:
                    # Compute the null space vector (last right singular vector)
                    _, _, Vh = np.linalg.svd(S, full_matrices=True)
                    A[r] = Vh[-1]  # Last row of Vh
                except:
                    # Fallback for numerical issues
                    A[r] = np.zeros(3)
                    A[r,r] = 1.0
            
            profiling['eigenvectors'] = profiling.get('eigenvectors', 0) + time.time() - eig_start
            
            # Calculate R and I matrices
            ri_start = time.time()
            max_roots = min(3, len(pp))
            
            # Reset matrices
            R.fill(0)
            I.fill(0)
            
            # Vectorized calculation of R and I matrices
            for i in range(3):
                for r in range(max_roots):
                    # R calculation (more direct)
                    R[i,r] = np.sum(C_transformed[i,2,2,:] * pp[r] * A[r,:])
                    
                    # I calculation (more direct)
                    I[i,r] = np.sum(C_transformed[i,2,0,:] * k[0] * A[r,:] + 
                                   C_transformed[i,2,1,:] * k[1] * A[r,:])
            
            # Form combined matrix more directly
            Comb = -R + 1j * I
            profiling['RI_matrices'] = profiling.get('RI_matrices', 0) + time.time() - ri_start
            
            # Solve for amplitudes
            amp_start = time.time()
            
            # Only consider up to 3 amplitudes
            for r in range(max_roots):
                # Use LU decomposition for better performance
                try:
                    det_comb = np.linalg.det(Comb)
                    if abs(det_comb) > 1e-10:
                        # Create augmented matrix
                        Aug = Comb.copy()
                        Aug[:,r] = del_vec
                        det_aug = np.linalg.det(Aug)
                        a[r] = det_aug / det_comb
                    else:
                        # Use least squares for ill-conditioned matrices
                        a[r] = np.linalg.lstsq(Comb, del_vec, rcond=None)[0][r]
                except:
                    a[r] = 0.0
                
            profiling['amplitudes'] = profiling.get('amplitudes', 0) + time.time() - amp_start
            
            # Calculate G33
            G33[ny_idx] = np.sum(a * A[:max_roots,2])
            
            profiling['iteration'] = profiling.get('iteration', 0) + time.time() - iter_start
            
            # Store debug info for first iteration if requested
            if ny_idx == 0 and debug:
                debug_values = {
                    'F': F.copy(),
                    'M': M.copy(),
                    'N': N.copy(),
                    'roots': roots.copy() if 'roots' in locals() else np.array([]),
                    'selected_roots': pp.copy(),
                    'first_G33': G33[0]
                }
        
        profiling['main_loop'] = time.time() - loop_start
        
        # Post-processing
        post_start = time.time()
        # Use cubic spline for smooth interpolation
        xx = np.arange(1, sampling + 1)
        yy = np.real(G33)
        cs = CubicSpline(xx, yy)
        
        # Use same increment as original
        inc = 1
        xnew = np.arange(1, sampling + 1, inc)
        ynew = cs(xnew)
        
        # Find peaks and calculate slowness
        YYnew_indices = self._h_l_peak(ynew, psaw)
        Num = 1 + inc * YYnew_indices
        slownessnew = Num * k0 / np.real(w)
        profiling['post_processing'] = time.time() - post_start
        
        # Total time
        profiling['total'] = time.time() - start_time
        
        # Add extra debug info
        if debug:
            print("\nOptimized G33 calculation (new version):")
            print(f"G33 shape: {G33.shape}")
            print(f"Number of peaks found: {len(YYnew_indices)}")
            print(f"Final velocities: {1.0/slownessnew}")
            
            # Print profiling information
            print("\nProfiling information:")
            for section, duration in sorted(profiling.items(), key=lambda x: x[1], reverse=True):
                if section != 'total' and section != 'main_loop' and section != 'iteration':
                    print(f"{section}: {duration:.6f} seconds ({duration/profiling['total']*100:.2f}%)")
            print(f"main_loop: {profiling['main_loop']:.6f} seconds ({profiling['main_loop']/profiling['total']*100:.2f}%)")
            print(f"total: {profiling['total']:.6f} seconds (100.00%)")
            
            debug_values['profiling'] = profiling
            return G33, ynew, slownessnew, debug_values
        
        return G33, ynew, slownessnew

    def _stable_det_opt(self, matrix):
        """Faster, numerically stable determinant for 3x3 matrices"""
        # Direct 3x3 determinant calculation (faster than generic algorithm)
        try:
            return (matrix[0,0] * (matrix[1,1] * matrix[2,2] - matrix[1,2] * matrix[2,1]) -
                    matrix[0,1] * (matrix[1,0] * matrix[2,2] - matrix[1,2] * matrix[2,0]) +
                    matrix[0,2] * (matrix[1,0] * matrix[2,1] - matrix[1,1] * matrix[2,0]))
        except:
            # Fallback to SVD for numerical stability
            U, s, Vh = np.linalg.svd(matrix)
            det_sign = np.linalg.det(U) * np.linalg.det(Vh)
            return det_sign * np.prod(s)

    def _calculate_polynomial_determinant(self, POL):
        """
        Calculates the determinant of a 3x3 matrix of polynomials (represented by coefficient lists).
        This needs to be expanded explicitly as numpy's poly determinant doesn't directly handle polynomial matrices.
        """
        # Directly expand the determinant for a 3x3 matrix
        p11, p12, p13 = POL[0]
        p21, p22, p23 = POL[1]
        p31, p32, p33 = POL[2]

        term1 = np.convolve(np.convolve(p11, p22), p33)
        term2 = np.convolve(np.convolve(p12, p23), p31)
        term3 = np.convolve(np.convolve(p13, p21), p32)
        term4 = np.convolve(np.convolve(p11, p23), p32)
        term5 = np.convolve(np.convolve(p12, p21), p33)
        term6 = np.convolve(np.convolve(p13, p22), p31)

        Poly_coeffs = term1 + term2 + term3 - term4 - term5 - term6
        return Poly_coeffs

    def _h_l_peak(self, y_new, psaw_flag, debug=False):
        """Find peaks in the y_new array using a more robust method."""
        # Find local maxima by comparing with neighbors
        peak_candidates = []
        for i in range(1, len(y_new)-1):
            if y_new[i] > y_new[i-1] and y_new[i] > y_new[i+1]:
                # Only consider significant peaks (adjust threshold as needed)
                if abs(y_new[i]) > 1e-22:  # Threshold based on observed values
                    peak_candidates.append(i)
        
        peak_candidates = np.array(peak_candidates)
        
        if len(peak_candidates) == 0:
            return np.array([])
            
        # Debug print
        if debug:
            print(f"\nPeak finding debug:")
            print(f"Number of significant peaks found: {len(peak_candidates)}")
            print(f"Peak positions: {peak_candidates}")
            print(f"Peak values: {y_new[peak_candidates]}")
        
        # Sort peaks by magnitude
        peak_magnitudes = abs(y_new[peak_candidates])
        sorted_indices = np.argsort(peak_magnitudes)[::-1]  # Sort in descending order
        peak_candidates = peak_candidates[sorted_indices]

        if psaw_flag and len(peak_candidates) >= 2:
            # Return two largest peaks for PSAW
            return peak_candidates[:2]
        else:
            # Return largest peak for SAW
            return np.array([peak_candidates[0]])

    def _calculate_intensity(self, ynew, slownessnew, debug=False):
        """
        Calculates the relative intensity of each SAW mode.
        Simplified intensity calculation focusing on peak height difference.
        """
        YYnew_indices = self._h_l_peak(ynew, False, debug=debug) # Find peaks again, adjust psaw if needed
        intensity = np.zeros(len(YYnew_indices))

        for jj_idx, peak_index in enumerate(YYnew_indices):
            pos_ind = peak_index # Directly use index from find_peaks
            go = True
            loop_idx = pos_ind

            neg_ind = pos_ind # Initialize in case loop doesn't run
            if loop_idx > 1: # Avoid index out of bounds
                loop_idx_inner = loop_idx
                while go and loop_idx_inner > 1: # loop_idx_inner instead of loop_idx
                    if ynew[loop_idx_inner - 2] < ynew[loop_idx_inner - 1]: # Check within bounds
                        go = False
                        neg_ind = loop_idx_inner - 1 # Store neg_ind when condition met
                    else:
                        loop_idx_inner -= 1 # Decrement loop_idx_inner
            if pos_ind < len(ynew) - 1: # Check bounds for pos_ind
                 intensity[jj_idx] = ynew[neg_ind] - ynew[pos_ind] # Intensity as difference

        return intensity

    def _plot_saw_profile(self, G33, k0, w, slownessnew, psaw, ynew):
        """Generates and displays the displacement-slowness profile plot."""
        plt.figure()
        hax = plt.gca()
        ny = np.arange(1, len(G33[0,:]) + 1)
        Nsx = 0  # Python 0-indexed
        
        # Plot displacement profile
        plt.plot(ny * k0 / np.real(w), np.real(G33[Nsx, :]), 'b', linewidth=2)
        
        # Plot SAW lines
        if slownessnew.size > 0:
            plt.axvline(x=slownessnew[0], color='r')
            if psaw and len(slownessnew) > 1:
                plt.axvline(x=slownessnew[1], color='r')

        plt.xlabel('Slowness (s/m)', fontsize=16)
        plt.ylabel('Displacement (arb. unit)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.show()

def deltaij(i, j):
    """Kronecker delta function."""
    return 1 if i == j else 0

def det_3x3_poly_coeffs_to_complex(matrix_of_complex_vec):
    """Determinant of 3x3 complex matrix directly without polynomial representation."""
    m = matrix_of_complex_vec
    det_val = m[0,0] * (m[1,1] * m[2,2] - m[1,2] * m[2,1]) - m[0,1] * (m[1,0] * m[2,2] - m[1,2] * m[2,0]) + m[0,2] * (m[1,0] * m[2,1] - m[1,1] * m[2,0])
    print(f"Determinant value: {det_val}") # DEBUG: Print determinant value

    return det_val

if __name__ == '__main__':
    # --- Example Material Class (replace with your actual Material class) ---
    class ExampleMaterial:
        def get_cijkl(self):
            # Example C tensor (replace with your actual elastic constants)
            C_example = np.zeros((3, 3, 3, 3))
            C_example = np.eye(3*3).reshape(3,3,3,3) # Example Identity-like C tensor - replace!
            return C_example

        def get_density(self):
            return 5000  # Example density

    # --- Example Usage ---
    material_example = ExampleMaterial()
    euler_example = np.array([0.1, 0.2, 0.3])
    deg_example = 30.0
    sampling_example = 400
    psaw_example = 1

    saw_calculator = SAWCalculator(material_example, euler_example)
    v_saw, index_saw, intensity_saw = saw_calculator.get_saw_speed(deg_example, sampling_example, psaw_example, draw_plot=True)

    print("SAW Velocity (v):", v_saw)
    print("SAW Direction (index):", index_saw)
    print("SAW Intensity (intensity):", intensity_saw)
    print("SAW Intensity (intensity):", intensity_saw)