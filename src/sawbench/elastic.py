"""
Elastic tensor calculations for sawbench.

This module provides functionality to calculate elastic tensors from
VASP calculations using the finite difference method.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from ase.io import read
from ase import Atoms

class ElasticTensor:
    """
    Elastic tensor class for storing and manipulating elastic constants.
    
    Attributes:
        tensor : np.ndarray
            6x6 elastic stiffness tensor in Voigt notation (GPa)
        crystal_system : str
            Detected crystal system
    """
    
    def __init__(self, tensor: np.ndarray):
        """
        Initialize elastic tensor.
        
        Parameters:
        -----------
        tensor : np.ndarray
            6x6 elastic stiffness tensor in GPa
        """
        if tensor.shape != (6, 6):
            raise ValueError("Elastic tensor must be 6x6 matrix")
        
        self.tensor = tensor.copy()
        self.crystal_system = self._detect_crystal_system()
    
    def _detect_crystal_system(self, tolerance: float = 0.1) -> str:
        """Detect crystal system from elastic tensor symmetry."""
        C = self.tensor
        
        # Check for cubic symmetry
        c11_avg = (C[0,0] + C[1,1] + C[2,2]) / 3
        c11_var = np.var([C[0,0], C[1,1], C[2,2]]) / c11_avg**2
        
        c12_avg = (C[0,1] + C[0,2] + C[1,2]) / 3  
        c12_var = np.var([C[0,1], C[0,2], C[1,2]]) / (c12_avg**2 + 1e-10)
        
        c44_avg = (C[3,3] + C[4,4] + C[5,5]) / 3
        c44_var = np.var([C[3,3], C[4,4], C[5,5]]) / c44_avg**2
        
        if c11_var < tolerance and c12_var < tolerance and c44_var < tolerance:
            return "cubic"
        else:
            return "general"
    
    def get_elastic_properties(self) -> Dict[str, float]:
        """
        Calculate elastic properties from the tensor.
        
        Returns:
        --------
        properties : dict
            Dictionary of elastic properties (all in GPa except Poisson's ratio)
        """
        C = self.tensor
        properties = {}
        
        if self.crystal_system == "cubic":
            properties['C11'] = C[0, 0]
            properties['C12'] = C[0, 1] 
            properties['C44'] = C[3, 3]
            
            # Bulk modulus: K = (C11 + 2*C12)/3
            properties['bulk_modulus'] = (C[0, 0] + 2*C[0, 1]) / 3
            
            # Shear modulus (Voigt): G = (C11 - C12 + 3*C44)/5
            properties['shear_modulus_voigt'] = (C[0, 0] - C[0, 1] + 3*C[3, 3]) / 5
            
            # Shear modulus (Reuss)
            s11 = (C[0, 0] + C[1, 1] - 2*C[0, 1]) / ((C[0, 0] + C[1, 1])*(C[0, 0] + C[1, 1] - 2*C[0, 1]) - 2*C[0, 1]**2)
            s44 = 1 / C[3, 3]
            properties['shear_modulus_reuss'] = 15 / (4*s11 + 3*s44)
            
            # Hill average
            properties['shear_modulus'] = (properties['shear_modulus_voigt'] + properties['shear_modulus_reuss']) / 2
            
        else:
            # General case - use Hill averages
            try:
                S = np.linalg.inv(C)
                properties['bulk_modulus'] = 1 / (S[0,0] + S[1,1] + S[2,2] + 2*(S[0,1] + S[0,2] + S[1,2]))
                properties['shear_modulus'] = ((C[0,0] + C[1,1] + C[2,2]) - (C[0,1] + C[0,2] + C[1,2]) + 3*(C[3,3] + C[4,4] + C[5,5])) / 15
                
            except np.linalg.LinAlgError:
                print("Warning: Could not invert elastic tensor")
                return properties
        
        # Young's modulus and Poisson's ratio
        if 'bulk_modulus' in properties and 'shear_modulus' in properties:
            K = properties['bulk_modulus']
            G = properties['shear_modulus']
            properties['youngs_modulus'] = 9*K*G / (3*K + G)
            properties['poisson_ratio'] = (3*K - 2*G) / (2*(3*K + G))
        
        return properties
    
    def __str__(self):
        """String representation of elastic tensor."""
        props = self.get_elastic_properties()
        
        result = f"ElasticTensor ({self.crystal_system} symmetry)\n"
        result += "="*50 + "\n\n"
        
        result += "Stiffness Tensor (GPa):\n"
        result += "     C11     C12     C13     C14     C15     C16\n"
        for i in range(6):
            row_str = f"C{i+1}j "
            for j in range(6):
                row_str += f"{self.tensor[i,j]:8.2f}"
            result += row_str + "\n"
        
        result += f"\nElastic Properties:\n"
        result += f"Crystal system: {self.crystal_system}\n"
        
        if self.crystal_system == "cubic":
            result += f"C₁₁ = {props['C11']:.2f} GPa\n"
            result += f"C₁₂ = {props['C12']:.2f} GPa\n"  
            result += f"C₄₄ = {props['C44']:.2f} GPa\n"
        
        if 'bulk_modulus' in props:
            result += f"Bulk modulus (K) = {props['bulk_modulus']:.2f} GPa\n"
        if 'shear_modulus' in props:
            result += f"Shear modulus (G) = {props['shear_modulus']:.2f} GPa\n"
        if 'youngs_modulus' in props:
            result += f"Young's modulus (E) = {props['youngs_modulus']:.2f} GPa\n"
        if 'poisson_ratio' in props:
            result += f"Poisson's ratio (ν) = {props['poisson_ratio']:.3f}\n"
        
        return result


def from_vasp_dir(base_path: Union[str, Path], strain_magnitude: Optional[float] = None,
                  reference_outcar: Optional[str] = None) -> ElasticTensor:
    """
    Calculate elastic tensor from VASP finite difference calculations.
    
    This function reads VASP OUTCAR files from a directory structure containing
    strained calculations and computes the elastic tensor using finite differences.
    
    Parameters:
    -----------
    base_path : str or Path
        Base directory containing VASP calculations
    strain_magnitude : float, optional
        Magnitude of strain applied in calculations. If None, will be auto-detected
        from the lattice parameters (default: None)
    reference_outcar : str, optional
        Path to reference OUTCAR file. If None, uses base_path/OUTCAR or ep1_plus/OUTCAR
    
    Returns:
    --------
    elastic_tensor : ElasticTensor
        Calculated elastic tensor object
    
    Directory Structure Expected:
    ----------------------------
    base_path/
    ├── OUTCAR                    # Reference (unstrained) calculation
    ├── ep1_plus/OUTCAR          # +εxx strain
    ├── ep1_minus/OUTCAR         # -εxx strain
    ├── ep2_plus/OUTCAR          # +εyy strain
    ├── ep2_minus/OUTCAR         # -εyy strain
    ├── ep3_plus/OUTCAR          # +εzz strain
    ├── ep3_minus/OUTCAR         # -εzz strain
    ├── ep4_plus/OUTCAR          # +γyz/2 shear strain
    ├── ep4_minus/OUTCAR         # -γyz/2 shear strain
    ├── ep5_plus/OUTCAR          # +γxz/2 shear strain
    ├── ep5_minus/OUTCAR         # -γxz/2 shear strain
    ├── ep6_plus/OUTCAR          # +γxy/2 shear strain
    └── ep6_minus/OUTCAR         # -γxy/2 shear strain
    
    Example:
    --------
    >>> from sawbench.elastic import from_vasp_dir
    >>> # Auto-detect strain magnitude
    >>> elastic_tensor = from_vasp_dir('/path/to/calculations')
    >>> print(elastic_tensor)
    >>> props = elastic_tensor.get_elastic_properties()
    >>> print(f"Bulk modulus: {props['bulk_modulus']:.2f} GPa")
    """
    base_path = Path(base_path)
    
    # Auto-detect strain magnitude if not provided
    if strain_magnitude is None:
        print("Auto-detecting strain magnitude from lattice parameters...")
        strain_magnitude = _detect_strain_magnitude(base_path)
        print(f"Detected strain magnitude: {strain_magnitude:.6f} ({strain_magnitude*100:.3f}%)")
    
    # Find reference calculation
    if reference_outcar is None:
        # Try base_path/OUTCAR first (unstrained reference)
        reference_outcar = base_path / 'OUTCAR'
        if not reference_outcar.exists():
            # Fallback to ep1_plus (assuming it's a reasonable reference)
            reference_outcar = base_path / 'ep1_plus' / 'OUTCAR'
            print("Warning: No unstrained reference found, using ep1_plus as reference")
    
    if not os.path.exists(reference_outcar):
        raise FileNotFoundError(f"Reference OUTCAR not found at {reference_outcar}")
    
    print(f"Reading reference calculation from {reference_outcar}")
    ref_atoms = read(str(reference_outcar), index=-1)
    ref_stress = ref_atoms.get_stress(voigt=True)  # ASE returns stress in eV/Å³
    
    # Convert stress from eV/Å³ to GPa
    # 1 eV/Å³ = 160.21766208 GPa
    ref_stress_gpa = ref_stress * 160.21766208
    
    print(f"Reference stress (GPa): {ref_stress_gpa}")
    
    # Read strained calculations
    stress_data = {}
    
    for strain_idx in range(1, 7):  # Strain components 1-6
        for sign in ['plus', 'minus']:
            dir_name = f"ep{strain_idx}_{sign}"
            outcar_path = base_path / dir_name / 'OUTCAR'
            
            if not outcar_path.exists():
                print(f"Warning: {outcar_path} not found, skipping...")
                continue
            
            print(f"Reading {dir_name}")
            atoms = read(str(outcar_path), index=-1)
            stress = atoms.get_stress(voigt=True) * 160.21766208  # Convert to GPa
            
            stress_data[(strain_idx, sign)] = stress
    
    print(f"Loaded {len(stress_data)} strained calculations")
    
    # Calculate elastic tensor using finite differences
    elastic_tensor = np.zeros((6, 6))
    
    for j in range(1, 7):  # Strain components j = 1,2,3,4,5,6
        plus_key = (j, 'plus')
        minus_key = (j, 'minus')
        
        if plus_key not in stress_data or minus_key not in stress_data:
            print(f"Warning: Missing data for strain component {j}, skipping...")
            continue
        
        stress_plus = stress_data[plus_key]
        stress_minus = stress_data[minus_key]
        
        # Calculate derivatives: C_ij = ∂σ_i/∂ε_j
        for i in range(6):  # Stress components i = 0,1,2,3,4,5
            dstress_dstrain = (stress_plus[i] - stress_minus[i]) / (2 * strain_magnitude)
            elastic_tensor[i, j-1] = dstress_dstrain
    
    print("Elastic tensor calculation completed!")
    print(f"Used strain magnitude: {strain_magnitude:.6f} ({strain_magnitude*100:.3f}%)")
    
    return ElasticTensor(elastic_tensor)


def _detect_strain_magnitude(base_path: Path) -> float:
    """
    Auto-detect strain magnitude from lattice parameters.
    
    Parameters:
    -----------
    base_path : Path
        Base directory containing VASP calculations
        
    Returns:
    --------
    strain_magnitude : float
        Detected strain magnitude
    """
    try:
        # Read reference structure from ep1_plus (initial structure)
        ref_outcar = base_path / 'ep1_plus' / 'OUTCAR'
        ref_atoms = read(str(ref_outcar), index=0)  # Initial structure
        ref_cell = ref_atoms.get_cell()
        
        # Check ep1_plus and ep1_minus for actual strain in x-direction
        strains = {}
        for sign in ['plus', 'minus']:
            outcar_path = base_path / f'ep1_{sign}' / 'OUTCAR'
            if outcar_path.exists():
                atoms = read(str(outcar_path), index=0)
                cell = atoms.get_cell()
                
                # Calculate strain in x-direction
                strain_x = (cell[0,0] - ref_cell[0,0]) / ref_cell[0,0]
                strains[f'ep1_{sign}'] = strain_x
        
        # Estimate actual strain magnitude
        if 'ep1_plus' in strains and 'ep1_minus' in strains:
            detected_strain = abs(strains['ep1_plus'] - strains['ep1_minus']) / 2
            return detected_strain
            
    except Exception as e:
        print(f"Warning: Could not auto-detect strain magnitude: {e}")
    
    # Default fallback
    print("Using default strain magnitude of 0.01")
    return 0.01


def tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 tensor to 6-component Voigt notation.
    
    Parameters:
    -----------
    tensor : np.ndarray
        3x3 tensor
        
    Returns:
    --------
    voigt : np.ndarray
        6-component Voigt vector
    """
    if tensor.shape != (3, 3):
        raise ValueError("Input must be 3x3 tensor")
    
    voigt = np.zeros(6)
    voigt[0] = tensor[0, 0]  # XX
    voigt[1] = tensor[1, 1]  # YY
    voigt[2] = tensor[2, 2]  # ZZ
    voigt[3] = tensor[1, 2]  # YZ
    voigt[4] = tensor[0, 2]  # XZ
    voigt[5] = tensor[0, 1]  # XY
    
    return voigt


def voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
    """
    Convert 6-component Voigt notation to 3x3 tensor.
    
    Parameters:
    -----------
    voigt : np.ndarray
        6-component Voigt vector
        
    Returns:
    --------
    tensor : np.ndarray
        3x3 tensor
    """
    if voigt.shape != (6,):
        raise ValueError("Input must be 6-component vector")
    
    tensor = np.zeros((3, 3))
    tensor[0, 0] = voigt[0]  # XX
    tensor[1, 1] = voigt[1]  # YY
    tensor[2, 2] = voigt[2]  # ZZ
    tensor[1, 2] = tensor[2, 1] = voigt[3]  # YZ
    tensor[0, 2] = tensor[2, 0] = voigt[4]  # XZ
    tensor[0, 1] = tensor[1, 0] = voigt[5]  # XY
    
    return tensor


def calculate_elastic_tensor(atoms, calculator, strain_magnitude=0.005, 
                           optimize_reference=True, optimize_strained=False,
                           fmax=0.01, max_steps=200) -> ElasticTensor:
    """
    Calculate elastic tensor using ASE atoms object and calculator.
    
    This function applies the same strain pattern as the VASP finite difference
    method but uses ASE calculators for the stress calculations.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Reference atomic structure (should be relaxed)
    calculator : ase.calculators.Calculator
        ASE calculator object (MACE, VASP, etc.)
    strain_magnitude : float
        Magnitude of strain to apply (default: 0.005 = 0.5%)
    optimize_reference : bool
        Whether to optimize the reference structure first (default: True)
    optimize_strained : bool
        Whether to optimize strained structures (default: False)
    fmax : float
        Force convergence criterion for optimization (default: 0.01 eV/Å)
    max_steps : int
        Maximum optimization steps (default: 200)
    
    Returns:
    --------
    elastic_tensor : ElasticTensor
        Calculated elastic tensor object
    
    Example:
    --------
    >>> from ase.io import read
    >>> from mace.calculators import MACECalculator
    >>> from sawbench.elastic import calculate_elastic_tensor
    >>> 
    >>> # Load structure and calculator
    >>> atoms = read('CONTCAR')
    >>> calc = MACECalculator(model_paths=['path/to/model'])
    >>> 
    >>> # Calculate elastic tensor
    >>> elastic_tensor = calculate_elastic_tensor(atoms, calc)
    >>> print(elastic_tensor)
    """
    import numpy as np
    from ase.optimize import FIRE
    from ase.constraints import FixSymmetry
    
    print("ASE Elastic Tensor Calculation")
    print("="*40)
    print(f"Strain magnitude: {strain_magnitude:.6f} ({strain_magnitude*100:.3f}%)")
    print(f"Optimize reference: {optimize_reference}")
    print(f"Optimize strained: {optimize_strained}")
    
    # Set up reference structure
    reference_atoms = atoms.copy()
    reference_atoms.set_calculator(calculator)
    
    if optimize_reference:
        print("\nOptimizing reference structure...")
        # Add symmetry constraints if needed
        try:
            reference_atoms.set_constraint(FixSymmetry(reference_atoms))
        except:
            pass  # Skip if symmetry detection fails
        
        opt = FIRE(reference_atoms, logfile='reference_opt.log')
        opt.run(fmax=fmax, steps=max_steps)
        print(f"Reference optimization completed")
    
    # Get reference stress
    ref_stress = reference_atoms.get_stress(voigt=True)  # eV/Å³
    ref_stress_gpa = ref_stress * 160.21766208  # Convert to GPa
    print(f"Reference stress (GPa): {ref_stress_gpa}")
    
    # Define strain matrices (same as your script)
    delta = strain_magnitude
    strain_matrices = [
        np.diag([delta, 0, 0]),                    # ε11 (ep1)
        np.diag([0, delta, 0]),                    # ε22 (ep2)  
        np.diag([0, 0, delta]),                    # ε33 (ep3)
        np.array([[0,0,0],[0,0,delta],[0,delta,0]]),   # γ23/2 (ep4)
        np.array([[0,0,delta],[0,0,0],[delta,0,0]]),   # γ13/2 (ep5)
        np.array([[0,delta,0],[delta,0,0],[0,0,0]])    # γ12/2 (ep6)
    ]
    
    print(f"\nCalculating stresses for {len(strain_matrices)} strain components...")
    
    # Calculate stresses for all strains
    stress_data = {}
    
    for i, strain_matrix in enumerate(strain_matrices, 1):
        print(f"\nStrain component {i}:")
        
        for sign, multiplier in [('plus', 1), ('minus', -1)]:
            print(f"  Processing ep{i}_{sign}...")
            
            # Create strained structure
            strained_atoms = reference_atoms.copy()
            
            # Apply strain: new_cell = (I + multiplier * strain) @ old_cell
            strain_tensor = multiplier * strain_matrix
            new_cell = (np.eye(3) + strain_tensor) @ reference_atoms.get_cell()
            strained_atoms.set_cell(new_cell, scale_atoms=True)
            
            # Set calculator
            strained_atoms.set_calculator(calculator)
            
            # Optimize if requested
            if optimize_strained:
                print(f"    Optimizing ep{i}_{sign}...")
                try:
                    strained_atoms.set_constraint(FixSymmetry(strained_atoms))
                except:
                    pass
                
                opt = FIRE(strained_atoms, logfile=f'ep{i}_{sign}_opt.log')
                opt.run(fmax=fmax, steps=max_steps)
            
            # Calculate stress
            stress = strained_atoms.get_stress(voigt=True) * 160.21766208  # Convert to GPa
            stress_data[(i, sign)] = stress
            
            print(f"    Stress: [{stress[0]:.3f}, {stress[1]:.3f}, {stress[2]:.3f}, {stress[3]:.3f}, {stress[4]:.3f}, {stress[5]:.3f}] GPa")
    
    print(f"\nCalculating elastic tensor from {len(stress_data)} stress calculations...")
    
    # Calculate elastic tensor using finite differences
    elastic_tensor = np.zeros((6, 6))
    
    for j in range(1, 7):  # Strain components j = 1,2,3,4,5,6
        plus_key = (j, 'plus')
        minus_key = (j, 'minus')
        
        if plus_key not in stress_data or minus_key not in stress_data:
            print(f"Warning: Missing data for strain component {j}, skipping...")
            continue
        
        stress_plus = stress_data[plus_key]
        stress_minus = stress_data[minus_key]
        
        # Calculate derivatives: C_ij = ∂σ_i/∂ε_j
        for i in range(6):  # Stress components i = 0,1,2,3,4,5
            dstress_dstrain = (stress_plus[i] - stress_minus[i]) / (2 * strain_magnitude)
            elastic_tensor[i, j-1] = dstress_dstrain
    
    print("Elastic tensor calculation completed!")
    print(f"Used strain magnitude: {strain_magnitude:.6f} ({strain_magnitude*100:.3f}%)")
    
    return ElasticTensor(elastic_tensor)


def calculate_elastic_tensor_with_vasp(atoms, vasp_command='vasp_std', 
                                     strain_magnitude=0.005, 
                                     vasp_settings=None) -> ElasticTensor:
    """
    Calculate elastic tensor using VASP through ASE.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        Reference atomic structure
    vasp_command : str
        VASP executable command (default: 'vasp_std')
    strain_magnitude : float
        Strain magnitude (default: 0.005)
    vasp_settings : dict
        VASP INCAR settings (default: optimized for elastic constants)
    
    Returns:
    --------
    elastic_tensor : ElasticTensor
        Calculated elastic tensor
    """
    from ase.calculators.vasp import Vasp
    
    # Default VASP settings optimized for elastic constants
    if vasp_settings is None:
        vasp_settings = {
            'xc': 'PBE',
            'encut': 520,           # High cutoff for accurate stress
            'ediff': 1e-8,          # Tight electronic convergence
            'ediffg': -1e-3,        # Good ionic convergence
            'ismear': 0,            # Gaussian smearing
            'sigma': 0.05,
            'prec': 'Accurate',
            'ibrion': -1,           # No ionic relaxation for elastic constants
            'nsw': 0,               # No ionic steps
            'isif': 2,              # Calculate stress tensor
            'lreal': False,         # Reciprocal space (more accurate)
            'lwave': False,         # Don't write WAVECAR
            'lcharg': False,        # Don't write CHGCAR
        }
    
    # Create VASP calculator
    calc = Vasp(command=vasp_command, **vasp_settings)
    
    # Use the general elastic tensor function
    return calculate_elastic_tensor(
        atoms=atoms,
        calculator=calc,
        strain_magnitude=strain_magnitude,
        optimize_reference=False,  # Assume atoms are already relaxed
        optimize_strained=False    # Single-point calculations for VASP
    )
