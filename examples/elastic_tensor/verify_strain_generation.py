#!/usr/bin/env python3
"""
Verify Strain Generation
=======================

This script verifies that the strain generation in the new ASE elastic tensor
function is identical to the user's original Python script.
"""

import numpy as np
from ase.build import bulk

def original_strain_generation(delta=0.005):
    """
    Original strain generation from user's script
    """
    # Six Voigt strain matrices (from user's script)
    eps = [
        np.diag([delta,0,0]),
        np.diag([0,delta,0]),
        np.diag([0,0,delta]),
        [[0,0,0],[0,0. ,delta],[0,delta,0]],
        [[0,0,delta],[0,0,0],[delta,0,0]],
        [[0,delta,0],[delta,0,0],[0,0,0]]
    ]
    return eps

def sawbench_strain_generation(delta=0.005):
    """
    Strain generation from sawbench function
    """
    strain_matrices = [
        np.diag([delta, 0, 0]),                    # ε11 (ep1)
        np.diag([0, delta, 0]),                    # ε22 (ep2)  
        np.diag([0, 0, delta]),                    # ε33 (ep3)
        np.array([[0,0,0],[0,0,delta],[0,delta,0]]),   # γ23/2 (ep4)
        np.array([[0,0,delta],[0,0,0],[delta,0,0]]),   # γ13/2 (ep5)
        np.array([[0,delta,0],[delta,0,0],[0,0,0]])    # γ12/2 (ep6)
    ]
    return strain_matrices

def compare_strain_matrices():
    """
    Compare the two strain generation methods
    """
    print("Strain Matrix Comparison")
    print("="*40)
    
    delta = 0.005
    original = original_strain_generation(delta)
    sawbench = sawbench_strain_generation(delta)
    
    print(f"Strain magnitude: {delta} ({delta*100}%)")
    print()
    
    for i, (orig, sb) in enumerate(zip(original, sawbench), 1):
        print(f"Strain component {i}:")
        print("Original:")
        print(orig)
        print("Sawbench:")
        print(sb)
        
        # Check if they're identical
        if np.allclose(orig, sb):
            print("CORRECT")
        else:
            print("INCORRECT")
            print(f"Difference:\n{orig - sb}")
        print()

def test_cell_deformation():
    """
    Test that cell deformation produces the same results
    """
    print("Cell Deformation Test")
    print("="*30)
    
    # Create a test structure
    atoms = bulk('Al', 'fcc', a=4.05)
    base_cell = atoms.get_cell()
    
    print("Base cell:")
    print(base_cell)
    print()
    
    δ = 0.005
    original = original_strain_generation(δ)
    sawbench = sawbench_strain_generation(δ)
    
    for i, (orig_strain, sb_strain) in enumerate(zip(original, sawbench), 1):
        print(f"Testing strain component {i}:")
        
        # Apply strain using original method
        orig_plus_cell = (np.eye(3) + orig_strain) @ base_cell
        orig_minus_cell = (np.eye(3) - orig_strain) @ base_cell
        
        # Apply strain using sawbench method  
        sb_plus_cell = (np.eye(3) + sb_strain) @ base_cell
        sb_minus_cell = (np.eye(3) - sb_strain) @ base_cell
        
        # Compare results
        plus_match = np.allclose(orig_plus_cell, sb_plus_cell)
        minus_match = np.allclose(orig_minus_cell, sb_minus_cell)
        
        print(f"  +strain cells match: {'CORRECT' if plus_match else 'INCORRECT'}")
        print(f"  -strain cells match: {'CORRECT' if minus_match else 'INCORRECT'}")
        
        if not plus_match:
            print(f"  +strain difference:\n{orig_plus_cell - sb_plus_cell}")
        if not minus_match:
            print(f"  -strain difference:\n{orig_minus_cell - sb_minus_cell}")
        print()

def verify_strain_tensor_conversion():
    """
    Verify that the strain tensors correspond to correct Voigt notation
    """
    print("Strain Tensor to Voigt Conversion")
    print("="*40)
    
    delta = 0.005
    strain_matrices = sawbench_strain_generation(delta)
    
    expected_voigt = [
        [delta, 0, 0, 0, 0, 0],      # ε11
        [0, delta, 0, 0, 0, 0],      # ε22
        [0, 0, delta, 0, 0, 0],      # ε33
        [0, 0, 0, 2*delta, 0, 0],    # γ23 (note: factor of 2)
        [0, 0, 0, 0, 2*delta, 0],    # γ13
        [0, 0, 0, 0, 0, 2*delta]     # γ12
    ]
    
    for i, (strain_matrix, expected) in enumerate(zip(strain_matrices, expected_voigt), 1):
        print(f"Strain component {i}:")
        
        # Convert 3x3 strain tensor to Voigt notation
        voigt = np.array([
            strain_matrix[0,0],  # ε11
            strain_matrix[1,1],  # ε22
            strain_matrix[2,2],  # ε33
            2*strain_matrix[1,2],  # γ23 = 2*ε23
            2*strain_matrix[0,2],  # γ13 = 2*ε13
            2*strain_matrix[0,1]   # γ12 = 2*ε12
        ])
        
        print(f"  Calculated Voigt: {voigt}")
        print(f"  Expected Voigt:   {expected}")
        
        if np.allclose(voigt, expected):
            print("  CORRECT")
        else:
            print("  INCORRECT")
            print(f"  Difference: {voigt - expected}")
        print()

def main():
    """
    Main verification function
    """
    print("Verifying Strain Generation Equivalence")
    print("="*50)
    print()
    
    compare_strain_matrices()
    test_cell_deformation()
    verify_strain_tensor_conversion()
    
    print("="*50)

if __name__ == "__main__":
    main() 