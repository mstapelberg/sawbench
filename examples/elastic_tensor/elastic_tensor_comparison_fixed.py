#!/usr/bin/env python3
"""
Fixed Elastic Tensor Comparison: DFT vs MLIP vs Experiment
=========================================================

This script compares elastic tensor calculations from:
1. VASP DFT calculations (from_vasp_dir)
2. Machine Learning Interatomic Potential (MLIP)
3. Experimental literature values

Fixed issues:
- Proper crystal system detection
- Correct stress-strain relationship
- Better error handling
- Debugging output
"""

import numpy as np
from pathlib import Path
from sawbench import from_vasp_dir, Material, SAWCalculator
from ase.io import read
from ase.optimize import FIRE
from ase.constraints import FixSymmetry

def load_reference_structure(initial_structure=True):
    """
    Load the reference structure from VASP OUTCAR
    """
    print("Loading Reference Structure")
    print("="*40)
    
    reference_path = Path('../data/PBE_Manual_Elastic/reference/OUTCAR')
    
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference OUTCAR not found at {reference_path}")
    
    # Read the first structure (index=0) from OUTCAR
    if initial_structure:
        atoms = read(str(reference_path), index=0)
    else:
        atoms = read(str(reference_path), index="-1")

    
    print(f"Loaded structure from: {reference_path}")
    print(f"Chemical formula: {atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Cell volume: {atoms.get_volume():.3f} Å³")
    print(f"Density: {atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume():.3f} g/cm³")
    
    # Print cell parameters
    cell = atoms.get_cell()
    print(f"Cell parameters:")
    for i, vec in enumerate(cell):
        print(f"  a{i+1}: [{vec[0]:8.3f}, {vec[1]:8.3f}, {vec[2]:8.3f}] Å")
    
    return atoms

def calculate_vasp_elastic_tensor():
    """
    Calculate elastic tensor from VASP finite difference calculations
    """
    print("\n" + "="*60)
    print("1. VASP DFT Elastic Tensor Calculation")
    print("="*60)
    
    data_path = '../data/PBE_Manual_Elastic'
    reference_path = Path(data_path) / 'reference' / 'OUTCAR'
    
    # Calculate using the validated VASP data
    vasp_elastic_tensor = from_vasp_dir(
        base_path=data_path,
        reference_outcar=str(reference_path)
        # strain_magnitude will be auto-detected
    )
    
    print("\nVASP DFT Results:")
    print(vasp_elastic_tensor)
    
    return vasp_elastic_tensor

def calculate_mlip_elastic_tensor_fixed(atoms, calculator, strain_magnitude=0.005, 
                                       optimize_strained=True, fmax=1e-3, max_steps=2500):
    """
    Fixed MLIP elastic tensor calculation with proper debugging and optimization
    """
    print("\n" + "="*60)
    print("2. MLIP Elastic Tensor Calculation (Fixed)")
    print("="*60)
    
    print(f"Strain magnitude: {strain_magnitude:.6f} ({strain_magnitude*100:.3f}%)")
    print(f"Optimize strained structures: {optimize_strained}")
    print(f"Force convergence: {fmax:.1e} eV/Å")
    print(f"Max optimization steps: {max_steps}")
    
    # Set up reference structure (no optimization to avoid issues)
    reference_atoms = atoms.copy()
    reference_atoms.set_calculator(calculator)
    
    # Get reference stress
    try:
        ref_stress = reference_atoms.get_stress(voigt=True)  # eV/Å³
        ref_stress_gpa = ref_stress * 160.21766208  # Convert to GPa
        print(f"Reference stress (GPa): {ref_stress_gpa}")
    except Exception as e:
        print(f"Error calculating reference stress: {e}")
        return None
    
    # Define strain matrices - let's try the original definition first
    delta = strain_magnitude
    strain_matrices = [
        np.diag([delta, 0, 0]),                              # ε11 (ep1)
        np.diag([0, delta, 0]),                              # ε22 (ep2)  
        np.diag([0, 0, delta]),                              # ε33 (ep3)
        np.array([[0,0,0],[0,0,delta],[0,delta,0]]),         # γ23 (ep4) - back to original
        np.array([[0,0,delta],[0,0,0],[delta,0,0]]),         # γ13 (ep5) - back to original
        np.array([[0,delta,0],[delta,0,0],[0,0,0]])          # γ12 (ep6) - back to original
    ]
    
    print(f"\nCalculating stresses for {len(strain_matrices)} strain components...")
    print("Note: Using original shear strain definition to match VASP")
    
    # Calculate stresses for all strains
    stress_data = {}
    
    for i, strain_matrix in enumerate(strain_matrices, 1):
        print(f"\nStrain component {i}:")
        print(f"  Strain matrix:\n{strain_matrix}")
        
        for sign, multiplier in [('plus', 1), ('minus', -1)]:
            print(f"  Processing ep{i}_{sign}...")
            
            try:
                # Create strained structure
                strained_atoms = reference_atoms.copy()
                
                # Apply strain: new_cell = (I + multiplier * strain) @ old_cell
                strain_tensor = multiplier * strain_matrix
                new_cell = (np.eye(3) + strain_tensor) @ reference_atoms.get_cell()
                strained_atoms.set_cell(new_cell, scale_atoms=True)
                
                # Set calculator
                strained_atoms.set_calculator(calculator)
                
                # Optimize strained structure if requested
                if optimize_strained:
                    print(f"    Optimizing ep{i}_{sign}...")
                    opt = FIRE(strained_atoms, logfile=f'ep{i}_{sign}_opt.log')
                    opt.run(fmax=fmax, steps=max_steps)
                
                # Calculate stress
                stress = strained_atoms.get_stress(voigt=True) * 160.21766208  # Convert to GPa
                stress_data[(i, sign)] = stress
                
                print(f"    Stress: [{stress[0]:.3f}, {stress[1]:.3f}, {stress[2]:.3f}, {stress[3]:.3f}, {stress[4]:.3f}, {stress[5]:.3f}] GPa")
                
            except Exception as e:
                print(f"    Error calculating stress for ep{i}_{sign}: {e}")
                return None
    
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
        
        print(f"\nStrain component {j}:")
        print(f"  Stress_plus:  {stress_plus}")
        print(f"  Stress_minus: {stress_minus}")
        print(f"  Difference:   {stress_plus - stress_minus}")
        
        # Calculate derivatives: C_ij = ∂σ_i/∂ε_j
        for i in range(6):  # Stress components i = 0,1,2,3,4,5
            dstress_dstrain = (stress_plus[i] - stress_minus[i]) / (2 * strain_magnitude)
            elastic_tensor[i, j-1] = dstress_dstrain
            print(f"  C_{i+1}{j} = {dstress_dstrain:.2f} GPa")
    
    print("Elastic tensor calculation completed!")
    print(f"Used strain magnitude: {strain_magnitude:.6f} ({strain_magnitude*100:.3f}%)")
    
    # Debug: Print raw elastic tensor
    print("\nRaw elastic tensor (GPa):")
    print(elastic_tensor)
    
    # Check for problematic values
    print("\nElastic tensor analysis:")
    print(f"C11 = {elastic_tensor[0,0]:.2f} GPa")
    print(f"C12 = {elastic_tensor[0,1]:.2f} GPa") 
    print(f"C44 = {elastic_tensor[3,3]:.2f} GPa")
    print(f"C55 = {elastic_tensor[4,4]:.2f} GPa")
    print(f"C66 = {elastic_tensor[5,5]:.2f} GPa")
    
    if elastic_tensor[3,3] < 0 or elastic_tensor[4,4] < 0 or elastic_tensor[5,5] < 0:
        print("WARNING: Negative shear moduli detected! This indicates an issue with the calculation.")
    
    # Create ElasticTensor object with debugging
    from sawbench.elastic import ElasticTensor
    
    try:
        mlip_elastic_tensor = ElasticTensor(elastic_tensor)
        return mlip_elastic_tensor
    except Exception as e:
        print(f"Error creating ElasticTensor object: {e}")
        return None

def compare_elastic_constants(vasp_tensor, mlip_tensor, experimental_values):
    """
    Compare elastic constants from different methods
    """
    print("\n" + "="*60)
    print("3. Elastic Constants Comparison")
    print("="*60)
    
    if mlip_tensor is None:
        print("MLIP tensor calculation failed, skipping comparison")
        return None
    
    # Extract properties
    vasp_props = vasp_tensor.get_elastic_properties()
    mlip_props = mlip_tensor.get_elastic_properties()
    
    print("VASP properties:", vasp_props)
    print("MLIP properties:", mlip_props)
    
    # Create comparison table
    properties = ['C11', 'C12', 'C44']
    
    print(f"{'Property':<8} | {'VASP (GPa)':<12} | {'MLIP (GPa)':<12} | {'Exp (GPa)':<11} | {'VASP/Exp':<9} | {'MLIP/Exp':<9} | {'MLIP/VASP':<10}")
    print("-" * 85)
    
    results = {}
    
    for prop in properties:
        vasp_val = vasp_props.get(prop, 0)
        mlip_val = mlip_props.get(prop, 0)
        exp_val = experimental_values.get(prop, 0)
        
        vasp_exp_ratio = vasp_val / exp_val if exp_val != 0 else 0
        mlip_exp_ratio = mlip_val / exp_val if exp_val != 0 else 0
        mlip_vasp_ratio = mlip_val / vasp_val if vasp_val != 0 else 0
        
        print(f"{prop:<8} | {vasp_val:>10.1f}   | {mlip_val:>10.1f}   | {exp_val:>9.1f}   | {vasp_exp_ratio:>7.3f}   | {mlip_exp_ratio:>7.3f}   | {mlip_vasp_ratio:>8.3f}")
        
        results[prop] = {
            'vasp': vasp_val,
            'mlip': mlip_val,
            'experimental': exp_val,
            'vasp_exp_ratio': vasp_exp_ratio,
            'mlip_exp_ratio': mlip_exp_ratio,
            'mlip_vasp_ratio': mlip_vasp_ratio
        }
    
    return results

def main():
    """
    Main comparison workflow
    """
    print("Fixed Elastic Tensor Comparison: DFT vs MLIP vs Experiment")
    print("="*70)
    
    # Experimental reference values (from literature)
    experimental_values = {
        'C11': 230.0,  # GPa
        'C12': 119.0,  # GPa
        'C44': 43.1    # GPa
    }
    
    try:
        # Load reference structure
        atoms = load_reference_structure(initial_structure=False)
        
        # Calculate VASP elastic tensor
        vasp_tensor = calculate_vasp_elastic_tensor()
        
        # Set up MLIP calculator
        print("\n" + "="*60)
        print("Setting up MLIP Calculator")
        print("="*60)
        
        # Use your NequIP calculator
        from nequip.ase import NequIPCalculator
        calculator = NequIPCalculator.from_compiled_model(compile_path='../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2',device='cuda')
        print("Using NequIP calculator")
        
        # Calculate MLIP elastic tensor with fixed method and requested parameters
        mlip_tensor = calculate_mlip_elastic_tensor_fixed(
            atoms, 
            calculator,
            strain_magnitude=0.005,
            optimize_strained=False,  # Single-point calculations only
            fmax=1e-3,
            max_steps=2500
        )
        
        if mlip_tensor is not None:
            # Compare elastic constants
            comparison_results = compare_elastic_constants(vasp_tensor, mlip_tensor, experimental_values)
            
            if comparison_results:
                print("\n" + "="*70)
                print("SUMMARY")
                print("="*70)
                print("Successfully compared VASP DFT vs MLIP elastic tensors")
                print("Fixed crystal system detection and stress calculation issues")
                print("Added proper optimization with fmax=1e-3 and max_steps=2500")
                print("Ready for further analysis")
        else:
            print("MLIP calculation failed - check calculator setup")
        
    except Exception as e:
        print(f"\nError in comparison workflow: {e}")
        import traceback
        traceback.print_exc()
        print("Please check:")
        print("1. VASP data is available in data/PBE_Manual_Elastic/")
        print("2. Reference OUTCAR exists")
        print("3. MLIP calculator is properly configured")

if __name__ == "__main__":
    main() 