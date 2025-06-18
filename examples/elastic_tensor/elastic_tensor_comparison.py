#!/usr/bin/env python3
"""
Elastic Tensor Comparison: DFT vs MLIP vs Experiment
===================================================

This script compares elastic tensor calculations from:
1. VASP DFT calculations (from_vasp_dir)
2. Machine Learning Interatomic Potential (MLIP)
3. Experimental literature values

The goal is to evaluate accuracy for SAW velocity predictions.
"""

import numpy as np
from pathlib import Path
from sawbench import from_vasp_dir, calculate_elastic_tensor, Material, SAWCalculator
from ase.io import read

def load_reference_structure():
    """
    Load the reference structure from VASP OUTCAR
    """
    print("Loading Reference Structure")
    print("="*40)
    
    reference_path = Path('data/PBE_Manual_Elastic/reference/OUTCAR')
    
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference OUTCAR not found at {reference_path}")
    
    # Read the first structure (index=0) from OUTCAR
    atoms = read(str(reference_path), index=0)
    
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
    
    data_path = 'data/PBE_Manual_Elastic'
    reference_path = Path(data_path) / 'reference' / 'OUTCAR'
    
    # Calculate using the validated VASP data
    vasp_elastic_tensor = from_vasp_dir(
        base_path=data_path,
        reference_outcar=str(reference_path),
        strain_magnitude=0.005
    )
    
    print("\nVASP DFT Results:")
    print(vasp_elastic_tensor)
    
    return vasp_elastic_tensor

def calculate_mlip_elastic_tensor(atoms, calculator):
    """
    Calculate elastic tensor using MLIP
    """
    print("\n" + "="*60)
    print("2. MLIP Elastic Tensor Calculation")
    print("="*60)
    
    # Calculate using the same strain magnitude as VASP (0.5%)
    mlip_elastic_tensor = calculate_elastic_tensor(
        atoms=atoms,
        calculator=calculator,
        strain_magnitude=0.005,      # Same as VASP calculations
        optimize_strained=True,      # Relax strained structures (atoms only, fixed cell)
        fmax=0.001                    # Force convergence criterion
    )
    
    print("\nMLIP Results:")
    print(mlip_elastic_tensor)
    
    return mlip_elastic_tensor

def compare_elastic_constants(vasp_tensor, mlip_tensor, experimental_values):
    """
    Compare elastic constants from different methods
    """
    print("\n" + "="*60)
    print("3. Elastic Constants Comparison")
    print("="*60)
    
    # Extract properties
    vasp_props = vasp_tensor.get_elastic_properties()
    mlip_props = mlip_tensor.get_elastic_properties()
    
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
    
    # Calculate derived properties comparison
    print(f"\n{'Derived Properties':<20} | {'VASP':<12} | {'MLIP':<12} | {'MLIP/VASP':<10}")
    print("-" * 60)
    
    derived_props = ['bulk_modulus', 'shear_modulus', 'youngs_modulus', 'poisson_ratio']
    for prop in derived_props:
        vasp_val = vasp_props.get(prop, 0)
        mlip_val = mlip_props.get(prop, 0)
        ratio = mlip_val / vasp_val if vasp_val != 0 else 0
        
        if prop == 'poisson_ratio':
            print(f"{prop:<20} | {vasp_val:>10.3f}   | {mlip_val:>10.3f}   | {ratio:>8.3f}")
        else:
            print(f"{prop:<20} | {vasp_val:>10.1f}   | {mlip_val:>10.1f}   | {ratio:>8.3f}")
    
    return results

def calculate_saw_velocities(elastic_constants, density):
    """
    Calculate SAW velocities using different elastic constants
    """
    print("\n" + "="*60)
    print("4. SAW Velocity Predictions")
    print("="*60)
    
    saw_results = {}
    
    for method, constants in elastic_constants.items():
        if method == 'experimental':
            continue
            
        print(f"\n{method.upper()} SAW Calculation:")
        print("-" * 30)
        
        # Create material
        material = Material(
            formula='V-Ti_alloy',
            C11=constants['C11'] * 1e9,  # Convert GPa to Pa
            C12=constants['C12'] * 1e9,
            C44=constants['C44'] * 1e9,
            density=density,  # kg/m³
            crystal_class='cubic'
        )
        
        # Calculate SAW velocities
        saw_calc = SAWCalculator(material)
        
        # Calculate for different propagation directions
        directions = [
            ([1, 0, 0], "X-direction"),
            ([1, 1, 0], "XY-direction"), 
            ([1, 1, 1], "XYZ-direction")
        ]
        
        method_results = {}
        
        for direction, name in directions:
            try:
                velocities = saw_calc.calculate_saw_velocities(
                    propagation_direction=direction,
                    surface_normal=[0, 0, 1]
                )
                
                # Extract Rayleigh velocity (typically the slowest)
                rayleigh_velocity = min(velocities) if velocities else 0
                method_results[name] = rayleigh_velocity
                
                print(f"  {name:<15}: {rayleigh_velocity:.0f} m/s")
                
            except Exception as e:
                print(f"  {name:<15}: Error - {e}")
                method_results[name] = 0
        
        saw_results[method] = method_results
    
    return saw_results

def analyze_errors(elastic_constants, saw_results):
    """
    Analyze errors between methods
    """
    print("\n" + "="*60)
    print("5. Error Analysis")
    print("="*60)
    
    print("\nElastic Constants Errors:")
    print("-" * 30)
    
    for prop in ['C11', 'C12', 'C44']:
        vasp_val = elastic_constants['vasp'][prop]
        mlip_val = elastic_constants['mlip'][prop]
        exp_val = elastic_constants['experimental'][prop]
        
        vasp_error = abs(vasp_val - exp_val) / exp_val * 100
        mlip_error = abs(mlip_val - exp_val) / exp_val * 100
        mlip_vasp_error = abs(mlip_val - vasp_val) / vasp_val * 100
        
        print(f"{prop}:")
        print(f"  VASP vs Exp:  {vasp_error:5.1f}% error")
        print(f"  MLIP vs Exp:  {mlip_error:5.1f}% error") 
        print(f"  MLIP vs VASP: {mlip_vasp_error:5.1f}% error")
    
    print("\nSAW Velocity Errors:")
    print("-" * 25)
    
    for direction in saw_results['vasp'].keys():
        vasp_vel = saw_results['vasp'][direction]
        mlip_vel = saw_results['mlip'][direction]
        
        if vasp_vel > 0 and mlip_vel > 0:
            error = abs(mlip_vel - vasp_vel) / vasp_vel * 100
            print(f"{direction:<15}: {error:5.1f}% error (MLIP vs VASP)")

def main():
    """
    Main comparison workflow
    """
    print("Elastic Tensor Comparison: DFT vs MLIP vs Experiment")
    print("="*70)
    
    # Experimental reference values (from literature)
    experimental_values = {
        'C11': 230.0,  # GPa
        'C12': 119.0,  # GPa
        'C44': 43.1    # GPa
    }
    
    # Estimated density (you should replace with your measured value)
    density = 6200  # kg/m³ (typical for V-Ti alloys)
    
    try:
        # Load reference structure
        atoms = load_reference_structure()
        
        # Calculate VASP elastic tensor
        vasp_tensor = calculate_vasp_elastic_tensor()
        
        # Set up your MLIP calculator
        print("\n" + "="*60)
        print("Setting up MLIP Calculator")
        print("="*60)
        
        # Replace this with your actual MLIP calculator
        # Examples:
        # from mace.calculators import MACECalculator
        # calculator = MACECalculator(model_paths=['path/to/your/model'])
        
        from nequip.ase import NequIPCalculator  
        calculator = NequIPCalculator.from_compiled_model(compile_path='data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2',device='cuda')
        
        # Calculate MLIP elastic tensor
        mlip_tensor = calculate_mlip_elastic_tensor(atoms, calculator)
        
        # Organize results for comparison
        vasp_props = vasp_tensor.get_elastic_properties()
        mlip_props = mlip_tensor.get_elastic_properties()
        
        elastic_constants = {
            'vasp': {
                'C11': vasp_props['C11'],
                'C12': vasp_props['C12'], 
                'C44': vasp_props['C44']
            },
            'mlip': {
                'C11': mlip_props['C11'],
                'C12': mlip_props['C12'],
                'C44': mlip_props['C44']
            },
            'experimental': experimental_values
        }
        
        # Compare elastic constants
        comparison_results = compare_elastic_constants(vasp_tensor, mlip_tensor, experimental_values)
        
        # Calculate SAW velocities
        saw_results = calculate_saw_velocities(elastic_constants, density)
        
        # Analyze errors
        analyze_errors(elastic_constants, saw_results)
        
        # Save results
        print("\n" + "="*60)
        print("6. Saving Results")
        print("="*60)
        
        output_file = 'elastic_tensor_comparison_results.txt'
        with open(output_file, 'w') as f:
            f.write("Elastic Tensor Comparison Results\n")
            f.write("="*50 + "\n\n")
            
            f.write("VASP DFT Results:\n")
            f.write(str(vasp_tensor) + "\n\n")
            
            f.write("MLIP Results:\n") 
            f.write(str(mlip_tensor) + "\n\n")
            
            f.write("Comparison Summary:\n")
            f.write("-" * 20 + "\n")
            for prop in ['C11', 'C12', 'C44']:
                f.write(f"{prop}: VASP={comparison_results[prop]['vasp']:.1f} GPa, ")
                f.write(f"MLIP={comparison_results[prop]['mlip']:.1f} GPa, ")
                f.write(f"Exp={comparison_results[prop]['experimental']:.1f} GPa\n")
        
        print(f"Results saved to: {output_file}")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("Successfully compared VASP DFT vs MLIP elastic tensors")
        print("Calculated SAW velocities for both methods")
        print("Analyzed errors relative to experimental values")
        print("Ready for SAW prediction accuracy assessment")
        
    except Exception as e:
        print(f"\nError in comparison workflow: {e}")
        print("Please check:")
        print("1. VASP data is available in data/PBE_Manual_Elastic/")
        print("2. Reference OUTCAR exists")
        print("3. MLIP calculator is properly configured")

if __name__ == "__main__":
    main() 