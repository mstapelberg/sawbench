#!/usr/bin/env python3
"""
Elastic Tensor Validation Script
===============================

This script provides comprehensive validation of elastic tensor calculations
by checking multiple potential sources of error and comparing with different methods.
"""

import numpy as np
from pathlib import Path
from sawbench import from_vasp_dir
from ase.io import read
import matplotlib.pyplot as plt

def check_vasp_elastic_tensor(outcar_path):
    """
    Check if VASP calculated elastic tensor directly (IBRION=6, ISIF=3)
    """
    print("1. Checking for direct VASP elastic tensor calculation...")
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Look for elastic tensor in OUTCAR
        if "TOTAL ELASTIC MODULI" in content or "SYMMETRIZED ELASTIC MODULI" in content:
            print("   ✓ Found VASP-calculated elastic tensor in OUTCAR!")
            
            # Extract the tensor
            lines = content.split('\n')
            elastic_section = False
            tensor_data = []
            
            for line in lines:
                if "TOTAL ELASTIC MODULI (kBar)" in line:
                    elastic_section = True
                    continue
                if elastic_section and "---" in line and len(tensor_data) > 0:
                    break
                if elastic_section and len(line.split()) == 7:  # Direction + 6 values
                    parts = line.split()
                    if parts[0] in ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'ZX']:
                        values = [float(x) for x in parts[1:]]
                        tensor_data.append(values)
            
            if len(tensor_data) == 6:
                vasp_tensor = np.array(tensor_data) / 10  # Convert kBar to GPa
                print("   VASP Direct Elastic Tensor (GPa):")
                print(vasp_tensor)
                return vasp_tensor
        else:
            print("   ✗ No direct elastic tensor found in VASP OUTCAR")
            print("   This suggests you used finite difference method")
    
    except Exception as e:
        print(f"   Error reading OUTCAR: {e}")
    
    return None

def analyze_strain_magnitude(base_path):
    """
    Analyze the actual strain magnitude from POSCAR files or lattice parameters
    """
    print("\n2. Analyzing actual strain magnitude...")
    
    base_path = Path(base_path)
    strains = {}
    
    # Read reference structure
    try:
        ref_outcar = base_path / 'ep1_plus' / 'OUTCAR'
        ref_atoms = read(str(ref_outcar), index=0)  # Initial structure
        ref_cell = ref_atoms.get_cell()
        print(f"   Reference cell from ep1_plus initial structure:")
        print(f"   {ref_cell}")
        
        # Check ep1_plus and ep1_minus for actual strain
        for sign in ['plus', 'minus']:
            outcar_path = base_path / f'ep1_{sign}' / 'OUTCAR'
            if outcar_path.exists():
                atoms = read(str(outcar_path), index=0)
                cell = atoms.get_cell()
                
                # Calculate strain in x-direction
                strain_x = (cell[0,0] - ref_cell[0,0]) / ref_cell[0,0]
                strains[f'ep1_{sign}'] = strain_x
                print(f"   ep1_{sign}: strain_x = {strain_x:.6f} ({strain_x*100:.3f}%)")
        
        # Estimate actual strain magnitude
        if 'ep1_plus' in strains and 'ep1_minus' in strains:
            actual_strain = abs(strains['ep1_plus'] - strains['ep1_minus']) / 2
            print(f"   ✓ Estimated strain magnitude: {actual_strain:.6f} ({actual_strain*100:.3f}%)")
            return actual_strain
        
    except Exception as e:
        print(f"   Error analyzing strain: {e}")
        print("   Using default strain magnitude of 0.01")
    
    return 0.01

def check_stress_consistency(base_path):
    """
    Check stress values for consistency and physical reasonableness
    """
    print("\n3. Checking stress consistency...")
    
    base_path = Path(base_path)
    
    for strain_idx in range(1, 4):  # Check first 3 normal strains
        stress_plus = None
        stress_minus = None
        
        for sign in ['plus', 'minus']:
            outcar_path = base_path / f'ep{strain_idx}_{sign}' / 'OUTCAR'
            if outcar_path.exists():
                atoms = read(str(outcar_path), index=-1)
                stress = atoms.get_stress(voigt=True) * 160.21766208  # Convert to GPa
                
                if sign == 'plus':
                    stress_plus = stress
                else:
                    stress_minus = stress
                
                print(f"   ep{strain_idx}_{sign} stress (GPa): [{stress[0]:.3f}, {stress[1]:.3f}, {stress[2]:.3f}, {stress[3]:.3f}, {stress[4]:.3f}, {stress[5]:.3f}]")
        
        if stress_plus is not None and stress_minus is not None:
            stress_diff = stress_plus - stress_minus
            print(f"   ep{strain_idx} stress difference: [{stress_diff[0]:.3f}, {stress_diff[1]:.3f}, {stress_diff[2]:.3f}, {stress_diff[3]:.3f}, {stress_diff[4]:.3f}, {stress_diff[5]:.3f}]")

def validate_with_different_strains(base_path):
    """
    Try calculating elastic tensor with different strain magnitude assumptions
    """
    print("\n4. Testing different strain magnitudes...")
    
    strain_values = [0.005, 0.01, 0.015, 0.02, 0.03]
    results = {}
    
    for strain in strain_values:
        try:
            ref_outcar = Path(base_path) / 'ep1_plus' / 'OUTCAR'
            elastic_tensor = from_vasp_dir(
                base_path=base_path,
                strain_magnitude=strain,
                reference_outcar=str(ref_outcar)
            )
            
            props = elastic_tensor.get_elastic_properties()
            results[strain] = {
                'C11': props.get('C11', 0),
                'C12': props.get('C12', 0),
                'C44': props.get('C44', 0)
            }
            
            print(f"   Strain {strain*100:.1f}%: C11={props.get('C11', 0):.1f}, C12={props.get('C12', 0):.1f}, C44={props.get('C44', 0):.1f} GPa")
        
        except Exception as e:
            print(f"   Error with strain {strain}: {e}")
    
    return results

def compare_with_literature(calculated_values, literature_values):
    """
    Compare calculated values with literature and identify potential issues
    """
    print("\n5. Comparison with Literature:")
    print("   Property     | Calculated | Literature | Ratio")
    print("   -------------|------------|------------|--------")
    
    for prop in ['C11', 'C12', 'C44']:
        if prop in calculated_values and prop in literature_values:
            calc = calculated_values[prop]
            lit = literature_values[prop]
            ratio = calc / lit if lit != 0 else float('inf')
            print(f"   {prop:<12} | {calc:>8.1f}   | {lit:>8.1f}   | {ratio:>6.3f}")

def check_material_identification(base_path):
    """
    Try to identify the material from OUTCAR files
    """
    print("\n6. Material identification:")
    
    try:
        outcar_path = Path(base_path) / 'ep1_plus' / 'OUTCAR'
        
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Look for POTCAR information
        if "POTCAR:" in content:
            potcar_lines = [line.strip() for line in content.split('\n') if "POTCAR:" in line]
            elements = []
            for line in potcar_lines[:5]:  # First few lines usually contain element info
                if "PAW_PBE" in line:
                    parts = line.split()
                    for part in parts:
                        if "PAW_PBE" in part:
                            element = part.replace("PAW_PBE", "").split("_")[0]
                            if element and element not in elements:
                                elements.append(element)
            
            print(f"   Elements detected: {elements}")
        
        # Look for lattice parameters
        atoms = read(str(outcar_path), index=-1)
        cell = atoms.get_cell()
        volume = atoms.get_volume()
        
        print(f"   Lattice parameters: a={np.linalg.norm(cell[0]):.3f} Å")
        print(f"   Cell volume: {volume:.3f} Å³")
        print(f"   Number of atoms: {len(atoms)}")
        
    except Exception as e:
        print(f"   Error identifying material: {e}")

def main():
    """
    Main validation function
    """
    print("Elastic Tensor Validation")
    print("="*50)
    
    # Use the real VASP data
    base_path = 'examples/data/PBE_Manual_Elastic'
    
    if not Path(base_path).exists():
        print(f"Error: Data path {base_path} not found!")
        return
    
    # Literature values for comparison (these look like Vanadium values)
    literature = {
        'C11': 230.0,  # GPa
        'C12': 119.0,  # GPa
        'C44': 43.1    # GPa
    }
    
    # 1. Check for direct VASP calculation
    ref_outcar = Path(base_path) / 'ep1_plus' / 'OUTCAR'
    vasp_tensor = check_vasp_elastic_tensor(ref_outcar)
    
    # 2. Analyze actual strain magnitude
    actual_strain = analyze_strain_magnitude(base_path)
    
    # 3. Check stress consistency
    check_stress_consistency(base_path)
    
    # 4. Test different strain magnitudes
    strain_results = validate_with_different_strains(base_path)
    
    # 5. Calculate with best estimate
    print(f"\n7. Calculating with estimated strain magnitude ({actual_strain*100:.3f}%):")
    try:
        elastic_tensor = from_vasp_dir(
            base_path=base_path,
            strain_magnitude=actual_strain,
            reference_outcar=str(ref_outcar)
        )
        
        props = elastic_tensor.get_elastic_properties()
        print(f"   C11 = {props.get('C11', 0):.1f} GPa")
        print(f"   C12 = {props.get('C12', 0):.1f} GPa")
        print(f"   C44 = {props.get('C44', 0):.1f} GPa")
        
        # Compare with literature
        compare_with_literature(props, literature)
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # 6. Material identification
    check_material_identification(base_path)
    
    # 7. Recommendations
    print("\n8. Recommendations:")
    print("   Based on the analysis:")
    
    if vasp_tensor is not None:
        print("   ✓ Use the direct VASP elastic tensor if available")
        vasp_props = {
            'C11': vasp_tensor[0,0],
            'C12': vasp_tensor[0,1], 
            'C44': vasp_tensor[3,3]
        }
        compare_with_literature(vasp_props, literature)
    else:
        print("   • Check if strain magnitude assumption is correct")
        print("   • Verify that calculations converged properly")
        print("   • Consider using a proper unstrained reference")
        print("   • Check VASP settings (ENCUT, KPOINTS, convergence)")

if __name__ == "__main__":
    main() 