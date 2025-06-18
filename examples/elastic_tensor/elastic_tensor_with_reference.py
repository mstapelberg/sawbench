#!/usr/bin/env python3
"""
Elastic Tensor Calculation with Proper Unstrained Reference
==========================================================

This example demonstrates the elastic tensor calculation using the proper
unstrained reference OUTCAR from the relaxed cell, which should give the
most accurate results.
"""

from sawbench import from_vasp_dir
from pathlib import Path

def main():
    """
    Elastic tensor calculation with proper unstrained reference
    """
    
    print("Elastic Tensor Calculation with Unstrained Reference")
    print("="*60)
    
    # Path to the VASP elastic data
    data_path = Path('examples/data/PBE_Manual_Elastic')
    reference_path = data_path / 'reference' / 'OUTCAR'
    
    if not data_path.exists():
        print(f"Error: Data path {data_path} not found!")
        return
    
    if not reference_path.exists():
        print(f"Error: Reference OUTCAR not found at {reference_path}")
        return
    
    print(f"Processing VASP data from: {data_path}")
    print(f"Using unstrained reference: {reference_path}")
    print()
    
    # Calculate elastic tensor with proper unstrained reference
    print("1. Calculating elastic tensor with unstrained reference:")
    print("-" * 55)
    
    elastic_tensor = from_vasp_dir(
        base_path=data_path,
        # strain_magnitude=None,  # Auto-detect (default)
        reference_outcar=str(reference_path)  # Use proper unstrained reference
    )
    
    print("\n" + "="*70)
    print("ELASTIC TENSOR RESULTS (with Unstrained Reference)")
    print("="*70)
    print(elastic_tensor)
    
    # Extract properties for comparison
    props = elastic_tensor.get_elastic_properties()
    
    print("\n2. Comparison with Literature:")
    print("-" * 30)
    
    # Literature values (Vanadium-based alloy)
    literature = {
        'C11': 230.0,  # GPa
        'C12': 119.0,  # GPa  
        'C44': 43.1    # GPa
    }
    
    print("Property | Our Result | Literature | Ratio  | Status")
    print("---------|------------|------------|--------|--------")
    
    for prop in ['C11', 'C12', 'C44']:
        if prop in props:
            our_val = props[prop]
            lit_val = literature[prop]
            ratio = our_val / lit_val
            
            # Determine status
            if 0.85 <= ratio <= 1.15:
                status = "✅ Excellent"
            elif 0.8 <= ratio <= 1.2:
                status = "✅ Good"
            elif 0.7 <= ratio <= 1.3:
                status = "⚠️  OK"
            else:
                status = "❌ Poor"
            
            print(f"{prop:<8} | {our_val:>8.1f}   | {lit_val:>8.1f}   | {ratio:>6.3f} | {status}")
    
    print("\n3. Advantages of Using Unstrained Reference:")
    print("-" * 45)
    print("✅ True zero-strain state as reference")
    print("✅ More accurate stress baseline")
    print("✅ Better finite difference calculation")
    print("✅ Eliminates reference strain artifacts")
    print("✅ Follows standard elastic tensor methodology")
    
    print("\n4. Material Properties:")
    print("-" * 23)
    if 'bulk_modulus' in props:
        print(f"Bulk modulus (K):    {props['bulk_modulus']:.1f} GPa")
    if 'shear_modulus' in props:
        print(f"Shear modulus (G):   {props['shear_modulus']:.1f} GPa")
    if 'youngs_modulus' in props:
        print(f"Young's modulus (E): {props['youngs_modulus']:.1f} GPa")
    if 'poisson_ratio' in props:
        print(f"Poisson's ratio (ν): {props['poisson_ratio']:.3f}")
    
    # Calculate mechanical stability criteria
    print("\n5. Mechanical Stability (Born Criteria for Cubic):")
    print("-" * 50)
    C11 = props.get('C11', 0)
    C12 = props.get('C12', 0)
    C44 = props.get('C44', 0)
    
    # Born stability criteria for cubic crystals
    criterion1 = C11 > abs(C12)
    criterion2 = C11 + 2*C12 > 0
    criterion3 = C44 > 0
    
    print(f"C11 > |C12|:     {C11:.1f} > {abs(C12):.1f} = {'✅ Pass' if criterion1 else '❌ Fail'}")
    print(f"C11 + 2*C12 > 0: {C11 + 2*C12:.1f} > 0 = {'✅ Pass' if criterion2 else '❌ Fail'}")
    print(f"C44 > 0:         {C44:.1f} > 0 = {'✅ Pass' if criterion3 else '❌ Fail'}")
    
    stability = criterion1 and criterion2 and criterion3
    print(f"\nOverall stability: {'✅ STABLE' if stability else '❌ UNSTABLE'}")
    
    # Save results with reference information
    print(f"\n6. Saving Results:")
    print("-" * 18)
    
    output_file = 'vasp_elastic_with_reference.txt'
    with open(output_file, 'w') as f:
        f.write("Elastic Tensor Results (with Unstrained Reference)\n")
        f.write("="*55 + "\n\n")
        f.write(f"Reference OUTCAR: {reference_path}\n")
        f.write(f"Strain magnitude: Auto-detected\n\n")
        f.write(str(elastic_tensor))
        f.write(f"\n\nComparison with Literature:\n")
        f.write("-" * 30 + "\n")
        
        for prop in ['C11', 'C12', 'C44']:
            if prop in props:
                our_val = props[prop]
                lit_val = literature[prop]
                ratio = our_val / lit_val
                f.write(f"{prop}: {our_val:.1f} GPa (Literature: {lit_val:.1f} GPa, Ratio: {ratio:.3f})\n")
        
        f.write(f"\n\nMechanical Stability:\n")
        f.write(f"C11 > |C12|: {criterion1}\n")
        f.write(f"C11 + 2*C12 > 0: {criterion2}\n") 
        f.write(f"C44 > 0: {criterion3}\n")
        f.write(f"Overall: {'STABLE' if stability else 'UNSTABLE'}\n")
        
        f.write(f"\n\nFor sawbench Material class:\n")
        f.write("Material(\n")
        f.write("    formula='V-Ti_alloy',  # Based on literature comparison\n")
        if 'C11' in props:
            f.write(f"    C11={props['C11']*1e9:.0f},  # Pa\n")
        if 'C12' in props:
            f.write(f"    C12={props['C12']*1e9:.0f},  # Pa\n")
        if 'C44' in props:
            f.write(f"    C44={props['C44']*1e9:.0f},  # Pa\n")
        f.write("    density=your_measured_density,  # kg/m³\n")
        f.write(f"    crystal_class='{elastic_tensor.crystal_system}'\n")
        f.write(")\n")
    
    print(f"✅ Results saved to: {output_file}")
    
    print("\n7. Summary:")
    print("-" * 11)
    print("• Used proper unstrained reference OUTCAR")
    print("• Auto-detected strain magnitude from lattice parameters")
    print("• Calculated full 6×6 elastic tensor")
    print("• Verified mechanical stability")
    print("• Results ready for sawbench Material class")
    
    print("\n" + "="*70)
    print("Elastic tensor calculation complete with unstrained reference!")
    print("="*70)

if __name__ == "__main__":
    main() 