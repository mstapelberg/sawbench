#!/usr/bin/env python3
"""
Example: Elastic Tensor Calculation with sawbench
=================================================

This example shows how to calculate elastic tensors using the integrated
sawbench.elastic module, which uses ASE's built-in VASP reading capabilities.
"""

from sawbench import from_vasp_dir, ElasticTensor
import numpy as np
from pathlib import Path

def main():
    """
    Example of calculating elastic tensor from VASP calculations.
    """
    
    print("Sawbench Elastic Tensor Calculation Example")
    print("="*50)
    
    # Example 1: Check for real VASP data first
    print("\n1. Using Real VASP Data (if available):")
    print("-" * 40)
    
    # Check if we have the real VASP elastic data
    real_data_path = Path('examples/data/PBE_Manual_Elastic')
    
    if real_data_path.exists():
        print(f"Found real VASP elastic data at: {real_data_path}")
        
        try:
            # For this dataset, we'll use ep1_plus as reference since we don't have 
            # a separate unstrained calculation
            reference_outcar = real_data_path / 'ep1_plus' / 'OUTCAR'
            
            # Calculate elastic tensor from real VASP data
            elastic_tensor = from_vasp_dir(
                base_path=real_data_path,
                strain_magnitude=0.005,  # Assuming 0.5% strain was used
                reference_outcar=str(reference_outcar)
            )
            
            # Print the results
            print("\n" + "="*60)
            print("REAL VASP ELASTIC TENSOR RESULTS")
            print("="*60)
            print(elastic_tensor)
            
            # Access individual properties
            props = elastic_tensor.get_elastic_properties()
            
            print(f"\nKey Elastic Properties from Your VASP Data:")
            if 'bulk_modulus' in props:
                print(f"Bulk modulus: {props['bulk_modulus']:.2f} GPa")
            if 'shear_modulus' in props:
                print(f"Shear modulus: {props['shear_modulus']:.2f} GPa")
            if 'youngs_modulus' in props:
                print(f"Young's modulus: {props['youngs_modulus']:.2f} GPa")
            if 'poisson_ratio' in props:
                print(f"Poisson's ratio: {props['poisson_ratio']:.3f}")
            
            # Access the raw tensor
            print(f"\nElastic Tensor Shape: {elastic_tensor.tensor.shape}")
            print(f"C11 = {elastic_tensor.tensor[0,0]:.2f} GPa")
            
            if elastic_tensor.crystal_system == "cubic":
                print(f"C12 = {elastic_tensor.tensor[0,1]:.2f} GPa")
                print(f"C44 = {elastic_tensor.tensor[3,3]:.2f} GPa")
            
            # Save results for your material
            print(f"\nSaving results to 'vasp_elastic_results.txt'...")
            with open('vasp_elastic_results.txt', 'w') as f:
                f.write(str(elastic_tensor))
                f.write(f"\n\nFor use with sawbench Material class:\n")
                f.write(f"Material(\n")
                f.write(f"    formula='YourMaterial',\n")
                if 'C11' in props:
                    f.write(f"    C11={props['C11']*1e9:.0f},  # Pa\n")
                if 'C12' in props:
                    f.write(f"    C12={props['C12']*1e9:.0f},  # Pa\n")
                if 'C44' in props:
                    f.write(f"    C44={props['C44']*1e9:.0f},  # Pa\n")
                f.write(f"    density=your_density,  # kg/m³\n")
                f.write(f"    crystal_class='{elastic_tensor.crystal_system}'\n")
                f.write(f")\n")
            
            print("Results saved!")
            
        except Exception as e:
            print(f"Error processing VASP data: {e}")
            print("This might be due to missing reference calculation or other issues.")
    
    else:
        print(f"Real VASP data not found at: {real_data_path}")
        print("Running basic demo instead...")
        
        try:
            # Fallback to basic usage
            elastic_tensor = from_vasp_dir(
                base_path='.',           # Current directory
                strain_magnitude=0.01    # 1% strain used in VASP calculations
            )
            
            print(elastic_tensor)
            
        except FileNotFoundError as e:
            print(f"Expected error (no OUTCAR files): {e}")
            print("This is normal if you don't have actual VASP calculations in this directory.")
    
    
    # Example 2: Custom path and parameters
    print("\n\n2. Custom Configuration:")
    print("-" * 25)
    
    custom_path = "/path/to/vasp/calculations"
    print(f"For calculations in: {custom_path}")
    print("You would use:")
    print(f"""
    elastic_tensor = from_vasp_dir(
        base_path='{custom_path}',
        strain_magnitude=0.005,  # 0.5% strain
        reference_outcar='/custom/path/to/reference/OUTCAR'
    )
    """)
    
    
    # Example 3: Working with elastic tensor directly
    print("\n3. Working with ElasticTensor objects:")
    print("-" * 35)
    
    # Create a sample elastic tensor (Silicon-like values for demonstration)
    sample_tensor = np.array([
        [165.8, 63.9, 63.9,  0.0,  0.0,  0.0],  # C1j
        [ 63.9,165.8, 63.9,  0.0,  0.0,  0.0],  # C2j  
        [ 63.9, 63.9,165.8,  0.0,  0.0,  0.0],  # C3j
        [  0.0,  0.0,  0.0, 79.6,  0.0,  0.0],  # C4j
        [  0.0,  0.0,  0.0,  0.0, 79.6,  0.0],  # C5j
        [  0.0,  0.0,  0.0,  0.0,  0.0, 79.6]   # C6j
    ])
    
    # Create ElasticTensor object
    demo_tensor = ElasticTensor(sample_tensor)
    
    print("Sample Silicon-like elastic tensor:")
    print(demo_tensor)
    
    
    # Example 4: Integration with SAW calculations
    print("\n4. Integration with SAW Calculations:")
    print("-" * 35)
    
    print("""
    # You can use the calculated elastic tensor with sawbench materials
    from sawbench import Material
    
    # Extract elastic constants
    props = elastic_tensor.get_elastic_properties()
    
    # Create material for SAW calculations
    material = Material(
        formula='MyMaterial',
        C11=props['C11'] * 1e9,      # Convert GPa to Pa
        C12=props['C12'] * 1e9,
        C44=props['C44'] * 1e9,
        density=your_density,         # kg/m³
        crystal_class='cubic'
    )
    
    # Now use with SAW calculator...
    """)
    
    
    print("\n5. Required Directory Structure:")
    print("-" * 30)
    print("""
    your_calculations/
    ├── OUTCAR                    # Reference (unstrained)
    ├── ep1_plus/OUTCAR          # +εxx strain
    ├── ep1_minus/OUTCAR         # -εxx strain
    ├── ep2_plus/OUTCAR          # +εyy strain
    ├── ep2_minus/OUTCAR         # -εyy strain
    ├── ep3_plus/OUTCAR          # +εzz strain
    ├── ep3_minus/OUTCAR         # -εzz strain
    ├── ep4_plus/OUTCAR          # +γyz/2 shear
    ├── ep4_minus/OUTCAR         # -γyz/2 shear
    ├── ep5_plus/OUTCAR          # +γxz/2 shear
    ├── ep5_minus/OUTCAR         # -γxz/2 shear
    ├── ep6_plus/OUTCAR          # +γxy/2 shear
    └── ep6_minus/OUTCAR         # -γxy/2 shear
    """)


if __name__ == "__main__":
    main() 