# Elastic Tensor Integration in Sawbench

## Overview

The elastic tensor functionality has been fully integrated into the sawbench package. This uses ASE's built-in VASP OUTCAR reading capabilities instead of manual parsing, making it more robust and reliable.

## Key Features

- **ASE Integration**: Uses `ase.io.read()` and `atoms.get_stress()` for reading VASP OUTCAR files
- **Sawbench Integration**: Part of the main sawbench package (`src/sawbench/elastic.py`)
- **Simple API**: Just use `from_vasp_dir()` function for most use cases
- **Automatic Crystal System Detection**: Detects cubic vs. general symmetry
- **Complete Elastic Properties**: Calculates bulk modulus, shear modulus, Young's modulus, Poisson's ratio

## Quick Usage

```python
from sawbench import from_vasp_dir

# Calculate elastic tensor from VASP finite difference calculations
elastic_tensor = from_vasp_dir(
    base_path='/path/to/vasp/calculations',
    strain_magnitude=0.01  # 1% strain used in VASP
)

# Print results
print(elastic_tensor)

# Get properties
props = elastic_tensor.get_elastic_properties()
print(f"Bulk modulus: {props['bulk_modulus']:.2f} GPa")
```

## Required Directory Structure

```
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
```

## Integration with SAW Calculations

The calculated elastic constants can be directly used with sawbench materials:

```python
from sawbench import Material, from_vasp_dir

# Calculate elastic tensor
elastic_tensor = from_vasp_dir('/path/to/calculations')
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

# Use with SAW calculator...
```

## Benefits of ASE Integration

1. **Robust Parsing**: ASE handles various VASP OUTCAR formats automatically
2. **Unit Conversion**: Automatic conversion from VASP units (eV/Å³) to GPa
3. **Error Handling**: Better error messages and debugging
4. **Consistency**: Uses the same I/O framework as the rest of sawbench
5. **Maintainability**: Less custom parsing code to maintain

## Files

- `src/sawbench/elastic.py` - Main elastic tensor module
- `examples/elastic_tensor_example.py` - Example usage script
- `examples/ELASTIC_TENSOR_README.md` - Detailed documentation (existing)

## What You Need

Besides the OUTCAR files from strained VASP calculations, you need:

1. **Strain magnitude** - The amount of strain applied (typically 0.5-2%)
2. **Reference calculation** - An unstrained OUTCAR for comparison
3. **Proper VASP setup** - Higher ENCUT, tight convergence, ISIF=2
