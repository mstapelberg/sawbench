# Elastic Tensor Calculation from VASP OUTCAR Files

This document explains how to calculate elastic tensors using the finite difference method from VASP OUTCAR files.

## Overview

The elastic tensor relates stress to strain in a material: **σ = C ε**

Where:
- **σ** is the stress tensor (6×1 in Voigt notation)  
- **C** is the elastic stiffness tensor (6×6)
- **ε** is the strain tensor (6×1 in Voigt notation)

This script uses finite differences to calculate C<sub>ij</sub> by applying small strains and measuring the resulting stress changes.

## Required Information

To use this script, you need:

1. **OUTCAR files** from VASP calculations with applied strains
2. **Strain magnitude** used in your calculations (typically 0.5-2%)
3. **Reference structure** (unstrained calculation)

## Directory Structure

Your calculation directory should be organized as follows:

```
/your/calculation/path/
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
```

## Strain Components (Voigt Notation)

| Index | Strain Component | Description | POSCAR Deformation |
|-------|------------------|-------------|--------------------|
| 1     | εxx              | Normal strain in x | Scale lattice vector a |
| 2     | εyy              | Normal strain in y | Scale lattice vector b |  
| 3     | εzz              | Normal strain in z | Scale lattice vector c |
| 4     | γyz/2            | Shear strain yz | Mix b and c vectors |
| 5     | γxz/2            | Shear strain xz | Mix a and c vectors |
| 6     | γxy/2            | Shear strain xy | Mix a and b vectors |

## VASP Calculation Setup

### 1. Reference Calculation (Unstrained)
```bash
# INCAR for reference calculation
ISMEAR = 0         # Gaussian smearing for insulators
SIGMA = 0.05       # or appropriate for your system
ENCUT = 600        # Higher cutoff often needed for elastic constants
PREC = Accurate    # High precision
EDIFF = 1E-7       # Tight electronic convergence
IBRION = -1        # No ionic relaxation for elastic constants
NSW = 0            # No ionic steps
ISIF = 2           # Calculate stress tensor
```

### 2. Strained Calculations

For each strain, modify the lattice vectors in POSCAR:

**Normal strains (ε₁₁, ε₂₂, ε₃₃):**
- Scale the corresponding lattice vector by (1 ± δ)
- Example for +1% strain in x: multiply first lattice vector by 1.01

**Shear strains (γ₂₃, γ₁₃, γ₁₂):**
- Add shear deformation to lattice vectors
- Example for γ₁₂ shear: add δ×b to lattice vector a

### Example Python script to generate strained POSCARs:

```python
import numpy as np

def create_strained_poscar(original_cell, strain_type, strain_magnitude):
    """
    Create a strained unit cell.
    
    strain_type: 1-6 for the six strain components
    strain_magnitude: strain amount (e.g., 0.01 for 1%)
    """
    cell = original_cell.copy()
    
    if strain_type == 1:  # εxx
        cell[0] *= (1 + strain_magnitude)
    elif strain_type == 2:  # εyy  
        cell[1] *= (1 + strain_magnitude)
    elif strain_type == 3:  # εzz
        cell[2] *= (1 + strain_magnitude)
    elif strain_type == 4:  # γyz/2
        cell[1] += strain_magnitude * cell[2]
    elif strain_type == 5:  # γxz/2
        cell[0] += strain_magnitude * cell[2] 
    elif strain_type == 6:  # γxy/2
        cell[0] += strain_magnitude * cell[1]
    
    return cell
```

## Usage

### Command Line Interface:

```bash
# Basic usage (assumes correct directory structure)
python calculate_elastic_tensor.py

# With custom parameters
python calculate_elastic_tensor.py --base-path /path/to/calculations --strain 0.01

# With custom reference OUTCAR
python calculate_elastic_tensor.py --reference /path/to/reference/OUTCAR --output my_results.txt
```

### Python Script:

```python
from calculate_elastic_tensor import ElasticTensorCalculator

# Initialize calculator
calc = ElasticTensorCalculator(
    base_path='/path/to/calculations',
    strain_magnitude=0.01  # 1% strain
)

# Load data and calculate
calc.load_reference_calculation()
calc.load_strained_calculations()
elastic_tensor = calc.calculate_elastic_tensor()

# Get results
calc.print_results()
properties = calc.get_elastic_properties()
```

## Output

The script provides:

1. **6×6 Elastic stiffness tensor** (GPa)
2. **Elastic properties:**
   - Bulk modulus (K)
   - Shear modulus (G) - Voigt and Reuss bounds
   - Young's modulus (E)
   - Poisson's ratio (ν)
   - Individual elastic constants (C₁₁, C₁₂, C₄₄ for cubic systems)

3. **Symmetry analysis** - checks if tensor is properly symmetric
4. **Crystal system detection** - identifies if approximately cubic

## Important Notes

### Convergence Requirements:
- **ENCUT**: Often needs to be higher than for total energy calculations
- **K-points**: Dense enough for accurate stress calculations  
- **EDIFF**: Tight electronic convergence (1E-7 or better)

### Strain Magnitude:
- Too small (< 0.2%): Numerical noise dominates
- Too large (> 3%): Non-linear effects become important
- Recommended: 0.5-2% strain

### Common Issues:
1. **Unconverged calculations**: Check OUTCAR for warnings
2. **Non-symmetric tensor**: May indicate convergence issues
3. **Unrealistic values**: Check strain setup and VASP parameters
4. **Missing OUTCAR files**: Script will skip missing calculations

### For Different Crystal Systems:
- **Cubic**: 3 independent constants (C₁₁, C₁₂, C₄₄)
- **Hexagonal**: 5 independent constants  
- **Orthorhombic**: 9 independent constants
- **General**: All 21 independent constants

## Validation

Compare your results with:
- Experimental values from literature
- Other DFT studies using similar functionals
- Check that C₁₁ > |C₁₂| for mechanical stability
- Verify Born stability criteria for your crystal system

## References

1. Wallace, D. C. (1972). Thermodynamics of crystals. Wiley.
2. Nye, J. F. (1985). Physical properties of crystals. Oxford University Press.
3. VASP Manual: Elastic constants calculation 