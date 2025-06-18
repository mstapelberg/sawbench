# Installation Guide

## Basic Installation

The core `sawbench` package has minimal dependencies and can be installed without any specific MLIP:

```bash
# Clone the repository
git clone https://github.com/mstapelberg/sawbench.git
cd sawbench

# Install core package
pip install -e .
```

## MLIP-Specific Installations

Since MACE and NequIP have conflicting dependencies (e3nn versions), you need to choose one or use separate environments.

### Option 1: Install with MACE

```bash
# Create environment
mamba create -n sawbench-mace python=3.11
mamba activate sawbench-mace

# Install sawbench with MACE support
pip install -e .[mace,mlip-examples]
```

### Option 2: Install with NequIP/Allegro

```bash
# Create environment  
mamba create -n sawbench-nequip python=3.11
mamba activate sawbench-nequip

# Install sawbench with NequIP support
pip install -e .[nequip,mlip-examples]
```

### Option 3: Examples Only (No MLIP)

If you just want to run examples with your own calculators:

```bash
# Install with example dependencies
pip install -e .[mlip-examples]
pip install -r requirements.txt
```

## Dependency Conflict Resolution

**Problem**: MACE requires `e3nn==0.4.4` while NequIP requires `e3nn>=0.5.6`

**Solution**: Use separate conda/mamba environments for each MLIP.

## Verification

Test your installation:

```python
# Test core functionality
from sawbench import Material, SAWCalculator, from_vasp_dir

# Test elastic tensor calculation (requires ASE)
from sawbench import calculate_elastic_tensor

# Test VASP elastic tensor reading
elastic_tensor = from_vasp_dir('path/to/vasp/calculations')
print(elastic_tensor)
```

## Development Installation

For development with all optional dependencies:

```bash
pip install -e .[dev,mlip-examples]
```

## Troubleshooting

1. **Import errors**: Make sure you're in the correct environment
2. **MLIP conflicts**: Use separate environments for MACE vs NequIP
3. **Missing ASE**: Install with `pip install ase>=3.22`
4. **VASP reading issues**: Ensure OUTCAR files are accessible

## Quick Start

1. **For MACE users**: `pip install -e .[mace,mlip-examples]` 
2. **For NequIP users**: `pip install -e .[nequip,mlip-examples]` 
3. **For other MLIP users**: `pip install -e .[mlip-examples]`