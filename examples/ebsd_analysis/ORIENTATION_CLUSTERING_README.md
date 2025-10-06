# Orientation Clustering for Sensitivity Analysis

## Overview

Based on your symmetry test results, we've implemented **symmetry-aware clustering** to identify the most prevalent grain orientations in your BCC Vanadium sample. This accounts for the fact that crystallographically equivalent orientations (related by cubic symmetry operations) give the same SAW velocities.

## Key Findings from Symmetry Test

✅ **All 24 symmetrically equivalent orientations give identical SAW velocities** (velocity spread: 0.00 m/s)  
✅ **This means we MUST account for symmetry when clustering** to avoid counting the same orientation 24 times  
✅ **Different crystallographic families give different velocities:**
- (100) family: 2532.56 m/s
- (110) family: 2545.45 m/s  
- (111) family: 2536.23 m/s

## How It Works

### 1. **Symmetry-Aware Distance Metric**
The clustering algorithm calculates the **minimum misorientation angle** between any two grain orientations, considering all 24 cubic symmetry operations:

```
misorientation(grain1, grain2) = min over all 24 symmetry ops (
    angle between (symmetry_op × grain1) and grain2
)
```

### 2. **DBSCAN Clustering**
- Groups grains within `clustering_threshold_deg` (default: 5°) of each other
- Weights clusters by grain area (larger grains = more important)
- Returns top N most prevalent orientations

### 3. **Output**
For each top orientation, you get:
- **Mean Euler angles** (both radians and degrees)
- **Number of grains** in the cluster
- **Total area** represented
- **Fraction of sample** (by count and by area)
- **Orientation spread** (how tightly clustered)

## Usage

### Basic Usage

```python
from sawbench import Material, interactive_grain_analysis

material = Material(
    formula='V',
    C11=229e9,
    C12=119e9,
    C44=43e9,
    density=6110,
    crystal_class='cubic'
)

grains_all, grains_roi, metadata = interactive_grain_analysis(
    ctf_path='path/to/your/file.ctf',
    material=material,
    wavelength=8.8e-6,
    find_top_n_orientations=5,  # Find top 5 orientations
    clustering_threshold_deg=5.0  # 5° clustering threshold
)

# Access results
top_orientations = metadata['top_orientations']
print(top_orientations)
```

### Accessing Top Orientations

```python
# The top_orientations DataFrame contains:
# - Rank: 1, 2, 3, ...
# - N_Grains: Number of grains in cluster
# - Total_Area_um2: Total area in μm²
# - Fraction_by_count: Fraction of total grains
# - Fraction_by_area: Fraction of total area
# - Mean_phi1_deg, Mean_Phi_deg, Mean_phi2_deg: Euler angles (degrees)
# - Mean_phi1_rad, Mean_Phi_rad, Mean_phi2_rad: Euler angles (radians)
# - Spread_deg: Average spread within cluster
# - Max_Spread_deg: Maximum spread within cluster

# Use these orientations for sensitivity analysis:
for idx, row in top_orientations.iterrows():
    euler_rad = (row['Mean_phi1_rad'], row['Mean_Phi_rad'], row['Mean_phi2_rad'])
    # Run SAW calculator with these angles
    # Vary material parameters to see effect on frequencies
```

## Output Files

When `output_dir` is specified, the following files are saved:

1. **`top_orientations.csv`**: Top N orientations with all statistics
2. **`grain_cluster_assignments.csv`**: Which cluster each grain belongs to
3. **`euler_angles_roi.csv`**: All Euler angles for ROI grains
4. **Standard outputs**: SAW frequency analysis plots, grain data CSVs, etc.

## Parameters

### `find_top_n_orientations`
- **Default**: `None` (no clustering)
- **Recommended**: `5` for initial analysis
- **Range**: 1-20 (more than 20 may not be meaningful)

### `clustering_threshold_deg`
- **Default**: `5.0°`
- **Recommended**: 3-7° for most EBSD data
- **Too small** (< 2°): May over-segment into too many clusters
- **Too large** (> 10°): May group distinct orientations together

## For Sensitivity Analysis

### Workflow

1. **Run clustering** to identify top 5 orientations
2. **Extract mean Euler angles** for each orientation
3. **For each orientation:**
   - Vary C11, C12, C44 by ±5% (or your uncertainty range)
   - Calculate SAW velocity for each perturbation
   - Compare to experimental measurements
4. **Identify which orientations** are most sensitive to parameter errors

### Example Sensitivity Script

```python
from sawbench import Material, SAWCalculator
import numpy as np

# Get top orientation from clustering
euler_rad = (1.5708, 1.5708, 0.0)  # Example: (90°, 90°, 0°)

# Nominal material parameters
C11_nominal = 229e9
C12_nominal = 119e9
C44_nominal = 43e9

# Vary parameters by ±5%
perturbations = np.linspace(-0.05, 0.05, 11)
velocities = []

for delta in perturbations:
    material = Material(
        formula='V',
        C11=C11_nominal * (1 + delta),
        C12=C12_nominal * (1 + delta),
        C44=C44_nominal * (1 + delta),
        density=6110,
        crystal_class='cubic'
    )
    
    calc = SAWCalculator(material, np.array(euler_rad))
    v, _, _ = calc.get_saw_speed(deg=135.0, sampling=4000, psaw=0)
    velocities.append(v[0])

# Analyze sensitivity
print(f"Velocity range: {min(velocities)} - {max(velocities)} m/s")
print(f"Sensitivity: {(max(velocities) - min(velocities)) / (2 * 0.05)} m/s per 5% change")
```

## Tips

1. **Start with default parameters** (5 orientations, 5° threshold)
2. **If you get too few clusters**: Increase `clustering_threshold_deg`
3. **If you get too many clusters**: Decrease `clustering_threshold_deg`
4. **For large datasets** (>500 grains): Clustering may take several minutes
5. **Check cluster spread**: If `Max_Spread_deg > 10°`, consider lowering threshold

## Troubleshooting

### "No clusters found"
- Try increasing `clustering_threshold_deg` to 7-10°
- Check that you have enough grains in ROI (need > `min_samples`)

### "Clustering takes too long"
- The algorithm is O(N² × 24) where N is number of grains
- For >1000 grains, consider analyzing a smaller ROI
- Or be patient - it's a one-time calculation!

### "All grains in one cluster"
- Decrease `clustering_threshold_deg` to 2-3°
- Your sample may be highly textured (which is useful to know!)

## References

- Bunge Euler angles: Z-X-Z convention (standard in EBSD)
- Cubic symmetry: 24 operations (Oh point group)
- Misorientation: Minimum angle between symmetry-equivalent variants

