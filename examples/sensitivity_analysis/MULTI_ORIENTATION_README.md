# Multi-Orientation Sensitivity Analysis

## Overview

This workflow performs **global variance-based sensitivity analysis** (Sobol' indices) to quantify how **EBSD measurement errors** affect SAW frequency predictions across different grain orientations.

### Key Questions Answered:
1. **Which parameters** (elastic constants, density, Euler angles, in-plane angle) most affect model-experiment agreement?
2. **How does sensitivity vary** across different grain orientations?
3. **How does sensitivity change** with the magnitude of EBSD measurement error (±1°, ±2.5°, ±5°)?

### Key Improvements Over Original Analysis:
- ✅ Uses **Bunge convention** (φ₁, Φ, φ₂) instead of (α, β, γ)
- ✅ Tests **28 representative grain orientations** (stratified by SAW frequency)
- ✅ Evaluates **multiple EBSD error scales** (±1°, ±2.5°, ±5°)
- ✅ Shows sensitivity **variation across orientations**
- ✅ More realistic: orientations come from **actual EBSD data**

---

## Workflow

### Step 1: Extract Representative Orientations (Optional)

If you want to regenerate the orientation list from your EBSD data:

```bash
cd /home/myless/Packages/sawbench/examples/sensitivity_analysis
python extract_representative_orientations.py
```

**What it does:**
- Loads grain data from `../ebsd_analysis/results/interactive_analysis_output_4000/grains_roi.csv`
- Creates ~29 frequency bins (stratified sampling)
- Selects one representative grain from each bin
- Outputs: `representative_grain_orientations.csv`

**Output:** A list of orientations in radians, ready to copy into the sensitivity script.

---

### Step 2: Run Multi-Orientation Sensitivity Analysis

```bash
python global_variance_sensitivity_multi_orientation.py
```

**Configuration (edit in script if needed):**
```python
# Number of representative orientations: 28 (pre-loaded)
EULER_ERROR_SCALES_DEG = [1.0, 2.5, 5.0]  # EBSD error magnitudes to test
N_SOBOL_SAMPLES = 512  # Increase to 1024-2048 for publication
NUM_WORKERS = 32  # Parallel workers
```

**What it does:**
- For each orientation and each error scale:
  - Generates Sobol samples in 8D parameter space:
    - K (bulk modulus), D (C₁₁-C₁₂), G (C₄₄), ρ (density)
    - φ₁_error, Φ_error, φ₂_error (EBSD fitting errors)
    - ψ_error (in-plane angle error)
  - Computes SAW frequencies for 9 measured angles
  - Calculates Wasserstein distance vs experiment
  - Analyzes Sobol sensitivity indices (S₁, S_T)

**Runtime:** ~1-4 hours depending on `N_SOBOL_SAMPLES` and CPU cores

**Output files:**
```
results/multi_orientation_sensitivity/
├── all_results.json              # Detailed results for all orientations/scales
├── summary_error_1.0deg.csv      # Average sensitivity at ±1°
├── summary_error_2.5deg.csv      # Average sensitivity at ±2.5°
└── summary_error_5.0deg.csv      # Average sensitivity at ±5°
```

---

### Step 3: Visualize Results

```bash
python plot_multi_orientation_sensitivity.py
```

**Generates 4 publication-quality figures:**

#### Figure 1: Average Sensitivity by Error Scale
- Side-by-side comparison of ±1°, ±2.5°, ±5° error scales
- Shows which parameters dominate at each scale
- Error bars show variation across orientations

#### Figure 2: Orientation Variation
- For each parameter, shows how S₁ and S_T vary across the 28 orientations
- Identifies if sensitivity is orientation-dependent
- Uses middle error scale (±2.5°)

#### Figure 3: Sensitivity vs Error Scale
- For each parameter, shows how sensitivity changes from ±1° to ±5°
- Reveals nonlinearity: does 5× larger error → 5× more sensitivity?
- Helps identify which parameters matter at which error scales

#### Figure 4: Heatmaps
- Parameters × Orientations heatmaps for S₁ and S_T
- Quickly spot orientation-dependent sensitivities
- One heatmap per error scale

**Output:** All figures saved to `results/multi_orientation_sensitivity/plots/`

---

## Understanding Sobol Indices

### S₁ (First-Order Index)
**Definition:** Fraction of output variance caused by varying parameter X_i alone.

**Interpretation:**
- S₁ = 0.6 → 60% of variance in Wasserstein distance comes from varying this parameter
- Measures **direct effect only** (no interactions)

### S_T (Total-Order Index)
**Definition:** Fraction of output variance involving parameter X_i (direct + all interactions).

**Interpretation:**
- S_T = 0.8 → 80% of variance involves this parameter (alone or with others)
- Measures **total contribution**
- Always: S₁ ≤ S_T

### Interaction Effects
**S_T - S₁** = contribution from parameter interactions

**Examples:**
- S₁ = 0.5, S_T = 0.5 → No interactions, purely additive effect
- S₁ = 0.3, S_T = 0.7 → Strong interactions with other parameters
- S₁ = 0.0, S_T = 0.4 → No direct effect, but important in combinations

---

## Expected Results & Interpretation

### Typical Findings:

**High Sensitivity (S_T > 0.5):**
- **D (anisotropy)**: Controls Zener ratio, strongly affects SAW speed anisotropy
- **K or G**: Elastic constants directly control wave velocity

**Moderate Sensitivity (0.1 < S_T < 0.5):**
- **ρ (density)**: Affects velocity through v = √(C/ρ)
- **ψ_error**: In-plane angle uncertainty

**Low Sensitivity (S_T < 0.1):**
- **φ₁, Φ, φ₂ errors**: Small Euler angle errors (±1-5°) may have minimal effect
  - **Why?** Small rotations around certain orientations don't change elastic properties much
  - **When they matter:** Near high-symmetry orientations or for specific propagation directions

### Orientation Dependence:

If sensitivity **varies strongly** across orientations:
- Your model predictions are **orientation-sensitive**
- EBSD errors matter more for some grains than others
- Consider weighting grains by their sensitivity in your analysis

If sensitivity is **similar** across orientations:
- Your conclusions are **robust** to grain selection
- Average sensitivity is representative

### Error Scale Dependence:

**Linear model:** S₁ and S_T should be similar across ±1°, ±2.5°, ±5°

**Nonlinear model:** Sensitivity changes with error magnitude
- If S_T(φ₁) increases dramatically from ±1° to ±5°: nonlinear response to orientation
- If S_T(D) stays constant: linear response to anisotropy changes

---

## Computational Cost

### Estimated evaluations:
```
N_orientations = 28
N_error_scales = 3
N_samples = 512
N_psi_angles = 9

Total SAW calculations ≈ 28 × 3 × 512 × 2 × 9 ≈ 774,144
```

With 32 cores @ ~0.01 sec/calculation → **2-4 hours**

### Tips to speed up:
1. **Reduce N_SOBOL_SAMPLES** to 256 for exploratory runs
2. **Reduce orientations** to 10-15 (use every 2nd or 3rd)
3. **Reduce error scales** to just [2.5] initially
4. **Lower SAW_SAMPLING** from 4000 to 400 (less accurate but 10× faster)

---

## Comparison with Original `global_variance_sensitivity.py`

| Aspect | Original | New Multi-Orientation |
|--------|----------|----------------------|
| **Baseline orientation** | Single: (0°, 0°, 0°) | 28 representative grains |
| **Euler convention** | (α, β, γ) unclear | Bunge (φ₁, Φ, φ₂) |
| **Error scale** | Fixed ±1° | Variable: ±1°, ±2.5°, ±5° |
| **Orientation coverage** | One point | Stratified by SAW frequency |
| **Output** | Single sensitivity table | Per-orientation + aggregated |
| **Visualization** | Manual | 4 automated figures |
| **Physical insight** | Limited | Orientation & scale dependence |

---

## Customization

### Change EBSD Data Source
Edit in `global_variance_sensitivity_multi_orientation.py`:
```python
REPRESENTATIVE_ORIENTATIONS_RAD = [
    # Replace with your orientations
    (phi1_rad, Phi_rad, phi2_rad),
    ...
]
```

### Change Material Properties
```python
C11_0, C12_0, C44_0 = 231.3, 117.5, 51.7  # GPa (Vanadium)
RHO_0 = 6100.0  # kg/m³
```

### Change Parameter Bounds
```python
"bounds": [
    [0.95*K0, 1.05*K0],      # Tighten/loosen elastic constant ranges
    ...
    [-10.0, 10.0],           # Test larger Euler angle errors
]
```

### Change Error Scales
```python
EULER_ERROR_SCALES_DEG = [0.5, 1.0, 2.0, 5.0, 10.0]  # More fine-grained
```

---

## Troubleshooting

### "No experimental data loaded"
- Check `EXPERIMENTAL_HDF5_PATH` points to your FFT data file
- Sensitivity will still run but may be uninformative without experiment

### "Only X/Y valid samples"
- Some parameter combinations violate cubic stability criteria
- If < 50% valid: expand elastic constant bounds or check material definition

### Very low Euler angle sensitivity
- Expected! Small rotations (±1-5°) around your baseline may not change elastic properties much
- Try larger error scales (±10-15°) or check if baseline is near a high-symmetry orientation

### Runtime too long
- Reduce `N_SOBOL_SAMPLES` to 256
- Reduce number of orientations to 10
- Lower `SAW_SAMPLING` to 400

---

## Citation & References

**Sobol Sensitivity Analysis:**
- Saltelli, A., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Herman, J., & Usher, W. (2017). SALib: Sensitivity Analysis Library in Python. *JOSS*.

**Euler Angles (Bunge Convention):**
- Bunge, H. J. (1982). *Texture Analysis in Materials Science*. Butterworths.

**SAW Calculations:**
- Farnell, G. W., & Adler, E. L. (1972). Elastic wave propagation in thin layers. *Physical Acoustics*, Vol. 9.

---

## Contact

For questions about this sensitivity analysis workflow:
- Check main README: `../README.md`
- Original sensitivity scripts: `derivative_sensitivity.py`, `global_variance_sensitivity.py`
- EBSD analysis: `../ebsd_analysis/README.md`

