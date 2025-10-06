# Visualization Guide for Multi-Scale Sensitivity Analysis

## Your Questions Answered

### 1. How Does Option A Work?

**Your implementation already does Option A!** Here's what happens:

```python
for error_scale in [1.0, 2.5, 5.0]:  # 3 separate analyses
    for orientation in 28_orientations:
        run_sobol_analysis(error_scale, orientation)
    aggregate_results(error_scale)
```

**Result:** You get **3 separate sensitivity rankings**, one for each error scale.

**Key insight:** Parameters may switch importance! For example:
- At ±1°: D (anisotropy) dominates because small orientation errors don't matter much
- At ±5°: Euler angles become more important because larger rotations affect predictions

### 2. Visualization Recommendations

You mentioned your current plot is a **simple horizontal bar chart**. Here are 3 options for visualizing 3 error scales:

---

#### **Option 1: Side-by-Side Panels (RECOMMENDED)**

**Script:** `plot_simple_comparison.py`

```
┌────────────────────┬────────────────────┬────────────────────┐
│    ±1° Error       │    ±2.5° Error     │    ±5° Error       │
├────────────────────┼────────────────────┼────────────────────┤
│ D ████████ 0.65    │ D ████████ 0.68    │ K ████████ 0.72    │
│ K ███████ 0.58     │ K ████████ 0.70    │ D ████████ 0.70    │
│ G █████ 0.42       │ G ██████ 0.48      │ φ₁ ██████ 0.55     │
│ ψ ████ 0.35        │ ψ █████ 0.40       │ G ██████ 0.52      │
│ ρ ███ 0.22         │ φ₁ ████ 0.32       │ ψ █████ 0.45       │
│ φ₁ ██ 0.15         │ ρ ███ 0.25         │ ρ ███ 0.28         │
│ Φ █ 0.08           │ Φ ██ 0.18          │ Φ ████ 0.35        │
│ φ₂ █ 0.06          │ φ₂ ██ 0.15         │ φ₂ ███ 0.30        │
└────────────────────┴────────────────────┴────────────────────┘
```

**Advantages:**
- ✅ Easy to compare rankings across scales
- ✅ Shows if parameters switch importance (e.g., D drops from #1 to #2)
- ✅ Clean, publication-ready
- ✅ Color-coded by parameter type (elastic=blue, EBSD=orange, sample=magenta)

**How to generate:**
```bash
python plot_simple_comparison.py
```

**Output:** `results/multi_orientation_sensitivity/plots/simple_comparison.png` (and .pdf)

---

#### **Option 2: Single Panel with Error Scale as Color Intensity**

```
Parameter      ├──────────────────────────────────┤
D              ████████████ (darker = ±5°)
               ██████████ (medium = ±2.5°)  
               ████████ (lighter = ±1°)
K              ███████████
               █████████
               ███████
...
```

**Advantages:**
- ✅ Compact
- ✅ Shows trend: does sensitivity increase/decrease with error scale?

**Disadvantages:**
- ⚠️ Can be hard to read with 3+ scales
- ⚠️ Loses exact numerical comparison

---

#### **Option 3: Line Plot (Sensitivity vs Error Scale)**

**Script:** Already in `plot_multi_orientation_sensitivity.py` (Figure 3)

```
Sensitivity │     ╱── K
Index       │   ╱╱
            │ ╱╱ D
            │╱ ──── G
            └─────────────
             1°  2.5°  5°
```

**Advantages:**
- ✅ Shows trend clearly
- ✅ Good for presentations (tells a story)

**Disadvantages:**
- ⚠️ Harder to compare many parameters at once

---

### 3. Sample Angle (ψ) - The Missing Parameter!

You're absolutely right to call this out! **ψ_err_deg** is critical but often overlooked.

#### What Each Error Type Means:

| Parameter | Physical Meaning | Error Type | Expected Sensitivity |
|-----------|------------------|------------|---------------------|
| **K, D, G** | Elastic constants | Material uncertainty | HIGH (directly affects wave speed) |
| **ρ** | Density | Material uncertainty | MODERATE (v ∝ √(C/ρ)) |
| **φ₁, Φ, φ₂** | Grain orientation | EBSD fitting error (per-grain, random) | LOW-MODERATE (depends on error scale) |
| **ψ** | Sample rotation | EBSD-TGS alignment error (systematic) | MODERATE-HIGH (affects all grains) |

#### Why ψ May Surprise You:

**ψ_err has only ±0.5° range** (vs ±1-5° for Euler angles), BUT:

1. **Systematic error:** All grains rotated together → doesn't average out
2. **Changes SAW propagation direction** relative to all grains simultaneously
3. **May show high S_T even with small bounds** because it's correlated across measurements

**Expected result:** ψ_err sensitivity should be **comparable to or higher than** individual Euler angle errors.

---

## Recommended Workflow

### Step 1: Run Analysis (2-4 hours)
```bash
python global_variance_sensitivity_multi_orientation.py
```

### Step 2: Generate Simple Comparison Plot
```bash
python plot_simple_comparison.py
```

**This gives you Figure 1:** Side-by-side bar charts for ±1°, ±2.5°, ±5°

### Step 3: Generate Full Analysis Suite
```bash
python plot_multi_orientation_sensitivity.py
```

**This gives you Figures 1-4:**
1. Average sensitivity comparison (similar to simple plot but with more details)
2. Sensitivity variation across orientations
3. Sensitivity vs error scale (line plots)
4. Heatmaps (parameters × orientations)

### Step 4: Interpret Results

**Look for:**

1. **Rank changes across error scales**
   - Does D dominate at all scales?
   - Do Euler angles become important at ±5°?
   - Does ψ have consistent importance?

2. **Interaction effects (S_T - S_1)**
   - Large difference → parameter important through interactions
   - Small difference → purely additive effect

3. **Orientation dependence (Figure 2 & 4)**
   - High variance across orientations → sensitivity depends on which grain you're in
   - Low variance → robust conclusions

---

## Color Coding in Plots

**Semantic colors group parameters by physical meaning:**

| Color | Parameter Type | Examples |
|-------|---------------|----------|
| 🔵 **Blue** | Elastic constants | K, D, G |
| 💚 **Teal** | Density | ρ |
| 🟠 **Orange** | EBSD orientation errors | φ₁, Φ, φ₂ |
| 💜 **Magenta** | Sample alignment error | ψ |

**Why this helps:**
- Quickly see if material properties (blue/teal) or measurement errors (orange/magenta) dominate
- Distinguishes **per-grain random errors** (EBSD) from **systematic errors** (sample alignment)

---

## Expected Results & Physical Interpretation

### Scenario 1: Elastic Constants Dominate
```
±1°:  D > K > G >> ψ > φ₁,Φ,φ₂
±5°:  D > K > G > ψ > φ₁,Φ,φ₂
```

**Interpretation:** Material uncertainty is your main issue, not measurement errors.  
**Action:** Better constrain elastic constants (e.g., from DFT, literature, or elastic tensor measurements).

---

### Scenario 2: Sample Alignment Matters
```
±1°:  D > ψ > K > G > φ₁,Φ,φ₂
±5°:  D > ψ > K > φ₁ > G > Φ,φ₂
```

**Interpretation:** EBSD-TGS alignment error dominates over individual grain orientation errors.  
**Action:** Improve fiducial markers, use cross-correlation alignment, or optimize sample mounting.

---

### Scenario 3: EBSD Errors Become Important at Large Scales
```
±1°:  D > K > G > ψ > φ₁,Φ,φ₂  (Euler angles low)
±5°:  D > φ₁ > K > Φ > G > ψ   (Euler angles jump up!)
```

**Interpretation:** Nonlinear sensitivity to orientation. Large EBSD fitting errors significantly degrade predictions.  
**Action:** Improve EBSD indexing quality, use higher-resolution scans, or apply orientation corrections.

---

### Scenario 4: Orientation-Dependent Sensitivity (Figure 4 shows stripes)
```
Heatmap shows: Some grains have high φ₁ sensitivity, others low
```

**Interpretation:** Sensitivity depends on where you are in orientation space.  
**Action:** Weight predictions by local sensitivity, or focus analysis on low-sensitivity orientations.

---

## Publication-Ready Figure Caption Template

### For Side-by-Side Comparison (Figure 1):

> **Figure X.** Global sensitivity analysis (Sobol' indices) showing the contribution of model parameters to prediction uncertainty at three EBSD measurement error scales. Bars show total-order sensitivity index (S_T, gray outline) and first-order index (S_1, filled), averaged across 28 representative grain orientations. Error bars represent standard deviation across orientations. Parameters are ranked by S_T (descending). Colors indicate parameter type: elastic constants (blue), density (teal), EBSD orientation errors (orange), sample alignment error (magenta). At small error scales (±1°), elastic constants dominate; at larger scales (±5°), measurement errors become increasingly important.

---

## Quick Troubleshooting

**Q: All Euler angle sensitivities are < 0.1, even at ±5°**  
**A:** This is possible! Small rotations around your specific baseline orientations may not significantly affect SAW propagation. Your sample may be in a "low-sensitivity" region of orientation space.

**Q: ψ_err has higher sensitivity than expected**  
**A:** This is actually expected for a systematic error. Even ±0.5° applied to all grains can shift predictions more than ±5° random errors that average out.

**Q: Rankings don't change much across error scales**  
**A:** Indicates linear model response. Good for robustness, but less information about nonlinearity.

**Q: Huge variation across orientations (error bars are large)**  
**A:** Some grains are more sensitive than others. Consider per-orientation analysis or stratified reporting.

---

## Any More Questions?

Ready to run the analysis! Let me know if you need help with:
- Adjusting computational parameters (speed vs accuracy tradeoff)
- Interpreting specific results
- Creating custom visualizations
- Publication figure formatting

