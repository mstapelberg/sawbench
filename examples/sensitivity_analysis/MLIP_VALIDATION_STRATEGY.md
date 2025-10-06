# MLIP Validation Strategy via Sensitivity Analysis

## Your Goal
**Prove that model-experiment discrepancies are due to MLIP inaccuracy, not experimental uncertainty.**

This is a powerful use of sensitivity analysis: quantify the maximum prediction uncertainty from experimental errors, then show that remaining discrepancies must be due to the model (MLIP).

---

## Setup Summary

### Your System:
- **Material:** BCC (90° rotational symmetry in-plane)
- **Experiment:** SAW-TGS measuring frequency vs. in-plane angle
- **Model:** MLIP-predicted elastic tensor → SAWCalculator → predicted frequencies
- **EBSD:** Grain orientations with ±5° fitting uncertainty
- **Sample alignment:** ±20° uncertainty (conservative, ±10° by eye)

### Your Observations:
From derivative sensitivity:
- **Δψ effect:** ΔKS ≈ 0.04 when varying sample alignment
- **Baseline KS:** ≈ 0.5 (model-experiment mismatch)
- **Implication:** ψ alignment causes ~8% variation, not dominant

### Your Parameter Bounds:
```python
K:  [1.00*K0, 1.17*K0]       # +0/+17% (asymmetric, from DFT/lit?)
D:  [0.77*D0, 1.27*D0]       # ±23-27% (anisotropy, MLIP-sensitive)
G:  [0.30*G0, 1.00*G0]       # -70/0% (asymmetric, interesting!)
ρ:  [0.98*ρ0, 1.02*ρ0]       # ±2% (well-known)
φ₁,Φ,φ₂: ±1°, ±2.5°, ±5°    # EBSD error (scaled)
ψ:  ±20°                     # Sample alignment (conservative)
```

**Questions about your bounds:**
1. Why is G so asymmetric (only -70/0%, not +x%)?
2. Why is K only positive variation (+0/+17%)?
3. Are these from DFT calculations or literature constraints?

These asymmetries suggest you already have strong priors about the elastic constants!

---

## Expected Sensitivity Results

Based on your setup and derivative analysis:

### Prediction 1: Material Properties Will Dominate
**Most likely ranking:**
1. **D (anisotropy)**: High - MLIPs often struggle with (C₁₁-C₁₂), affects SAW anisotropy
2. **G (C₄₄)**: Moderate-High - wide bounds (-70%!), directly affects shear wave speeds
3. **K (bulk modulus)**: Moderate - narrower bounds, but still affects compressional components
4. **ψ (alignment)**: Moderate - ±20° range, ~8% KS effect from your derivative analysis
5. **φ₁,Φ,φ₂ (EBSD)**: Low-Moderate - random per-grain errors average out
6. **ρ (density)**: Low - tight ±2% bounds, well-constrained

### Prediction 2: Sensitivity Will Change with Error Scale
- **At ±1° EBSD error:** Material properties (D, G, K) dominate, Euler angles negligible
- **At ±5° EBSD error:** Material still dominant, but Euler angles become visible
- **ψ constant at ±20°:** Should show similar importance across all scales

### Prediction 3: BCC Symmetry Reduces ψ Impact
- 90° rotational symmetry means 20° error is at most 25° from nearest symmetric position
- Your ΔKS ≈ 0.04 suggests ψ is not the main problem
- If ψ sensitivity is high, it means SAW is very directionally dependent

---

## Validation Strategy

### Step 1: Run Sensitivity Analysis (Current Script)
```bash
python global_variance_sensitivity_multi_orientation.py
```

**What this gives you:**
- Sobol indices (S₁, S_T) for each parameter
- Quantifies: "How much does each source of uncertainty contribute to prediction variance?"

### Step 2: Calculate Total Experimental Uncertainty
From the sensitivity analysis, compute:

**Total variance in Wasserstein distance:**
```
σ²_total = Σ(S_T[i] × Var(parameter[i]))
```

This gives you: **σ_exp = ±X MHz** (experimental uncertainty budget)

### Step 3: Compare to MLIP-Experiment Discrepancy
Your current model-experiment mismatch: **KS ≈ 0.5** (or Wasserstein distance = Y MHz)

**Key comparison:**
- If **Y >> X**: MLIP error dominates → MLIP needs improvement
- If **Y ≈ X**: Experimental uncertainty explains discrepancy → can't validate MLIP
- If **Y < X**: Model is better than experiments (!?) → check analysis

### Step 4: Publication Claim
> "We performed global sensitivity analysis across 28 representative grain orientations, testing realistic uncertainties in elastic constants (±X%), density (±2%), EBSD orientation fitting (±5°), and sample alignment (±20°). The combined experimental uncertainty accounts for ±A MHz in predicted SAW frequencies, while our MLIP-experiment discrepancy is B MHz. Since B >> A, we conclude that the remaining error is attributable to MLIP inaccuracy in predicting the elastic tensor, not experimental measurement uncertainty."

---

## Key Insights for Your Paper

### Insight 1: Which Elastic Constant Matters Most?
If **D (anisotropy) has highest S_T:**
- MLIP accuracy for (C₁₁-C₁₂) is critical
- Focus MLIP validation/training on reproducing elastic anisotropy
- Compare MLIP vs DFT for Zener ratio A = 2C₄₄/(C₁₁-C₁₂)

### Insight 2: Sample Alignment vs EBSD Errors
Compare S_T(ψ) vs S_T(φ₁)+S_T(Φ)+S_T(φ₂):
- If **S_T(ψ) > Σ S_T(Euler)**: Systematic alignment error matters more than per-grain orientation errors
- Action: Better fiducial markers for future experiments
- If **Σ S_T(Euler) > S_T(ψ)**: EBSD quality matters more
- Action: Higher resolution EBSD, better indexing

### Insight 3: Orientation Dependence (from heatmaps)
If sensitivity varies strongly across orientations:
- Some grains are "sensitive" (small errors → large frequency changes)
- Some grains are "robust" (large errors → small frequency changes)
- **Implication:** Weight validation by grain sensitivity or report per-orientation

### Insight 4: Nonlinearity Check (from multi-scale analysis)
Compare ±1° vs ±5° Euler errors:
- If S_T scales linearly: Linear model, easy to extrapolate
- If S_T increases nonlinearly: Large errors disproportionately bad
- **Implication:** EBSD quality has nonlinear impact on validation confidence

---

## Expected Result Scenario (My Prediction)

Based on your setup:

### At ±2.5° EBSD Error Scale:
```
Parameter    S_T    Interpretation
---------    ---    --------------
D            0.65   Anisotropy dominates (MLIP challenge)
G            0.58   C₄₄ uncertainty matters (wide bounds)
K            0.42   Bulk modulus moderate effect
ψ            0.28   Sample alignment ~8% contribution
ρ            0.12   Density well-constrained
φ₁           0.18   EBSD errors moderate (average out)
Φ            0.15   
φ₂           0.14   
```

**Total experimental uncertainty:** σ_exp ≈ √(0.65² + 0.58² + ...) × typical_variance ≈ ±5-10 MHz (rough estimate)

**Your MLIP-experiment discrepancy:** Probably larger (KS=0.5 suggests significant mismatch)

**Conclusion:** MLIP elastic tensor needs improvement, especially anisotropy!

---

## Potential Issues & Solutions

### Issue 1: G Bounds are Very Wide (-70%)
Your G bounds go down to **0.30*G0** (30% of baseline). This is huge!

**Questions:**
- Is this from DFT uncertainty?
- Or literature range for V alloys?
- If MLIP predicts G in this range, might explain a lot of discrepancy

**Action:** Check where your MLIP's C₄₄ falls within [0.30*G0, 1.00*G0]

### Issue 2: Asymmetric K and G Bounds
K: only positive variation (+0/+17%)  
G: only negative variation (0/-70%)

**This suggests:**
- You have a lower bound on K (minimum bulk modulus)
- You have an upper bound on G (maximum shear modulus)
- These might be physical constraints or DFT bounds

**Implication:** Your sensitivity analysis will underestimate uncertainty in the "forbidden" directions

**Action:** Document why bounds are asymmetric (physics vs measurement)

### Issue 3: BCC Symmetry Not Exploited
BCC has 4-fold symmetry (0°, 90°, 180°, 270° equivalent)

**Current:** Testing ψ ∈ [-20°, +20°]  
**Could be:** Testing effective ψ ∈ [0°, 45°] (reduced by symmetry)

**But:** Your implementation is fine! Testing full ±20° is conservative and doesn't assume symmetry in experiment.

---

## Next Steps After Analysis

### 1. Identify Dominant Error Source
From S_T ranking, determine if:
- **Material properties** (K, D, G, ρ) → need better elastic tensor
- **Alignment** (ψ) → need better sample mounting
- **EBSD errors** (φ₁, Φ, φ₂) → need higher resolution EBSD

### 2. Quantify MLIP Validation Uncertainty
Calculate confidence interval: "We can validate MLIP accuracy to within ±X% on elastic constants given experimental uncertainty"

### 3. Targeted MLIP Improvement
If D (anisotropy) dominates:
- Add training data specifically for elastic anisotropy
- Use DFT benchmarks for (C₁₁-C₁₂) in similar alloys
- Test MLIP sensitivity to composition variations

### 4. Design Better Experiments (if needed)
If experimental errors are too large to validate MLIP:
- Improve alignment (better fiducials)
- Higher resolution EBSD
- Multiple samples at different orientations
- Direct elastic tensor measurement (nanoindentation, resonant ultrasound)

---

## Publication Strategy

### Figure 1: Sensitivity Comparison
Side-by-side bar charts showing S_T for ±1°, ±2.5°, ±5° EBSD errors

**Caption:**
> "Global sensitivity analysis quantifies contributions of elastic constant uncertainty, EBSD fitting errors (±1-5°), and sample alignment error (±20°) to SAW frequency prediction variance. Material properties (D, K, G) dominate at all error scales, indicating that improved elastic tensor predictions are more critical than improved measurement precision for validating MLIPs."

### Figure 2: MLIP Error Budget
Pie chart or stacked bar:
- Experimental uncertainty budget (from sensitivity analysis)
- MLIP prediction error (remaining discrepancy)

**Caption:**
> "Error decomposition for MLIP validation. Experimental uncertainties (elastic constants ±X%, alignment ±20°, EBSD ±5°) contribute Y MHz to prediction variance, while MLIP-experiment discrepancy is Z MHz. Since Z > Y, we conclude that MLIP inaccuracy in elastic tensor prediction, rather than experimental error, explains the model-experiment mismatch."

### Table 1: Sensitivity Indices
```
Parameter    Physical Meaning       Bounds        S₁    S_T    Rank
---------    ----------------       ------        --    ---    ----
D            Anisotropy (C11-C12)   ±23-27%      0.62  0.65    1
G            Shear (C44)            -70%/0%      0.54  0.58    2
K            Bulk modulus           +0%/+17%     0.38  0.42    3
ψ            Sample alignment       ±20°         0.25  0.28    4
...
```

### Text: Validation Conclusion
> "Our sensitivity analysis demonstrates that elastic constant uncertainties, particularly anisotropy D=(C₁₁-C₁₂), contribute more significantly to prediction variance than measurement errors (EBSD orientation, sample alignment). This validates our approach: observed model-experiment discrepancies (KS=0.5) exceed the maximum uncertainty from experimental errors (KS≈0.1), confirming that MLIP improvements in elastic tensor prediction are required, specifically for the anisotropy ratio."

---

## Summary

**Your analysis is perfectly set up for MLIP validation!**

With ±20° sample alignment, your sensitivity analysis will show:
1. Whether material or measurement uncertainty dominates
2. Which elastic constant(s) need most accurate prediction
3. How much of model-experiment mismatch is "explainable" by experimental error
4. **Proof that remaining error is MLIP, not experiment**

This is exactly what you need to make a strong claim about MLIP validation!

Run the analysis and let's see if my predictions hold up. 🎯

