#!/usr/bin/env python3
"""
saw_sensitivity_panels.py

Generate five complementary sensitivity figures for SAW frequency:
(a) ternary trade-off map with iso-contours and a projected gradient field
(b) local elasticities ("tornado" bar chart) at the base point
(c) one-at-a-time (±50%) curves for each elastic constant
(d) Zener anisotropy scan: normalized Rayleigh speed vs A = 2 C44 / (C11 - C12)
(e) global (rank) sensitivity via random sampling in ±50% box

Place this file alongside your modules:
 - materials.py
 - euler_transformations.py
 - saw_calculator.py

Run:
  python saw_sensitivity_panels.py --outdir ./saw_sensitivity_figs

Notes
-----
- Panel (a) uses the same trade-off simplex mapping as your current ternary script:
    C11 = C11_base * (0.5 + w11), etc., with w11+w12+w44=1.
  This highlights how increasing one constant at the expense of the others shifts f.
- All frequency evaluations call your SAWCalculator -> get_saw_speed() pipeline.

Author: (you)
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import mpltern  # pip install mpltern

from sawbench.materials import Material           
from sawbench.saw_calculator import SAWCalculator 

# ---------------------------
# Core frequency computation
# ---------------------------

def compute_saw_frequency_mhz(
    C11_GPa: float,
    C12_GPa: float,
    C44_GPa: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    deg_inplane: float = 0.0,
    sampling: int = 400,
    wavelength_m: float = 8.8e-6,
) -> float:
    """Compute predicted SAW frequency (MHz) for given cubic constants and density.

    Returns NaN if the SAW pipeline fails.
    """
    try:
        material = Material(
            formula="Cubic",
            C11=C11_GPa * 1e9,
            C12=C12_GPa * 1e9,
            C44=C44_GPa * 1e9,
            density=density_kg_m3,
            crystal_class="cubic",
        )
        calc = SAWCalculator(material=material, euler_angles=np.asarray(euler_angles_rad))

        # Try optimized path then fallback, mirroring your own usage.
        try:
            v, _, _ = calc.get_saw_speed(
                deg_inplane, sampling=sampling, psaw=0,
                draw_plot=False, debug=False, use_optimized=True)
        except Exception:
            v = np.array([])

        if v is None or len(v) == 0 or not np.isfinite(v[0]):
            try:
                v, _, _ = calc.get_saw_speed(
                    deg_inplane, sampling=sampling, psaw=0,
                    draw_plot=False, debug=False, use_optimized=False)
            except Exception:
                return float("nan")

        if v is None or len(v) == 0 or not np.isfinite(v[0]):
            return float("nan")

        return float(v[0] / wavelength_m / 1e6)  # MHz
    except Exception:
        return float("nan")


# ---------------------------
# Helper constructs
# ---------------------------

def generate_ternary_grid(resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform triangular (barycentric) grid (t, l, r), with t+l+r=1."""
    ts, ls, rs = [], [], []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            ts.append(i / resolution)
            ls.append(j / resolution)
            rs.append(k / resolution)
    return np.array(ts), np.array(ls), np.array(rs)


def elasticity_sensitivities(
    base_c11: float,
    base_c12: float,
    base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
    eps: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Return base frequency and dimensionless elasticities ∂ln f / ∂ln Ck."""
    f0 = compute_saw_frequency_mhz(
        base_c11, base_c12, base_c44,
        density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
        deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
    out = {}
    for name, val in [("C11", base_c11), ("C12", base_c12), ("C44", base_c44)]:
        args = dict(C11_GPa=base_c11, C12_GPa=base_c12, C44_GPa=base_c44)
        args[name + "_GPa"] = val * (1 + eps)
        f_up = compute_saw_frequency_mhz(
            **args, density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
            deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
        args[name + "_GPa"] = val * (1 - eps)
        f_dn = compute_saw_frequency_mhz(
            **args, density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
            deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
        out[name] = (f_up - f_dn) / (2 * eps * f0)
    return f0, out


def one_at_a_time_curves(
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
    n: int = 41,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return normalized x (ratio to base) and f(x) for each constant varied alone."""
    xs = np.linspace(0.5, 1.5, n)
    curves = {}
    for idx, (name, base) in enumerate([("C11", base_c11), ("C12", base_c12), ("C44", base_c44)]):
        fvals = []
        for x in xs:
            c11, c12, c44 = base_c11, base_c12, base_c44
            if name == "C11": c11 = base * x
            if name == "C12": c12 = base * x
            if name == "C44": c44 = base * x
            f = compute_saw_frequency_mhz(
                c11, c12, c44,
                density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
                deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
            fvals.append(f)
        curves[name] = (xs, np.array(fvals))
    return curves


def zener_scan(
    A_vals: np.ndarray,
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hold K and μ at base, vary Zener ratio A; return A and v_R / sqrt(μ/ρ)."""
    K = (base_c11 + 2 * base_c12) / 3.0
    mu = base_c44
    outA, outNorm = [], []
    for A in A_vals:
        if A <= 0: 
            continue
        C12 = K - (2.0 / 3.0) * (mu / A)
        C11 = K + (4.0 / 3.0) * (mu / A)
        # basic cubic stability checks
        if not (C44_positive := mu > 0): 
            continue
        if not (C11 > abs(C12) and (C11 - C12) > 0):
            continue
        f_mhz = compute_saw_frequency_mhz(
            C11, C12, mu,
            density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
            deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
        if not np.isfinite(f_mhz): 
            continue
        v = f_mhz * 1e6 * wavelength_m
        outA.append(A)
        outNorm.append(v / np.sqrt(mu * 1e9 / density_kg_m3))
    return np.array(outA), np.array(outNorm)


def rank_sensitivity(
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
    n: int = 250,
    seed: int = 0,
) -> Dict[str, float]:
    """Random global sensitivity via Spearman rank correlation (absolute value)."""
    rng = np.random.default_rng(seed)
    C11s = base_c11 * (0.5 + rng.random(n))
    C12s = base_c12 * (0.5 + rng.random(n))
    C44s = base_c44 * (0.5 + rng.random(n))
    f = np.zeros(n)
    for i in range(n):
        f[i] = compute_saw_frequency_mhz(
            C11s[i], C12s[i], C44s[i],
            density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
            deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
    def spearman(x: np.ndarray) -> float:
        rx = np.argsort(np.argsort(x))
        rf = np.argsort(np.argsort(f))
        rx = (rx - rx.mean()) / (rx.std() + 1e-12)
        rf = (rf - rf.mean()) / (rf.std() + 1e-12)
        return float(np.mean(rx * rf))
    return {"C11": abs(spearman(C11s)),
            "C12": abs(spearman(C12s)),
            "C44": abs(spearman(C44s))}


# ---------------------------
# Plotting (panels a–e)
# ---------------------------

def panel_a_ternary(
    outdir: str,
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
    res: int = 28,
    focus_range: Optional[Tuple[float, float]] = (200.0, 500.0),
    cmap: str = "viridis",
    gradient_points: int = 60,
) -> None:
    """(a) Ternary heatmap + iso-contours + projected gradient field."""
    t, l, r = generate_ternary_grid(res)
    freq = np.empty_like(t)
    for idx, (a, b, c) in enumerate(zip(t, l, r)):
        C11 = base_c11 * (0.5 + a)
        C12 = base_c12 * (0.5 + b)
        C44 = base_c44 * (0.5 + c)
        freq[idx] = compute_saw_frequency_mhz(
            C11, C12, C44,
            density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
            deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)

    fig = plt.figure(figsize=(7.8, 7.0))
    ax = fig.add_subplot(111, projection="ternary")

    finite = np.isfinite(freq)
    # Heatmap
    if np.sum(finite) >= 3:
        hm = ax.tripcolor(t[finite], l[finite], r[finite], freq[finite],
                          cmap=cmap, shading="gouraud")
    else:
        hm = ax.scatter(t[finite], l[finite], r[finite], c=freq[finite], s=14, cmap=cmap)

    # Iso-contours
    finite_vals = freq[finite]
    if finite_vals.size >= 8:
        lo, hi = (focus_range if focus_range else (np.percentile(finite_vals, 10),
                                                   np.percentile(finite_vals, 90)))
        lo = max(lo, float(np.min(finite_vals)))
        hi = min(hi, float(np.max(finite_vals)))
        try:
            CS = ax.tricontour(t[finite], l[finite], r[finite], freq[finite],
                               levels=np.linspace(lo, hi, 8), linewidths=0.7, colors="k")
            ax.clabel(CS, inline=True, fontsize=8, fmt="%.0f")
        except Exception:
            pass

    # Axes labels and ticks (mapped to GPa)
    ax.set_tlabel("C11")
    ax.set_llabel("C12")
    ax.set_rlabel("C44")
    ticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.taxis.set_ticks(ticks); ax.laxis.set_ticks(ticks); ax.raxis.set_ticks(ticks)
    ax.taxis.set_ticklabels([f"{base_c11*(0.5+x):.1f}" for x in ticks])
    ax.laxis.set_ticklabels([f"{base_c12*(0.5+x):.1f}" for x in ticks])
    ax.raxis.set_ticklabels([f"{base_c44*(0.5+x):.1f}" for x in ticks])

    cbar = fig.colorbar(hm, ax=ax, pad=0.10, shrink=0.85)
    cbar.set_label("Predicted SAW frequency (MHz)")

    title = (f"Ternary trade-off: f(C11/C12/C44)\n"
             f"Base (GPa): C11={base_c11:.1f}, C12={base_c12:.1f}, C44={base_c44:.1f}; "
             f"ρ={density_kg_m3:.0f} kg/m³; λ={wavelength_m*1e6:.2f} µm; angle={deg_inplane:.1f}°")
    ax.set_title(title, pad=16)

    # Projected gradient field (sparse sampling)
    rng = np.random.default_rng(0)
    cand = np.where(finite)[0]
    rng.shuffle(cand)
    cand = cand[:min(gradient_points, cand.size)]
    # Chain-rule Jacobian for mapping from w -> C (trade-off simplex)
    J = np.diag([base_c11, base_c12, base_c44])

    def fd_df_dC(C11, C12, C44, eps=0.01):
        f0 = compute_saw_frequency_mhz(C11, C12, C44,
                                       density_kg_m3=density_kg_m3,
                                       euler_angles_rad=euler_angles_rad,
                                       deg_inplane=deg_inplane,
                                       sampling=sampling, wavelength_m=wavelength_m)
        out = []
        for name, val in [("C11", C11), ("C12", C12), ("C44", C44)]:
            c11, c12, c44 = C11, C12, C44
            if name == "C11":
                fu = compute_saw_frequency_mhz(C11*(1+eps), C12, C44, density_kg_m3=density_kg_m3,
                                               euler_angles_rad=euler_angles_rad, deg_inplane=deg_inplane,
                                               sampling=sampling, wavelength_m=wavelength_m)
                fd = compute_saw_frequency_mhz(C11*(1-eps), C12, C44, density_kg_m3=density_kg_m3,
                                               euler_angles_rad=euler_angles_rad, deg_inplane=deg_inplane,
                                               sampling=sampling, wavelength_m=wavelength_m)
            elif name == "C12":
                fu = compute_saw_frequency_mhz(C11, C12*(1+eps), C44, density_kg_m3=density_kg_m3,
                                               euler_angles_rad=euler_angles_rad, deg_inplane=deg_inplane,
                                               sampling=sampling, wavelength_m=wavelength_m)
                fd = compute_saw_frequency_mhz(C11, C12*(1-eps), C44, density_kg_m3=density_kg_m3,
                                               euler_angles_rad=euler_angles_rad, deg_inplane=deg_inplane,
                                               sampling=sampling, wavelength_m=wavelength_m)
            else:
                fu = compute_saw_frequency_mhz(C11, C12, C44*(1+eps), density_kg_m3=density_kg_m3,
                                               euler_angles_rad=euler_angles_rad, deg_inplane=deg_inplane,
                                               sampling=sampling, wavelength_m=wavelength_m)
                fd = compute_saw_frequency_mhz(C11, C12, C44*(1-eps), density_kg_m3=density_kg_m3,
                                               euler_angles_rad=euler_angles_rad, deg_inplane=deg_inplane,
                                               sampling=sampling, wavelength_m=wavelength_m)
            # central difference for ∂f/∂Ck
            out.append((fu - fd) / (2 * eps * val))
        return np.array(out), f0

    # Draw small arrows/segments
    for idx in cand:
        a, b, c = float(t[idx]), float(l[idx]), float(r[idx])
        C11 = base_c11 * (0.5 + a)
        C12 = base_c12 * (0.5 + b)
        C44 = base_c44 * (0.5 + c)
        gC, f0 = fd_df_dC(C11, C12, C44)  # ∇_C f (MHz/GPa)
        gw = J @ gC                         # ∇_w f (MHz)
        # Project to simplex plane (zero-sum direction)
        gw = gw - np.mean(gw)
        if np.linalg.norm(gw) < 1e-12:
            continue
        direction = gw / (np.linalg.norm(gw) + 1e-12)
        step = 0.06  # small step in barycentric space
        a2, b2, c2 = a + step*direction[0], b + step*direction[1], c + step*direction[2]
        # Keep inside the simplex
        # Project back to sum=1
        s = a2 + b2 + c2
        a2, b2, c2 = a2/s, b2/s, c2/s
        # Clip gently toward interior
        eps_clip = 1e-3
        a2 = np.clip(a2, eps_clip, 1-eps_clip)
        b2 = np.clip(b2, eps_clip, 1-eps_clip)
        c2 = 1 - a2 - b2
        try:
            # mpltern quiver may not be present in all versions; fall back to a segment
            ax.quiver([a], [b], [c], [a2-a], [b2-b], [c2-c], angles='xy', scale_units='xy', scale=1, width=0.004)
        except Exception:
            ax.plot([a, a2], [b, b2], [c, c2], lw=0.7, color='k')

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "panel_a_ternary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_b_tornado(
    outdir: str,
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
) -> None:
    """(b) Local elasticities S = ∂ln f / ∂ln Ck as a bar chart."""
    f0, sens = elasticity_sensitivities(
        base_c11, base_c12, base_c44,
        density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
        deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
    names = ["C11", "C12", "C44"]
    vals = [sens[n] for n in names]

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(names, vals)
    ax.set_ylabel(r"$S=\partial \ln f / \partial \ln C$")
    ax.set_title(f"Local elasticities at base (f₀={f0:.1f} MHz)")
    ax.axhline(0.0, lw=0.8, ls="--")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "panel_b_tornado.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_c_oat(
    outdir: str,
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
) -> None:
    """(c) One-at-a-time curves for each constant over ±50%."""
    curves = one_at_a_time_curves(
        base_c11, base_c12, base_c44,
        density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
        deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, name in zip(axes, ["C11", "C12", "C44"]):
        xs, ys = curves[name]
        ax.plot(xs, ys)
        ax.set_xlabel(f"{name}/base")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("f (MHz)")
            ax.set_yscale("log")
    fig.suptitle("One-at-a-time sensitivity (others fixed at base)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "panel_c_oat.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_d_zener(
    outdir: str,
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
) -> None:
    """(d) Normalized Rayleigh speed vs Zener ratio A (anisotropy scan)."""
    A_vals = np.linspace(0.5, 3.0, 36)
    A, vnorm = zener_scan(A_vals, base_c11, base_c12, base_c44,
                          density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
                          deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(A, vnorm, marker="o", ms=3, lw=1)
    ax.set_xlabel("Zener anisotropy A = 2 C44 / (C11 - C12)")
    ax.set_ylabel(r"$v_R/\sqrt{\mu/\rho}$")
    ax.grid(True, alpha=0.3)
    ax.set_title("Anisotropy-only sensitivity (K, μ held at base)")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "panel_d_zener.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_e_rank(
    outdir: str,
    base_c11: float, base_c12: float, base_c44: float,
    *,
    density_kg_m3: float,
    euler_angles_rad: Tuple[float, float, float],
    deg_inplane: float,
    sampling: int,
    wavelength_m: float,
    n: int = 250,
) -> None:
    """(e) Global rank sensitivity (absolute Spearman coefficients)."""
    rank = rank_sensitivity(
        base_c11, base_c12, base_c44,
        density_kg_m3=density_kg_m3, euler_angles_rad=euler_angles_rad,
        deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m, n=n)

    names = ["C11", "C12", "C44"]
    vals = [rank[n] for n in names]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(names, vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("abs Spearman ρ")
    ax.set_title(f"Global rank sensitivity (n={n})")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "panel_e_rank.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Generate sensitivity panels (a–e) for SAW frequency.")
    p.add_argument("--c11_base_gpa", type=float, default=231.3)
    p.add_argument("--c12_base_gpa", type=float, default=117.5)
    p.add_argument("--c44_base_gpa", type=float, default=51.7)
    p.add_argument("--density", type=float, default=6100.0)
    p.add_argument("--wavelength_um", type=float, default=8.8)
    p.add_argument("--deg", type=float, default=0.0, help="in-plane rotation (degrees)")
    p.add_argument("--sampling", type=int, default=400, choices=[400, 4000, 40000])
    p.add_argument("--res", type=int, default=28, help="ternary grid resolution (edge nodes)")
    p.add_argument("--outdir", type=str, default="./saw_sensitivity_figs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_rank", type=int, default=250, help="samples for panel (e)")
    p.add_argument("--no_ternary", action="store_true", help="skip panel (a)")
    args = p.parse_args()

    base_c11 = float(args.c11_base_gpa)
    base_c12 = float(args.c12_base_gpa)
    base_c44 = float(args.c44_base_gpa)
    density = float(args.density)
    wavelength_m = float(args.wavelength_um) * 1e-6
    deg_inplane = float(args.deg)
    sampling = int(args.sampling)
    res = int(args.res)
    outdir = str(args.outdir)
    np.random.seed(args.seed)

    # orientation: use identity (0,0,0) here; plug in your preferred Euler if desired
    euler_angles_rad = (0.0, 0.0, 0.0)

    if not args.no_ternary:
        panel_a_ternary(outdir, base_c11, base_c12, base_c44,
                        density_kg_m3=density, euler_angles_rad=euler_angles_rad,
                        deg_inplane=deg_inplane, sampling=sampling,
                        wavelength_m=wavelength_m, res=res)

    panel_b_tornado(outdir, base_c11, base_c12, base_c44,
                    density_kg_m3=density, euler_angles_rad=euler_angles_rad,
                    deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)

    panel_c_oat(outdir, base_c11, base_c12, base_c44,
                density_kg_m3=density, euler_angles_rad=euler_angles_rad,
                deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)

    panel_d_zener(outdir, base_c11, base_c12, base_c44,
                  density_kg_m3=density, euler_angles_rad=euler_angles_rad,
                  deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m)

    panel_e_rank(outdir, base_c11, base_c12, base_c44,
                 density_kg_m3=density, euler_angles_rad=euler_angles_rad,
                 deg_inplane=deg_inplane, sampling=sampling, wavelength_m=wavelength_m,
                 n=args.n_rank)

    print(f"Saved panels to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
