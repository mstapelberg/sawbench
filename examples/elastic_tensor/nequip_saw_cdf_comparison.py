#!/usr/bin/env python3
"""
Compare SAW frequency CDFs across NequIP potentials
===================================================

This script:
- Loads NequIP potentials from a directory (config-aware vs exploit, plus tuned variants)
- Builds a 512-atom V–Ti random alloy supercell (50/50) and relaxes it per potential
- Computes the elastic tensor (approx. cubic) using sawbench
- Builds Material objects and samples random orientations to compute SAW frequency distributions
- Plots CDFs comparing base vs tuned weights for both families
- Saves figure and a JSON summary in results/

Usage:
  python nequip_saw_cdf_comparison.py \
    --pot-dir /home/myless/Packages/sawbench/examples/data/potentials/config_aware_vs_non_test \
    --device cuda \
    --saw-angle-deg 135 \
    --wavelength 8.8e-6

Notes:
- Requires nequip to be installed and the models to be deployable via NequIPCalculator.from_compiled_model
- Heavy computation: relaxation + elastic tensor calculation for each potential
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from nequip.ase import NequIPCalculator
except ImportError:
    NequIPCalculator = None

from ase.build import bulk
from ase import Atoms

from sawbench import (
    Material,
    calculate_elastic_tensor,
    calculate_elastic_tensor_from_vasp,
    relax_atoms,
    BravaisType,
    EV_A3_TO_GPA,
    plot_frequency_cdfs,
    load_fft_data_from_hdf5,
    extract_experimental_peak_parameters,
    load_ebsd_map,
    calculate_saw_frequencies_for_ebsd_grains,
)


# --- Constants and defaults ---
DEFAULT_POT_DIR = \
    "/home/myless/Packages/sawbench/examples/data/potentials/config_aware_vs_non_test"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NUM_SAW_WORKERS: int = 32
NUM_SAMPLES: int = 4000


def discover_potentials(pot_dir: Path) -> Dict[str, List[Path]]:
    """
    Group potentials into families per architecture based on filename pattern,
    and filter to only include models with mlp512:
      losstype_lmaxX_nlayersY_mlp512*.nequip.zip
    Example losstype: config_aware, exploit, config_aware_tuned, msetw, etc.
    """
    arch_groups: Dict[str, List[Path]] = {}
    for p in sorted(pot_dir.glob("*.nequip.zip")):
        name = p.name
        # Only include models with mlp512 in the filename
        if "mlp512" not in name:
            continue
        # Exclude zbl models unless they are mse_ prefixed (e.g., mse_zbl_...)
        if "zbl" in name.lower():
            if not name.startswith("mse_"):
                continue
        parts = name.split("_")
        # Expect pattern: losstype_lmaxX_nlayersY_mlpZ... .nequip.zip
        # We'll find the tokens containing lmax, nlayers, mlp and build a key
        lmax_tok = next((t for t in parts if t.startswith("lmax")), None)
        nl_tok = next((t for t in parts if t.startswith("nlayers")), None)
        mlp_tok = next((t for t in parts if t.startswith("mlp")), None)
        if lmax_tok and nl_tok and mlp_tok:
            key = f"{lmax_tok}_{nl_tok}_{mlp_tok}"
        else:
            # fallback: group by whole filename stem sans losstype prefix if present
            key = "unknown_architecture"
        arch_groups.setdefault(key, []).append(p)
    return arch_groups


def build_v_ti_supercell_512(a_lattice: float = 3.01, seed: int | None = None) -> Atoms:
    """
    Build a 512-atom random alloy supercell for V–Ti (1.2 at% Ti) in bcc structure.

    We construct a bcc primitive cell (1 atom) and repeat 8x8x8 → 512 atoms.
    """
    base = bulk("V", crystalstructure="bcc", a=a_lattice, cubic=True) 
    sc = base.repeat((4, 4, 4))  # 128 atoms total

    # Randomly substitute 1.2 at% of the atoms to Ti
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    num_atoms = len(sc)
    # 1.2 at% of 512 atoms = 6.144, so round to 6 atoms
    num_ti = int(round(num_atoms * 0.012))
    indices = np.arange(num_atoms)
    rng.shuffle(indices)
    ti_idx = indices[:num_ti]
    symbols = sc.get_chemical_symbols()
    for idx in ti_idx:
        symbols[idx] = "Ti"
    sc.set_chemical_symbols(symbols)
    return sc


def compute_elastic_and_material(
    atoms: Atoms,
    calculator: Any,
    *,
    bravais: BravaisType = BravaisType.CUBIC,
    max_strain_normal: float = 0.005,
    max_strain_shear: float = 0.02,
    n_deform: int = 3,
    relax_fmax: float = 1e-3,
    relax_steps: int = 500,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Relax the alloy structure with the given calculator, compute elastic tensor (GPa),
    and return (C_tensor_gpa, {C11,C12,C44}, density_kg_m3).
    """
    # Relax
    atoms_relaxed = atoms.copy()
    atoms_relaxed.calc = calculator
    atoms_relaxed = relax_atoms(atoms_relaxed, fmax_threshold=relax_fmax, steps=relax_steps)

    # Elastic tensor (in eV/Å^3); convert to GPa
    C = calculate_elastic_tensor(
        calculator=calculator,
        atoms=atoms_relaxed,
        bravais_type=bravais,
        max_strain_normal=max_strain_normal,
        max_strain_shear=max_strain_shear,
        n_deform=n_deform,
    )
    C_gpa = C * EV_A3_TO_GPA

    # Approximate cubic constants from diagonal/shear positions
    C11 = float(C_gpa[0, 0])
    C12 = float(C_gpa[0, 1])
    C44 = float(C_gpa[3, 3])

    # Use fixed density for V–Ti alloy
    density_kg_m3 = 6000.0

    return C_gpa, {"C11": C11, "C12": C12, "C44": C44}, density_kg_m3


def compute_saw_frequencies_for_ebsd(
    material: Material,
    ebsd_map,
    wavelength_m: float,
    saw_angle_deg: float,
    sampling: int,
    num_workers: int,
) -> np.ndarray:
    """Use EBSD grain orientations to predict SAW distribution (Hz)."""
    df = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=material,
        wavelength=wavelength_m,
        saw_calc_angle_deg=saw_angle_deg,
        num_workers=num_workers,
        saw_calc_sampling=sampling,
    )
    if df is None or df.empty:
        return np.array([])
    if 'Peak SAW Frequency (Hz)' not in df.columns:
        return np.array([])
    return df['Peak SAW Frequency (Hz)'].dropna().to_numpy()


def build_material_from_C_and_density(
    name: str, C_gpa: Dict[str, float], density_kg_m3: float
) -> Material:
    GPA_TO_PA = 1e9
    return Material(
        formula=name,
        C11=C_gpa["C11"] * GPA_TO_PA,
        C12=C_gpa["C12"] * GPA_TO_PA,
        C44=C_gpa["C44"] * GPA_TO_PA,
        density=density_kg_m3,
        crystal_class="cubic",
    )


def is_cubic_tensor_stable(cubic_gpa: Dict[str, float]) -> bool:
    """Mechanical stability criteria for cubic crystals.
    Conditions: C11 - C12 > 0, C11 + 2 C12 > 0, C44 > 0.
    """
    try:
        C11 = float(cubic_gpa["C11"])
        C12 = float(cubic_gpa["C12"])
        C44 = float(cubic_gpa["C44"])
        if not (np.isfinite(C11) and np.isfinite(C12) and np.isfinite(C44)):
            return False
        return (C11 - C12) > 0 and (C11 + 2.0 * C12) > 0 and (C44 > 0)
    except Exception:
        return False


def plot_cdf_groups(
    grouped_freqs_mhz: Dict[str, List[np.ndarray]],
    labels_for_lines: Dict[str, List[str]],
    out_path: Path,
    xlim: Tuple[float, float] | None = None,
    overlay_series: Dict[str, np.ndarray] | None = None,
    overlay_labels: Dict[str, str] | None = None,
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: config_aware vs tuned
    left_ax = axs[0]
    left_data = grouped_freqs_mhz.get("config_aware_panel", [])
    left_labels = labels_for_lines.get("config_aware_panel", [])
    left_colors = plt.cm.viridis(np.linspace(0, 1, len(left_data)))
    plot_frequency_cdfs(
        left_ax,
        [np.sort(d) for d in left_data],
        labels=left_labels,
        colors=left_colors,
        title="Config-aware: Base vs Tuned",
        xlim=xlim,
    )

    # Overlays on left panel
    if overlay_series:
        ov_colors = {"DFT": "black", "EXP": "red", "PredEXP": "blue"}
        for key in ["DFT", "EXP", "PredEXP"]:
            if key in overlay_series and overlay_series[key].size:
                data_sorted = np.sort(overlay_series[key])
                cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                left_ax.plot(data_sorted, cdf, color=ov_colors.get(key, "gray"),
                             linewidth=2.0, label=overlay_labels.get(key, key))
        left_ax.legend(loc='best')

    # Right: exploit vs tuned
    right_ax = axs[1]
    right_data = grouped_freqs_mhz.get("exploit_panel", [])
    right_labels = labels_for_lines.get("exploit_panel", [])
    right_colors = plt.cm.plasma(np.linspace(0, 1, len(right_data)))
    plot_frequency_cdfs(
        right_ax,
        [np.sort(d) for d in right_data],
        labels=right_labels,
        colors=right_colors,
        title="Exploit: Base vs Tuned",
        xlim=xlim,
    )

    # Overlays on right panel
    if overlay_series:
        ov_colors = {"DFT": "black", "EXP": "red", "PredEXP": "blue"}
        for key in ["DFT", "EXP", "PredEXP"]:
            if key in overlay_series and overlay_series[key].size:
                data_sorted = np.sort(overlay_series[key])
                cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                right_ax.plot(data_sorted, cdf, color=ov_colors.get(key, "gray"),
                              linewidth=2.0, label=overlay_labels.get(key, key))
        right_ax.legend(loc='best')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare SAW CDFs across NequIP potentials")
    parser.add_argument("--pot-dir", type=str, default=DEFAULT_POT_DIR, help="Directory with *.nequip.zip models")
    parser.add_argument("--device", type=str, default="cuda", help="Device for NequIP (cuda|cpu)")
    parser.add_argument("--saw-angle-deg", type=float, default=135.0, help="SAW propagation angle (deg)")
    parser.add_argument("--wavelength", type=float, default=8.8e-6, help="SAW wavelength (m)")
    parser.add_argument("--sampling", type=int, default=4000, help="Sampling for SAWCalculator (unused in EBSD mode)")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of workers for SAWCalculator")
    parser.add_argument("--lattice-a", type=float, default=3.01, help="Initial bcc lattice constant (Å)")
    # Replicates & relaxation controls
    parser.add_argument("--replicates", type=int, default=1, help="Number of alloy replicates to average per model")
    parser.add_argument("--replicate-seed", type=int, default=42, help="Base RNG seed for replicate generation")
    parser.add_argument("--replicate-aggregate", type=str, choices=["mean", "median"], default="mean", help="Aggregation method for elasticity tensors across replicates")
    parser.add_argument("--relax-fmax", type=float, default=1e-3, help="Force threshold for relaxation (eV/Å)")
    parser.add_argument("--relax-steps", type=int, default=500, help="Max relaxation steps")
    # DFT and experimental inputs
    parser.add_argument("--dft-dir", type=str, default=None, help="Directory with VASP OUTCARs for elastic tensor (reference, ep* subdirs)")
    parser.add_argument("--dft-strain", type=float, default=0.005, help="Strain magnitude used in DFT directories (e.g., 0.005)")
    parser.add_argument("--dft-bravais", type=str, default="cubic", help="Bravais type for DFT tensor (cubic|hexagonal|...)")
    parser.add_argument("--exp-hdf5", type=str, default=None, help="Experimental FFT HDF5 to extract peak freqs")
    parser.add_argument("--exp-min-mhz", type=float, default=200.0, help="Min MHz filter for EXP peaks")
    parser.add_argument("--exp-max-mhz", type=float, default=400.0, help="Max MHz filter for EXP peaks")
    parser.add_argument("--ebsd-path", type=str, default="/home/myless/Packages/sawbench/examples/data/V-1_2Ti_EBSD_Map.ctf", help="EBSD file path for grain orientations (e.g., .ctf)")
    parser.add_argument("--ebsd-type", type=str, default="OxfordText", help="EBSD file type")
    parser.add_argument("--ebsd-boundary", type=float, default=5.0, help="Boundary misorientation (deg)")
    parser.add_argument("--ebsd-min-grain", type=int, default=10, help="Min grain pixel size")
    args = parser.parse_args()

    if NequIPCalculator is None:
        raise ImportError("nequip is not installed. Please install nequip to run this script.")

    pot_dir = Path(args.pot_dir)
    if not pot_dir.exists():
        raise FileNotFoundError(f"Potential directory not found: {pot_dir}")

    groups = discover_potentials(pot_dir)
    print("Discovered potentials by architecture:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} models")

    # Build one example alloy structure to log size info (replicates will build their own)
    example_atoms = build_v_ti_supercell_512(a_lattice=args.lattice_a, seed=args.replicate_seed)
    print(f"Example V–Ti alloy supercell: {len(example_atoms)} atoms, initial volume {example_atoms.get_volume():.1f} Å^3")

    # Resolve EBSD path robustly (handle relative paths and spaces)
    script_dir = Path(__file__).resolve().parent
    ebsd_path = Path(args.ebsd_path)
    if not ebsd_path.exists():
        base_name = ebsd_path.name
        candidate_paths = [
            (script_dir / args.ebsd_path),
            (script_dir / ".." / "data" / base_name),
            (script_dir.parent / "data" / base_name),
        ]
        found = None
        for cand in candidate_paths:
            cand = cand.resolve()
            if cand.exists():
                found = cand
                break
        if found is None:
            # Last resort: limited search near examples root
            try:
                examples_root = script_dir.parent.resolve()
                for cand in examples_root.rglob(base_name):
                    found = cand
                    break
            except Exception:
                pass
        if found is not None:
            ebsd_path = found
    print(f"Resolved EBSD path: {ebsd_path}")
    # Some loaders expect base path without extension (e.g., defdap for OxfordText)
    ebsd_base = ebsd_path
    if ebsd_path.suffix.lower() == ".ctf":
        ebsd_base = ebsd_path.with_suffix("")
    # Load EBSD map for orientations
    ebsd_map = load_ebsd_map(
        str(ebsd_base), args.ebsd_type, args.ebsd_boundary, args.ebsd_min_grain
    )
    if not ebsd_map:
        raise RuntimeError(f"Failed to load EBSD map from {ebsd_path}; cannot compute EBSD-based SAW distributions.")

    # Process each model individually, store frequencies and summaries
    run_summary: Dict[str, Dict] = {}

    def process_model(model_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
        print(f"\n--- Processing model: {model_path.name} ---")
        try:
            # Prefer from_compiled_model; fallback to private method if needed
            try:
                calc = NequIPCalculator.from_compiled_model(str(model_path), device=args.device)
            except Exception:
                calc = NequIPCalculator._from_packaged_model(package_path=str(model_path), device=args.device)
        except Exception as e:
            print(f"Failed to load calculator for {model_path.name}: {e}")
            return np.array([]), {}

        # Replicate loop: build different supercells, compute elasticity, validate, then aggregate
        kept_tensors: List[np.ndarray] = []
        kept_cubics: List[Dict[str, float]] = []
        attempted_tensors: List[np.ndarray] = []
        attempted_cubics: List[Dict[str, float]] = []
        density_kg_m3 = 6100.0
        for r in range(max(1, int(args.replicates))):
            try:
                atoms_rep = build_v_ti_supercell_512(a_lattice=args.lattice_a, seed=int(args.replicate_seed) + r)
                C_full_gpa, cubic_consts_rep, density_kg_m3 = compute_elastic_and_material(
                    atoms=atoms_rep,
                    calculator=calc,
                    bravais=BravaisType.CUBIC,
                    max_strain_normal=0.005,
                    max_strain_shear=0.02,
                    n_deform=3,
                    relax_fmax=args.relax_fmax,
                    relax_steps=args.relax_steps,
                )
                attempted_tensors.append(C_full_gpa)
                attempted_cubics.append(cubic_consts_rep)
                if is_cubic_tensor_stable(cubic_consts_rep):
                    kept_tensors.append(C_full_gpa)
                    kept_cubics.append(cubic_consts_rep)
                else:
                    print(f"  Replicate {r+1}: Unstable cubic tensor -> dropped (C11={cubic_consts_rep['C11']:.2f}, C12={cubic_consts_rep['C12']:.2f}, C44={cubic_consts_rep['C44']:.2f})")
            except Exception as e:
                print(f"  Replicate {r+1}: Elastic tensor computation failed: {e}")
                continue

        # Fallback if no stable tensors
        tensors_for_agg = kept_tensors if kept_tensors else attempted_tensors
        # cubics_for_agg retained for potential future reporting if needed
        if not tensors_for_agg:
            print(f"No successful elasticity tensors for {model_path.name}; skipping.")
            return np.array([]), {}

        stack_C = np.stack(tensors_for_agg, axis=0)
        if args.replicate_aggregate == "median":
            C_gpa_full = np.median(stack_C, axis=0)
        else:
            C_gpa_full = np.mean(stack_C, axis=0)

        # Aggregated cubic constants
        C11 = float(C_gpa_full[0, 0])
        C12 = float(C_gpa_full[0, 1])
        C44 = float(C_gpa_full[3, 3])
        cubic_consts = {"C11": C11, "C12": C12, "C44": C44}

        # Build material from aggregated constants and compute SAW frequencies
        mat = build_material_from_C_and_density(model_path.stem, cubic_consts, density_kg_m3)
        freqs_hz = compute_saw_frequencies_for_ebsd(
            mat, ebsd_map=ebsd_map, wavelength_m=args.wavelength, saw_angle_deg=args.saw_angle_deg, sampling=args.sampling, num_workers=NUM_SAW_WORKERS
        )
        freqs_mhz = freqs_hz / 1e6

        # Convert full tensor to nested list for JSON serialization
        elasticity_tensor = C_gpa_full.tolist()

        # Calculate elastic moduli for cubic crystals from aggregated constants
        bulk_modulus = (C11 + 2 * C12) / 3
        shear_modulus = C44
        youngs_modulus = (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)
        poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (6 * bulk_modulus + 2 * shear_modulus)

        run_summary[model_path.name] = {
            "model": model_path.name,
            "elasticity": {
                "C11": float(C_gpa_full[0, 0]),
                "C12": float(C_gpa_full[0, 1]),
                "C13": float(C_gpa_full[0, 2]),
                "C14": float(C_gpa_full[0, 3]),
                "C15": float(C_gpa_full[0, 4]),
                "C16": float(C_gpa_full[0, 5]),
                "C21": float(C_gpa_full[1, 0]),
                "C22": float(C_gpa_full[1, 1]),
                "C23": float(C_gpa_full[1, 2]),
                "C24": float(C_gpa_full[1, 3]),
                "C25": float(C_gpa_full[1, 4]),
                "C26": float(C_gpa_full[1, 5]),
                "C31": float(C_gpa_full[2, 0]),
                "C32": float(C_gpa_full[2, 1]),
                "C33": float(C_gpa_full[2, 2]),
                "C34": float(C_gpa_full[2, 3]),
                "C35": float(C_gpa_full[2, 4]),
                "C36": float(C_gpa_full[2, 5]),
                "C41": float(C_gpa_full[3, 0]),
                "C42": float(C_gpa_full[3, 1]),
                "C43": float(C_gpa_full[3, 2]),
                "C44": float(C_gpa_full[3, 3]),
                "C45": float(C_gpa_full[3, 4]),
                "C46": float(C_gpa_full[3, 5]),
                "C51": float(C_gpa_full[4, 0]),
                "C52": float(C_gpa_full[4, 1]),
                "C53": float(C_gpa_full[4, 2]),
                "C54": float(C_gpa_full[4, 3]),
                "C55": float(C_gpa_full[4, 4]),
                "C56": float(C_gpa_full[4, 5]),
                "C61": float(C_gpa_full[5, 0]),
                "C62": float(C_gpa_full[5, 1]),
                "C63": float(C_gpa_full[5, 2]),
                "C64": float(C_gpa_full[5, 3]),
                "C65": float(C_gpa_full[5, 4]),
                "C66": float(C_gpa_full[5, 5]),
                "full_tensor": elasticity_tensor
            },
            "elastic_moduli_gpa": {
                "youngs_modulus": float(youngs_modulus),
                "bulk_modulus": float(bulk_modulus),
                "shear_modulus": float(shear_modulus),
                "poisson_ratio": float(poisson_ratio)
            },
            "cubic_constants_gpa": cubic_consts,
            "replicates": {
                "requested": int(args.replicates),
                "kept": int(len(kept_tensors)) if kept_tensors else int(len(attempted_tensors)),
                "dropped": int(len(attempted_tensors) - len(kept_tensors)),
                "aggregation": args.replicate_aggregate,
                "per_replicate_cubic_gpa": kept_cubics if kept_cubics else attempted_cubics,
            },
            "density_kg_m3": density_kg_m3,
            "num_freqs": int(len(freqs_mhz)),
            "frequencies_mhz": freqs_mhz.tolist() if len(freqs_mhz) > 0 else [],
            "freq_mhz_summary": {
                "mean": float(np.nanmean(freqs_mhz)) if len(freqs_mhz) else np.nan,
                "median": float(np.nanmedian(freqs_mhz)) if len(freqs_mhz) else np.nan,
                "min": float(np.nanmin(freqs_mhz)) if len(freqs_mhz) else np.nan,
                "max": float(np.nanmax(freqs_mhz)) if len(freqs_mhz) else np.nan,
            },
        }
        return freqs_mhz, cubic_consts

    # Build overlay: DFT SAW freqs (EBSD-based) + EXP freqs
    overlay_series_global: Dict[str, np.ndarray] = {}
    overlay_labels_global: Dict[str, str] = {}

    # DFT overlay
    if args.dft_dir:
        try:
            bravais_map = {
                "cubic": BravaisType.CUBIC,
                "hexagonal": BravaisType.HEXAGONAL,
                "trigonal": BravaisType.TRIGONAL,
                "tetragonal": BravaisType.TETRAGONAL,
                "orthorhombic": BravaisType.ORTHORHOMBIC,
                "monoclinic": BravaisType.MONOCLINIC,
                "triclinic": BravaisType.TRICLINIC,
            }
            bravais = bravais_map.get(args.dft_bravais.lower(), BravaisType.CUBIC)
            C_dft_gpa = calculate_elastic_tensor_from_vasp(
                directory_path=args.dft_dir,
                strain_amount=float(args.dft_strain),
                bravais_type=bravais,
            )
            # Approximate cubic constants (or fall back to principal entries)
            C11 = float(C_dft_gpa[0, 0])
            C12 = float(C_dft_gpa[0, 1])
            C44 = float(C_dft_gpa[3, 3])
            density_kg_m3 = 6100.0
            mat_dft = build_material_from_C_and_density("DFT", {"C11": C11, "C12": C12, "C44": C44}, density_kg_m3)
            freqs_dft_hz = compute_saw_frequencies_for_ebsd(
                mat_dft, ebsd_map=ebsd_map, wavelength_m=args.wavelength, saw_angle_deg=args.saw_angle_deg, sampling=args.sampling, num_workers=NUM_SAW_WORKERS
            )
            overlay_series_global["DFT"] = freqs_dft_hz / 1e6
            overlay_labels_global["DFT"] = "DFT"
            
            # Store DFT elastic data for reference
            overlay_series_global["DFT_elastic_data"] = {
                "full_tensor": C_dft_gpa.tolist(),
                "cubic_constants": {"C11": C11, "C12": C12, "C44": C44},
                "density_kg_m3": density_kg_m3
            }
        except Exception as e:
            print(f"DFT overlay failed: {e}")

    # PredEXP overlay: experimentally derived elastic constants and density
    try:
        predexp_cubic = {"C11": 229.0, "C12": 119.0, "C44": 43.0}  # GPa
        predexp_density = 6110.0  # kg/m^3
        mat_predexp = build_material_from_C_and_density("PredEXP", predexp_cubic, predexp_density)
        freqs_predexp_hz = compute_saw_frequencies_for_ebsd(
            mat_predexp, ebsd_map=ebsd_map, wavelength_m=args.wavelength, saw_angle_deg=args.saw_angle_deg, sampling=args.sampling, num_workers=NUM_SAW_WORKERS
        )
        overlay_series_global["PredEXP"] = freqs_predexp_hz / 1e6
        overlay_labels_global["PredEXP"] = "PredEXP"
        overlay_series_global["PredEXP_elastic_data"] = {
            "full_tensor": [
                [predexp_cubic["C11"], predexp_cubic["C12"], predexp_cubic["C12"], 0.0, 0.0, 0.0],
                [predexp_cubic["C12"], predexp_cubic["C11"], predexp_cubic["C12"], 0.0, 0.0, 0.0],
                [predexp_cubic["C12"], predexp_cubic["C12"], predexp_cubic["C11"], 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, predexp_cubic["C44"], 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, predexp_cubic["C44"], 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, predexp_cubic["C44"]],
            ],
            "cubic_constants": predexp_cubic,
            "density_kg_m3": predexp_density,
        }
    except Exception as e:
        print(f"PredEXP overlay failed: {e}")

    # EXP overlay: use modular workflow approach (dominant peaks with filtering)
    if args.exp_hdf5:
        try:
            fft = load_fft_data_from_hdf5(args.exp_hdf5)
            if fft:
                exp_freq_axis, _, _, exp_amp, (Ny, Nx) = fft
                dominant = []
                for iy in range(Ny):
                    for ix in range(Nx):
                        amp_trace = exp_amp[:, iy, ix]
                        peak_params_all = extract_experimental_peak_parameters(
                            exp_freq_axis, amp_trace, num_peaks_to_extract=3
                        )
                        valid_mask = ~np.isnan(peak_params_all[:, 0])
                        if np.any(valid_mask):
                            actual = peak_params_all[valid_mask, :]
                            # strongest peak by amplitude
                            strongest_idx = np.argmax(actual[:, 0])
                            mu = actual[strongest_idx, 1]
                            if np.isfinite(mu):
                                dominant.append(mu)
                dom = np.array(dominant)
                if dom.size:
                    # Filter to requested MHz window
                    min_hz = args.exp_min_mhz * 1e6
                    max_hz = args.exp_max_mhz * 1e6
                    dom = dom[(dom >= min_hz) & (dom <= max_hz)]
                    overlay_series_global["EXP"] = dom / 1e6
                    overlay_labels_global["EXP"] = "Experiment"
        except Exception as e:
            print(f"Experimental overlay failed: {e}")

    # For each architecture, gather series by losstype and plot a figure
    for arch_key, model_paths in groups.items():
        # Map losstype -> concatenated freqs
        losstype_to_freqs: Dict[str, List[np.ndarray]] = {}
        for mp in model_paths:
            losstype = mp.name.split("_")[0] if "_" in mp.name else "unknown"
            # Normalize to {mse, ca, catw, msetw}
            if losstype not in {"mse", "ca", "catw", "msetw"}:
                if losstype.startswith("config"):
                    losstype = "ca" if "tuned" not in losstype else "catw"
                elif losstype.startswith("exploit"):
                    losstype = "mse"
            freqs_mhz, _ = process_model(mp)
            if freqs_mhz.size:
                losstype_to_freqs.setdefault(losstype, []).append(freqs_mhz)

        # Prepare datasets and labels
        datasets_mhz = []
        labels = []
        for losstype, series_list in sorted(losstype_to_freqs.items()):
            datasets_mhz.append(np.concatenate(series_list))
            labels.append(losstype)

        if not datasets_mhz:
            continue

        # Single-panel CDF with overlays
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets_mhz)))
        # Compute dynamic x-limits from all datasets (models + DFT + EXP)
        x_series_list: List[np.ndarray] = []
        x_series_list.extend(datasets_mhz)
        if "DFT" in overlay_series_global:
            x_series_list.append(overlay_series_global["DFT"])  # MHz already
        if "EXP" in overlay_series_global:
            x_series_list.append(overlay_series_global["EXP"])  # MHz already
        if "PredEXP" in overlay_series_global:
            x_series_list.append(overlay_series_global["PredEXP"])  # MHz already
        min_val = np.inf
        max_val = -np.inf
        for arr in x_series_list:
            if arr is None or getattr(arr, "size", 0) == 0:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            local_min = float(np.min(finite))
            local_max = float(np.max(finite))
            if local_min < min_val:
                min_val = local_min
            if local_max > max_val:
                max_val = local_max
        xlim = (min_val, max_val) if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val else None
        plot_frequency_cdfs(
            ax,
            [np.sort(d) for d in datasets_mhz],
            labels=labels,
            colors=colors,
            title=f"SAW CDFs: {arch_key}",
            xlim=xlim,
        )
        # Overlays
        if overlay_series_global:
            ov_colors = {"DFT": "black", "EXP": "red", "PredEXP": "blue"}
            for key in ["DFT", "EXP", "PredEXP"]:
                if key in overlay_series_global and overlay_series_global[key].size:
                    data_sorted = np.sort(overlay_series_global[key])
                    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                    ax.plot(data_sorted, cdf, color=ov_colors.get(key, "gray"),
                            linewidth=2.0, label=overlay_labels_global.get(key, key))
            ax.legend(loc='best')

        out_path = RESULTS_DIR / f"saw_cdf_arch_{arch_key}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

    # Add reference data (DFT and experimental) to summary
    reference_data = {}
    
    # Add DFT data if available
    if "DFT" in overlay_series_global and overlay_series_global["DFT"].size > 0:
        dft_freqs = overlay_series_global["DFT"]
        
        # Calculate DFT elastic moduli if elastic data is available
        dft_elastic_moduli = {}
        if "DFT_elastic_data" in overlay_series_global:
            elastic_data = overlay_series_global["DFT_elastic_data"]
            cubic_consts = elastic_data["cubic_constants"]
            C11 = cubic_consts["C11"]
            C12 = cubic_consts["C12"]
            C44 = cubic_consts["C44"]
            
            # Calculate elastic moduli for cubic crystals
            bulk_modulus = (C11 + 2 * C12) / 3
            shear_modulus = C44
            youngs_modulus = (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)
            poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (6 * bulk_modulus + 2 * shear_modulus)
            
            dft_elastic_moduli = {
                "youngs_modulus": float(youngs_modulus),
                "bulk_modulus": float(bulk_modulus),
                "shear_modulus": float(shear_modulus),
                "poisson_ratio": float(poisson_ratio)
            }
        
        reference_data["DFT"] = {
            "model": "DFT",
            "frequencies_mhz": dft_freqs.tolist(),
            "num_freqs": int(len(dft_freqs)),
            "freq_mhz_summary": {
                "mean": float(np.nanmean(dft_freqs)) if len(dft_freqs) else np.nan,
                "median": float(np.nanmedian(dft_freqs)) if len(dft_freqs) else np.nan,
                "min": float(np.nanmin(dft_freqs)) if len(dft_freqs) else np.nan,
                "max": float(np.nanmax(dft_freqs)) if len(dft_freqs) else np.nan,
            },
        }
        
        # Add elastic data if available
        if "DFT_elastic_data" in overlay_series_global:
            elastic_data = overlay_series_global["DFT_elastic_data"]
            reference_data["DFT"].update({
                "elasticity": {
                    "C11": float(elastic_data["cubic_constants"]["C11"]),
                    "C12": float(elastic_data["cubic_constants"]["C12"]),
                    "C13": float(elastic_data["cubic_constants"]["C12"]),  # C13 = C12 for cubic
                    "C14": 0.0,
                    "C15": 0.0,
                    "C16": 0.0,
                    "C21": float(elastic_data["cubic_constants"]["C12"]),  # C21 = C12 for cubic
                    "C22": float(elastic_data["cubic_constants"]["C11"]),  # C22 = C11 for cubic
                    "C23": float(elastic_data["cubic_constants"]["C12"]),  # C23 = C12 for cubic
                    "C24": 0.0,
                    "C25": 0.0,
                    "C26": 0.0,
                    "C31": float(elastic_data["cubic_constants"]["C12"]),  # C31 = C12 for cubic
                    "C32": float(elastic_data["cubic_constants"]["C12"]),  # C32 = C12 for cubic
                    "C33": float(elastic_data["cubic_constants"]["C11"]),  # C33 = C11 for cubic
                    "C34": 0.0,
                    "C35": 0.0,
                    "C36": 0.0,
                    "C41": 0.0,
                    "C42": 0.0,
                    "C43": 0.0,
                    "C44": float(elastic_data["cubic_constants"]["C44"]),
                    "C45": 0.0,
                    "C46": 0.0,
                    "C51": 0.0,
                    "C52": 0.0,
                    "C53": 0.0,
                    "C54": 0.0,
                    "C55": float(elastic_data["cubic_constants"]["C44"]),  # C55 = C44 for cubic
                    "C56": 0.0,
                    "C61": 0.0,
                    "C62": 0.0,
                    "C63": 0.0,
                    "C64": 0.0,
                    "C65": 0.0,
                    "C66": float(elastic_data["cubic_constants"]["C44"]),  # C66 = C44 for cubic
                    "full_tensor": elastic_data["full_tensor"]
                },
                "elastic_moduli_gpa": dft_elastic_moduli,
                "cubic_constants_gpa": elastic_data["cubic_constants"],
                "density_kg_m3": elastic_data["density_kg_m3"]
            })
    
    # Add experimental data if available
    if "EXP" in overlay_series_global and overlay_series_global["EXP"].size > 0:
        exp_freqs = overlay_series_global["EXP"]
        reference_data["EXP"] = {
            "model": "Experiment",
            "frequencies_mhz": exp_freqs.tolist(),
            "num_freqs": int(len(exp_freqs)),
            "freq_mhz_summary": {
                "mean": float(np.nanmean(exp_freqs)) if len(exp_freqs) else np.nan,
                "median": float(np.nanmedian(exp_freqs)) if len(exp_freqs) else np.nan,
                "min": float(np.nanmin(exp_freqs)) if len(exp_freqs) else np.nan,
                "max": float(np.nanmax(exp_freqs)) if len(exp_freqs) else np.nan,
            },
        }
    
    # Add PredEXP data if available
    if "PredEXP" in overlay_series_global and overlay_series_global["PredEXP"].size > 0:
        predexp_freqs = overlay_series_global["PredEXP"]
        pred_elastic = overlay_series_global.get("PredEXP_elastic_data", {})
        # Calculate elastic moduli from provided cubic constants
        dct = pred_elastic.get("cubic_constants", {"C11": 229.0, "C12": 119.0, "C44": 43.0})
        B = (dct["C11"] + 2 * dct["C12"]) / 3
        G = dct["C44"]
        E = (9 * B * G) / (3 * B + G)
        nu = (3 * B - 2 * G) / (6 * B + 2 * G)
        reference_data["PredEXP"] = {
            "model": "PredEXP",
            "frequencies_mhz": predexp_freqs.tolist(),
            "num_freqs": int(len(predexp_freqs)),
            "freq_mhz_summary": {
                "mean": float(np.nanmean(predexp_freqs)) if len(predexp_freqs) else np.nan,
                "median": float(np.nanmedian(predexp_freqs)) if len(predexp_freqs) else np.nan,
                "min": float(np.nanmin(predexp_freqs)) if len(predexp_freqs) else np.nan,
                "max": float(np.nanmax(predexp_freqs)) if len(predexp_freqs) else np.nan,
            },
            "elasticity": {
                "C11": float(dct["C11"]),
                "C12": float(dct["C12"]),
                "C13": float(dct["C12"]),
                "C14": 0.0,
                "C15": 0.0,
                "C16": 0.0,
                "C21": float(dct["C12"]),
                "C22": float(dct["C11"]),
                "C23": float(dct["C12"]),
                "C24": 0.0,
                "C25": 0.0,
                "C26": 0.0,
                "C31": float(dct["C12"]),
                "C32": float(dct["C12"]),
                "C33": float(dct["C11"]),
                "C34": 0.0,
                "C35": 0.0,
                "C36": 0.0,
                "C41": 0.0,
                "C42": 0.0,
                "C43": 0.0,
                "C44": float(dct["C44"]),
                "C45": 0.0,
                "C46": 0.0,
                "C51": 0.0,
                "C52": 0.0,
                "C53": 0.0,
                "C54": 0.0,
                "C55": float(dct["C44"]),
                "C56": 0.0,
                "C61": 0.0,
                "C62": 0.0,
                "C63": 0.0,
                "C64": 0.0,
                "C65": 0.0,
                "C66": float(dct["C44"]),
                "full_tensor": pred_elastic.get("full_tensor")
            },
            "elastic_moduli_gpa": {
                "youngs_modulus": float(E),
                "bulk_modulus": float(B),
                "shear_modulus": float(G),
                "poisson_ratio": float(nu)
            },
            "cubic_constants_gpa": dct,
            "density_kg_m3": float(pred_elastic.get("density_kg_m3", 6110.0))
        }
    
    # Combine model results with reference data
    full_summary = {
        "model_results": run_summary,
        "reference_data": reference_data
    }
    
    # Save JSON summary
    summary_path = RESULTS_DIR / "saw_cdf_nequip_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()


