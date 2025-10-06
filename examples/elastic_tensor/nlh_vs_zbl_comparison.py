#!/usr/bin/env python3
"""
Compare SAW frequency CDFs: NLH vs ZBL NequIP potentials
========================================================

This script compares two specific NequIP potentials:
- ca_lmax2_nlayers2_mlp512_nlh_epoch169.nequip.zip (NLH)
- ca_lmax2_nlayers2_mlp512_zbl_epoch205.nequip.zip (ZBL)

The script:
- Loads the two specific NequIP potentials
- Builds a 512-atom V–Ti random alloy supercell and relaxes it per potential
- Computes elastic tensors using sawbench
- Builds Material objects and samples EBSD grain orientations for SAW frequencies
- Plots CDFs comparing NLH vs ZBL with experimental data overlay
- Saves figure and JSON summary in results/

Usage:
  # Full comparison with SAW frequencies
  python nlh_vs_zbl_comparison.py \
    --nlh-model /path/to/ca_lmax2_nlayers2_mlp512_nlh_epoch169.nequip.zip \
    --zbl-model /path/to/ca_lmax2_nlayers2_mlp512_zbl_epoch205.nequip.zip \
    --device cuda \
    --saw-angle-deg 135 \
    --wavelength 8.8e-6 \
    --exp-hdf5 /path/to/experimental_data.h5
  
  # Elastic tensor comparison only (faster)
  python nlh_vs_zbl_comparison.py \
    --nlh-model /path/to/ca_lmax2_nlayers2_mlp512_nlh_epoch169.nequip.zip \
    --zbl-model /path/to/ca_lmax2_nlayers2_mlp512_zbl_epoch205.nequip.zip \
    --device cuda \
    --elastic-only

  # Include DFT elastic tensor for comparison
  python nlh_vs_zbl_comparison.py \
    --nlh-model /path/to/nlh.nequip.zip \
    --zbl-model /path/to/zbl.nequip.zip \
    --device cuda \
    --elastic-only \
    --dft-dir /path/to/vasp_elastic_calc \
    --dft-strain 0.005 \
    --dft-bravais cubic

Notes:
- Requires nequip to be installed and the models to be deployable via NequIPCalculator.from_compiled_model
- Heavy computation: relaxation + elastic tensor calculation for each potential
- Experimental data is loaded from HDF5 and processed to extract dominant peak frequencies
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default experimental data paths
DEFAULT_EXP_PATHS = [
    "/home/myless/Documents/saw_freq_analysis/fftData.h5",
    "/Users/myless/Dropbox (MIT)/Research/2025/Spring_2025/TGS-Mapping/processed_analysis/v-1_2ti/fftData.h5"
]

# Default EBSD path
DEFAULT_EBSD_PATH = "/home/myless/Packages/sawbench/examples/data/V-1_2Ti_EBSD_Map.ctf"


def build_v_ti_supercell_512(a_lattice: float = 3.01) -> Atoms:
    """
    Build a 512-atom random alloy supercell for V–Ti (1.2 at% Ti) in bcc structure.

    We construct a bcc primitive cell (1 atom) and repeat 4x4x4 → 128 atoms.
    """
    base = bulk("V", crystalstructure="bcc", a=a_lattice, cubic=True) 
    sc = base.repeat((4, 4, 4))  # 128 atoms total

    # Randomly substitute 1.2 at% of the atoms to Ti
    rng = np.random.default_rng(42)
    num_atoms = len(sc)
    # 1.2 at% of 128 atoms = 1.536, so round to 2 atoms
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
    max_strain_shear: float = 0.01,
    n_deform: int = 5,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Relax the alloy structure with the given calculator, compute elastic tensor (GPa),
    and return (C_tensor_gpa, {C11,C12,C44}, density_kg_m3).
    """
    # Relax
    atoms_relaxed = atoms.copy()
    atoms_relaxed.calc = calculator
    atoms_relaxed = relax_atoms(atoms_relaxed, fmax_threshold=1e-3, steps=500)

    # Elastic tensor (in eV/Å^3); convert to GPa
    C = calculate_elastic_tensor(
        calculator=calculator,
        atoms=atoms_relaxed,
        bravais_type=BravaisType.TRICLINIC,
        max_strain_normal=max_strain_normal,
        max_strain_shear=max_strain_shear,
        n_deform=n_deform,
    )
    C_gpa = C * EV_A3_TO_GPA
    cubic_consts = project_to_cubic(C_gpa)

    # Approximate cubic constants from diagonal/shear positions
    C11 = float(cubic_consts["C11"])
    C12 = float(cubic_consts["C12"])
    C44 = float(cubic_consts["C44"])

    # Use fixed density for V–Ti alloy
    density_kg_m3 = 6100.0

    return C_gpa, {"C11": C11, "C12": C12, "C44": C44}, density_kg_m3


def compute_saw_frequencies_for_ebsd(
    material: Material,
    ebsd_map,
    wavelength_m: float,
    saw_angle_deg: float,
) -> np.ndarray:
    """Use EBSD grain orientations to predict SAW distribution (Hz)."""
    df = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=material,
        wavelength=wavelength_m,
        saw_calc_angle_deg=saw_angle_deg,
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

def project_to_cubic(C: np.ndarray) -> dict[str, float]:
    # C is 6x6 in Voigt order
    C11 = float((C[0,0] + C[1,1] + C[2,2]) / 3.0)
    C12 = float((C[0,1] + C[0,2] + C[1,2]) / 3.0)
    C44 = float((C[3,3] + C[4,4] + C[5,5]) / 3.0)
    return {"C11": C11, "C12": C12, "C44": C44}


def load_experimental_frequencies(
    hdf5_path: str,
    min_mhz: float = 200.0,
    max_mhz: float = 400.0,
    num_peaks: int = 3
) -> Optional[np.ndarray]:
    """
    Load experimental frequencies from HDF5 file.
    
    Args:
        hdf5_path: Path to experimental HDF5 file
        min_mhz: Minimum frequency filter in MHz
        max_mhz: Maximum frequency filter in MHz
        num_peaks: Number of peaks to extract per pixel
        
    Returns:
        Array of experimental frequencies in MHz, or None if loading fails
    """
    try:
        fft = load_fft_data_from_hdf5(hdf5_path)
        if not fft:
            print(f"Failed to load experimental data from {hdf5_path}")
            return None
            
        exp_freq_axis, _, _, exp_amp, (Ny, Nx) = fft
        print(f"Loaded experimental data: {Ny}x{Nx} pixels, {len(exp_freq_axis)} frequency points")
        
        dominant = []
        for iy in range(Ny):
            for ix in range(Nx):
                amp_trace = exp_amp[:, iy, ix]
                peak_params_all = extract_experimental_peak_parameters(
                    exp_freq_axis, amp_trace, num_peaks_to_extract=num_peaks
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
            min_hz = min_mhz * 1e6
            max_hz = max_mhz * 1e6
            dom = dom[(dom >= min_hz) & (dom <= max_hz)]
            print(f"Extracted {len(dom)} experimental frequencies in {min_mhz}-{max_mhz} MHz range")
            return dom / 1e6  # Convert to MHz
        else:
            print("No valid experimental frequencies found")
            return None
            
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return None


def plot_comparison_cdf(
    nlh_freqs: np.ndarray,
    zbl_freqs: np.ndarray,
    exp_freqs: Optional[np.ndarray],
    out_path: Path,
    xlim: Tuple[float, float] = (150, 550)
):
    """Plot CDF comparison of NLH vs ZBL with experimental overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Prepare data for plotting
    datasets = [np.sort(nlh_freqs), np.sort(zbl_freqs)]
    labels = ["NLH", "ZBL"]
    colors = ["#1f77b4", "#ff7f0e"]  # Blue and orange
    
    # Plot model CDFs
    plot_frequency_cdfs(
        ax,
        datasets,
        labels=labels,
        colors=colors,
        title="SAW Frequency CDFs: NLH vs ZBL",
        xlim=xlim,
    )
    
    # Add experimental overlay if available
    if exp_freqs is not None and len(exp_freqs) > 0:
        exp_sorted = np.sort(exp_freqs)
        cdf = np.arange(1, len(exp_sorted) + 1) / len(exp_sorted)
        ax.plot(exp_sorted, cdf, color="red", linewidth=2.0, 
                label=f"Experiment (n={len(exp_freqs)})", linestyle="--")
        ax.legend(loc='best')
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {out_path}")


def process_model(
    model_path: Path,
    atoms: Atoms,
    ebsd_map,
    wavelength_m: float,
    saw_angle_deg: float,
    device: str,
    elastic_only: bool = False
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Process a single NequIP model and return frequencies and summary.
    
    Returns:
        (frequencies_mhz, elastic_constants, summary_dict)
    """
    print(f"\n--- Processing model: {model_path.name} ---")
    
    try:
        # Load calculator
        try:
            calc = NequIPCalculator.from_compiled_model(str(model_path), device=device)
        except Exception:
            calc = NequIPCalculator._from_packaged_model(package_path=str(model_path), device=device, chemical_symbols=["Ti", "V"])
    except Exception as e:
        print(f"Failed to load calculator for {model_path.name}: {e}")
        return np.array([]), {}, {}

    try:
        # Compute elastic properties
        C_gpa_full, cubic_consts, density_kg_m3 = compute_elastic_and_material(
            atoms=atoms, calculator=calc, bravais=BravaisType.CUBIC,
            max_strain_normal=0.005, max_strain_shear=0.02, n_deform=3,
        )
    except Exception as e:
        print(f"Elastic tensor computation failed for {model_path.name}: {e}")
        return np.array([]), {}, {}

    # Build material and compute SAW frequencies (unless elastic_only)
    freqs_mhz = np.array([])
    if not elastic_only:
        mat = build_material_from_C_and_density(model_path.stem, cubic_consts, density_kg_m3)
        freqs_hz = compute_saw_frequencies_for_ebsd(
            mat, ebsd_map=ebsd_map, wavelength_m=wavelength_m, saw_angle_deg=saw_angle_deg
        )
        freqs_mhz = freqs_hz / 1e6

    # Create summary
    summary = {
        "elastic_constants_gpa": cubic_consts,
        "density_kg_m3": density_kg_m3,
        "num_freqs": int(len(freqs_mhz)),
    }
    
    if not elastic_only:
        summary["freq_mhz_summary"] = {
            "mean": float(np.nanmean(freqs_mhz)) if len(freqs_mhz) else np.nan,
            "median": float(np.nanmedian(freqs_mhz)) if len(freqs_mhz) else np.nan,
            "min": float(np.nanmin(freqs_mhz)) if len(freqs_mhz) else np.nan,
            "max": float(np.nanmax(freqs_mhz)) if len(freqs_mhz) else np.nan,
            "std": float(np.nanstd(freqs_mhz)) if len(freqs_mhz) else np.nan,
        }

    return freqs_mhz, cubic_consts, summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare NLH vs ZBL NequIP potentials")
    parser.add_argument("--nlh-model", type=str, required=True, 
                       help="Path to NLH model (.nequip.zip)")
    parser.add_argument("--zbl-model", type=str, required=True,
                       help="Path to ZBL model (.nequip.zip)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device for NequIP (cuda|cpu)")
    parser.add_argument("--saw-angle-deg", type=float, default=135.0, 
                       help="SAW propagation angle (deg)")
    parser.add_argument("--wavelength", type=float, default=8.8e-6, 
                       help="SAW wavelength (m)")
    parser.add_argument("--lattice-a", type=float, default=3.01, 
                       help="Initial bcc lattice constant (Å)")
    parser.add_argument("--exp-hdf5", type=str, default=None,
                       help="Experimental HDF5 file path")
    parser.add_argument("--exp-min-mhz", type=float, default=200.0,
                       help="Min MHz filter for experimental data")
    parser.add_argument("--exp-max-mhz", type=float, default=400.0,
                       help="Max MHz filter for experimental data")
    parser.add_argument("--ebsd-path", type=str, default=DEFAULT_EBSD_PATH,
                       help="EBSD file path for grain orientations")
    parser.add_argument("--ebsd-type", type=str, default="OxfordText",
                       help="EBSD file type")
    parser.add_argument("--ebsd-boundary", type=float, default=5.0,
                       help="Boundary misorientation (deg)")
    parser.add_argument("--ebsd-min-grain", type=int, default=10,
                       help="Min grain pixel size")
    parser.add_argument("--elastic-only", action="store_true",
                       help="Skip SAW calculations and only compare elastic tensors")
    # Optional DFT elastic tensor directory
    parser.add_argument("--dft-dir", type=str, default="../data/PBE_Manual_Elastic",
                       help="Directory containing VASP OUTCARs for elastic tensor (reference + ep* subdirs)")
    parser.add_argument("--dft-strain", type=float, default=0.005,
                       help="Strain magnitude used in the DFT elastic calculation (e.g., 0.005)")
    parser.add_argument("--dft-bravais", type=str, default="cubic",
                       help="Bravais type for DFT tensor (cubic|hexagonal|trigonal|tetragonal|orthorhombic|monoclinic|triclinic)")
    args = parser.parse_args()

    if NequIPCalculator is None:
        raise ImportError("nequip is not installed. Please install nequip to run this script.")

    # Validate model paths
    nlh_path = Path(args.nlh_model)
    zbl_path = Path(args.zbl_model)
    
    if not nlh_path.exists():
        raise FileNotFoundError(f"NLH model not found: {nlh_path}")
    if not zbl_path.exists():
        raise FileNotFoundError(f"ZBL model not found: {zbl_path}")

    print(f"NLH model: {nlh_path}")
    print(f"ZBL model: {zbl_path}")

    # Build shared initial alloy structure
    base_atoms = build_v_ti_supercell_512(a_lattice=args.lattice_a)
    print(f"Built V–Ti alloy supercell: {len(base_atoms)} atoms, initial volume {base_atoms.get_volume():.1f} Å^3")

    # Load EBSD map (only if not in elastic-only mode)
    ebsd_map = None
    if not args.elastic_only:
        script_dir = Path(__file__).resolve().parent
        ebsd_path = Path(args.ebsd_path)
        if not ebsd_path.exists():
            # Try to find EBSD file in common locations
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
                raise FileNotFoundError(f"EBSD file not found: {ebsd_path}")
            ebsd_path = found
        
        print(f"Using EBSD path: {ebsd_path}")
        ebsd_base = ebsd_path.with_suffix("") if ebsd_path.suffix.lower() == ".ctf" else ebsd_path
        
        ebsd_map = load_ebsd_map(
            str(ebsd_base), args.ebsd_type, args.ebsd_boundary, args.ebsd_min_grain
        )
        if not ebsd_map:
            raise RuntimeError(f"Failed to load EBSD map from {ebsd_path}")
    else:
        print("Elastic-only mode: skipping EBSD map loading")

    # Load experimental data if provided (skip in elastic-only mode)
    exp_freqs = None
    if not args.elastic_only:
        if args.exp_hdf5:
            exp_freqs = load_experimental_frequencies(
                args.exp_hdf5, args.exp_min_mhz, args.exp_max_mhz
            )
        else:
            # Try default paths
            for default_path in DEFAULT_EXP_PATHS:
                if Path(default_path).exists():
                    print(f"Using default experimental data: {default_path}")
                    exp_freqs = load_experimental_frequencies(
                        default_path, args.exp_min_mhz, args.exp_max_mhz
                    )
                    break
            if exp_freqs is None:
                print("No experimental data found, plotting models only")
    else:
        print("Elastic-only mode: skipping experimental data loading")

    # Process both models
    run_summary = {}
    
    # Process NLH model
    nlh_freqs, nlh_consts, nlh_summary = process_model(
        nlh_path, base_atoms, ebsd_map, args.wavelength, args.saw_angle_deg, args.device, args.elastic_only
    )
    run_summary["nlh"] = nlh_summary
    
    # Process ZBL model
    zbl_freqs, zbl_consts, zbl_summary = process_model(
        zbl_path, base_atoms, ebsd_map, args.wavelength, args.saw_angle_deg, args.device, args.elastic_only
    )
    run_summary["zbl"] = zbl_summary

    # Optionally compute DFT elastic tensor and cubic projection
    dft_consts = None
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
            dft_bravais = bravais_map.get(args.dft_bravais.lower(), BravaisType.CUBIC)
            C_dft_gpa = calculate_elastic_tensor_from_vasp(
                directory_path=args.dft_dir,
                strain_amount=float(args.dft_strain),
                bravais_type=dft_bravais,
            )
            dft_consts = project_to_cubic(C_dft_gpa)
            run_summary["dft"] = {"elastic_constants_gpa": dft_consts}
        except Exception as e:
            print(f"DFT elastic tensor extraction failed: {e}")

    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nElastic constants (GPa):")
    print(f"NLH - C11={nlh_consts['C11']:.1f}, C12={nlh_consts['C12']:.1f}, C44={nlh_consts['C44']:.1f}")
    print(f"ZBL - C11={zbl_consts['C11']:.1f}, C12={zbl_consts['C12']:.1f}, C44={zbl_consts['C44']:.1f}")
    if dft_consts is not None:
        print(f"DFT - C11={dft_consts['C11']:.1f}, C12={dft_consts['C12']:.1f}, C44={dft_consts['C44']:.1f}")
    
    # Calculate differences
    print(f"\nDifferences (ZBL - NLH):")
    print(f"C11: {zbl_consts['C11'] - nlh_consts['C11']:+.1f} GPa")
    print(f"C12: {zbl_consts['C12'] - nlh_consts['C12']:+.1f} GPa")
    print(f"C44: {zbl_consts['C44'] - nlh_consts['C44']:+.1f} GPa")
    if dft_consts is not None:
        print(f"\nDifferences to DFT (Model - DFT):")
        print(f"NLH: C11 {nlh_consts['C11'] - dft_consts['C11']:+.1f}, C12 {nlh_consts['C12'] - dft_consts['C12']:+.1f}, C44 {nlh_consts['C44'] - dft_consts['C44']:+.1f}")
        print(f"ZBL: C11 {zbl_consts['C11'] - dft_consts['C11']:+.1f}, C12 {zbl_consts['C12'] - dft_consts['C12']:+.1f}, C44 {zbl_consts['C44'] - dft_consts['C44']:+.1f}")
    
    if not args.elastic_only:
        if len(nlh_freqs) > 0 and len(zbl_freqs) > 0:
            print(f"\nSAW frequencies:")
            print(f"NLH: {len(nlh_freqs)} frequencies, mean={np.mean(nlh_freqs):.1f} MHz, std={np.std(nlh_freqs):.1f} MHz")
            print(f"ZBL: {len(zbl_freqs)} frequencies, mean={np.mean(zbl_freqs):.1f} MHz, std={np.std(zbl_freqs):.1f} MHz")
            
            if exp_freqs is not None and len(exp_freqs) > 0:
                print(f"EXP: {len(exp_freqs)} frequencies, mean={np.mean(exp_freqs):.1f} MHz, std={np.std(exp_freqs):.1f} MHz")

    # Create comparison plot (only if not elastic-only)
    if not args.elastic_only:
        out_path = RESULTS_DIR / "nlh_vs_zbl_comparison.png"
        plot_comparison_cdf(nlh_freqs, zbl_freqs, exp_freqs, out_path)
    else:
        print("\nElastic-only mode: skipping SAW frequency plot generation")

    # Save JSON summary
    summary_path = RESULTS_DIR / "nlh_vs_zbl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
