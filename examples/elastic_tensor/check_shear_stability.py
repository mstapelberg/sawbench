import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Conditional imports for MLIP calculators
try:
    from nequip.ase import NequIPCalculator
except ImportError:
    NequIPCalculator = None

try:
    from mace.calculators import mace_mp, MACECalculator
except ImportError:
    mace_mp = None
    MACECalculator = None

# Import from sawbench package
from sawbench.elastic import (
    get_cart_deformed_cell,
    full_3x3_to_voigt_6_stress,
    EV_A3_TO_GPA
)
from ase.io import read

# Assumes the script is run from 'examples/elastic_tensor/'
OUTPUT_DIR = Path("results/stability_checks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Define all models and their configurations
MODELS_CONFIG = {
    'mace_foundation': {
        'name': 'MACE Foundation (medium)',
        'type': 'mlip',
        'calculator_class': mace_mp,
        'model_path': "medium",
        'init_kwargs': {'default_dtype': "float64"}
    },
    'mace_finetuned': {
        'name': 'MACE Finetuned (gen6)',
        'type': 'mlip',
        'calculator_class': MACECalculator,
        'model_path': "../data/potentials/gen_6_model_0_L1_isolated-2026-01-16-finetuned_fp64_nh10000_lr1e-4_stagetwo.model",
        'init_kwargs': {'default_dtype': "float32"}
    },
    'allegro': {
        'name': 'Allegro/NequIP',
        'type': 'mlip',
        'calculator_class': NequIPCalculator,
        'model_path': "../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2",
        'init_kwargs': {}
    }
}

def check_shear_stability(model_config: dict, device: str):
    """
    Applies shear deformation to a structure and plots energy/stress vs. strain
    to diagnose elastic stability for C44.
    """
    print(f"--- Checking shear stability for: {model_config['name']} ---")
    
    # 1. Initialize Calculator
    MLIPCalculatorClass = model_config['calculator_class']
    if MLIPCalculatorClass is None:
        raise ImportError(f"Calculator for '{model_config['name']}' not installed.")

    model_path = model_config['model_path']
    if model_config['calculator_class'] == NequIPCalculator:
        calculator = MLIPCalculatorClass.from_compiled_model(model_path, device=device)
    elif model_config['calculator_class'] == MACECalculator: # Finetuned
        calculator = MLIPCalculatorClass(model_paths=[model_path], device=device, **model_config['init_kwargs'])
    elif model_config['calculator_class'] == mace_mp: # Foundation
        calculator = MLIPCalculatorClass(model=model_path, device=device, **model_config['init_kwargs'])
    else:
        raise TypeError(f"Unknown calculator class for {model_config['name']}")

    # 2. Load relaxed reference structure
    # Using a pre-relaxed structure is important
    initial_atoms = read('../data/PBE_Manual_Elastic/reference/OUTCAR', index=0)
    initial_atoms.calc = calculator
    
    # Get energy of the reference, un-strained structure
    base_energy = initial_atoms.get_potential_energy()

    # 3. Apply deformations and calculate properties
    # Axis 3 corresponds to yz shear, which is related to C44
    shear_axis = 3 
    strains = np.linspace(-0.02, 0.02, 15) # -2% to +2% strain
    energies = []
    stresses_yz = []

    print("Applying shear strains and calculating energy/stress...")
    for strain in strains:
        deformed_atoms = get_cart_deformed_cell(atoms=initial_atoms, axis=shear_axis, size=strain)
        deformed_atoms.calc = calculator
        
        # Energy relative to the base structure
        energy = deformed_atoms.get_potential_energy()
        energies.append(energy - base_energy)
        
        # Get the Voigt stress component for yz shear
        stress_voigt = full_3x3_to_voigt_6_stress(deformed_atoms.get_stress(voigt=False))
        stresses_yz.append(stress_voigt[shear_axis] * EV_A3_TO_GPA)

    # 4. Fit and Plot
    # Fit a quadratic to the energy-strain data: E = a*x^2 + b*x + c
    # The second derivative (2a) is related to the elastic constant.
    energy_fit_coeffs = np.polyfit(strains, energies, 2)
    energy_fit_curve = np.polyval(energy_fit_coeffs, strains)
    
    # Fit a line to the stress-strain data: Stress = m*x + c
    # The slope (m) is related to the elastic constant.
    stress_fit_coeffs = np.polyfit(strains, stresses_yz, 1)
    stress_fit_line = np.polyval(stress_fit_coeffs, strains)
    
    C44_from_stress_fit = stress_fit_coeffs[0] / 2.0  # C44 = (d(sigma_yz) / d(gamma_yz)) / 2

    print(f"\n--- Results for {model_config['name']} ---")
    print(f"Quadratic fit to Energy vs Strain (E = ax^2+bx+c): a = {energy_fit_coeffs[0]:.4f}")
    print(f"Linear fit to Stress vs Strain (S = mx+c):   m = {stress_fit_coeffs[0]:.4f}")
    print(f"C44 estimated from stress plot slope: {C44_from_stress_fit:.2f} GPa")


    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Shear Stability Analysis for {model_config["name"]}', fontsize=16)

    # Energy vs. Strain Plot
    ax1.plot(strains, energies, 'o', label='Calculated Data')
    ax1.plot(strains, energy_fit_curve, '-', label=f'Quadratic Fit (Curvature a={energy_fit_coeffs[0]:.2f})')
    ax1.set_xlabel('Shear Strain ($\\gamma_{yz}$)')
    ax1.set_ylabel('Relative Energy (eV)')
    ax1.set_title('Energy vs. Shear Strain')
    ax1.grid(True)
    ax1.legend()
    # Annotate stability
    stability_text = "Stable (a > 0)" if energy_fit_coeffs[0] > 0 else "Unstable (a < 0)"
    ax1.text(0.05, 0.95, stability_text, transform=ax1.transAxes, 
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

    # Stress vs. Strain Plot
    ax2.plot(strains, stresses_yz, 'o', label='Calculated Data')
    ax2.plot(strains, stress_fit_line, '-', label=f'Linear Fit (Slope m={stress_fit_coeffs[0]:.2f})')
    ax2.set_xlabel('Shear Strain ($\\gamma_{yz}$)')
    ax2.set_ylabel('Shear Stress ($\\sigma_{yz}$) [GPa]')
    ax2.set_title('Stress vs. Shear Strain')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = OUTPUT_DIR / f'shear_stability_{model_config["name"].replace(" ", "_").lower()}.png'
    plt.savefig(output_filename)
    print(f"Saved diagnostic plot to: {output_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose shear stability (C44) for a given MLIP model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS_CONFIG.keys(),
        help="The model to check."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to run the calculation on (e.g., 'cuda' or 'cpu')."
    )
    args = parser.parse_args()

    model_config = MODELS_CONFIG[args.model]
    check_shear_stability(model_config, args.device) 