import numpy as np
import json
import os
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

try:
    from pyace import PyACECalculator
except ImportError:
    PyACECalculator = None
    print("PyACECalculator not found. Please install pyace to use this model.")

# Import from sawbench package
from sawbench import (
    load_ebsd_map,
    calculate_saw_frequencies_for_ebsd_grains,
    Material,
    calculate_elastic_tensor,
    calculate_elastic_tensor_from_vasp,
    relax_atoms,
    BravaisType,
    EV_A3_TO_GPA,
)
from ase.io import read

# --- Static Configuration ---
# Assumes the script is run from the 'examples/ebsd_analysis/' directory.
#EBSD_FILENAME = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected.ctf"
EBSD_FILENAME = 'V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected'

# Try to find the EBSD data file
path1 = f'/home/myless/Documents/saw_freq_analysis/{EBSD_FILENAME}'
path2 = f'../data/{EBSD_FILENAME}' # Relative to ebsd_analysis/
path3 = EBSD_FILENAME              # Relative to ebsd_analysis/

if os.path.exists(path1):
    EBSD_DATA_PATH = path1
    print(f"Found EBSD data at: {path1}")
elif os.path.exists(path2):
    EBSD_DATA_PATH = path2
    print(f"Found EBSD data at: {path2}")
elif os.path.exists(path3):
    EBSD_DATA_PATH = path3
    print(f"Found EBSD data at: {path3}")
else:
    EBSD_DATA_PATH = EBSD_FILENAME # Default to original to reproduce error message
    print(f"Warning: EBSD file '{EBSD_FILENAME}' not found in likely locations.")
    print(f"  - Tried: {path1}")
    print(f"  - Tried: {path2}")
    print(f"  - Tried: {path3}")

EBSD_DATA_TYPE = "OxfordText"
EBSD_BOUNDARY_DEF_ANGLE = 5.0
EBSD_MIN_GRAIN_PX_SIZE = 10
WAVELENGTH_M = 8.8e-6
SAW_CALC_ANGLE_DEG = 135
# Saves results to 'sawbench/results/'
OUTPUT_DIR = Path("results")

# Define all models and their configurations
MODELS_CONFIG = {
    'dft': {
        'name': 'DFT (PBE)',
        'type': 'pre-calculated',
        'vasp_path': "../data/PBE_Manual_Elastic",
        'strain': 0.005,
        'density_g_cm3': 6.11 # Use experimental density
    },
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
    'mace_fromscratch_gen6': {
        'name': 'MACE From Scratch (gen6)',
        'type': 'mlip',
        'calculator_class': MACECalculator,
        'model_path': "../data/potentials/gen_6_model_0_L1_isolated-2026-01-16_stagetwo.model",
        'init_kwargs': {'default_dtype': "float32"}
    },
    'allegro': {
        'name': 'Allegro From Scratch (gen 7)',
        'type': 'mlip',
        'calculator_class': NequIPCalculator,
        'model_path': "../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2",
        'init_kwargs': {}
    },
    'ace': {
        'name': 'ACE TaTiVW',
        'type': 'mlip',
        'calculator_class': PyACECalculator,
        'model_path': "../data/potentials/output_potential.yace",
        'init_kwargs': {}
    },
    'nequip_foundation': {
        'name': 'NequIP Foundation Model',
        'type': 'mlip',
        'calculator_class': NequIPCalculator,
        'model_path': "../data/potentials/Nequip_MPTrj_SK_16_06_25.nequip.pt2",
        'init_kwargs': {}
    },
    'allegro_foundation': {
        'name': 'Allegro Foundation Model',
        'type': 'mlip',
        'calculator_class': NequIPCalculator,
        'model_path': "../data/potentials/Allegro_MPTrj_SK_16_06_25.nequip.pt2",
        'init_kwargs': {}
    }

}

# Unit conversion constants
GPA_TO_PA = 1e9
G_CM3_TO_KG_M3 = 1000

def get_elastic_properties(model_config: dict, device: str) -> dict:
    """
    Gets the elastic properties for a given model configuration.
    For MLIPs, this involves relaxing a structure and calculating the tensor.
    For DFT, it involves parsing VASP outputs.
    """
    if model_config['type'] == 'mlip':
        print(f"\n--- Calculating properties for MLIP: {model_config['name']} ---")
        MLIPCalculatorClass = model_config['calculator_class']
        if MLIPCalculatorClass is None:
            raise ImportError(f"Calculator for '{model_config['name']}' not installed.")

        # Initialize calculator based on its type
        model_path = model_config['model_path']
        if model_config['calculator_class'] == NequIPCalculator:
             calculator = MLIPCalculatorClass.from_compiled_model(model_path, device=device)
        elif model_config['calculator_class'] == MACECalculator: # Finetuned
             calculator = MLIPCalculatorClass(model_paths=[model_path], device=device, **model_config['init_kwargs'])
        elif model_config['calculator_class'] == mace_mp: # Foundation
             calculator = MLIPCalculatorClass(model=model_path, device=device, **model_config['init_kwargs'])
        elif model_config['calculator_class'] == PyACECalculator:
             # PyACECalculator takes the model path as a direct argument.
             # It does not use a 'device' argument like the other calculators.
             calculator = MLIPCalculatorClass(model_path, **model_config['init_kwargs'])
        else:
            raise TypeError(f"Unknown calculator class for {model_config['name']}")

        # Relax initial structure
        initial_atoms = read('../data/PBE_Manual_Elastic/reference/OUTCAR', index=0)
        initial_atoms.calc = calculator
        relaxed_atoms = relax_atoms(initial_atoms, fmax_threshold=1e-4, steps=1000)

        # Calculate elastic tensor
        C_tensor_ev_a3 = calculate_elastic_tensor(
            calculator=calculator,
            atoms=relaxed_atoms,
            bravais_type=BravaisType.CUBIC,
            max_strain_shear=0.005,
        )
        C_tensor_gpa = C_tensor_ev_a3 * EV_A3_TO_GPA
        
        # Calculate density
        volume = relaxed_atoms.get_volume()
        mass = sum(relaxed_atoms.get_masses())
        #density_g_cm3 = mass * 1.66054 / (volume * 1e-24)
        density_g_cm3 = 6.11

        return {
            "C11": C_tensor_gpa[0, 0],
            "C12": C_tensor_gpa[0, 1],
            "C44": C_tensor_gpa[3, 3],
            "density_g_cm3": density_g_cm3
        }

    elif model_config['type'] == 'pre-calculated':
        print(f"\n--- Getting pre-calculated properties for: {model_config['name']} ---")
        C_tensor_gpa = calculate_elastic_tensor_from_vasp(
            directory_path=model_config['vasp_path'],
            strain_amount=model_config['strain'],
            bravais_type=BravaisType.CUBIC,
        )
        return {
            "C11": C_tensor_gpa[0, 0],
            "C12": C_tensor_gpa[0, 1],
            "C44": C_tensor_gpa[3, 3],
            "density_g_cm3": model_config['density_g_cm3']
        }
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate SAW frequency distribution data for a given model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS_CONFIG.keys(),
        help="The model to use for the calculation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to run the calculation on (e.g., 'cuda' or 'cpu')."
    )
    args = parser.parse_args()

    # --- 1. Get Elastic Properties ---
    model_config = MODELS_CONFIG[args.model]
    properties = get_elastic_properties(model_config, args.device)
    
    print("\n--- Calculated/Loaded Properties ---")
    print(f"Model: {model_config['name']}")
    print(f"  C11: {properties['C11']:.2f} GPa")
    print(f"  C12: {properties['C12']:.2f} GPa")
    print(f"  C44: {properties['C44']:.2f} GPa")
    print(f"  Density: {properties['density_g_cm3']:.2f} g/cm³")

    # --- 2. Load EBSD Map ---
    ebsd_map = load_ebsd_map(
        EBSD_DATA_PATH, EBSD_DATA_TYPE, EBSD_BOUNDARY_DEF_ANGLE, EBSD_MIN_GRAIN_PX_SIZE
    )
    if not ebsd_map:
        print("Fatal: Failed to load EBSD data. Exiting.")
        return

    # --- 3. Calculate SAW Frequencies ---
    material = Material(
        formula=model_config['name'],
        C11=properties['C11'] * GPA_TO_PA,
        C12=properties['C12'] * GPA_TO_PA,
        C44=properties['C44'] * GPA_TO_PA,
        density=properties['density_g_cm3'] * G_CM3_TO_KG_M3,
        crystal_class='cubic'
    )
    
    df_ebsd_grains = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=material,
        wavelength=WAVELENGTH_M,
        saw_calc_angle_deg=SAW_CALC_ANGLE_DEG
    )

    predicted_freq_array_hz = np.array([])
    if not df_ebsd_grains.empty and 'Peak SAW Frequency (Hz)' in df_ebsd_grains.columns:
        predicted_freq_array_hz = df_ebsd_grains['Peak SAW Frequency (Hz)'].dropna().to_numpy()
    
    print(f"\nCalculated {len(predicted_freq_array_hz)} predicted SAW frequencies for {len(ebsd_map.grainList)} grains.")

    # --- 4. Save Results to JSON ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"saw_frequencies_{args.model}.json"
    
    results_to_save = {
        "model_name": model_config['name'],
        "model_key": args.model,
        "properties_gpa": {k: v for k, v in properties.items() if k != 'density_g_cm3'},
        "density_kg_m3": properties['density_g_cm3'] * G_CM3_TO_KG_M3,
        "saw_frequencies_hz": predicted_freq_array_hz.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
        
    print(f"\nSuccessfully saved results to: {output_path}")
    print("Data generation complete.")

if __name__ == "__main__":
    main()
