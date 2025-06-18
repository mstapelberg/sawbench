import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk

# Conditional imports for MLIP calculators
try:
    from nequip.ase import NequIPCalculator
except ImportError:
    NequIPCalculator = None

try:
    from mace.calculators import mace_mp
    from mace.calculators import MACECalculator
except ImportError:
    mace_mp = None

# Import from this project
from sawbench.elastic import (
    calculate_elastic_tensor,
    relax_atoms,
    BravaisType,
    calculate_elastic_tensor_from_vasp,
    EV_A3_TO_GPA,
)
from sawbench.materials import Material
from sawbench.saw_calculator import SAWCalculator

# --- Configuration ---
MLIP_BACKEND = 'mace-gen6'  # Options: 'nequip', 'mace'

# Set up the material and MLIP model
ELEMENT = "V"
CRYSTAL_STRUCTURE = "bcc"
LATTICE_CONSTANT = 3.01
DEVICE = "cuda"  # Use 'cuda' for GPU or 'cpu'

MLIP_CONFIG = {
    'nequip': {
        'name': 'NequIP',
        'calculator_class': NequIPCalculator,
        'model_path': "../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2",
    },
    'mace': {
        'name': 'MACE (medium)',
        'calculator_class': mace_mp,
        'model_path': "medium",  # For mace_mp, this is the model name
    },
    'mace-gen6': {
        'name': 'MACE (gen6)',
        'calculator_class': MACECalculator,
        'model_path': "../data/potentials/gen_6_model_0_L1_isolated-2026-01-16-finetuned_fp64_nh10000_lr1e-4_stagetwo.model",  # For mace_mp, this is the model name
    }
}

# Define DFT reference values for Copper (PBE)
# Source: https://materialsproject.org/materials/mp-30
EXP_ELASTIC_CONSTANTS_GPA = {"C11": 230.98, "C12": 120.17, "C44": 43.77}
EXP_DENSITY_G_CM3 = 6.11  # g/cm^3

DFT_DATA_PATH = "../data/PBE_Manual_Elastic"
DFT_STRAIN_AMOUNT = 0.005


# SAW calculation settings
SURFACE_NORMAL = [0, 0, 1]
# For a (001) surface, a 0-degree in-plane angle corresponds to the [100] direction,
# and a 45-degree angle corresponds to the [110] direction.
PROPAGATION_ANGLES_DEG = np.linspace(0, 90, 46)  # From [100] to [010]

# Unit conversion constants
GPA_TO_PA = 1e9
G_CM3_TO_KG_M3 = 1000


def get_mlip_properties(atoms, calculator):
    """
    Relax a structure and calculate its elastic tensor and density using an MLIP.
    """
    print("\n--- Relaxing structure with MLIP ---")
    relaxed_atoms = relax_atoms(atoms, fmax_threshold=1e-4)
    
    print("\n--- Calculating elastic tensor with MLIP ---")
    C_tensor_ev_a3 = calculate_elastic_tensor(
        calculator=calculator,
        atoms=relaxed_atoms,
        bravais_type=BravaisType.CUBIC,
        max_strain_shear=0.005,  # Further reduce shear strain for stability
    )

    # Convert to GPa
    C_tensor_gpa = C_tensor_ev_a3 * EV_A3_TO_GPA

    # Extract Voigt constants
    mlip_C11_gpa = C_tensor_gpa[0, 0]
    mlip_C12_gpa = C_tensor_gpa[0, 1]
    mlip_C44_gpa = C_tensor_gpa[3, 3]

    mlip_constants_gpa = {
        "C11": mlip_C11_gpa,
        "C12": mlip_C12_gpa,
        "C44": mlip_C44_gpa,
    }

    print("\n--- DEBUG: Calculated MLIP Properties ---")
    print(f"MLIP Elastic Constants (GPa): {mlip_constants_gpa}")
    
    # Density of the relaxed structure
    volume = relaxed_atoms.get_volume()  # Angstrom^3
    mass = sum(relaxed_atoms.get_masses())  # amu
    # Convert amu/A^3 to g/cm^3
    density_g_cm3 = mass * 1.66054 / (volume * 1e-24)

    return mlip_constants_gpa, density_g_cm3


def get_dft_properties(directory_path, strain_amount):
    """
    Calculate the elastic tensor and density from a directory of VASP OUTCARs.
    """
    print("\n--- Calculating elastic tensor from VASP outputs ---")
    C_tensor_gpa = calculate_elastic_tensor_from_vasp(
        directory_path=directory_path,
        strain_amount=strain_amount,
        bravais_type=BravaisType.CUBIC,
    )

    dft_constants_gpa = {
        "C11": C_tensor_gpa[0, 0],
        "C12": C_tensor_gpa[0, 1],
        "C44": C_tensor_gpa[3, 3],
    }

    # Read density from the reference OUTCAR
    from ase.io import read
    ref_atoms = read(f"{directory_path}/reference/OUTCAR", format="vasp-out")
    volume = ref_atoms.get_volume()
    mass = sum(ref_atoms.get_masses())
    #density_g_cm3 = mass * 1.66054 / (volume * 1e-24)
    density_g_cm3 = 6.11
    
    return dft_constants_gpa, density_g_cm3


def calculate_saw_speeds(material, angles_deg):
    """
    Calculate SAW speeds for a material over a range of propagation angles.
    """
    # For a (001) surface, the crystal axes are aligned with the sample axes.
    # No rotation is needed.
    euler_angles_rad = np.array([0.0, 0.0, 0.0])
    
    saw_calculator = SAWCalculator(material, euler_angles_rad)
    
    velocities = []
    print(f"Calculating SAW speeds for {material.formula}...")
    for i, angle in enumerate(angles_deg):
        # The get_saw_speed method returns a list of velocities,
        # the primary SAW is typically the first and slowest.
        v_saw, _, _ = saw_calculator.get_saw_speed(deg=angle, sampling=4000)
        velocities.append(v_saw[0] if len(v_saw) > 0 else np.nan)
        print(f"  Angle {angle: >5.1f}°: {velocities[-1]:.0f} m/s ({i+1}/{len(angles_deg)})")

    return np.array(velocities)


def plot_results(angles, exp_speeds, mlip_speeds, dft_speeds, mlip_name):
    """
    Plot the SAW speed comparison graph.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(angles, exp_speeds, "-o", label="EXP (300K Reference)", markersize=4)
    ax.plot(angles, mlip_speeds, "-s", label=mlip_name, markersize=4)
    ax.plot(angles, dft_speeds, "-^", label="DFT (PBE)", markersize=4)

    ax.set_xlabel("Propagation Angle on (001) Surface (degrees)")
    ax.set_ylabel("SAW Velocity (m/s)")
    ax.set_title(f"SAW Velocity in Vanadium ({ELEMENT}) on (001) Surface")
    ax.legend()
    ax.grid(True)
    
    # Annotate key directions
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.text(0.5, 0.95, '[100]', transform=ax.get_xaxis_transform(), ha='left', va='top')
    ax.axvline(45, color='k', linestyle='--', linewidth=0.8)
    ax.text(45/90 + 0.01, 0.95, '[110]', transform=ax.get_xaxis_transform(), ha='left', va='top')
    ax.axvline(90, color='k', linestyle='--', linewidth=0.8)
    ax.text(1.0, 0.95, '[010]', transform=ax.get_xaxis_transform(), ha='right', va='top')

    plt.tight_layout()
    plt.savefig("saw_speed_comparison_v.png")
    print("\nSaved plot to saw_speed_comparison_v.png")
    plt.show()


def main():
    """
    Main workflow to calculate and compare elastic tensors and SAW speeds.
    """
    from ase.io import read
    # 1. CALCULATE MLIP PROPERTIES
    # ---------------------------------
    config = MLIP_CONFIG[MLIP_BACKEND]
    mlip_name = config['name']
    MLIPCalculatorClass = config['calculator_class']

    print(f"--- Initializing {mlip_name.upper()} Calculation ---")
    
    if MLIPCalculatorClass is None:
        raise ImportError(
            f"The calculator for '{MLIP_BACKEND}' is not installed in this environment."
        )

    if MLIP_BACKEND == 'nequip':
        calculator = MLIPCalculatorClass.from_compiled_model(
            compile_path=config['model_path'], device=DEVICE
        )
    elif MLIP_BACKEND == 'mace':
        calculator = MLIPCalculatorClass(
            model=config['model_path'], device=DEVICE, default_dtype="float32", use_cueq=True
        )
    elif MLIP_BACKEND == 'mace-gen6':
        calculator = MLIPCalculatorClass(
            model_paths=[config['model_path']], device=DEVICE, default_dtype="float32", use_cueq=True
        )
    else:
        raise ValueError(f"Invalid MLIP_BACKEND specified: {MLIP_BACKEND}")

    initial_atoms = read('../data/PBE_Manual_Elastic/reference/OUTCAR', index=0)
    initial_atoms.calc = calculator

    mlip_constants_gpa, mlip_density_g_cm3 = get_mlip_properties(
        initial_atoms, calculator
    )
    
    dft_constants_gpa, dft_density_g_cm3 = get_dft_properties(
        DFT_DATA_PATH, DFT_STRAIN_AMOUNT
    )

    # 2. PREPARE MATERIALS FOR SAW CALCULATION
    # ------------------------------------------
    print("\n--- Preparing Materials for SAW Calculator ---")
    # Experimental Material
    exp_material = Material(
        formula=f"{ELEMENT}-EXP",
        C11=EXP_ELASTIC_CONSTANTS_GPA["C11"] * GPA_TO_PA,
        C12=EXP_ELASTIC_CONSTANTS_GPA["C12"] * GPA_TO_PA,
        C44=EXP_ELASTIC_CONSTANTS_GPA["C44"] * GPA_TO_PA,
        density=EXP_DENSITY_G_CM3 * G_CM3_TO_KG_M3,
        crystal_class="cubic",
    )
    print(f"EXP Material: {exp_material}")

    # MLIP Material
    mlip_material = Material(
        formula=f"{ELEMENT}-{mlip_name.upper()}",
        C11=mlip_constants_gpa["C11"] * GPA_TO_PA,
        C12=mlip_constants_gpa["C12"] * GPA_TO_PA,
        C44=mlip_constants_gpa["C44"] * GPA_TO_PA,
        density=EXP_DENSITY_G_CM3 * G_CM3_TO_KG_M3,
        crystal_class="cubic",
    )
    print(f"MLIP Material: {mlip_material}")

    # DFT Material
    dft_material = Material(
        formula=f"{ELEMENT}-DFT",
        C11=dft_constants_gpa["C11"] * GPA_TO_PA,
        C12=dft_constants_gpa["C12"] * GPA_TO_PA,
        C44=dft_constants_gpa["C44"] * GPA_TO_PA,
        density=EXP_DENSITY_G_CM3 * G_CM3_TO_KG_M3,
        crystal_class="cubic",
    )
    print(f"DFT Material: {dft_material}")

    # 3. CALCULATE SAW SPEEDS
    # -------------------------
    exp_saw_speeds = calculate_saw_speeds(exp_material, PROPAGATION_ANGLES_DEG)
    mlip_saw_speeds = calculate_saw_speeds(mlip_material, PROPAGATION_ANGLES_DEG)
    dft_saw_speeds = calculate_saw_speeds(dft_material, PROPAGATION_ANGLES_DEG)

    # 4. PLOT RESULTS
    # -----------------
    print("\n--- Plotting Results ---")
    plot_results(PROPAGATION_ANGLES_DEG, exp_saw_speeds, mlip_saw_speeds, dft_saw_speeds, mlip_name)
    
    # Print summary
    print("\n" + "="*80)
    print("Summary of Results")
    print("="*80)
    header = f"{'Constant':>5} | {'EXP (GPa)':>12} | {mlip_name.upper()+' (GPa)':>15} | {'DFT (GPa)':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'C11':>5} | {EXP_ELASTIC_CONSTANTS_GPA['C11']:12.2f} | {mlip_constants_gpa['C11']:15.2f} | {dft_constants_gpa['C11']:12.2f}")
    print(f"{'C12':>5} | {EXP_ELASTIC_CONSTANTS_GPA['C12']:12.2f} | {mlip_constants_gpa['C12']:15.2f} | {dft_constants_gpa['C12']:12.2f}")
    print(f"{'C44':>5} | {EXP_ELASTIC_CONSTANTS_GPA['C44']:12.2f} | {mlip_constants_gpa['C44']:15.2f} | {dft_constants_gpa['C44']:12.2f}")
    
    print(f"\n{'Property':>10} | {'EXP':>15} | {mlip_name.upper():>15} | {'DFT':>15}")
    print("-" * (len(header)))
    #print(f"{'Density':>10} | {EXP_DENSITY_G_CM3*G_CM3_TO_KG_M3:15.0f} | {mlip_density_g_cm3*G_CM3_TO_KG_M3:15.0f} | {dft_density_g_cm3*G_CM3_TO_KG_M3:15.0f} (kg/m^3)")
    print(f"{'SAW [100]':>10} | {exp_saw_speeds[0]:15.0f} | {mlip_saw_speeds[0]:15.0f} | {dft_saw_speeds[0]:15.0f} (m/s)")
    print(f"{'SAW [110]':>10} | {exp_saw_speeds[45]:15.0f} | {mlip_saw_speeds[45]:15.0f} | {dft_saw_speeds[45]:15.0f} (m/s)")
    print("=" * len(header))


if __name__ == "__main__":
    main() 