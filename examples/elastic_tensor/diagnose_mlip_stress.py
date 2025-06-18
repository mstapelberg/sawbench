#!/usr/bin/env python3
"""
Diagnostic script to test MLIP stress calculations
=================================================

This script tests the MLIP model's ability to calculate stresses
and compares with VASP reference values.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
from ase import Atoms
from nequip.ase import NequIPCalculator
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

# Conversion factor from eV/Å^3 to GPa
EV_A3_TO_GPA = 160.21766208

# --- Configuration ---
VASP_OUTCAR_PATH = "../data/PBE_Manual_Elastic/reference/OUTCAR"
NEQUIP_MODEL_PATH = "../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2"
DEVICE = "cuda"

def run_mlip_relaxation(atoms, calculator):
    """
    Performs a full relaxation (cell and positions) with an MLIP calculator
    and returns the trajectory of key physical quantities.
    """
    print("\n--- Running MLIP Relaxation ---")
    
    # Use a copy to avoid modifying the original atoms object
    atoms_to_relax = atoms.copy()
    atoms_to_relax.calc = calculator

    # Lists to store trajectory data
    trajectory_data = {
        'energies': [],
        'fmax_values': [],
        'pressures_gpa': []
    }

    def log_step(a=atoms_to_relax):
        """Callback function to log data at each optimization step."""
        energy = a.get_potential_energy()
        forces = a.get_forces()
        stress = a.get_stress(voigt=False)
        
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        pressure_gpa = -np.trace(stress) / 3.0 * EV_A3_TO_GPA
        
        trajectory_data['energies'].append(energy)
        trajectory_data['fmax_values'].append(fmax)
        trajectory_data['pressures_gpa'].append(pressure_gpa)
        
        step = len(trajectory_data['energies'])
        print(f"  MLIP Step {step:3d}: E = {energy:10.4f} eV,  f_max = {fmax:8.4f} eV/Å,  P = {pressure_gpa:8.4f} GPa")

    # Set up the optimizer
    cell_filter = FrechetCellFilter(atoms_to_relax)
    optimizer = FIRE(cell_filter)
    
    # Attach the logger
    optimizer.attach(log_step, interval=1)
    
    # Run the optimization (a bit loose to see the trend)
    optimizer.run(fmax=1e-3, steps=100)
    
    return trajectory_data

def evaluate_mlip_on_vasp_traj(vasp_traj, calculator):
    """
    Evaluates the MLIP energy, forces, and stress on each frame of a
    pre-calculated VASP trajectory.
    """
    print("\n--- Evaluating MLIP on VASP Trajectory ---")
    mlip_on_vasp_data = {
        'energies': [],
        'fmax_values': [],
        'pressures_gpa': []
    }

    for i, atoms in enumerate(vasp_traj):
        atoms.calc = calculator
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        pressure_gpa = -np.trace(stress) / 3.0 * EV_A3_TO_GPA
        
        mlip_on_vasp_data['energies'].append(energy)
        mlip_on_vasp_data['fmax_values'].append(fmax)
        mlip_on_vasp_data['pressures_gpa'].append(pressure_gpa)
        
        print(f"  VASP Step {i+1:3d}: MLIP E = {energy:10.4f} eV,  MLIP f_max = {fmax:8.4f} eV/Å,  MLIP P = {pressure_gpa:8.4f} GPa")
        
    return mlip_on_vasp_data

def get_vasp_trajectory_data(vasp_traj):
    """Extracts energy, fmax, and pressure from a VASP trajectory."""
    print("\n--- Extracting Data from VASP Trajectory ---")
    vasp_data = {
        'energies': [],
        'fmax_values': [],
        'pressures_gpa': []
    }
    for i, atoms in enumerate(vasp_traj):
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        
        fmax = np.sqrt((forces ** 2).sum(axis=1).max())
        pressure_gpa = -np.trace(stress) / 3.0 * EV_A3_TO_GPA

        vasp_data['energies'].append(energy)
        vasp_data['fmax_values'].append(fmax)
        vasp_data['pressures_gpa'].append(pressure_gpa)
        print(f"  VASP Step {i+1:3d}: VASP E = {energy:10.4f} eV,  VASP f_max = {fmax:8.4f} eV/Å,  VASP P = {pressure_gpa:8.4f} GPa")

    return vasp_data

def plot_trajectories(vasp_data, mlip_relax_data, mlip_on_vasp_data):
    """
    Plots the comparison of VASP vs. MLIP relaxation trajectories.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot Energies
    axs[0].plot(vasp_data['energies'], '-o', label='VASP Relaxation', markersize=4)
    axs[0].plot(mlip_relax_data['energies'], '-s', label='MLIP Relaxation', markersize=4)
    axs[0].plot(mlip_on_vasp_data['energies'], '--^', label='MLIP on VASP Traj.', markersize=4, alpha=0.7)
    axs[0].set_ylabel('Potential Energy (eV)')
    axs[0].legend()
    axs[0].set_title('Energy during Relaxation')

    # Plot Max Forces
    axs[1].plot(vasp_data['fmax_values'], '-o', label='VASP Relaxation', markersize=4)
    axs[1].plot(mlip_relax_data['fmax_values'], '-s', label='MLIP Relaxation', markersize=4)
    axs[1].plot(mlip_on_vasp_data['fmax_values'], '--^', label='MLIP on VASP Traj.', markersize=4, alpha=0.7)
    axs[1].set_ylabel('Max Force (eV/Å)')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_title('Max Force during Relaxation')

    # Plot Pressures
    axs[2].plot(vasp_data['pressures_gpa'], '-o', label='VASP Relaxation', markersize=4)
    axs[2].plot(mlip_relax_data['pressures_gpa'], '-s', label='MLIP Relaxation', markersize=4)
    axs[2].plot(mlip_on_vasp_data['pressures_gpa'], '--^', label='MLIP on VASP Traj.', markersize=4, alpha=0.7)
    axs[2].axhline(0, color='k', linestyle='--', linewidth=1)
    axs[2].set_xlabel('Relaxation Step')
    axs[2].set_ylabel('Pressure (GPa)')
    axs[2].legend()
    axs[2].set_title('Pressure during Relaxation')
    
    plt.tight_layout()
    plt.savefig("relaxation_trajectory_comparison.png")
    print("\nSaved diagnostic plot to relaxation_trajectory_comparison.png")
    plt.show()

def main():
    """
    Main function to run the diagnostic comparison.
    """
    # 1. Load VASP relaxation trajectory
    vasp_trajectory = read(VASP_OUTCAR_PATH, index=":")
    print(f"Loaded {len(vasp_trajectory)} frames from VASP OUTCAR.")
    
    # 2. Initialize MLIP calculator
    calculator = NequIPCalculator.from_compiled_model(
        compile_path=NEQUIP_MODEL_PATH, device=DEVICE
    )
    
    # 3. Get data from VASP trajectory
    vasp_data = get_vasp_trajectory_data(vasp_trajectory)

    # 4. Run MLIP relaxation from the same starting point
    initial_atoms = vasp_trajectory[0]
    mlip_relax_data = run_mlip_relaxation(initial_atoms, calculator)
    
    # 5. Evaluate the MLIP on the VASP trajectory frames
    mlip_on_vasp_data = evaluate_mlip_on_vasp_traj(vasp_trajectory, calculator)
    
    # 6. Plot the results
    plot_trajectories(vasp_data, mlip_relax_data, mlip_on_vasp_data)
    
if __name__ == "__main__":
    main() 