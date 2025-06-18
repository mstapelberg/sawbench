from ase.io import read
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


mlip = 'mace'
atoms_list = read('../data/job_gen_7-2025-04-29_test.xyz', index=':')
if mlip == 'nequip':
    from nequip.ase import NequIPCalculator
    calc = NequIPCalculator.from_compiled_model(compile_path='../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2',device='cuda')
elif mlip == 'mace':
    from mace.calculators import mace_mp
    calc = mace_mp(model='medium', device='cuda', default_dtype='float64', use_cueq=True)
else:
    raise ValueError(f"Invalid MLIP backend: {mlip}")

ref_force = []
ref_energy_per_atom = []
ref_stress = []

pred_force = []
pred_energy_per_atom = []
pred_stress = []

# Store per-structure data for error analysis
structure_data = []

for i, atoms in enumerate(atoms_list):
    if i % 100 == 0:
        print(f"Processing {i} of {len(atoms_list)}")
    
    # Reference data
    ref_forces_struct = atoms.arrays['REF_force']
    ref_energy_struct = atoms.info['REF_energy']/len(atoms)
    ref_stresses_struct = atoms.info['REF_stress']
    
    # Predictions
    atoms.calc = calc
    pred_forces_struct = atoms.get_forces()
    pred_energy_struct = atoms.get_potential_energy()/len(atoms)
    pred_stresses_struct = atoms.get_stress(voigt=False).flatten()
    
    # Calculate per-structure RMSE
    force_rmse_struct = np.sqrt(np.mean((ref_forces_struct - pred_forces_struct)**2))
    energy_rmse_struct = abs(ref_energy_struct - pred_energy_struct)
    stress_rmse_struct = np.sqrt(np.mean((ref_stresses_struct - pred_stresses_struct)**2))
    
    # Store structure-level data
    structure_data.append({
        'index': i,
        'force_rmse': force_rmse_struct,
        'energy_rmse': energy_rmse_struct,
        'stress_rmse': stress_rmse_struct,
        'n_atoms': len(atoms)
    })
    
    # Store flattened data for overall analysis
    ref_force.extend(ref_forces_struct.flatten())
    ref_energy_per_atom.append(ref_energy_struct)
    ref_stress.extend(ref_stresses_struct.flatten())
    
    pred_force.extend(pred_forces_struct.flatten())
    pred_energy_per_atom.append(pred_energy_struct)
    pred_stress.extend(pred_stresses_struct.flatten())

# Convert to numpy arrays
ref_force = np.array(ref_force)
pred_force = np.array(pred_force)
ref_energy_per_atom = np.array(ref_energy_per_atom)
pred_energy_per_atom = np.array(pred_energy_per_atom)
ref_stress = np.array(ref_stress)
pred_stress = np.array(pred_stress)

# Calculate overall RMSE and R²
def calculate_metrics(ref, pred):
    rmse = np.sqrt(np.mean((ref - pred)**2))
    r2 = r2_score(ref, pred)
    return rmse, r2

rmse_force, r2_force = calculate_metrics(ref_force, pred_force)
rmse_stress, r2_stress = calculate_metrics(ref_stress, pred_stress)
rmse_energy, r2_energy = calculate_metrics(ref_energy_per_atom, pred_energy_per_atom)

print(f"Overall RMSE Force: {rmse_force:.4f}, R² Force: {r2_force:.4f}")
print(f"Overall RMSE Stress: {rmse_stress:.4f}, R² Stress: {r2_stress:.4f}")
print(f"Overall RMSE Energy per Atom: {rmse_energy:.4f}, R² Energy per Atom: {r2_energy:.4f}")

# Extract per-structure RMSE arrays
force_rmse_per_struct = np.array([s['force_rmse'] for s in structure_data])
energy_rmse_per_struct = np.array([s['energy_rmse'] for s in structure_data])
stress_rmse_per_struct = np.array([s['stress_rmse'] for s in structure_data])

# Identify top N structures with highest errors
N = 50
def get_top_error_structures(rmse_array, metric_name):
    top_indices = np.argsort(rmse_array)[-N:][::-1]  # Get indices of N highest errors
    print(f"\nTop {N} structures with highest {metric_name} RMSE:")
    print(f"{'Rank':<5} {'Structure Index':<15} {'RMSE':<12} {'N_atoms':<8}")
    print("-" * 45)
    for rank, idx in enumerate(top_indices, 1):
        struct_data = structure_data[idx]
        print(f"{rank:<5} {struct_data['index']:<15} {rmse_array[idx]:<12.6f} {struct_data['n_atoms']:<8}")
    return top_indices

force_top_indices = get_top_error_structures(force_rmse_per_struct, "Force")
energy_top_indices = get_top_error_structures(energy_rmse_per_struct, "Energy")
stress_top_indices = get_top_error_structures(stress_rmse_per_struct, "Stress")

# Create comprehensive plots
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 12))

# Top row: Parity plots
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(ref_force, pred_force, alpha=0.6, s=15, c='#1f77b4', edgecolors='none')
force_min, force_max = min(ref_force.min(), pred_force.min()), max(ref_force.max(), pred_force.max())
ax1.plot([force_min, force_max], [force_min, force_max], 'r--', lw=2, alpha=0.8, label='Perfect correlation')
ax1.set_xlabel('Reference Force (eV/Å)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Force (eV/Å)', fontsize=11, fontweight='bold')
ax1.set_title(f'Force Parity Plot\nOverall RMSE = {rmse_force:.4f}, R² = {r2_force:.4f}', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')

ax2 = plt.subplot(2, 3, 2)
ax2.scatter(ref_stress, pred_stress, alpha=0.6, s=15, c='#ff7f0e', edgecolors='none')
stress_min, stress_max = min(ref_stress.min(), pred_stress.min()), max(ref_stress.max(), pred_stress.max())
ax2.plot([stress_min, stress_max], [stress_min, stress_max], 'r--', lw=2, alpha=0.8, label='Perfect correlation')
ax2.set_xlabel('Reference Stress (GPa)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Stress (GPa)', fontsize=11, fontweight='bold')
ax2.set_title(f'Stress Parity Plot\nOverall RMSE = {rmse_stress:.4f}, R² = {r2_stress:.4f}', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_aspect('equal', adjustable='box')

ax3 = plt.subplot(2, 3, 3)
ax3.scatter(ref_energy_per_atom, pred_energy_per_atom, alpha=0.6, s=15, c='#2ca02c', edgecolors='none')
energy_min, energy_max = min(ref_energy_per_atom.min(), pred_energy_per_atom.min()), max(ref_energy_per_atom.max(), pred_energy_per_atom.max())
ax3.plot([energy_min, energy_max], [energy_min, energy_max], 'r--', lw=2, alpha=0.8, label='Perfect correlation')
ax3.set_xlabel('Reference Energy per Atom (eV)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Energy per Atom (eV)', fontsize=11, fontweight='bold')
ax3.set_title(f'Energy Parity Plot\nOverall RMSE = {rmse_energy:.4f}, R² = {r2_energy:.4f}', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')

# Bottom row: RMSE histograms
ax4 = plt.subplot(2, 3, 4)
n_bins = min(50, len(force_rmse_per_struct)//5)
ax4.hist(force_rmse_per_struct, bins=n_bins, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5)
ax4.axvline(np.mean(force_rmse_per_struct), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(force_rmse_per_struct):.4f}')
ax4.axvline(np.median(force_rmse_per_struct), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(force_rmse_per_struct):.4f}')
ax4.set_xlabel('Force RMSE per Structure (eV/Å)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title(f'Force RMSE Distribution\n({len(force_rmse_per_struct)} structures)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

ax5 = plt.subplot(2, 3, 5)
n_bins = min(50, len(stress_rmse_per_struct)//5)
ax5.hist(stress_rmse_per_struct, bins=n_bins, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=0.5)
ax5.axvline(np.mean(stress_rmse_per_struct), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(stress_rmse_per_struct):.4f}')
ax5.axvline(np.median(stress_rmse_per_struct), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(stress_rmse_per_struct):.4f}')
ax5.set_xlabel('Stress RMSE per Structure (GPa)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title(f'Stress RMSE Distribution\n({len(stress_rmse_per_struct)} structures)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

ax6 = plt.subplot(2, 3, 6)
n_bins = min(50, len(energy_rmse_per_struct)//5)
ax6.hist(energy_rmse_per_struct, bins=n_bins, alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=0.5)
ax6.axvline(np.mean(energy_rmse_per_struct), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(energy_rmse_per_struct):.4f}')
ax6.axvline(np.median(energy_rmse_per_struct), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(energy_rmse_per_struct):.4f}')
ax6.set_xlabel('Energy RMSE per Structure (eV/atom)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title(f'Energy RMSE Distribution\n({len(energy_rmse_per_struct)} structures)', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()

plt.tight_layout()
plt.savefig('comprehensive_parity_analysis_test_macemp0.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Total structures analyzed: {len(structure_data)}")
print(f"\nForce RMSE per structure:")
print(f"  Mean: {np.mean(force_rmse_per_struct):.6f} eV/Å")
print(f"  Median: {np.median(force_rmse_per_struct):.6f} eV/Å")
print(f"  Std: {np.std(force_rmse_per_struct):.6f} eV/Å")
print(f"  Max: {np.max(force_rmse_per_struct):.6f} eV/Å")

print(f"\nStress RMSE per structure:")
print(f"  Mean: {np.mean(stress_rmse_per_struct):.6f} GPa")
print(f"  Median: {np.median(stress_rmse_per_struct):.6f} GPa")
print(f"  Std: {np.std(stress_rmse_per_struct):.6f} GPa")
print(f"  Max: {np.max(stress_rmse_per_struct):.6f} GPa")

print(f"\nEnergy RMSE per structure:")
print(f"  Mean: {np.mean(energy_rmse_per_struct):.6f} eV/atom")
print(f"  Median: {np.median(energy_rmse_per_struct):.6f} eV/atom")
print(f"  Std: {np.std(energy_rmse_per_struct):.6f} eV/atom")
print(f"  Max: {np.max(energy_rmse_per_struct):.6f} eV/atom")




