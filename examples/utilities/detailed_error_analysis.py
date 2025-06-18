from ase.io import read
from nequip.ase import NequIPCalculator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from collections import Counter, defaultdict
import pandas as pd
from ase.data import chemical_symbols, atomic_numbers
import seaborn as sns

# Load data
print("Loading data...")
atoms_list = read('../data/job_gen_7-2025-04-29_train.xyz', index=':')
calc = NequIPCalculator.from_compiled_model(compile_path='../data/potentials/gen_7_2025-06-11_r5_stress100.nequip.pt2', device='cuda')

# Store comprehensive data for analysis
structure_data = []
config_type_data = defaultdict(list)

print(f"Analyzing {len(atoms_list)} structures...")

for i, atoms in enumerate(atoms_list):
    if i % 1000 == 0:
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
    
    # Extract detailed structural information
    config_type = atoms.info.get('config_type', 'unknown')
    elements = list(set(atoms.get_chemical_symbols()))
    composition = dict(Counter(atoms.get_chemical_symbols()))
    
    # Calculate additional properties
    cell_volume = atoms.get_volume()
    density = len(atoms) / cell_volume if cell_volume > 0 else 0
    
    # Stress tensor analysis
    stress_tensor = ref_stresses_struct.reshape(3, 3)
    hydrostatic_pressure = np.trace(stress_tensor) / 3
    deviatoric_stress = stress_tensor - np.eye(3) * hydrostatic_pressure
    von_mises_stress = np.sqrt(3/2 * np.sum(deviatoric_stress**2))
    
    # Force analysis
    max_force = np.max(np.linalg.norm(ref_forces_struct, axis=1))
    mean_force = np.mean(np.linalg.norm(ref_forces_struct, axis=1))
    
    # Store comprehensive data
    struct_info = {
        'index': i,
        'force_rmse': force_rmse_struct,
        'energy_rmse': energy_rmse_struct,
        'stress_rmse': stress_rmse_struct,
        'n_atoms': len(atoms),
        'config_type': config_type,
        'elements': sorted(elements),
        'composition': composition,
        'cell_volume': cell_volume,
        'density': density,
        'ref_energy_per_atom': ref_energy_struct,
        'pred_energy_per_atom': pred_energy_struct,
        'hydrostatic_pressure': hydrostatic_pressure,
        'von_mises_stress': von_mises_stress,
        'max_force': max_force,
        'mean_force': mean_force,
        'stress_tensor_trace': np.trace(stress_tensor),
        'stress_tensor_det': np.linalg.det(stress_tensor)
    }
    
    structure_data.append(struct_info)
    config_type_data[config_type].append(struct_info)

print("Analysis complete. Generating detailed reports...")

# Convert to DataFrame for easier analysis
df = pd.DataFrame(structure_data)

# Analysis parameters
N_TOP = 50

def analyze_top_errors(df, error_column, metric_name):
    """Analyze top N structures with highest errors for a given metric"""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: TOP {N_TOP} {metric_name.upper()} ERRORS")
    print(f"{'='*80}")
    
    # Get top error structures
    top_errors = df.nlargest(N_TOP, error_column)
    
    # Basic statistics
    print(f"\n{metric_name} Error Statistics for Top {N_TOP} structures:")
    print(f"Mean: {top_errors[error_column].mean():.6f}")
    print(f"Median: {top_errors[error_column].median():.6f}")
    print(f"Std: {top_errors[error_column].std():.6f}")
    print(f"Min: {top_errors[error_column].min():.6f}")
    print(f"Max: {top_errors[error_column].max():.6f}")
    
    # Config type analysis
    print(f"\nConfig Type Distribution (Top {N_TOP} {metric_name} errors):")
    config_counts = top_errors['config_type'].value_counts()
    total_configs = df['config_type'].value_counts()
    
    for config_type, count in config_counts.items():
        percentage_in_top = (count / N_TOP) * 100
        percentage_in_dataset = (total_configs[config_type] / len(df)) * 100
        enrichment = percentage_in_top / percentage_in_dataset if percentage_in_dataset > 0 else float('inf')
        print(f"  {config_type}: {count} ({percentage_in_top:.1f}% of top errors, "
              f"{percentage_in_dataset:.1f}% of dataset, enrichment: {enrichment:.2f}x)")
    
    # Size distribution analysis
    print(f"\nSystem Size Analysis (Top {N_TOP} {metric_name} errors):")
    size_bins = [0, 10, 50, 100, 150, 200, 300, float('inf')]
    size_labels = ['1-10', '11-50', '51-100', '101-150', '151-200', '201-300', '300+']
    
    top_size_dist = pd.cut(top_errors['n_atoms'], bins=size_bins, labels=size_labels).value_counts()
    total_size_dist = pd.cut(df['n_atoms'], bins=size_bins, labels=size_labels).value_counts()
    
    for size_range in size_labels:
        if size_range in top_size_dist.index and size_range in total_size_dist.index:
            top_count = top_size_dist[size_range]
            total_count = total_size_dist[size_range]
            if total_count > 0:
                enrichment = (top_count / N_TOP) / (total_count / len(df))
                print(f"  {size_range} atoms: {top_count} structures (enrichment: {enrichment:.2f}x)")
    
    # Elemental composition analysis
    print(f"\nElemental Composition Analysis (Top {N_TOP} {metric_name} errors):")
    all_elements = set()
    for elements_list in top_errors['elements']:
        all_elements.update(elements_list)
    
    element_counts = {}
    for element in all_elements:
        count = sum(1 for elements_list in top_errors['elements'] if element in elements_list)
        total_count = sum(1 for elements_list in df['elements'] if element in elements_list)
        enrichment = (count / N_TOP) / (total_count / len(df)) if total_count > 0 else float('inf')
        element_counts[element] = (count, enrichment)
    
    # Sort by enrichment
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1][1], reverse=True)
    for element, (count, enrichment) in sorted_elements:
        print(f"  {element}: {count}/{N_TOP} structures (enrichment: {enrichment:.2f}x)")
    
    # Property correlation analysis
    print(f"\nProperty Correlations with {metric_name} Error:")
    numeric_columns = ['n_atoms', 'density', 'ref_energy_per_atom', 'hydrostatic_pressure', 
                      'von_mises_stress', 'max_force', 'mean_force']
    
    correlations = []
    for col in numeric_columns:
        if col in top_errors.columns:
            corr = top_errors[error_column].corr(top_errors[col])
            correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for prop, corr in correlations:
        print(f"  {prop}: {corr:.4f}")
    
    # Extreme cases analysis
    print(f"\nExtreme Cases Analysis (Top 10 {metric_name} errors):")
    extreme_cases = top_errors.head(10)
    
    for idx, row in extreme_cases.iterrows():
        print(f"\nRank {len(extreme_cases) - list(extreme_cases.index).index(idx)}: Structure {row['index']}")
        print(f"  {metric_name} RMSE: {row[error_column]:.6f}")
        print(f"  Config type: {row['config_type']}")
        print(f"  N_atoms: {row['n_atoms']}")
        print(f"  Elements: {row['elements']}")
        print(f"  Density: {row['density']:.4f} atoms/Ų")
        print(f"  Ref energy/atom: {row['ref_energy_per_atom']:.4f} eV")
        print(f"  Hydrostatic pressure: {row['hydrostatic_pressure']:.4f} GPa")
        print(f"  von Mises stress: {row['von_mises_stress']:.4f} GPa")
        print(f"  Max force: {row['max_force']:.4f} eV/Å")
    
    return top_errors

# Perform detailed analysis for each error type
force_analysis = analyze_top_errors(df, 'force_rmse', 'Force')
energy_analysis = analyze_top_errors(df, 'energy_rmse', 'Energy')
stress_analysis = analyze_top_errors(df, 'stress_rmse', 'Stress')

# Cross-correlation analysis
print(f"\n{'='*80}")
print("CROSS-CORRELATION ANALYSIS")
print(f"{'='*80}")

# Find structures that appear in multiple top-error lists
force_top_indices = set(force_analysis['index'].values)
energy_top_indices = set(energy_analysis['index'].values)
stress_top_indices = set(stress_analysis['index'].values)

overlap_fe = force_top_indices.intersection(energy_top_indices)
overlap_fs = force_top_indices.intersection(stress_top_indices)
overlap_es = energy_top_indices.intersection(stress_top_indices)
overlap_all = force_top_indices.intersection(energy_top_indices).intersection(stress_top_indices)

print(f"Overlap between top {N_TOP} error structures:")
print(f"  Force & Energy: {len(overlap_fe)} structures")
print(f"  Force & Stress: {len(overlap_fs)} structures")
print(f"  Energy & Stress: {len(overlap_es)} structures")
print(f"  All three metrics: {len(overlap_all)} structures")

if overlap_all:
    print(f"\nStructures with high errors in ALL metrics:")
    for idx in overlap_all:
        row = df[df['index'] == idx].iloc[0]
        print(f"  Structure {idx}: config_type={row['config_type']}, "
              f"n_atoms={row['n_atoms']}, elements={row['elements']}")

# Dataset-wide statistics for comparison
print(f"\n{'='*80}")
print("DATASET-WIDE STATISTICS FOR COMPARISON")
print(f"{'='*80}")

print(f"Total structures: {len(df)}")
print(f"Config types: {df['config_type'].value_counts().to_dict()}")
print(f"System sizes: min={df['n_atoms'].min()}, max={df['n_atoms'].max()}, mean={df['n_atoms'].mean():.1f}")
print(f"Energy range: {df['ref_energy_per_atom'].min():.4f} to {df['ref_energy_per_atom'].max():.4f} eV/atom")
print(f"Pressure range: {df['hydrostatic_pressure'].min():.2f} to {df['hydrostatic_pressure'].max():.2f} GPa")
print(f"von Mises stress range: {df['von_mises_stress'].min():.2f} to {df['von_mises_stress'].max():.2f} GPa")

# Error correlation analysis
print(f"\nError Correlations across all structures:")
error_correlations = df[['force_rmse', 'energy_rmse', 'stress_rmse']].corr()
print(error_correlations)

# Save detailed results to CSV for further analysis
detailed_results = df.copy()
detailed_results['elements_str'] = detailed_results['elements'].apply(lambda x: ','.join(x))
detailed_results.drop(['elements', 'composition'], axis=1, inplace=True)

# Add rankings
detailed_results['force_rank'] = detailed_results['force_rmse'].rank(ascending=False)
detailed_results['energy_rank'] = detailed_results['energy_rmse'].rank(ascending=False)
detailed_results['stress_rank'] = detailed_results['stress_rmse'].rank(ascending=False)

detailed_results.to_csv('detailed_error_analysis.csv', index=False)
print(f"\nDetailed results saved to 'detailed_error_analysis.csv'")

# Create summary visualization
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Error distribution comparisons
for i, (error_col, metric_name, color) in enumerate([
    ('force_rmse', 'Force', '#1f77b4'),
    ('energy_rmse', 'Energy', '#2ca02c'), 
    ('stress_rmse', 'Stress', '#ff7f0e')
]):
    ax = axes[0, i]
    top_errors = df.nlargest(N_TOP, error_col)
    
    # Plot distributions
    ax.hist(df[error_col], bins=50, alpha=0.6, label='All structures', color='lightgray', density=True)
    ax.hist(top_errors[error_col], bins=20, alpha=0.8, label=f'Top {N_TOP} errors', color=color, density=True)
    ax.set_xlabel(f'{metric_name} RMSE')
    ax.set_ylabel('Density')
    ax.set_title(f'{metric_name} Error Distribution')
    ax.legend()
    ax.set_yscale('log')

# Config type analysis
for i, (error_col, metric_name, color) in enumerate([
    ('force_rmse', 'Force', '#1f77b4'),
    ('energy_rmse', 'Energy', '#2ca02c'), 
    ('stress_rmse', 'Stress', '#ff7f0e')
]):
    ax = axes[1, i]
    top_errors = df.nlargest(N_TOP, error_col)
    
    # Config type enrichment
    config_counts = top_errors['config_type'].value_counts()
    total_configs = df['config_type'].value_counts()
    
    enrichments = []
    labels = []
    for config_type in config_counts.index:
        if config_type in total_configs.index:
            enrichment = (config_counts[config_type] / N_TOP) / (total_configs[config_type] / len(df))
            enrichments.append(enrichment)
            labels.append(config_type)
    
    bars = ax.bar(range(len(enrichments)), enrichments, color=color, alpha=0.7)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.8, label='No enrichment')
    ax.set_xlabel('Config Type')
    ax.set_ylabel('Enrichment Factor')
    ax.set_title(f'{metric_name} Error Enrichment by Config Type')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

plt.tight_layout()
plt.savefig('detailed_error_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Summary visualization saved to 'detailed_error_analysis_summary.png'")
print(f"Full results available in 'detailed_error_analysis.csv'") 