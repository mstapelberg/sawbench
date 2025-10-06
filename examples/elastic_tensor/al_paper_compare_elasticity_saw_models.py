import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

# Set global font and font sizes - use Helvetica and increase all by 2 points
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Load the JSON data
RESULTS_DIR = Path("results")
json_file = Path("results/saw_cdf_nequip_summary.json")
with open(json_file, 'r') as f:
    data = json.load(f)

# Extract the models we want to compare
model_names = {
    'mse': 'mse_lmax2_nlayers2_mlp512_zbl_epoch107.nequip.zip',
    'msetw': 'msetw_lmax2_nlayers2_mlp512_nlh.nequip.zip',
    'ca': 'ca_lmax2_nlayers2_mlp512_nlh_epoch169.nequip.zip',
    'catw': 'catw_lmax2_nlayers2_mlp512_nlh_epoch128.nequip.zip',
    'DFT': 'DFT',
    'PredEXP': 'PredEXP',
    'EXP': 'EXP'
}

# Get the model data
models_data = {}
for key, model_name in model_names.items():
    if model_name in data['model_results']:
        models_data[key] = data['model_results'][model_name]
    elif model_name in data['reference_data']:
        models_data[key] = data['reference_data'][model_name]
    else:
        print(f"Warning: Model {model_name} not found")
        continue

def extract_cubic_constants(model_data):
    """Return dict with keys C11, C12, C44 if available, else None."""
    if 'cubic_constants_gpa' in model_data:
        cc = model_data['cubic_constants_gpa']
        if all(k in cc and cc[k] is not None for k in ('C11', 'C12', 'C44')):
            return {'C11': cc['C11'], 'C12': cc['C12'], 'C44': cc['C44']}
    if 'elasticity' in model_data:
        el = model_data['elasticity']
        c11 = el.get('C11')
        c12 = el.get('C12')
        # Prefer cubic C44 if present; otherwise Voigt C44
        c44 = el.get('C44')
        if c11 is not None and c12 is not None and c44 is not None:
            return {'C11': c11, 'C12': c12, 'C44': c44}
    return None

def is_mechanically_stable(constants):
    """Check cubic stability: C44>0, C11-C12>0, C11+2*C12>0."""
    if constants is None:
        return False
    c11 = constants.get('C11')
    c12 = constants.get('C12')
    c44 = constants.get('C44')
    if c11 is None or c12 is None or c44 is None:
        return False
    return (c44 > 0) and ((c11 - c12) > 0) and ((c11 + 2 * c12) > 0)

# Function to plot elastic constants
def plot_elastic_constants(models_data):
    """Plot C11, C12, C44 values for each model in horizontal bar chart"""

    # Extract constants
    c11_values = []
    c12_values = []
    c44_values = []
    model_labels = []

    for model_name, model_data in models_data.items():
        constants = extract_cubic_constants(model_data)
        if constants:
            c11_values.append(constants['C11'])
            c12_values.append(constants['C12'])
            c44_values.append(constants['C44'])
            model_labels.append(model_name.upper())

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(model_labels))
    bar_height = 0.25

    # Plot bars
    bars1 = ax.barh(y_pos - bar_height, c11_values, bar_height, label='C11', color='#2A33C3')
    bars2 = ax.barh(y_pos, c12_values, bar_height, label='C12', color='#8F2D56')
    bars3 = ax.barh(y_pos + bar_height, c44_values, bar_height, label='C44', color='#6E8B00')

    # Add values on bars
    for i, (c11, c12, c44) in enumerate(zip(c11_values, c12_values, c44_values)):
        ax.text(c11 + 1, y_pos[i] - bar_height, f'{c11:.1f}', va='center', fontsize=9)
        ax.text(c12 + 1, y_pos[i], f'{c12:.1f}', va='center', fontsize=9)
        ax.text(c44 + 1, y_pos[i] + bar_height, f'{c44:.1f}', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Elastic Constants (GPa)', fontsize=12, fontweight='bold')
    #ax.set_title('Elastic Constants Comparison: C11, C12, C44')
    ax.legend()

    plt.tight_layout()
    return fig

# Function to plot histograms
def plot_histograms(models_data):
    """Plot overlaid frequency histograms for all models"""

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00']

    # Plot EXP first with lower alpha so other models can stack on top
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])
        exp_mean = np.mean(exp_freqs)
        exp_std = np.std(exp_freqs)
        ax.hist(exp_freqs, bins=50, alpha=0.2,
               color='black', edgecolor='black', linewidth=0.5,
               label='EXP')

    # Plot other models (skip mechanically unstable)
    plotted = 0
    for model_name, model_data in models_data.items():
        if model_name == 'EXP' or 'frequencies_mhz' not in model_data:
            continue

        constants = extract_cubic_constants(model_data)
        if not is_mechanically_stable(constants):
            continue

        freqs = np.array(model_data['frequencies_mhz'])

        ax.hist(freqs, bins=50, alpha=0.7,
                color=colors[plotted % len(colors)],
                edgecolor='none', linewidth=0,
                label=f'{model_name.upper()}')

        mean_freq = np.mean(freqs)
        std_freq = np.std(freqs)

        ax.text(0.98, 0.98 - plotted*0.08, f'{model_name.upper()}: μ={mean_freq:.1f}±{std_freq:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors[plotted % len(colors)]))
        plotted += 1

    # Add EXP statistics to the text area (at the bottom)
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        num_models = sum(1 for m, d in models_data.items() if m != 'EXP' and 'frequencies_mhz' in d and is_mechanically_stable(extract_cubic_constants(d)))
        ax.text(0.98, 0.98 - num_models*0.08 - 0.08,
               f'EXP: μ={exp_mean:.1f}±{exp_std:.1f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Count')
    ax.set_title('Frequency Distribution Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Function to plot CDFs
def plot_cdfs(models_data):
    """Plot overlaid cumulative distribution functions for all models"""

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00']

    # Always plot EXP (no stability check)
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])
        exp_sorted = np.sort(exp_freqs)
        y_exp = np.arange(1, len(exp_freqs) + 1) / len(exp_freqs)
        ax.plot(exp_sorted, y_exp, color='black', linewidth=2, label='EXP')
        exp_mean = np.mean(exp_freqs)
        exp_median = np.median(exp_freqs)
        ax.axvline(exp_mean, color='black', linestyle='--', alpha=0.5)
        ax.axvline(exp_median, color='black', linestyle=':', alpha=0.5)

    plotted = 0
    for model_name, model_data in models_data.items():
        if 'frequencies_mhz' not in model_data or model_name == 'EXP':
            continue
        constants = extract_cubic_constants(model_data)
        if not is_mechanically_stable(constants):
            continue
        freqs = np.array(model_data['frequencies_mhz'])
        freqs_sorted = np.sort(freqs)

        y = np.arange(1, len(freqs) + 1) / len(freqs)

        ax.plot(freqs_sorted, y, color=colors[plotted % len(colors)], linewidth=2,
                label=f'{model_name.upper()}')

        mean_freq = np.mean(freqs)
        median_freq = np.median(freqs)
        ax.axvline(mean_freq, color=colors[plotted % len(colors)], linestyle='--', alpha=0.5)
        ax.axvline(median_freq, color=colors[plotted % len(colors)], linestyle=':', alpha=0.5)
        plotted += 1

    ax.set_xlabel('Frequency (MHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    #ax.set_title('Cumulative Distribution Function Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add summary statistics table in the upper left
    summary_text = "Model Statistics:\n"
    # Always include EXP in summary if present
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])
        summary_text += f"EXP: μ={np.mean(exp_freqs):.1f}, med={np.median(exp_freqs):.1f}\n"
    for model_name, model_data in models_data.items():
        if model_name == 'EXP':
            continue
        if 'frequencies_mhz' in model_data and is_mechanically_stable(extract_cubic_constants(model_data)):
            freqs = np.array(model_data['frequencies_mhz'])
            mean_freq = np.mean(freqs)
            median_freq = np.median(freqs)
            summary_text += f"{model_name.upper()}: μ={mean_freq:.1f}, med={median_freq:.1f}\n"

    ax.text(0.99, 0.2, summary_text, transform=ax.transAxes,
           verticalalignment='bottom', horizontalalignment='right', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    return fig

# Create the plots
fig1 = plot_elastic_constants(models_data)
fig2 = plot_histograms(models_data)
fig3 = plot_cdfs(models_data)

# Show the plots
plt.show()

# Optionally save the plots
fig1.savefig(os.path.join(RESULTS_DIR, 'elastic_constants_comparison.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(RESULTS_DIR, 'frequency_histograms.png'), dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(RESULTS_DIR, 'frequency_cdfs.png'), dpi=300, bbox_inches='tight')

print("Plots created successfully!")
print("Available models:", list(models_data.keys()))