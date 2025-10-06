import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set global font and font sizes - use Helvetica and increase all by 2 points
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

def format_parameter_label(param_name: str) -> str:
    """Convert parameter names to proper Greek symbols with subscripts."""
    # Mapping for parameter names to Greek symbols with subscripts
    param_mapping = {
        'K': r'$\mathbf{K}$',
        'D': r'$\mathbf{D}$', 
        'G': r'$\mathbf{G}$',
        'rho': r'$\boldsymbol{\rho}$',
        'alpha_deg': r'$\boldsymbol{\alpha}_{\mathrm{err}}$',
        'beta_deg': r'$\boldsymbol{\beta}_{\mathrm{err}}$',
        'gamma_deg': r'$\boldsymbol{\gamma}_{\mathrm{err}}$',
        'psi_err_deg': r'$\boldsymbol{\Psi}_{\mathrm{err}}$'
    }
    
    return param_mapping.get(param_name, f'${param_name}$')

def plot_sensitivity_bars(csv_path: str, *, outfile: str | None = None) -> None:
    """Plot sorted horizontal bars of sensitivity indices from a CSV.

    Expects columns: parameter, S1, S1_conf, ST, ST_conf
    - Bars: S1 (solid), with error caps of +/- S1_conf
    - Behind each S1 bar, draw a faint outline bar indicating ST
    Sorted by ST descending.
    """
    # Define color scheme
    color_scheme = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56", "#6E8B00"]
    
    df = pd.read_csv(csv_path)
    required = {"parameter", "S1", "S1_conf", "ST", "ST_conf"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Sort by ST DESC, then S1 DESC
    df_sorted = df.sort_values(["ST", "S1"], ascending=[False, False]).reset_index(drop=True)

    params = df_sorted["parameter"].astype(str).to_list()
    S1 = df_sorted["S1"].to_numpy(dtype=float)
    S1_err = df_sorted["S1_conf"].to_numpy(dtype=float)
    ST = df_sorted["ST"].to_numpy(dtype=float)

    y = np.arange(len(params))

    fig, ax = plt.subplots(1, 1, figsize=(8, max(3.5, 0.5 * len(params))))

    # ST faint outline behind
    ax.barh(y, ST, color="none", edgecolor="0.7", linewidth=2.0, height=0.7, label="ST (outline)")

    # S1 bars with color scheme - cycle through colors for each parameter
    bar_colors = [color_scheme[i % len(color_scheme)] for i in range(len(params))]
    ax.barh(y, S1, color=bar_colors, alpha=0.9, height=0.5, label="S1")

    # Error caps for S1 - use same colors as bars
    for i, (s1_val, s1_err_val, y_pos) in enumerate(zip(S1, S1_err, y)):
        ax.errorbar(s1_val, y_pos, xerr=s1_err_val, fmt='none', 
                   ecolor=bar_colors[i], elinewidth=1.5, capsize=3)

    ax.set_xlabel("Sensitivity Index", fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    # Format parameter names to Greek symbols and make them bold
    formatted_params = [format_parameter_label(param) for param in params]
    ax.set_yticklabels(formatted_params, fontweight='bold')
    ax.invert_yaxis()  # highest at top
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.legend(loc="lower right")
    #ax.set_title("Global Sensitivity Indices (S1 with CI; ST outline)")
    fig.tight_layout()

    if outfile is None:
        # Save beside the CSV by default
        base = os.path.splitext(csv_path)[0]
        outfile = f"{base}_bars.png"
    fig.savefig(outfile, dpi=200)
    print(f"Saved: {os.path.abspath(outfile)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot S1/ST sensitivity bars from CSV.")
    parser.add_argument("csv", help="Path to sensitivity_analysis.csv")
    parser.add_argument("--out", default=None, help="Output image path (png). Optional")
    args = parser.parse_args()

    plot_sensitivity_bars(args.csv, outfile=args.out)


