#!/usr/bin/env python3
"""
plot_peak_heatmap.py

Loads peak-frequency data from an HDF5 file and corner coordinates from a CSV,
then plots a heatmap with discrete color levels and an overlaid polygon,
and saves the figure.
"""

import argparse
import os
import sys

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import EngFormatter
from scipy import stats # For KDE, IQR, MAD


def load_peak_data(h5_path):
    """
    Load peak frequency data and coordinate vectors from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file containing datasets '/peakFreq', '/X', '/Ycoord'.

    Returns
    -------
    peak_freq : np.ndarray
        2D array of shape (Ny, Nx) with peak frequencies.
    x : np.ndarray
        1D array of length Nx with X positions (mm).
    y : np.ndarray
        1D array of length Ny with Y positions (mm).
    """
    if not os.path.exists(h5_path):
        sys.exit(f"Error: HDF5 file not found: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        peak_freq = f['/peak_freq'][()]
        x = f['/X'][()]
        y = f['/Ycoord'][()]
    return peak_freq, x, y


def load_corners(csv_path):
    """
    Load corner coordinates from a CSV and close the polygon.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns ['Corner','X_coord','Y_coord','Fiducial'].

    Returns
    -------
    xc : np.ndarray
        1D array of X coordinates, closed (first point repeated at end).
    yc : np.ndarray
        1D array of Y coordinates, closed.
    """
    if not os.path.exists(csv_path):
        sys.exit(f"Error: CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {'X_coord', 'Y_coord'}.issubset(df.columns):
        sys.exit(f"Error: CSV must contain 'X_coord' and 'Y_coord' columns.")
    xc = df['X_coord'].to_list()
    yc = df['Y_coord'].to_list()
    # close polygon
    xc.append(xc[0])
    yc.append(yc[0])
    return np.array(xc), np.array(yc)


def plot_heatmap(peak_freq, x, y, xc, yc, vmin, vmax, levels, title=None):
    """
    Create a heatmap of peak_freq over (x, y) with discrete color levels,
    clipped between vmin and vmax, and overlay the corner polygon.

    Parameters
    ----------
    peak_freq : np.ndarray
        2D array of shape (Ny, Nx).
    x : np.ndarray
        1D array of length Nx.
    y : np.ndarray
        1D array of length Ny.
    xc : np.ndarray
        1D array of closed polygon X coords.
    yc : np.ndarray
        1D array of closed polygon Y coords.
    vmin : float
        Lower bound for color scale (Hz).
    vmax : float
        Upper bound for color scale (Hz).
    levels : int
        Number of discrete color bins.
    title : str, optional
        Plot title.
    """
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(8, 6))

    # create discrete colormap
    cmap = plt.get_cmap('viridis', levels)
    norm = BoundaryNorm(np.linspace(vmin, vmax, levels + 1), ncolors=levels)

    mesh = ax.pcolormesh(
        X, Y, peak_freq,
        shading='auto',
        cmap=cmap,
        norm=norm
    )

    formatter = EngFormatter(unit='Hz')
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.linspace(vmin, vmax, min(levels + 1, 11)), format=formatter)
    cbar.set_label("Peak Frequency")

    ax.plot(xc, yc, color='black', linewidth=2, label='Corner Polygon')
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_histogram_and_stats(peak_freq_data, output_base_filename, plot_title_prefix, vmin, vmax):
    """
    Calculates statistics (mean, std dev, median, mode via KDE, IQR, MAD),
    prints them, and plots a histogram of the peak frequency data,
    focusing the histogram display on the [vmin, vmax] range.

    Parameters
    ----------
    peak_freq_data : np.ndarray
        2D array of peak frequencies.
    output_base_filename : str
        Base filename for the output histogram image (e.g., "heatmap").
    plot_title_prefix : str
        Prefix for the plot title.
    vmin : float
        Lower bound for histogram x-axis and data filtering for plot.
    vmax : float
        Upper bound for histogram x-axis and data filtering for plot.
    """
    # Flatten the 2D array and remove NaNs for overall statistics
    valid_peak_freqs_all = peak_freq_data[~np.isnan(peak_freq_data)].flatten()

    if valid_peak_freqs_all.size == 0:
        print("No valid (non-NaN) peak frequency data available for histogram and stats.")
        return

    formatter = EngFormatter(unit='Hz')

    # Calculate traditional and robust statistics on the entire valid dataset
    mean_freq_all = np.mean(valid_peak_freqs_all)
    std_freq_all = np.std(valid_peak_freqs_all)
    median_freq_all = np.median(valid_peak_freqs_all)
    iqr_freq_all = stats.iqr(valid_peak_freqs_all)
    mad_freq_all = stats.median_abs_deviation(valid_peak_freqs_all)

    # Estimate mode using Kernel Density Estimation
    try:
        kde = stats.gaussian_kde(valid_peak_freqs_all)
        # Evaluate KDE over a fine grid within the data range
        kde_x_vals = np.linspace(valid_peak_freqs_all.min(), valid_peak_freqs_all.max(), 500)
        kde_y_vals = kde(kde_x_vals)
        mode_freq_all = kde_x_vals[np.argmax(kde_y_vals)]
        mode_available = True
    except Exception as e: # Catch potential errors in KDE (e.g., if all data points are identical)
        print(f"Warning: Could not estimate mode using KDE: {e}")
        mode_freq_all = np.nan # Or some other placeholder
        mode_available = False


    print(f"\nOverall Peak Frequency Statistics (all valid data):")
    print(f"  Mean: {formatter(mean_freq_all)}")
    print(f"  Standard Deviation: {formatter(std_freq_all)}")
    print(f"  Median: {formatter(median_freq_all)}")
    if mode_available:
        print(f"  Mode (KDE estimate): {formatter(mode_freq_all)}")
    print(f"  Interquartile Range (IQR): {formatter(iqr_freq_all)}")
    print(f"  Median Absolute Deviation (MAD): {formatter(mad_freq_all)}")


    # Filter data for histogram based on vmin and vmax
    ranged_peak_freqs = valid_peak_freqs_all[(valid_peak_freqs_all >= vmin) & (valid_peak_freqs_all <= vmax)]

    if ranged_peak_freqs.size == 0:
        print(f"No peak frequency data available within the specified range [{formatter(vmin)}, {formatter(vmax)}] for histogram.")
        return

    # Plot histogram
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    
    ax_hist.hist(ranged_peak_freqs, bins='auto', color='skyblue', edgecolor='black', density=True, label='Data Histogram (density)')
    
    # Plot lines for mean, median, and mode
    ax_hist.axvline(mean_freq_all, color='red', linestyle='dashed', linewidth=1.5, label=f'Overall Mean: {formatter(mean_freq_all)}')
    ax_hist.axvline(median_freq_all, color='green', linestyle='dotted', linewidth=1.5, label=f'Overall Median: {formatter(median_freq_all)}')
    if mode_available:
        ax_hist.axvline(mode_freq_all, color='purple', linestyle='dashdot', linewidth=1.5, label=f'Overall Mode (KDE): {formatter(mode_freq_all)}')

    # Plot KDE curve if available, scaled to match histogram density
    if mode_available: # Re-evaluate KDE for plotting over the specified vmin, vmax if desired, or use original
        plot_kde_x = np.linspace(vmin, vmax, 500)
        plot_kde_y = kde(plot_kde_x)
        ax_hist.plot(plot_kde_x, plot_kde_y, color='orange', linestyle='-', linewidth=2, label='KDE')

    # Add text for std dev, IQR, MAD
    stats_text = f'Overall Std Dev: {formatter(std_freq_all)}\n' \
                 f'Overall IQR: {formatter(iqr_freq_all)}\n' \
                 f'Overall MAD: {formatter(mad_freq_all)}'
    ax_hist.text(0.05, 0.95, stats_text,
                 transform=ax_hist.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    ax_hist.set_title(f"{plot_title_prefix} - Peak Frequency Distribution (Range: {formatter(vmin)} to {formatter(vmax)})")
    ax_hist.set_xlabel("Peak Frequency")
    ax_hist.set_ylabel("Density / Count (for histogram bars if not density)") # ylabel updated if density=True
    ax_hist.xaxis.set_major_formatter(formatter)
    ax_hist.set_xlim(vmin, vmax)
    ax_hist.legend(fontsize='small')
    ax_hist.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    hist_output_filename = f"{os.path.splitext(output_base_filename)[0]}_histogram.png"
    try:
        fig_hist.savefig(hist_output_filename, dpi=300)
        print(f"Histogram saved to {hist_output_filename}")
    except Exception as e:
        sys.exit(f"Error saving histogram figure: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot peak-frequency heatmap with discrete color levels and corner polygon."
    )
    parser.add_argument(
        "--h5", default="peakFreq.h5",
        help="HDF5 file containing '/peakFreq','/X','/Ycoord' (default: peakFreq.h5)"
    )
    parser.add_argument(
        "--csv", default="corners.csv",
        help="CSV file with ['X_coord','Y_coord'] cols (default: corners.csv)"
    )
    parser.add_argument(
        "--out", default="heatmap.png",
        help="Output image file path (default: heatmap.png)"
    )
    parser.add_argument(
        "--vmin", type=float, default=250e6,
        help="Lower bound of color scale in Hz (default: 250e6)"
    )
    parser.add_argument(
        "--vmax", type=float, default=350e6,
        help="Upper bound of color scale in Hz (default: 350e6)"
    )
    parser.add_argument(
        "--levels", type=int, default=100,
        help="Number of discrete color levels (default: 100)"
    )
    parser.add_argument(
        "--title", default="Peak Frequency Heatmap",
        help="Plot title (default: 'Peak Frequency Heatmap')"
    )
    parser.add_argument(
        "--histogram", action="store_true",
        help="Generate and save a histogram of peak frequencies with mean/std dev."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    peak_freq, x, y = load_peak_data(args.h5)
    xc, yc = load_corners(args.csv)

    # Print min and max peak frequency
    min_freq_data = np.nanmin(peak_freq)
    max_freq_data = np.nanmax(peak_freq)
    print(f"Minimum peak frequency in data: {min_freq_data:.2e} Hz")
    print(f"Maximum peak frequency in data: {max_freq_data:.2e} Hz")

    if args.histogram:
        plot_histogram_and_stats(peak_freq, args.out, args.title, args.vmin, args.vmax)

    fig, ax = plot_heatmap(
        peak_freq, x, y, xc, yc,
        vmin=args.vmin,
        vmax=args.vmax,
        levels=args.levels,
        title=args.title
    )

    try:
        fig.savefig(args.out, dpi=300)
        print(f"Heatmap saved to {args.out}")
    except Exception as e:
        sys.exit(f"Error saving figure: {e}")


if __name__ == "__main__":
    main()