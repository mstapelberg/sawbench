import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Optional, Sequence
from matplotlib.colors import BoundaryNorm

def plot_frequency_histogram(
    ax: Axes, 
    data_mhz: np.ndarray, 
    title: str, 
    bins: int = 30,
    label: Optional[str] = None,
    color: Optional[str] = None,
    xlabel: str = 'Frequency (MHz)',
    ylabel: str = 'Counts',
    xlim: Optional[Tuple[float, float]] = None,
    alpha: float = 0.7,
    density: bool = False
) -> None:
    """
    Plots a histogram of frequency data on a given Matplotlib Axes object.
    Can display raw counts, probability density, or frequency.

    Args:
        ax (Axes): The Matplotlib Axes object to plot on.
        data_mhz (np.ndarray): 1D array of frequency data in MHz.
        title (str): Title for the histogram.
        bins (int): Number of bins for the histogram. Defaults to 30.
        label (Optional[str]): Label for the dataset, used for the legend. Defaults to None.
        color (Optional[str]): Color for the histogram. If None, uses Matplotlib's default color cycle.
        xlabel (str): Label for the x-axis. Defaults to 'Frequency (MHz)'.
        ylabel (str): Label for the y-axis (used when density=False). Defaults to 'Counts'.
        xlim (Optional[Tuple[float, float]]): X-axis limits. Defaults to None.
        alpha (float): Transparency of the histogram bars. Defaults to 0.7.
        density (bool): If True, normalize the histogram to form a probability density.
                        If 'frequency', normalizes by total counts.
                        If False, show raw counts. Defaults to False.
    """
    valid_data = data_mhz[~np.isnan(data_mhz)]
    if valid_data.size > 0:
        weights = None
        y_label_text = ylabel
        if density == 'frequency':
            weights = np.ones_like(valid_data) / len(valid_data)
            y_label_text = 'Frequency'
        elif density:
            y_label_text = 'Probability Density'

        ax.hist(valid_data, bins=bins, alpha=alpha, edgecolor='black', label=label, density=density if isinstance(density, bool) else False, color=color, weights=weights)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        
        ax.set_ylabel(y_label_text)
        
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)
    else:
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
    ax.ticklabel_format(style='plain', axis='x') # Avoid scientific notation for MHz

def plot_frequency_cdfs(
    ax: Axes,
    datasets_mhz_sorted: List[np.ndarray],
    labels: List[str],
    colors: List[str],
    title: str = 'Cumulative Distribution Functions (CDFs)',
    xlabel: str = 'Frequency (MHz)',
    ylabel: str = 'Cumulative Probability',
    xlim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plots one or more Cumulative Distribution Functions (CDFs) on a given Matplotlib Axes object.

    Args:
        ax (Axes): The Matplotlib Axes object to plot on.
        datasets_mhz_sorted (List[np.ndarray]): A list of 1D arrays, where each array contains
                                             sorted frequency data in MHz.
        labels (List[str]): A list of labels corresponding to each dataset.
        colors (List[str]): A list of colors corresponding to each dataset.
        title (str): Title for the CDF plot. Defaults to 'Cumulative Distribution Functions (CDFs)'.
        xlabel (str): Label for the x-axis. Defaults to 'Frequency (MHz)'.
        ylabel (str): Label for the y-axis. Defaults to 'Cumulative Probability'.
        xlim (Optional[Tuple[float, float]]): X-axis limits. Defaults to None.
    """
    if not datasets_mhz_sorted or len(datasets_mhz_sorted) != len(labels) or len(datasets_mhz_sorted) != len(colors):
        ax.text(0.5, 0.5, "Invalid input for CDFs", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    plotted_any = False
    for data_sorted, label, color in zip(datasets_mhz_sorted, labels, colors):
        valid_data = data_sorted[~np.isnan(data_sorted)]
        if valid_data.size > 0:
            cdf = np.arange(1, len(valid_data) + 1) / len(valid_data)
            ax.plot(valid_data, cdf, color=color, label=label)
            plotted_any = True
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, "No data for CDF plot", ha='center', va='center', transform=ax.transAxes)
    if xlim:
        ax.set_xlim(xlim)
    ax.ticklabel_format(style='plain', axis='x')

def plot_cdf_difference(
    ax: Axes,
    data1_mhz_sorted: np.ndarray,
    data2_mhz_sorted: List[np.ndarray],
    label1: str = "Dataset 1",
    labels2: Optional[List[str]] = None,
    title: str = 'CDF Difference',
    xlabel: str = 'Frequency (MHz)',
    ylabel: str = 'Difference in Cumulative Probability',
    xlim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plots the difference between a reference CDF and one or more other CDFs.

    Args:
        ax (Axes): The Matplotlib Axes object to plot on.
        data1_mhz_sorted (np.ndarray): Reference dataset (sorted, 1D array in MHz).
        data2_mhz_sorted (List[np.ndarray]): List of other datasets to compare.
        label1 (str): Name for the reference dataset.
        labels2 (Optional[List[str]]): List of names for the other datasets.
        title (str): Title for the plot. Defaults to 'CDF Difference'.
        xlabel (str): Label for the x-axis. Defaults to 'Frequency (MHz)'.
        ylabel (str): Label for the y-axis. Defaults to 'Difference in Cumulative Probability'.
        xlim (Optional[Tuple[float, float]]): X-axis limits. Defaults to None.
    """
    valid_data1 = data1_mhz_sorted[~np.isnan(data1_mhz_sorted)]
    if valid_data1.size == 0:
        ax.text(0.5, 0.5, "Reference data is empty.", ha='center', va='center')
        ax.set_title(title)
        return

    # Create a combined frequency axis from all datasets for accurate interpolation
    all_valid_data = [valid_data1] + [d[~np.isnan(d)] for d in data2_mhz_sorted if d.size > 0]
    if not all_valid_data:
        ax.text(0.5, 0.5, "All datasets are empty.", ha='center', va='center')
        ax.set_title(title)
        return
        
    all_freqs_mhz = np.sort(np.unique(np.concatenate(all_valid_data)))

    # Interpolate the reference CDF
    cdf1_vals = np.interp(all_freqs_mhz, valid_data1, np.linspace(0, 1, len(valid_data1)), left=0, right=1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(data2_mhz_sorted)))
    
    for i, data2 in enumerate(data2_mhz_sorted):
        valid_data2 = data2[~np.isnan(data2)]
        if valid_data2.size == 0:
            continue

        cdf2_vals = np.interp(all_freqs_mhz, valid_data2, np.linspace(0, 1, len(valid_data2)), left=0, right=1)
        cdf_difference = cdf1_vals - cdf2_vals
        
        current_label = f'{label1} - {labels2[i]}' if labels2 and i < len(labels2) else f'Diff {i+1}'
        ax.plot(all_freqs_mhz, cdf_difference, color=colors[i], label=current_label)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    if xlim:
        ax.set_xlim(xlim)
    ax.ticklabel_format(style='plain', axis='x')

def plot_ebsd_property_map(
    ax: Axes,
    map_data: np.ndarray,
    title: str,
    cbar_label: str = 'Property Value',
    step_size_um: Optional[float] = None,
    cmap: str = 'viridis',
    origin: str = 'lower',
    aspect: str = 'equal'
) -> None:
    """
    Plots a 2D EBSD property map (e.g., SAW frequency) on a given Matplotlib Axes object.

    Args:
        ax (Axes): The Matplotlib Axes object to plot on.
        map_data (np.ndarray): 2D numpy array containing the map data.
        title (str): Title for the map.
        cbar_label (str): Label for the colorbar. Defaults to 'Property Value'.
        step_size_um (Optional[float]): Step size in micrometers. If provided, axis labels
                                      will be in micrometers. Defaults to None (pixel units).
        cmap (str): Colormap to use. Defaults to 'viridis'.
        origin (str): Origin for imshow. Defaults to 'lower'.
        aspect (str): Aspect ratio for imshow. Defaults to 'equal'.
    """
    if map_data is None or map_data.ndim != 2:
        ax.text(0.5, 0.5, "Invalid map data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    im = ax.imshow(map_data, cmap=cmap, origin=origin, aspect=aspect)
    ax.set_title(title)
    
    if step_size_um:
        y_dim, x_dim = map_data.shape
        ax.set_xlabel(f'X ($\\mu$m)')
        ax.set_ylabel(f'Y ($\\mu$m)')
        # Adjust ticks to show physical units, e.g., by setting extent or custom ticks
        # For simplicity with imshow, often easier to label "X Pixel (step_size um)"
        # Or, adjust extent:
        # extent = [0, x_dim * step_size_um, 0, y_dim * step_size_um]
        # im = ax.imshow(map_data, cmap=cmap, origin=origin, aspect=aspect, extent=extent)
        ax.set_xlabel(f'X Pixel (step: {step_size_um:.2f} $\\mu$m)')
        ax.set_ylabel(f'Y Pixel (step: {step_size_um:.2f} $\\mu$m)')

    else:
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        
    cbar = plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.ticklabel_format(style='plain', axis='x') # For colorbar if it goes scientific 

def plot_experimental_heatmap(
    ax: plt.Axes,
    peak_freq_map: np.ndarray,
    x_coords: np.ndarray, 
    y_coords: np.ndarray,
    title: str = "Experimental Peak Frequency Map",
    cbar_label: str = "Peak Frequency (MHz)",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    levels: int = 100
) -> None:
    """
    Plot experimental peak frequency heatmap with discrete color levels.
    
    Args:
        ax: Matplotlib axes to plot on
        peak_freq_map: 2D array of peak frequencies 
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates  
        title: Plot title
        cbar_label: Colorbar label
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        levels: Number of discrete color levels
    """
    # Convert to MHz for display
    if "MHz" in cbar_label:
        peak_freq_map_display = peak_freq_map / 1e6
        if vmin is not None:
            vmin = vmin / 1e6
        if vmax is not None:
            vmax = vmax / 1e6
    else:
        peak_freq_map_display = peak_freq_map
    
    # Auto-scale if not provided
    if vmin is None:
        vmin = np.nanmin(peak_freq_map_display)
    if vmax is None:
        vmax = np.nanmax(peak_freq_map_display)
        
    # Create coordinate meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create discrete colormap
    cmap = plt.get_cmap('viridis', levels)
    norm = BoundaryNorm(np.linspace(vmin, vmax, levels + 1), ncolors=levels)
    
    # Plot heatmap
    mesh = ax.pcolormesh(
        X, Y, peak_freq_map_display,
        shading='auto',
        cmap=cmap, 
        norm=norm
    )
    
    # Add colorbar
    cbar = plt.colorbar(
        mesh, ax=ax, 
        ticks=np.linspace(vmin, vmax, min(levels + 1, 11))
    )
    cbar.set_label(cbar_label)
    
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.set_title(title)
    ax.set_aspect('equal') 