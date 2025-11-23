#!/usr/bin/env python3
"""
Plot raw TGS signals, compute FFTs, and fit peaks.

Data structure:
- X: (61,) X spatial coordinates
- Ycoord: (61,) Y spatial coordinates  
- T: (10000,) time array in seconds
- Y: (61, 61, 10000) signal data [X, Y, time]
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, windows, savgol_filter
from scipy.optimize import curve_fit
from pathlib import Path
import argparse
import random

# Color scheme
COLOR_BLUE = '#2A33C3'
COLOR_ORANGE = '#A35D00'
COLOR_TEAL = '#0B7285'
COLOR_PINK = '#8F2D56'
COLOR_GREEN = '#6E8B00'
COLOR_SLATE_LIGHT = '#E2E8F0'
COLOR_SLATE_DARK = '#4A5568'


def load_raw_signal(filepath):
    """Load raw signal data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        T = f['T'][:]
        X = f['X'][:]
        Ycoord = f['Ycoord'][:]
        Y = f['Y'][:]  # Shape: (61, 61, 10000)
    
    return T, X, Ycoord, Y


def plot_raw_signal(t, signal, x_pos, y_pos, ax=None, title=None, freqs=None, amplitude=None, 
                   freq_range_mhz=None, popt=None, popt1=None, popt2=None):
    """
    Plot a single raw time-domain signal with optional fitted FFT overlay.
    
    Args:
        t: Time array (will be trimmed to t >= 0)
        signal: Signal array
        x_pos, y_pos: Position coordinates
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        freqs: Optional frequency array for FFT overlay
        amplitude: Optional amplitude array for FFT overlay
        freq_range_mhz: Optional frequency range (min, max) in MHz for FFT overlay
        popt: Optional fitted Gaussian parameters [A, mu, sigma, offset] for overlay
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Trim data to t >= 0
    mask = t >= 0
    t_trimmed = t[mask]
    signal_trimmed = signal[mask]
    
    ax.plot(t_trimmed * 1e6, signal_trimmed, color=COLOR_BLUE, linewidth=1.0)  # Convert to microseconds
    ax.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    if title is None:
        ax.set_title(f'Raw Signal at X={x_pos:.2f}, Y={y_pos:.2f}', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Bold the main plot axes
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2, labelsize=11)
    
    # Add fitted FFT overlay in upper right if provided
    if freqs is not None and amplitude is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # Make overlay bigger: 50% width, 45% height
        ax_inset = inset_axes(ax, width="50%", height="45%", loc='upper right', borderpad=3)
        
        # Convert to MHz
        freqs_mhz = freqs / 1e6
        
        # Apply frequency range filter if specified
        if freq_range_mhz is not None:
            freq_min, freq_max = freq_range_mhz
            mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
            freqs_mhz_plot = freqs_mhz[mask]
            amplitude_plot = amplitude[mask]
        else:
            freqs_mhz_plot = freqs_mhz
            amplitude_plot = amplitude
        
        # Plot data
        ax_inset.plot(freqs_mhz_plot, amplitude_plot, color=COLOR_BLUE, linewidth=1, label='Data', alpha=0.7)
        
        # Plot fitted curve(s) if available
        if popt1 is not None and popt2 is not None:
            # Two peaks case
            freqs_fit = np.linspace(freqs_mhz_plot.min(), freqs_mhz_plot.max(), 1000)
            # Extract individual Gaussians (without offset for individual plots)
            fit_curve1 = popt1[0] * np.exp(-0.5 * ((freqs_fit - popt1[1]) / popt1[2])**2) + popt1[3]
            fit_curve2 = popt2[0] * np.exp(-0.5 * ((freqs_fit - popt2[1]) / popt2[2])**2) + popt2[3]
            # Combined fit: sum of peaks minus one offset (since both share the same baseline)
            fit_curve_total = (popt1[0] * np.exp(-0.5 * ((freqs_fit - popt1[1]) / popt1[2])**2) +
                              popt2[0] * np.exp(-0.5 * ((freqs_fit - popt2[1]) / popt2[2])**2) +
                              popt1[3])  # Single offset
            
            ax_inset.plot(freqs_fit, fit_curve1, color=COLOR_PINK, linestyle='--', linewidth=2, 
                         label=f'Peak 1: {popt1[1]:.3f} MHz', alpha=0.8)
            ax_inset.plot(freqs_fit, fit_curve2, color=COLOR_ORANGE, linestyle='--', linewidth=2, 
                         label=f'Peak 2: {popt2[1]:.3f} MHz', alpha=0.8)
            ax_inset.plot(freqs_fit, fit_curve_total, color=COLOR_PINK, linestyle=':', linewidth=1.5, 
                         label='Combined Fit', alpha=0.6)
            
            # Mark peaks
            ax_inset.axvline(popt1[1], color=COLOR_GREEN, linestyle=':', linewidth=2, alpha=0.7)
            ax_inset.axvline(popt2[1], color=COLOR_GREEN, linestyle=':', linewidth=2, alpha=0.7)
            ax_inset.plot(popt1[1], popt1[0], color=COLOR_GREEN, marker='o', markersize=8)
            ax_inset.plot(popt2[1], popt2[0], color=COLOR_GREEN, marker='o', markersize=8)
            ax_inset.legend(fontsize=9, loc='upper right')
        elif popt is not None:
            # Single peak case
            freqs_fit = np.linspace(freqs_mhz_plot.min(), freqs_mhz_plot.max(), 1000)
            fit_curve = gaussian(freqs_fit, *popt)
            ax_inset.plot(freqs_fit, fit_curve, color=COLOR_PINK, linestyle='--', linewidth=2, label='Gaussian Fit')
            
            peak_freq = popt[1]
            peak_amp = popt[0]
            ax_inset.axvline(peak_freq, color=COLOR_GREEN, linestyle=':', linewidth=2, 
                           label=f'Peak: {peak_freq:.3f} MHz')
            ax_inset.plot(peak_freq, peak_amp, color=COLOR_GREEN, marker='o', markersize=8)
            ax_inset.legend(fontsize=9, loc='upper right')
        
        ax_inset.set_xlabel('Frequency (MHz)', fontsize=11, fontweight='bold')
        ax_inset.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax_inset.tick_params(width=2, labelsize=10)
        ax_inset.grid(True, alpha=0.3)
        if freq_range_mhz is not None:
            ax_inset.set_xlim(freq_range_mhz)
        
        # Bold the overlay axes
        for spine in ax_inset.spines.values():
            spine.set_linewidth(2)
    
    return ax


def compute_fft(t, signal, n_fft=None, window='blackman', smooth=None):
    """
    Compute FFT of a time-domain signal with optional zero-padding for higher resolution.
    Only uses data where t >= 0.
    
    Args:
        t: Time array (will be trimmed to t >= 0)
        signal: Signal array
        n_fft: Number of FFT points. If None, uses signal length. 
               If larger than signal length, zero-pads for higher resolution.
        window: Window function to apply before FFT to reduce spectral leakage.
                Options: 'hanning', 'hamming', 'blackman', 'bartlett', 'none'
                Default: 'blackman' (best side-lobe suppression for exponential decay)
        smooth: Smoothing window size for Savitzky-Golay filter. If None, no smoothing.
                Should be an odd integer. Larger values = more smoothing.
    
    Returns:
        freqs_positive: Positive frequency array
        amplitude: Amplitude spectrum (smoothed if smooth is specified)
    """
    # Trim data to t >= 0
    mask = t >= 0
    t_trimmed = t[mask]
    signal_trimmed = signal[mask]
    
    if len(t_trimmed) < 2:
        raise ValueError("Not enough data points after trimming to t >= 0")
    
    dt = t_trimmed[1] - t_trimmed[0]  # Time step
    N = len(signal_trimmed)
    
    # Apply window function to reduce spectral leakage
    if window and window.lower() != 'none':
        if window.lower() == 'hanning':
            win = windows.hann(N)
        elif window.lower() == 'hamming':
            win = windows.hamming(N)
        elif window.lower() == 'blackman':
            win = windows.blackman(N)
        elif window.lower() == 'bartlett':
            win = windows.bartlett(N)
        else:
            raise ValueError(f"Unknown window type: {window}. Use 'hanning', 'hamming', 'blackman', 'bartlett', or 'none'")
        
        # Apply window
        signal_windowed = signal_trimmed * win
    else:
        signal_windowed = signal_trimmed
    
    # Use zero-padding if n_fft is specified and larger than signal length
    if n_fft is None:
        n_fft = N
    elif n_fft < N:
        n_fft = N  # Can't use fewer points than signal length
    
    # Zero-pad signal if needed
    if n_fft > N:
        padded_signal = np.zeros(n_fft)
        padded_signal[:N] = signal_windowed
        signal_to_fft = padded_signal
    else:
        signal_to_fft = signal_windowed
    
    # Compute FFT
    fft_vals = fft(signal_to_fft, n=n_fft)
    freqs = fftfreq(n_fft, dt)
    
    # Take positive frequencies only
    positive_freq_mask = freqs > 0
    freqs_positive = freqs[positive_freq_mask]
    fft_positive = fft_vals[positive_freq_mask]
    
    # Amplitude spectrum
    amplitude = np.abs(fft_positive)
    
    # Apply smoothing if requested (helps reduce high-frequency ripples)
    if smooth is not None and smooth > 1:
        # Ensure smooth is odd and not larger than the data
        smooth = int(smooth)
        if smooth % 2 == 0:
            smooth += 1  # Make odd
        if smooth > len(amplitude):
            smooth = len(amplitude) if len(amplitude) % 2 == 1 else len(amplitude) - 1
        
        if smooth >= 3:
            # Use Savitzky-Golay filter for smoothing (preserves peak shape better than moving average)
            # Polynomial order 2 works well for smooth spectra
            amplitude = savgol_filter(amplitude, smooth, 2)
    
    return freqs_positive, amplitude


def plot_fft(freqs, amplitude, x_pos, y_pos, ax=None, title=None, freq_range_mhz=None):
    """Plot FFT amplitude spectrum."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to MHz for plotting
    freqs_mhz = freqs / 1e6
    
    # Apply frequency range filter if specified
    if freq_range_mhz is not None:
        freq_min, freq_max = freq_range_mhz
        mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
        freqs_mhz = freqs_mhz[mask]
        amplitude = amplitude[mask]
    
    ax.plot(freqs_mhz, amplitude, color=COLOR_TEAL, linewidth=1)
    ax.set_xlabel('Frequency (MHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    if title is None:
        ax.set_title(f'FFT Spectrum at X={x_pos:.2f}, Y={y_pos:.2f}', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Bold the axes
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2, labelsize=11)
    
    return ax


def find_single_peak_position(T, Y, X, Ycoord, n_fft, window, smooth, freq_range_mhz, max_attempts=1000):
    """
    Randomly search for a position with a single dominant peak in the FFT spectrum.
    
    Returns:
        (x_idx, y_idx, x_pos, y_pos, signal, freqs, amplitude) or None if not found
    """
    print(f"\nSearching for position with single dominant peak (max {max_attempts} attempts)...")
    
    for attempt in range(max_attempts):
        # Random position
        x_idx = random.randint(0, len(X) - 1)
        y_idx = random.randint(0, len(Ycoord) - 1)
        
        signal = Y[x_idx, y_idx, :]
        freqs, amplitude = compute_fft(T, signal, n_fft=n_fft, window=window, smooth=smooth)
        
        # Convert to MHz and filter frequency range
        freqs_mhz = freqs / 1e6
        if freq_range_mhz is not None:
            freq_min, freq_max = freq_range_mhz
            mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
            freqs_mhz_filtered = freqs_mhz[mask]
            amplitude_filtered = amplitude[mask]
        else:
            freqs_mhz_filtered = freqs_mhz
            amplitude_filtered = amplitude
        
        if len(amplitude_filtered) < 10:
            continue
        
        # Find peaks with reasonable prominence
        prominence_threshold = amplitude_filtered.max() * 0.15  # At least 15% of max
        peaks, properties = find_peaks(amplitude_filtered, prominence=prominence_threshold)
        
        # Look for exactly 1 dominant peak (or 1 peak that's much stronger than others)
        if len(peaks) == 1:
            # Single peak found
            peak_idx = peaks[0]
            peak_freq = freqs_mhz_filtered[peak_idx]
            peak_amp = amplitude_filtered[peak_idx]
            
            x_pos = X[x_idx]
            y_pos = Ycoord[y_idx]
            print(f"  Found single peak at position ({x_idx}, {y_idx}):")
            print(f"    Peak: {peak_freq:.3f} MHz (amplitude: {peak_amp:.2e})")
            return (x_idx, y_idx, x_pos, y_pos, signal, freqs, amplitude)
        elif len(peaks) > 1:
            # Check if one peak is dominant (at least 2x stronger than second peak)
            peak_amplitudes = amplitude_filtered[peaks]
            sorted_peaks = peaks[np.argsort(peak_amplitudes)[::-1]]
            
            peak1_amp = amplitude_filtered[sorted_peaks[0]]
            peak2_amp = amplitude_filtered[sorted_peaks[1]]
            
            if peak1_amp >= 2.0 * peak2_amp:
                # First peak is dominant
                peak_idx = sorted_peaks[0]
                peak_freq = freqs_mhz_filtered[peak_idx]
                peak_amp = amplitude_filtered[peak_idx]
                
                x_pos = X[x_idx]
                y_pos = Ycoord[y_idx]
                print(f"  Found dominant single peak at position ({x_idx}, {y_idx}):")
                print(f"    Peak: {peak_freq:.3f} MHz (amplitude: {peak_amp:.2e})")
                return (x_idx, y_idx, x_pos, y_pos, signal, freqs, amplitude)
        
        if (attempt + 1) % 100 == 0:
            print(f"  Attempted {attempt + 1} positions...")
    
    print(f"  Could not find position with single dominant peak after {max_attempts} attempts")
    return None


def find_two_peak_position(T, Y, X, Ycoord, n_fft, window, smooth, freq_range_mhz, max_attempts=1000):
    """
    Randomly search for a position with two dominant peaks in the FFT spectrum.
    
    Returns:
        (x_idx, y_idx, x_pos, y_pos, signal, freqs, amplitude) or None if not found
    """
    print(f"\nSearching for position with two dominant peaks (max {max_attempts} attempts)...")
    
    for attempt in range(max_attempts):
        # Random position
        x_idx = random.randint(0, len(X) - 1)
        y_idx = random.randint(0, len(Ycoord) - 1)
        
        signal = Y[x_idx, y_idx, :]
        freqs, amplitude = compute_fft(T, signal, n_fft=n_fft, window=window, smooth=smooth)
        
        # Convert to MHz and filter frequency range
        freqs_mhz = freqs / 1e6
        if freq_range_mhz is not None:
            freq_min, freq_max = freq_range_mhz
            mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
            freqs_mhz_filtered = freqs_mhz[mask]
            amplitude_filtered = amplitude[mask]
        else:
            freqs_mhz_filtered = freqs_mhz
            amplitude_filtered = amplitude
        
        if len(amplitude_filtered) < 10:
            continue
        
        # Find peaks with reasonable prominence
        prominence_threshold = amplitude_filtered.max() * 0.15  # At least 15% of max
        peaks, properties = find_peaks(amplitude_filtered, prominence=prominence_threshold)
        
        if len(peaks) >= 2:
            # Check that peaks are reasonably separated and significant
            peak_amplitudes = amplitude_filtered[peaks]
            sorted_peaks = peaks[np.argsort(peak_amplitudes)[::-1]]  # Sort by amplitude descending
            
            # Get top 2 peaks
            peak1_idx = sorted_peaks[0]
            peak2_idx = sorted_peaks[1]
            
            peak1_freq = freqs_mhz_filtered[peak1_idx]
            peak2_freq = freqs_mhz_filtered[peak2_idx]
            peak1_amp = amplitude_filtered[peak1_idx]
            peak2_amp = amplitude_filtered[peak2_idx]
            
            # Check that both peaks are significant (at least 20% of max)
            if peak2_amp >= amplitude_filtered.max() * 0.2:
                x_pos = X[x_idx]
                y_pos = Ycoord[y_idx]
                print(f"  Found two peaks at position ({x_idx}, {y_idx}):")
                print(f"    Peak 1: {peak1_freq:.3f} MHz (amplitude: {peak1_amp:.2e})")
                print(f"    Peak 2: {peak2_freq:.3f} MHz (amplitude: {peak2_amp:.2e})")
                return (x_idx, y_idx, x_pos, y_pos, signal, freqs, amplitude)
        
        if (attempt + 1) % 100 == 0:
            print(f"  Attempted {attempt + 1} positions...")
    
    print(f"  Could not find position with two dominant peaks after {max_attempts} attempts")
    return None


def gaussian(x, A, mu, sigma, offset):
    """Gaussian function for peak fitting."""
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset


def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, offset):
    """Sum of two Gaussian functions for fitting two peaks."""
    return (A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2) + 
            A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2) + offset)


def fit_two_peaks(freqs, amplitude, freq_range_mhz=None):
    """
    Fit two Gaussian peaks to the FFT spectrum.
    
    Returns:
        popt1: Optimal parameters for first peak [A, mu, sigma, offset]
        popt2: Optimal parameters for second peak [A, mu, sigma, offset]
        peak_freqs: Tuple of (peak1_freq, peak2_freq) in MHz
        peak_amps: Tuple of (peak1_amp, peak2_amp)
    """
    # Convert to MHz
    freqs_mhz = freqs / 1e6
    
    # Apply frequency range filter if specified
    if freq_range_mhz is not None:
        freq_min, freq_max = freq_range_mhz
        mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
        freqs_mhz = freqs_mhz[mask]
        amplitude = amplitude[mask]
    
    if len(freqs_mhz) < 6:
        return None, None, None, None
    
    # Find peaks
    prominence_threshold = amplitude.max() * 0.15
    peaks, properties = find_peaks(amplitude, prominence=prominence_threshold)
    
    if len(peaks) < 2:
        return None, None, None, None
    
    # Get top 2 peaks
    peak_amplitudes = amplitude[peaks]
    sorted_peaks = peaks[np.argsort(peak_amplitudes)[::-1]]
    peak1_idx = sorted_peaks[0]
    peak2_idx = sorted_peaks[1]
    
    peak1_freq = freqs_mhz[peak1_idx]
    peak2_freq = freqs_mhz[peak2_idx]
    peak1_amp = amplitude[peak1_idx]
    peak2_amp = amplitude[peak2_idx]
    
    # Initial guesses for double Gaussian fit
    sigma_guess = (freqs_mhz.max() - freqs_mhz.min()) / 20
    offset_guess = amplitude.min()
    
    initial_guess = [
        peak1_amp, peak1_freq, sigma_guess,  # Peak 1
        peak2_amp, peak2_freq, sigma_guess,  # Peak 2
        offset_guess  # Offset
    ]
    
    try:
        # Fit double Gaussian
        popt, pcov = curve_fit(
            double_gaussian,
            freqs_mhz,
            amplitude,
            p0=initial_guess,
            maxfev=5000
        )
        
        # Extract parameters for each peak
        A1, mu1, sigma1, A2, mu2, sigma2, offset = popt
        
        popt1 = [A1, mu1, sigma1, offset]
        popt2 = [A2, mu2, sigma2, offset]
        
        peak_freqs = (mu1, mu2)
        peak_amps = (A1, A2)
        
        return popt1, popt2, peak_freqs, peak_amps
    except (RuntimeError, ValueError):
        # If double fit fails, try fitting each peak separately
        try:
            # Fit peak 1
            A1_guess = peak1_amp
            mu1_guess = peak1_freq
            sigma1_guess = sigma_guess
            offset1_guess = offset_guess
            popt1, _ = curve_fit(
                gaussian,
                freqs_mhz,
                amplitude,
                p0=[A1_guess, mu1_guess, sigma1_guess, offset1_guess],
                maxfev=5000
            )
            
            # Fit peak 2
            A2_guess = peak2_amp
            mu2_guess = peak2_freq
            sigma2_guess = sigma_guess
            offset2_guess = offset_guess
            popt2, _ = curve_fit(
                gaussian,
                freqs_mhz,
                amplitude,
                p0=[A2_guess, mu2_guess, sigma2_guess, offset2_guess],
                maxfev=5000
            )
            
            peak_freqs = (popt1[1], popt2[1])
            peak_amps = (popt1[0], popt2[0])
            
            return popt1, popt2, peak_freqs, peak_amps
        except (RuntimeError, ValueError):
            return None, None, None, None


def fit_peak(freqs, amplitude, initial_guess=None, freq_range_mhz=None):
    """
    Fit a Gaussian peak to the FFT spectrum.
    
    Returns:
        popt: Optimal parameters [A, mu, sigma, offset]
        pcov: Covariance matrix
        peak_freq: Fitted peak frequency (MHz)
        peak_amp: Fitted peak amplitude
    """
    # Convert to MHz
    freqs_mhz = freqs / 1e6
    
    # Apply frequency range filter if specified
    if freq_range_mhz is not None:
        freq_min, freq_max = freq_range_mhz
        mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
        freqs_mhz = freqs_mhz[mask]
        amplitude = amplitude[mask]
    
    if len(freqs_mhz) < 3:
        return None, None, None, None
    
    # Find initial peak using scipy
    peaks, properties = find_peaks(amplitude, prominence=amplitude.max() * 0.1)
    
    if len(peaks) == 0:
        return None, None, None, None
    
    # Use highest peak
    main_peak_idx = peaks[np.argmax(amplitude[peaks])]
    
    if initial_guess is None:
        A_guess = amplitude[main_peak_idx]
        mu_guess = freqs_mhz[main_peak_idx]
        sigma_guess = (freqs_mhz.max() - freqs_mhz.min()) / 20  # Rough estimate
        offset_guess = amplitude.min()
        initial_guess = [A_guess, mu_guess, sigma_guess, offset_guess]
    
    try:
        # Fit Gaussian
        popt, pcov = curve_fit(
            gaussian, 
            freqs_mhz, 
            amplitude, 
            p0=initial_guess,
            maxfev=5000
        )
        
        peak_freq = popt[1]  # mu parameter
        peak_amp = popt[0]   # A parameter
        
        return popt, pcov, peak_freq, peak_amp
    except (RuntimeError, ValueError):
        return None, None, None, None


def plot_fft_with_fit(freqs, amplitude, popt, x_pos, y_pos, ax=None, title=None, freq_range_mhz=None):
    """Plot FFT spectrum with fitted Gaussian peak."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to MHz
    freqs_mhz = freqs / 1e6
    
    # Apply frequency range filter if specified
    if freq_range_mhz is not None:
        freq_min, freq_max = freq_range_mhz
        mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
        freqs_mhz = freqs_mhz[mask]
        amplitude = amplitude[mask]
    
    # Plot data
    ax.plot(freqs_mhz, amplitude, color=COLOR_TEAL, linewidth=1, label='Data', alpha=0.7)
    
    # Plot fit if available
    if popt is not None:
        freqs_fit = np.linspace(freqs_mhz.min(), freqs_mhz.max(), 1000)
        fit_curve = gaussian(freqs_fit, *popt)
        ax.plot(freqs_fit, fit_curve, color=COLOR_ORANGE, linestyle='--', linewidth=2, label='Gaussian Fit')
        
        peak_freq = popt[1]
        peak_amp = popt[0]
        ax.axvline(peak_freq, color=COLOR_GREEN, linestyle=':', linewidth=2, 
                   label=f'Peak: {peak_freq:.3f} MHz')
        ax.plot(peak_freq, peak_amp, color=COLOR_PINK, marker='o', markersize=10)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    if title is None:
        ax.set_title(f'FFT with Peak Fit at X={x_pos:.2f}, Y={y_pos:.2f}', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bold the axes
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2, labelsize=11)
    
    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot raw TGS signals, FFTs, and fit peaks')
    parser.add_argument('--file', type=str, 
                       default='/home/myless/Documents/saw_freq_analysis/rawSignal.h5',
                       help='Path to HDF5 file')
    parser.add_argument('--x-idx', type=int, default=30, 
                       help='X index (0-60) to plot')
    parser.add_argument('--y-idx', type=int, default=30,
                       help='Y index (0-60) to plot')
    parser.add_argument('--freq-range', type=float, nargs=2, default=[200, 400],
                       metavar=('MIN', 'MAX'),
                       help='Frequency range in MHz for plotting/fitting (default: 200 400). FFT computed over full range.')
    parser.add_argument('--n-fft', type=int, default=None,
                       help='Number of FFT points for zero-padding. Default: 10x signal length for high resolution.')
    parser.add_argument('--window', type=str, default='blackman',
                       choices=['hanning', 'hamming', 'blackman', 'bartlett', 'none'],
                       help='Window function to reduce spectral leakage (default: blackman). Use "none" to disable.')
    parser.add_argument('--smooth', type=int, default=51,
                       help='Smoothing window size (odd integer) for Savitzky-Golay filter. Larger = more smoothing. Default: 51. Use 0 to disable.')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for two-peak search (default: 42)')
    parser.add_argument('--n-plot', type=int, default=0,
                       help='Number of single-peak and twin-peak plots to generate (saved in outdir/sweep). Default: 0 (disabled)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.file}")
    T, X, Ycoord, Y = load_raw_signal(args.file)
    
    print(f"\nData loaded:")
    print(f"  Time range: {T.min()*1e6:.3f} to {T.max()*1e6:.3f} μs")
    print(f"  X range: {X.min():.2f} to {X.max():.2f}")
    print(f"  Y range: {Ycoord.min():.2f} to {Ycoord.max():.2f}")
    print(f"  Signal shape: {Y.shape}")
    print(f"  Signal range: {Y.min():.2f} to {Y.max():.2f}")
    
    # Get signal at specified position
    x_idx = args.x_idx
    y_idx = args.y_idx
    signal = Y[x_idx, y_idx, :]
    x_pos = X[x_idx]
    y_pos = Ycoord[y_idx]
    
    print(f"\nAnalyzing signal at:")
    print(f"  X index: {x_idx}, X position: {x_pos:.2f}")
    print(f"  Y index: {y_idx}, Y position: {y_pos:.2f}")
    
    # Set default n_fft for high resolution (10x signal length)
    n_fft = args.n_fft if args.n_fft is not None else len(signal) * 10
    dt = T[1] - T[0]
    freq_resolution = 1.0 / (n_fft * dt)  # Frequency resolution in Hz
    
    # Set smoothing parameter (0 means no smoothing)
    smooth_param = args.smooth if args.smooth > 0 else None
    
    print(f"\nFFT parameters:")
    print(f"  Signal length: {len(signal)} points")
    print(f"  FFT points: {n_fft} (zero-padded for higher resolution)")
    print(f"  Frequency resolution: {freq_resolution/1e6:.6f} MHz")
    print(f"  Window function: {args.window}")
    if smooth_param:
        print(f"  Smoothing: Savitzky-Golay filter (window size: {smooth_param})")
    else:
        print(f"  Smoothing: None")
    
    # Create output directory if specified
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute FFT first (over full range) for overlay
    print("\nComputing FFT over full frequency range...")
    freqs, amplitude = compute_fft(T, signal, n_fft=n_fft, window=args.window, smooth=smooth_param)
    
    # Convert freq_range to tuple if it's a list
    freq_range_mhz = tuple(args.freq_range) if args.freq_range is not None else None
    
    # Fit peak first (needed for overlay)
    print("\nFitting peak...")
    popt, pcov, peak_freq, peak_amp = fit_peak(freqs, amplitude, freq_range_mhz=freq_range_mhz)
    
    # Plot 1: Raw signal with fitted FFT overlay
    print("\nPlotting raw signal with fitted FFT overlay...")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_raw_signal(T, signal, x_pos, y_pos, ax=ax1, 
                   freqs=freqs, amplitude=amplitude, freq_range_mhz=freq_range_mhz, popt=popt)
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'raw_signal.png', dpi=150)
        print(f"  Saved: {output_dir / 'raw_signal.png'}")
    plt.show()
    
    # Plot 2: FFT (computed over full range)
    print(f"\nFFT statistics:")
    print(f"  Frequency range: {freqs.min()/1e6:.2f} to {freqs.max()/1e6:.2f} MHz")
    print(f"  Number of frequency bins: {len(freqs)}")
    print(f"  Max amplitude: {amplitude.max():.2e} at {freqs[np.argmax(amplitude)]/1e6:.6f} MHz")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_fft(freqs, amplitude, x_pos, y_pos, ax=ax2, freq_range_mhz=freq_range_mhz)
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'fft_spectrum.png', dpi=150)
        print(f"  Saved: {output_dir / 'fft_spectrum.png'}")
    plt.show()
    
    # Plot 3: FFT with peak fit
    if peak_freq is not None:
        print(f"  Peak frequency: {peak_freq:.3f} MHz")
        print(f"  Peak amplitude: {peak_amp:.2e}")
        print(f"  FWHM: {2.355 * popt[2]:.3f} MHz")  # FWHM = 2.355 * sigma
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        plot_fft_with_fit(freqs, amplitude, popt, x_pos, y_pos, ax=ax3, 
                         freq_range_mhz=freq_range_mhz)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'fft_with_fit.png', dpi=150)
            print(f"  Saved: {output_dir / 'fft_with_fit.png'}")
        plt.show()
    else:
        print("  Could not fit peak - try adjusting frequency range or position")
    
    # Plot 4: Multiple positions comparison
    print("\nCreating comparison plot at multiple positions...")
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    positions = [
        (30, 30, "Center"),
        (15, 15, "Corner 1"),
        (45, 45, "Corner 2"),
        (30, 15, "Edge")
    ]
    
    for idx, (x_i, y_i, label) in enumerate(positions):
        if x_i >= len(X) or y_i >= len(Ycoord):
            continue
        
        sig = Y[x_i, y_i, :]
        freqs_pos, amp_pos = compute_fft(T, sig, n_fft=n_fft, window=args.window, smooth=smooth_param)
        
        ax = axes[idx]
        plot_fft(freqs_pos, amp_pos, X[x_i], Ycoord[y_i], ax=ax, 
                title=f'{label} (X={X[x_i]:.2f}, Y={Ycoord[y_i]:.2f})',
                freq_range_mhz=freq_range_mhz)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'comparison_multiple_positions.png', dpi=150)
        print(f"  Saved: {output_dir / 'comparison_multiple_positions.png'}")
    plt.show()
    
    # Plot 5: Search for and plot a position with two dominant peaks
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    print(f"\nRandom seed set to: {args.random_seed}")
    
    two_peak_result = find_two_peak_position(
        T, Y, X, Ycoord, n_fft, args.window, smooth_param, freq_range_mhz, max_attempts=1000
    )
    
    if two_peak_result is not None:
        x_idx_2peak, y_idx_2peak, x_pos_2peak, y_pos_2peak, signal_2peak, freqs_2peak, amplitude_2peak = two_peak_result
        
        # Fit both peaks separately
        print("\nFitting both peaks...")
        popt1_2peak, popt2_2peak, peak_freqs_2peak, peak_amps_2peak = fit_two_peaks(
            freqs_2peak, amplitude_2peak, freq_range_mhz=freq_range_mhz
        )
        
        if popt1_2peak is not None and popt2_2peak is not None:
            print(f"  Peak 1: {peak_freqs_2peak[0]:.3f} MHz (amplitude: {peak_amps_2peak[0]:.2e})")
            print(f"  Peak 2: {peak_freqs_2peak[1]:.3f} MHz (amplitude: {peak_amps_2peak[1]:.2e})")
            print(f"  Peak 1 FWHM: {2.355 * popt1_2peak[2]:.3f} MHz")
            print(f"  Peak 2 FWHM: {2.355 * popt2_2peak[2]:.3f} MHz")
            
            print("\nPlotting raw signal with FFT overlay for two-peak position...")
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            plot_raw_signal(T, signal_2peak, x_pos_2peak, y_pos_2peak, ax=ax5, 
                           freqs=freqs_2peak, amplitude=amplitude_2peak, 
                           freq_range_mhz=freq_range_mhz, popt1=popt1_2peak, popt2=popt2_2peak)
        else:
            print("  Could not fit both peaks - plotting without fits")
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            plot_raw_signal(T, signal_2peak, x_pos_2peak, y_pos_2peak, ax=ax5, 
                           freqs=freqs_2peak, amplitude=amplitude_2peak, 
                           freq_range_mhz=freq_range_mhz)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'raw_signal_two_peaks.png', dpi=150)
            print(f"  Saved: {output_dir / 'raw_signal_two_peaks.png'}")
        plt.show()
    else:
        print("\nSkipping two-peak plot (none found)")
    
    # Generate sweep plots if requested
    if args.n_plot > 0:
        if output_dir is None:
            print("\nWarning: --n-plot requires --output-dir to be specified. Skipping sweep plots.")
        else:
            sweep_dir = output_dir / 'sweep'
            sweep_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"Generating {args.n_plot} single-peak and {args.n_plot} twin-peak plots")
            print(f"Saving to: {sweep_dir}")
            print(f"{'='*60}")
            
            # Generate single peak plots
            print(f"\nGenerating {args.n_plot} single-peak plots...")
            single_peak_count = 0
            for i in range(args.n_plot):
                result = find_single_peak_position(
                    T, Y, X, Ycoord, n_fft, args.window, smooth_param, freq_range_mhz, max_attempts=1000
                )
                
                if result is not None:
                    x_idx_sp, y_idx_sp, x_pos_sp, y_pos_sp, signal_sp, freqs_sp, amplitude_sp = result
                    
                    # Fit the peak
                    popt_sp, _, peak_freq_sp, _ = fit_peak(freqs_sp, amplitude_sp, freq_range_mhz=freq_range_mhz)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plot_raw_signal(T, signal_sp, x_pos_sp, y_pos_sp, ax=ax,
                                   freqs=freqs_sp, amplitude=amplitude_sp,
                                   freq_range_mhz=freq_range_mhz, popt=popt_sp)
                    plt.tight_layout()
                    
                    filename = sweep_dir / f'single_peak_{i+1:03d}_x{x_idx_sp}_y{y_idx_sp}.png'
                    plt.savefig(filename, dpi=150)
                    plt.close()
                    single_peak_count += 1
                    print(f"  Saved: {filename.name}")
                else:
                    print(f"  Could not find single peak for plot {i+1}")
            
            print(f"  Generated {single_peak_count}/{args.n_plot} single-peak plots")
            
            # Generate twin peak plots
            print(f"\nGenerating {args.n_plot} twin-peak plots...")
            twin_peak_count = 0
            for i in range(args.n_plot):
                result = find_two_peak_position(
                    T, Y, X, Ycoord, n_fft, args.window, smooth_param, freq_range_mhz, max_attempts=1000
                )
                
                if result is not None:
                    x_idx_tp, y_idx_tp, x_pos_tp, y_pos_tp, signal_tp, freqs_tp, amplitude_tp = result
                    
                    # Fit both peaks
                    popt1_tp, popt2_tp, peak_freqs_tp, peak_amps_tp = fit_two_peaks(
                        freqs_tp, amplitude_tp, freq_range_mhz=freq_range_mhz
                    )
                    
                    if popt1_tp is not None and popt2_tp is not None:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        plot_raw_signal(T, signal_tp, x_pos_tp, y_pos_tp, ax=ax,
                                       freqs=freqs_tp, amplitude=amplitude_tp,
                                       freq_range_mhz=freq_range_mhz, popt1=popt1_tp, popt2=popt2_tp)
                        plt.tight_layout()
                        
                        filename = sweep_dir / f'twin_peak_{i+1:03d}_x{x_idx_tp}_y{y_idx_tp}.png'
                        plt.savefig(filename, dpi=150)
                        plt.close()
                        twin_peak_count += 1
                        print(f"  Saved: {filename.name}")
                    else:
                        print(f"  Could not fit peaks for twin-peak plot {i+1}")
                else:
                    print(f"  Could not find twin peaks for plot {i+1}")
            
            print(f"  Generated {twin_peak_count}/{args.n_plot} twin-peak plots")
            print(f"\nSweep plots complete! Total: {single_peak_count} single-peak + {twin_peak_count} twin-peak plots")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

