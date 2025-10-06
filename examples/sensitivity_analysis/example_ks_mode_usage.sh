#!/bin/bash
# Example usage of the new KS mode in ternary_saw_frequency.py

# This example shows how to run the ternary plot in KS metric sensitivity mode
# The plot will show how the KS metric (comparing predicted EBSD grain frequencies
# to experimental data) changes as you vary K, D, G elastic constants.

python ternary_saw_frequency.py \
    --ks_mode \
    --plot_kdg \
    --ebsd_path "/home/myless/Packages/sawbench/examples/data/V-1_2Ti_EBSD_Map" \
    --experimental_h5 "/home/myless/Documents/saw_freq_analysis/fftData.h5" \
    --c11_base_gpa 229.0 \
    --c12_base_gpa 119.0 \
    --c44_base_gpa 43.0 \
    --density 6110.0 \
    --wavelength_um 8.8 \
    --deg 135.0 \
    --exp_min_mhz 200.0 \
    --exp_max_mhz 400.0 \
    --num_workers 32 \
    --sampling 400 \
    --res 30 \
    --outfile "results/ternary_ks_sensitivity_kdg.png"

# Notes:
# - The plot will use K/D/G axes (--plot_kdg)
# - Default propagation angle is now 135° (changed from 0°)
# - The colormap shows normalized change in KS metric: (KS - KS₀)/KS₀
# - Lower KS values (blue) indicate better agreement with experiment
# - The custom diverging colormap uses blue (#2A33C3) to white to red (#8F2D56)
# - Resolution --res 30 means 31 points per edge (496 total grid points)
# - Each grid point requires computing SAW frequencies for all EBSD grains
# - With 32 workers, this should take a reasonable amount of time

