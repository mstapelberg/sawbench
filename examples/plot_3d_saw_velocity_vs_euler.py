import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import os

# Import from sawbench package
from sawbench import (
    load_ebsd_map,
    calculate_saw_frequencies_for_ebsd_grains,
    Material,
)

# --- Configuration ---
# EBSD Data Parameters
# USER: Please verify this path. It should point to your EBSD data file/folder.
EBSD_DATA_PATH = "V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected"  # Example path
EBSD_DATA_TYPE = "OxfordText"
EBSD_BOUNDARY_DEF_ANGLE = 5.0
EBSD_MIN_GRAIN_PX_SIZE = 10

# Material and SAW Calculation Parameters (e.g., Vanadium)
NOMINAL_MATERIAL_PROPS = {
    'formula': 'V', 'C11': 229e9, 'C12': 119e9,
    'C44': 43e9, 'density': 6110, 'crystal_class': 'cubic'
}
WAVELENGTH_M = 8.8e-6
# For this specific plot, we typically use a single, representative SAW calculation angle (e.g., 0 degrees)
SAW_CALC_ANGLE_DEG = 0.0  # Degrees

# Plotting Parameters
PLOTS_OUTPUT_DIR = "plots_3d_euler" # Specific directory for these plots

# --- 3D Plotting Function for SAW Velocity vs Euler Angles ---
def plot_saw_velocity_euler_3d(
    df_saw_data: pd.DataFrame,
    wavelength_m: float,
    output_dir: str = PLOTS_OUTPUT_DIR,
    output_filename_base: str = "saw_velocity_euler_3d"
):
    """
    Generates and saves a static 3D scatter plot (Matplotlib) and displays
    an interactive 3D scatter plot (Plotly) of SAW velocity vs. Euler angles.

    Args:
        df_saw_data: DataFrame containing 'Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)', 'Peak SAW Frequency (Hz)'.
        wavelength_m: Wavelength in meters for velocity calculation.
        output_dir: Directory to save the static plot.
        output_filename_base: Base name for the saved static plot file.
    """
    print("--- Generating 3D SAW Velocity vs Euler Angle Plots ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {os.path.abspath(output_dir)}")

    required_cols = ['Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)', 'Peak SAW Frequency (Hz)']
    if not all(col in df_saw_data.columns for col in required_cols):
        print(f"Error: DataFrame for 3D plot is missing one or more required columns: {required_cols}.")
        print(f"Available columns: {df_saw_data.columns.tolist()}")
        print("Skipping 3D plots.")
        return

    phi1_rad = df_saw_data['Euler1 (rad)'].to_numpy()
    Phi_rad = df_saw_data['Euler2 (rad)'].to_numpy()
    phi2_rad = df_saw_data['Euler3 (rad)'].to_numpy()
    frequencies_hz = df_saw_data['Peak SAW Frequency (Hz)'].to_numpy()

    saw_velocity_mps = frequencies_hz * wavelength_m

    phi1_deg = np.rad2deg(phi1_rad)
    Phi_deg = np.rad2deg(Phi_rad)
    phi2_deg = np.rad2deg(phi2_rad)

    # 1. Static Matplotlib 3D Scatter Plot
    try:
        fig_mpl = plt.figure(figsize=(12, 10))
        ax_mpl = fig_mpl.add_subplot(111, projection='3d')
        scatter_mpl = ax_mpl.scatter(phi1_deg, Phi_deg, phi2_deg, c=saw_velocity_mps, cmap='viridis', s=10)
        ax_mpl.set_xlabel('phi1 (degrees)')
        ax_mpl.set_ylabel('Phi (degrees)')
        ax_mpl.set_zlabel('phi2 (degrees)')
        ax_mpl.set_title('SAW Velocity vs. Euler Angles (Bunge Convention)')
        cbar_mpl = fig_mpl.colorbar(scatter_mpl, ax=ax_mpl, shrink=0.6)
        cbar_mpl.set_label('SAW Velocity (m/s)')
        static_plot_filename = os.path.join(output_dir, f"{output_filename_base}_static.png")
        plt.savefig(static_plot_filename)
        print(f"Static 3D plot saved to: {os.path.abspath(static_plot_filename)}")
        plt.close(fig_mpl)
    except Exception as e:
        print(f"Error generating Matplotlib 3D plot: {e}")

    # 2. Interactive Plotly 3D Scatter Plot
    try:
        pio.renderers.default = "browser" 
        fig_plotly = go.Figure(data=[go.Scatter3d(
            x=phi1_deg, y=Phi_deg, z=phi2_deg, mode='markers',
            marker=dict(
                size=3, color=saw_velocity_mps, colorscale='Viridis',
                colorbar_title='SAW Velocity (m/s)', opacity=0.8
            ),
            text=[f"Vel: {v:.2f} m/s<br>phi1: {p1:.1f}°<br>Phi: {P:.1f}°<br>phi2: {p2:.1f}°"
                  for v, p1, P, p2 in zip(saw_velocity_mps, phi1_deg, Phi_deg, phi2_deg)],
            hoverinfo='text'
        )])
        fig_plotly.update_layout(
            title='Interactive SAW Velocity vs. Euler Angles (Bunge Convention)',
            scene=dict(
                xaxis_title='phi1 (degrees)', yaxis_title='Phi (degrees)', zaxis_title='phi2 (degrees)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig_plotly.show()
        print("Displaying interactive 3D plot.")
    except Exception as e:
        print(f"Error generating Plotly 3D plot: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting 3D SAW Velocity vs. Euler Angles Plotting Script ---")

    # 1. Load EBSD Data
    print(f"Attempting to load EBSD data from: {os.path.abspath(EBSD_DATA_PATH)}")
    ebsd_map = load_ebsd_map(
        EBSD_DATA_PATH, EBSD_DATA_TYPE,
        EBSD_BOUNDARY_DEF_ANGLE, EBSD_MIN_GRAIN_PX_SIZE
    )

    if not ebsd_map:
        print("Critical Error: EBSD data could not be loaded. Exiting.")
        exit()
    
    print(f"EBSD map loaded successfully. Number of grains: {len(ebsd_map.grainList)}")

    # 2. Prepare for SAW Calculation
    material_obj = Material(**NOMINAL_MATERIAL_PROPS)
    print(f"Material properties for '{material_obj.formula}' loaded.")

    # 3. Calculate SAW Frequencies (and get Euler angles)
    print(f"Calculating SAW frequencies for all grains at a fixed angle of {SAW_CALC_ANGLE_DEG:.1f} deg...")
    df_grain_data = calculate_saw_frequencies_for_ebsd_grains(
        ebsd_map_obj=ebsd_map,
        material=material_obj,
        wavelength=WAVELENGTH_M,
        saw_calc_angle_deg=SAW_CALC_ANGLE_DEG
    )

    if df_grain_data.empty:
        print("Warning: No SAW frequency data returned for grains. Cannot generate 3D plot.")
        exit()
    
    print(f"SAW frequency data calculated for {len(df_grain_data)} grains.")

    # 4. Generate 3D Plots
    plot_saw_velocity_euler_3d(
        df_grain_data,
        WAVELENGTH_M,
        output_dir=PLOTS_OUTPUT_DIR
    )

    print("--- 3D SAW Velocity vs. Euler Angles Plotting Script Finished ---") 