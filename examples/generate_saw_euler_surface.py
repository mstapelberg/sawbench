import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import os
from tqdm import tqdm
import pandas as pd
from scipy import stats

# Import from sawbench package
from sawbench.materials import Material
from sawbench.saw_calculator import SAWCalculator

# --- Configuration ---
# Material Parameters (e.g., Vanadium)
NOMINAL_MATERIAL_PROPS = {
    'formula': 'V', 'C11': 229e9, 'C12': 119e9,
    'C44': 43e9, 'density': 6110, 'crystal_class': 'cubic'
}

# Euler Angle Grid Parameters
# Ranges are inclusive for np.arange if step makes them so, otherwise use np.linspace
PHI1_DEG_RANGE = (0, 360)
PHI_DEG_RANGE = (0, 180)
PHI2_DEG_RANGE = (0, 360) # phi2 range for the 3D grid and slider
ANGLE_STEP_DEG = 10 # Step for phi1, Phi, and phi2

# SAW Calculation Parameters
SAW_PROPAGATION_ANGLE_ON_SURFACE_DEG = 0.0  # Propagation direction on the sample surface (x-y plane)

# Plotting Parameters
PLOTS_OUTPUT_DIR = "plots_euler_surface"
PLOT_FILE_BASE = "saw_velocity_surface"

# --- Helper Functions ---

def plot_interactive_saw_surface(
    phi1_steps_deg: np.ndarray,
    Phi_steps_deg: np.ndarray,
    phi2_steps_deg: np.ndarray,
    saw_velocities_grid_mps: np.ndarray, # Expects shape (phi2, Phi, phi1)
    material_name: str,
    saw_prop_angle_deg: float
):
    """
    Generates and displays an interactive 3D surface plot of SAW velocity
    with a slider for phi2.
    Args:
        phi1_steps_deg: 1D array of phi1 Euler angles (degrees).
        Phi_steps_deg: 1D array of Phi Euler angles (degrees).
        phi2_steps_deg: 1D array of phi2 Euler angles (degrees) for the slider.
        saw_velocities_grid_mps: 3D grid of SAW velocities (phi2_idx, Phi_idx, phi1_idx).
        material_name: Name of the material for plot title.
        saw_prop_angle_deg: SAW propagation angle for plot title.
    """
    pio.renderers.default = "browser"

    X_phi1_mesh, Y_Phi_mesh = np.meshgrid(phi1_steps_deg, Phi_steps_deg)

    fig = go.Figure()

    # Add initial surface trace (for the first phi2 value)
    fig.add_trace(go.Surface(
        x=X_phi1_mesh,
        y=Y_Phi_mesh,
        z=saw_velocities_grid_mps[0, :, :],
        colorscale='Viridis',
        colorbar=dict(title='SAW Velocity (m/s)'),
        cmin=np.nanmin(saw_velocities_grid_mps), # Consistent color scaling across slices
        cmax=np.nanmax(saw_velocities_grid_mps)
    ))

    # Create slider steps
    sliders_steps = []
    for i, phi2_val in enumerate(phi2_steps_deg):
        step = dict(
            method='restyle', # Updates data
            args=['z', [saw_velocities_grid_mps[i, :, :]]], # Update z data of the first trace
            label=f'{phi2_val:.1f}'
        )
        sliders_steps.append(step)

    fig.update_layout(
        title=f'Interactive SAW Velocity Surface (Material: {material_name}, SAW Prop: {saw_prop_angle_deg}°)<br>Use slider for phi2',
        scene=dict(
            xaxis_title='phi1 (degrees)',
            yaxis_title='Phi (degrees)',
            zaxis_title='SAW Velocity (m/s)',
            # Aspect ratio can be adjusted if needed
            # aspectratio=dict(x=1, y=1, z=0.7)
        ),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "phi2 (deg): "},
            pad={"t": 50},
            steps=sliders_steps
        )],
        margin=dict(l=0, r=0, b=0, t=100)
    )
    
    # Optional: Save to HTML
    # html_filename = os.path.join(PLOTS_OUTPUT_DIR, f"{PLOT_FILE_BASE}_interactive.html")
    # if not os.path.exists(PLOTS_OUTPUT_DIR):
    #     os.makedirs(PLOTS_OUTPUT_DIR)
    # fig.write_html(html_filename)
    # print(f"Interactive plot saved to {os.path.abspath(html_filename)}")

    fig.show()

def display_velocity_stats_and_hist(
    saw_velocities_grid_mps: np.ndarray, 
    material_name: str,
    angle_step_deg: float,
    phi1_range: tuple,
    Phi_range: tuple,
    phi2_range: tuple,
    output_dir: str = PLOTS_OUTPUT_DIR
):
    """
    Calculates, prints, and plots a histogram of SAW velocities.
    Args:
        saw_velocities_grid_mps (np.ndarray): 3D grid of SAW velocities.
        material_name (str): Name of the material for plot title.
        angle_step_deg (float): Step size used for angle grid generation.
        phi1_range, Phi_range, phi2_range (tuple): Ranges used for angle grid.
        output_dir (str): Directory to save the histogram plot.
    """
    all_velocities = saw_velocities_grid_mps.flatten()
    valid_velocities = all_velocities[~np.isnan(all_velocities)]

    if len(valid_velocities) == 0:
        print("\nNo valid SAW velocities found to calculate statistics or plot histogram.")
        return

    print("\n--- SAW Velocity Statistics ---")
    print(f"Data from Euler angle grid: phi1{phi1_range}, Phi{Phi_range}, phi2{phi2_range} (steps: {angle_step_deg}°)")
    print(f"Total calculated points (including NaNs): {len(all_velocities)}")
    print(f"Number of valid SAW velocities: {len(valid_velocities)}")
    
    mean_vel = np.mean(valid_velocities)
    median_vel = np.median(valid_velocities)
    min_vel = np.min(valid_velocities)
    max_vel = np.max(valid_velocities)
    range_vel = max_vel - min_vel
    std_dev_vel = np.std(valid_velocities)
    variance_vel = np.var(valid_velocities)
    skew_vel = stats.skew(valid_velocities)
    kurt_vel = stats.kurtosis(valid_velocities) # Fisher's definition (normal==0)

    print(f"Mean: {mean_vel:.2f} m/s")
    print(f"Median: {median_vel:.2f} m/s")
    print(f"Min: {min_vel:.2f} m/s")
    print(f"Max: {max_vel:.2f} m/s")
    print(f"Range: {range_vel:.2f} m/s")
    print(f"Standard Deviation: {std_dev_vel:.2f} m/s")
    print(f"Variance: {variance_vel:.2f} (m/s)^2")
    print(f"Skewness: {skew_vel:.3f}")
    print(f"Kurtosis (Fisher): {kurt_vel:.3f}")

    # Plot Histogram
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory for histogram: {os.path.abspath(output_dir)}")

    plt.figure(figsize=(10, 6))
    plt.hist(valid_velocities, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of SAW Velocities (Material: {material_name})\nEuler Grid: phi1{phi1_range}, Phi{Phi_range}, phi2{phi2_range}, Step {angle_step_deg}°')
    plt.xlabel("SAW Velocity (m/s)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    hist_filename = os.path.join(output_dir, f"{PLOT_FILE_BASE}_histogram.png")
    try:
        plt.savefig(hist_filename)
        print(f"Histogram saved to: {os.path.abspath(hist_filename)}")
    except Exception as e:
        print(f"Error saving histogram {hist_filename}: {e}")
    plt.close() # Close the histogram figure

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting SAW Velocity Euler Angle Surface Generation ---")

    material = Material(**NOMINAL_MATERIAL_PROPS)
    print(f"Using material: {material.formula}")

    phi1_deg_steps = np.arange(PHI1_DEG_RANGE[0], PHI1_DEG_RANGE[1] + ANGLE_STEP_DEG, ANGLE_STEP_DEG)
    Phi_deg_steps = np.arange(PHI_DEG_RANGE[0], PHI_DEG_RANGE[1] + ANGLE_STEP_DEG, ANGLE_STEP_DEG)
    phi2_deg_steps = np.arange(PHI2_DEG_RANGE[0], PHI2_DEG_RANGE[1] + ANGLE_STEP_DEG, ANGLE_STEP_DEG)

    # Initialize 3D grid for SAW velocities
    saw_velocities_grid_mps = np.full(
        (len(phi2_deg_steps), len(Phi_deg_steps), len(phi1_deg_steps)), 
        np.nan
    )

    print(f"\nCalculating SAW velocities for a grid of Euler angles:")
    print(f"phi1 steps: {len(phi1_deg_steps)} (Range: {PHI1_DEG_RANGE}, Step: {ANGLE_STEP_DEG}°)")
    print(f"Phi steps: {len(Phi_deg_steps)} (Range: {PHI_DEG_RANGE}, Step: {ANGLE_STEP_DEG}°)")
    print(f"phi2 steps: {len(phi2_deg_steps)} (Range: {PHI2_DEG_RANGE}, Step: {ANGLE_STEP_DEG}°)")
    total_calculations = len(phi1_deg_steps) * len(Phi_deg_steps) * len(phi2_deg_steps)
    print(f"Total SAW calculations to perform: {total_calculations}")

    with tqdm(total=total_calculations, desc="Calculating SAW Velocities") as pbar:
        for idx_phi2, phi2_val_deg in enumerate(phi2_deg_steps):
            for idx_Phi, Phi_val_deg in enumerate(Phi_deg_steps):
                for idx_phi1, phi1_val_deg in enumerate(phi1_deg_steps):
                    current_euler_deg = np.array([phi1_val_deg, Phi_val_deg, phi2_val_deg])
                    current_euler_rad = np.deg2rad(current_euler_deg)

                    try:
                        calculator = SAWCalculator(material, current_euler_rad)
                        v_mps_all_modes, _, _ = calculator.get_saw_speed(
                            deg=SAW_PROPAGATION_ANGLE_ON_SURFACE_DEG,
                            sampling=400 
                        )
                        
                        if v_mps_all_modes is not None and len(v_mps_all_modes) > 0 and pd.notna(v_mps_all_modes[0]):
                            saw_velocities_grid_mps[idx_phi2, idx_Phi, idx_phi1] = v_mps_all_modes[0]
                        # NaNs are pre-filled, so no explicit else needed if pd.notna is false or array is empty
                        
                    except Exception as e:
                        # Keep it as NaN if error, print less verbosely in the loop
                        # print(f"Error for Euler {current_euler_deg} deg: {e}. Stored NaN.") 
                        pass # Already NaN
                    pbar.update(1)
    
    print("\nSAW velocity calculations finished.")

    # Plotting the interactive surface
    if np.any(~np.isnan(saw_velocities_grid_mps)):
        print("\nGenerating interactive 3D surface plot...")
        plot_interactive_saw_surface(
            phi1_deg_steps, Phi_deg_steps, phi2_deg_steps, 
            saw_velocities_grid_mps,
            material_name=material.formula,
            saw_prop_angle_deg=SAW_PROPAGATION_ANGLE_ON_SURFACE_DEG
        )
    else:
        print("\nSkipping interactive plot as all calculated SAW velocities are NaN.")

    # Display statistics and histogram
    print("\nCalculating and displaying statistics...")
    display_velocity_stats_and_hist(
        saw_velocities_grid_mps, 
        material_name=material.formula,
        angle_step_deg=ANGLE_STEP_DEG,
        phi1_range=PHI1_DEG_RANGE,
        Phi_range=PHI_DEG_RANGE,
        phi2_range=PHI2_DEG_RANGE,
        output_dir=PLOTS_OUTPUT_DIR
    )

    print("\n--- SAW Velocity Euler Angle Surface Generation Finished ---") 