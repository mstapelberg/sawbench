import numpy as np 
import matplotlib.pyplot as plt 
import torch

#from saw_elastic_predictions.materials import Material
#from saw_elastic_predictions.saw_calculator import SAWCalculator
from sawbench.materials import Material
from sawbench.saw_calculator import SAWCalculator
from sawbench.io import read_ctf
from mace.calculators.mace import MACECalculator
from ase.build import bulk

import torch_sim as ts
from torch_sim.elastic import get_bravais_type
from torch_sim.models.mace import MaceModel
from torch_sim.units import MetalUnits
from torch_sim.integrators import MDState
from torch_sim.monte_carlo import swap_monte_carlo
from dataclasses import dataclass
from sawbench.grains import plot_ebsd_map_mpl, GrainCollection, ipf_color

compute_elastic_tensor = False

if compute_elastic_tensor:
    # Calculator 
    unit_conv = ts.units.UnitConversion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    mace_model = torch.load('../../forge/scratch/potentials/gen_6_model_0_L0_isolated-2026-01-16-finetuned_fp64_nh10000_lr1e-4_stagetwo.model', map_location=device)

    struct = bulk("V", "bcc", a = 3.01, cubic = True).repeat((4,4,4))
    # replace Ti with V to get around V-1.2Ti
    x_ti = 0.012
    # get select 1.2at% of the atoms in struct and replace with Ti
    indices = np.random.choice(np.arange(len(struct)), size=int(len(struct)*x_ti), replace=False)
    struct.symbols[indices] = "Ti"



    model = MaceModel(
        model = mace_model,
        device = device,
        compute_forces = True,
        compute_stress = True,
        dtype = dtype,
        enable_cueq = True if device == "cuda" else False,
    )

    # --- Updated Hybrid MC Swap + NVT MD Annealing Phase ---
    print("\nStarting Hybrid MC Swap + NVT MD Annealing Phase...")

    # Define annealing temperature and kT
    T_anneal_kelvin = 1073 # Target temperature in Kelvin, e.g., 1000 K
    annealing_kT = T_anneal_kelvin * MetalUnits.temperature # kT in energy units (eV if MetalUnits.temperature is k_B in eV/K)
    # Or, if you know the kT value directly in eV:
    # annealing_kT = 0.1 # Example: 0.1 eV for kT

    print(f"Annealing at T = {T_anneal_kelvin} K, corresponding kT = {annealing_kT:.4f} [model energy units]")

    n_annealing_steps = 300
    swap_frequency = 10 # Attempt swap every 10 steps

    # Initial state for annealing
    anneal_state_init = ts.io.atoms_to_state(atoms=struct, device=device, dtype=dtype)

    # Initialize NVT Langevin integrator
    nvt_init, nvt_step = ts.integrators.nvt_langevin(model=model, dt=0.001, kT=annealing_kT) # dt in ps
    md_state_initial = nvt_init(anneal_state_init, seed=42)

    # Define HybridSwapMCState
    @dataclass
    class HybridSwapMCState(MDState):
        """State for Monte Carlo simulations.
        Attributes:
            energy: Energy of the system
            last_permutation: Tensor indicating the last swap permutation.
        """
        last_permutation: torch.Tensor

    # Initialize the hybrid state
    current_anneal_state = HybridSwapMCState(
        **vars(md_state_initial),
        last_permutation=torch.zeros(
            md_state_initial.n_batches, device=md_state_initial.device, dtype=torch.bool
        ),
    )

    # Initialize Swap Monte Carlo (note: the example does swap_init(md_state) then makes hybrid_state)
    # Let's assume swap_step can take HybridSwapMCState directly.
    # If swap_init is necessary to set up the swapper, it might be called like this:
    _swap_init_fn, swap_step_fn = swap_monte_carlo(model=model, kT=annealing_kT, seed=42)
    # current_anneal_state = _swap_init_fn(current_anneal_state) # If swap_init needs to be called on the HybridSwapMCState

    anneal_generator = torch.Generator(device=device)
    anneal_generator.manual_seed(1234)

    for anneal_step in range(n_annealing_steps):
        if anneal_step % swap_frequency == 0:
            print(f"Anneal Step {anneal_step}/{n_annealing_steps}: Attempting MC Swap...")
            current_anneal_state = swap_step_fn(current_anneal_state, kT=torch.tensor(annealing_kT, device=device, dtype=dtype), generator=anneal_generator)
        else:
            if anneal_step % 10 == 0: # Print MD progress occasionally (reduced frequency)
                energy_val = current_anneal_state.energy.item() if current_anneal_state.energy is not None else float('nan')
                print(f"Anneal Step {anneal_step}/{n_annealing_steps}: MD step. Energy: {energy_val:.4f}")
            current_anneal_state = nvt_step(current_anneal_state, dt=torch.tensor(0.001, device=device, dtype=dtype), kT=torch.tensor(annealing_kT, device=device, dtype=dtype))
        
    print("Annealing phase complete.")
    # The state after annealing is `current_anneal_state`
    # --- End of Hybrid MC Swap + NVT MD Annealing Phase ---


    # Target force tolerance 
    fmax = 1E-3 # eV/A

    # Initialize FIRE optimizer with the state from annealing
    print("\nStarting FIRE relaxation...")
    fire_init, fire_update = ts.optimizers.frechet_cell_fire(model=model, scalar_pressure=0.0)


    fire_input_state = ts.integrators.MDState(
        positions=current_anneal_state.positions,
        cell=current_anneal_state.cell,
        momenta=current_anneal_state.momenta,  # Expected to be a Tensor
        energy=current_anneal_state.energy,    # Expected to be a Tensor
        forces=current_anneal_state.forces,    # Expected to be a Tensor
        masses=current_anneal_state.masses,
        atomic_numbers=current_anneal_state.atomic_numbers,
        
        # Attributes likely inherited from SimState's __init__
        pbc=getattr(current_anneal_state, 'pbc', True), 
        batch=getattr(current_anneal_state, 'batch', 
                    torch.zeros(current_anneal_state.positions.size(0), # Assuming positions is [N_atoms_total, 3]
                                dtype=torch.long, 
                                device=current_anneal_state.device)
                    ),
        # Optional SimState attributes
        #constraints=getattr(current_anneal_state, 'constraints', None),
        #ids=getattr(current_anneal_state, 'ids', None)
    )
    # If 'device' and 'dtype' are not part of MDState.__init__ because they are properties,
    # they should be removed from the explicit constructor call.
    # The same applies to any other properties vs. direct __init__ args.

    state = fire_init(state=fire_input_state)

    for step in range(500): # Or more steps if needed
        pressure = -torch.trace(state.stress.squeeze()) / 3 * unit_conv.eV_per_Ang3_to_GPa
        current_fmax = torch.max(torch.abs(state.forces.squeeze()))
        print(
            f"Step {step}, Energy: {state.energy.item():.4f}, "
            f"Pressure: {pressure.item():.4f}, "
            f"Fmax: {current_fmax.item():.4f}"
        )
        if current_fmax < fmax and abs(pressure) < 1e-2:
            break
        state = fire_update(state=state)

    # Get bravais type
    bravais_type = get_bravais_type(state)

    # Calculate elastic tensor
    elastic_tensor = ts.elastic.calculate_elastic_tensor(
        model, state=state, bravais_type=bravais_type
    )

    # Convert to GPa
    elastic_tensor = elastic_tensor * unit_conv.eV_per_Ang3_to_GPa

    # Calculate elastic moduli
    bulk_modulus, shear_modulus, poisson_ratio, pugh_ratio = (
        ts.elastic.calculate_elastic_moduli(elastic_tensor)
    )

    # Print elastic tensor
    print("\nElastic tensor (GPa):")
    elastic_tensor_np = elastic_tensor.cpu().numpy()
    for row in elastic_tensor_np:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    # Print mechanical moduli
    print(f"Bulk modulus (GPa): {bulk_modulus:.4f}")
    print(f"Shear modulus (GPa): {shear_modulus:.4f}")
    print(f"Poisson's ratio: {poisson_ratio:.4f}")
    print(f"Pugh's ratio (K/G): {pugh_ratio:.4f}")

    # calculate the predicted density of the structure after relaxation
    relaxed_atoms_list = ts.io.state_to_atoms(state=state)
    if not isinstance(relaxed_atoms_list, list): # Ensure it's a list
        relaxed_atoms_list = [relaxed_atoms_list]

    if not relaxed_atoms_list:
        raise ValueError("state_to_atoms returned an empty list.")
    relaxed_atoms = relaxed_atoms_list[0] # Take the first (and likely only) Atoms object

    density_g_cm3 = relaxed_atoms.get_masses().sum() * 1.66053906660 / relaxed_atoms.get_volume()
    relaxed_density_kg_m3 = density_g_cm3 * 1000

    print(f"Relaxed density: {density_g_cm3:.4f} g/cm³ ({relaxed_density_kg_m3:.2f} kg/m³)")

    C11 = elastic_tensor_np[0,0] * 1e9, # Assuming elastic_tensor_np is in GPa, convert to Pa
    C12 = elastic_tensor_np[0,1] * 1e9, # Assuming elastic_tensor_np is in GPa, convert to Pa
    C44 = elastic_tensor_np[3,3] * 1e9, # Assuming elastic_tensor_np is in GPa, convert to Pa
    density = relaxed_density_kg_m3,    # Use the calculated density in kg/m^3


else:
    C11 = 205.3418*1e9
    C12 = 109.0862*1e9
    C44 = (14.1573 + 14.1693 + 25.9939)/3 * 1e9
    density = 6.2766

v_1_2ti = Material(
    formula='V-1_2Ti',
    C11 = C11, # Assuming elastic_tensor_np is in GPa, convert to Pa
    C12 = C12, # Assuming elastic_tensor_np is in GPa, convert to Pa
    C44 = C44, # Assuming elastic_tensor_np is in GPa, convert to Pa
    density = density,    # Use the calculated density in kg/m^3
    crystal_class = 'cubic'
)


# get list of euler angles from ebsd file
data_file = '../../../Dropbox (MIT)/Research/2025/Spring_2025/EBSD+EDS/Data/V-1Ti/Peri-1_2Ti/v12ti_results_4/V1.2Ti_5.6 Specimen 1 Area 1 Montaged Data 2 Montaged Map Data-Corrected.ctf'

# read the ebsd file
header, ebsd_data = read_ctf(data_file)

# --- Test sawbench.grains functionality ---
print("\n--- Testing sawbench.grains ---")

# 1. Extract necessary data from ebsd_data
# Ensure column names match your CTF file output from read_ctf
try:
    euler_angles = ebsd_data[['Euler1', 'Euler2', 'Euler3']].values
    xy_coords = ebsd_data[['X', 'Y']].values
except KeyError as e:
    print(f"KeyError: One of 'Euler1', 'Euler2', 'Euler3', 'X', 'Y' not found in ebsd_data columns.")
    print(f"Available columns: {ebsd_data.columns.tolist()}")
    print("Skipping grains.py testing.")
    # Optionally, re-raise e or exit if these columns are critical
    raise e 

# 2. Determine map dimensions (x_dim, y_dim)
# CTF headers often store XCells and YCells. Let's try to get them.
# The header values in read_ctf are lists of strings, so take the first element and convert to int.
try:
    x_dim = int(header.get('XCells', [0])[0])
    y_dim = int(header.get('YCells', [0])[0])
    if x_dim == 0 or y_dim == 0:
        print("Warning: XCells or YCells not found or zero in header. Attempting to infer dimensions.")
        # Basic inference if X, Y are grid indices (might not be robust for all CTF)
        if xy_coords.size > 0:
            x_unique_sorted = np.sort(np.unique(xy_coords[:, 0]))
            y_unique_sorted = np.sort(np.unique(xy_coords[:, 1]))
            # Check if coords are step-like
            if len(x_unique_sorted) > 1 and len(y_unique_sorted) > 1:
                 x_steps = np.diff(x_unique_sorted)
                 y_steps = np.diff(y_unique_sorted)
                 if np.allclose(x_steps, x_steps[0]) and np.allclose(y_steps, y_steps[0]):
                     # Assuming 0-indexed if they are pixel counts
                     x_dim_inf = len(x_unique_sorted)
                     y_dim_inf = len(y_unique_sorted)
                     print(f"Inferred dimensions from X, Y unique values: x_dim={x_dim_inf}, y_dim={y_dim_inf}")
                     # Heuristic: If actual number of points matches product of unique X/Y counts
                     if x_dim_inf * y_dim_inf == len(xy_coords):
                         x_dim = x_dim_inf
                         y_dim = y_dim_inf
                         # Map XY coordinates to 0-indexed grid for GrainCollection if they are physical
                         # This step depends on how GrainCollection expects xy_coords vs x_dim/y_dim
                         # For now, assume GrainCollection can handle physical xy_coords if x_dim/y_dim are physical counts
                         # Or, if they are already 0-indexed pixels:
                         # xy_coords_for_gc = np.copy(xy_coords)
                         # xy_coords_for_gc[:,0] = (xy_coords[:,0] - np.min(xy_coords[:,0])) / x_steps[0]
                         # xy_coords_for_gc[:,1] = (xy_coords[:,1] - np.min(xy_coords[:,1])) / y_steps[0]
                     else:
                          print("Could not reliably infer grid dimensions. Grain boundary calculations might be affected.")
                          x_dim, y_dim = 0,0 # Mark as unknown
            else: # Not enough unique points to infer step
                print("Could not reliably infer grid dimensions from X,Y unique values.")
                x_dim, y_dim = 0,0 # Mark as unknown


    print(f"Using map dimensions: x_dim={x_dim}, y_dim={y_dim}")
except Exception as e_dim:
    print(f"Error getting/inferring dimensions: {e_dim}. Setting to 0,0.")
    x_dim, y_dim = 0, 0


# 3. Create GrainCollection
# Adjust misorientation_threshold_deg and min_samples_for_grain as needed
# For large datasets, this can take time. Consider downsampling for quick tests.
# e.g., euler_angles_sampled = euler_angles[::10], xy_coords_sampled = xy_coords[::10]
# Then pass sampled data to GrainCollection, but x_dim,y_dim would need adjustment or careful handling.

# For a full run:
print(f"Creating GrainCollection for {len(euler_angles)} points...")
grain_col = GrainCollection(
    euler_angles_deg=euler_angles,
    xy_coords=xy_coords, # Pass the original XY coordinates
    x_dim=x_dim,         # Pass determined/inferred physical grid cell counts
    y_dim=y_dim,
    misorientation_threshold_deg=5.0,
    min_samples_for_grain=10,
    segmentation_method='hdbscan'
)

print(f"Number of grains found: {len(grain_col)}")
if len(grain_col) > 0 and len(grain_col) < 10: # Print details for a few grains
    for g_id, grain_obj in grain_col.grains.items():
        print(grain_obj)

# 4. Calculate IPF colors (using Z-axis as reference direction [0,0,1])
print("Calculating IPF colors...")
ipf_colors = ipf_color(grain_col.all_euler_angles_deg, ref_direction=np.array([0,0,1]))

# 5. Get grain boundaries
# This requires x_dim and y_dim to be correctly set for grid operations.
print("Calculating grain boundaries...")
boundaries_xy = grain_col.get_grain_boundaries()
if boundaries_xy.size == 0 and (x_dim == 0 or y_dim == 0):
    print("Grain boundaries could not be computed (likely due to unknown map dimensions).")
elif boundaries_xy.size == 0:
    print("No grain boundaries found (or computed).")
else:
    print(f"Found {len(boundaries_xy)} boundary points.")


# 6. Plot IPF map with grain boundaries
print("Plotting IPF map...")
fig_ipf, ax_ipf = plot_ebsd_map_mpl(
    xy_coords=grain_col.all_xy_coords, # Use the original XY coordinates for plotting
    property_colors=ipf_colors,
    x_dim=grain_col.x_dim, # Pass the dimensions for aspect ratio and potential gridded display
    y_dim=grain_col.y_dim,
    title=f"IPF Map (Z-direction) - {header.get('JobMode', ['N/A'])[0]}",
    grain_boundaries_xy=boundaries_xy,
    boundary_color='white', # Adjust color for visibility
    boundary_linewidth=0.5,
    point_size=1 # Adjust point_size for large maps
)
plt.show()

# Optional: Plot a map colored by grain ID
if x_dim > 0 and y_dim > 0 :
    print("Plotting Grain ID map...")
    # Create a flat map of grain IDs, assigning unique color to each grain ID
    # Noise points (label -1) can be colored differently (e.g., black)
    grain_id_map_flat = np.copy(grain_col.grain_labels_flat).astype(float)
    grain_id_map_flat[grain_id_map_flat == -1] = np.nan # So noise points are not plotted or colored by colormap

    fig_gid, ax_gid = plot_ebsd_map_mpl(
        xy_coords=grain_col.all_xy_coords,
        property_colors=grain_id_map_flat, # Scalar values for grain IDs
        x_dim=grain_col.x_dim,
        y_dim=grain_col.y_dim,
        title="Grain ID Map",
        cbar_label="Grain ID",
        grain_boundaries_xy=boundaries_xy,
        boundary_color='black',
        boundary_linewidth=0.5,
        point_size=1
    )
    plt.show()
else:
    print("Skipping Grain ID map plot due to unknown dimensions.")


print("--- grains.py testing finished ---")
