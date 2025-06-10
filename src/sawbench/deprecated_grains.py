# This file is a copy of the original grains.py before refactoring to heavily use defdap.
# It's kept for reference or if any specific custom logic needs to be revisited.
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
# For Plotly, ensure it's installed: pip install plotly
# import plotly.graph_objects as go
# import plotly.express as px
from tqdm import tqdm
import pandas as pd
import scipy.signal as sig
import scipy.optimize as opt
from typing import TYPE_CHECKING

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan library not found. HDBSCAN segmentation method will not be available. Try 'pip install hdbscan'")

from .materials import Material
from .saw_calculator import SAWCalculator

if TYPE_CHECKING:
    from defdap import ebsd # For type hinting ebsd.Map


# --- Quaternion, Rotation Matrix, and Misorientation Utilities ---

def euler_to_quaternion(phi1, Phi, phi2, degrees=True):
    """Convert Bunge Euler angles (ZXZ' convention) to quaternion (scalar-last: x,y,z,w)."""
    if degrees:
        phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
    # Scipy's Rotation uses scalar-last for quaternions (x, y, z, w)
    # and expects ZXZ convention for Bunge angles.
    return Rotation.from_euler('ZXZ', np.stack([phi1, Phi, phi2], axis=-1), degrees=False).as_quat()

def euler_to_matrix(phi1, Phi, phi2, degrees=True):
    """Convert Bunge Euler angles (ZXZ' convention) to rotation matrix."""
    if degrees:
        phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
    return Rotation.from_euler('ZXZ', np.stack([phi1, Phi, phi2], axis=-1), degrees=False).as_matrix()

def misorientation_angle(q1, q2, degrees=True):
    """
    Compute misorientation angle between two sets of quaternions.
    q1, q2 are (N, 4) or (4,) arrays of quaternions (scalar-last).
    Returns angle in degrees by default.
    """
    # Ensure q1 and q2 are normalized
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    # Quaternion multiplication: q_delta = q2 * q1_conjugate
    # q_conjugate = [-x, -y, -z, w]
    q1_conj = q1 * np.array([-1, -1, -1, 1])
    
    # Simplified product for dot product approach:
    # The dot product between two quaternions |q1 . q2| = cos(angle/2)
    # Need to handle crystal symmetry for true misorientation, but this gives disorientation.
    dot_product = np.abs(np.sum(q1 * q2, axis=-1))
    dot_product = np.clip(dot_product, -1.0, 1.0) # Ensure valid input for arccos
    
    angle_rad = 2 * np.arccos(dot_product)
    
    # Smallest angle due to symmetry (for cubic, many more, simplified here)
    # angle_rad = np.min(angle_rad, np.pi - angle_rad) # this is not general enough

    if degrees:
        return np.rad2deg(angle_rad)
    return angle_rad

def misorientation_metric_for_dbscan(q_array_i, q_array_j):
    """
    Custom distance metric for DBSCAN: misorientation angle in degrees.
    q_array_i and q_array_j are single quaternion arrays (shape (4,)).
    """
    return misorientation_angle(q_array_i[np.newaxis,:], q_array_j[np.newaxis,:], degrees=True)[0]


# --- Grain Segmentation ---

def segment_grains_dbscan(euler_angles_deg, 
                          misorientation_threshold_deg=5.0, 
                          min_samples=10, 
                          method='sklearn_dbscan', # 'sklearn_dbscan' or 'hdbscan'
                          hdbscan_min_cluster_size=None, # Specific to hdbscan if different from min_samples
                          hdbscan_cluster_selection_epsilon=None # Specific to hdbscan
                         ):
    """
    Segment grains using DBSCAN or HDBSCAN on orientation data (Euler angles).
    Args:
        euler_angles_deg: (N, 3) array of Euler angles in degrees (phi1, Phi, phi2).
        misorientation_threshold_deg: Misorientation threshold in degrees for clustering.
                                      Used as 'eps' in sklearn_dbscan and 'cluster_selection_epsilon' in hdbscan.
        min_samples: Minimum number of pixels to form a grain.
                     Used as 'min_samples' in sklearn_dbscan and 'min_cluster_size' in hdbscan.
        method (str): 'sklearn_dbscan' or 'hdbscan'.
        hdbscan_min_cluster_size (int, optional): Overrides min_samples for hdbscan's min_cluster_size.
        hdbscan_cluster_selection_epsilon (float, optional): Overrides misorientation_threshold_deg for hdbscan's cluster_selection_epsilon.

    Returns:
        labels: (N,) array of grain labels (-1 for noise).
    """
    if euler_angles_deg.ndim == 1: # Single point
        return np.array([0]) if min_samples == 1 else np.array([-1])
    if len(euler_angles_deg) == 0: # Empty input
        return np.array([])
    if len(euler_angles_deg) < min_samples: # Not enough points to form any cluster
        return np.full(len(euler_angles_deg), -1)

    quats = euler_to_quaternion(euler_angles_deg[:, 0], euler_angles_deg[:, 1], euler_angles_deg[:, 2], degrees=True)

    if method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            print("Error: HDBSCAN method selected, but hdbscan library is not installed. Falling back to sklearn_dbscan.")
            method = 'sklearn_dbscan'
        else:
            print("Attempting HDBSCAN segmentation...")
            # Parameters for HDBSCAN:
            # min_cluster_size: similar to min_samples in DBSCAN.
            # metric: our custom misorientation function.
            # cluster_selection_epsilon: This is crucial for HDBSCAN to behave like DBSCAN
            #                            in terms of extracting flat clusters based on an epsilon.
            # allow_single_cluster: True, can be useful for small datasets or isolated grains.
            # core_dist_n_jobs: for parallel processing of core distances (part of HDBSCAN).
            
            _min_cluster_size = hdbscan_min_cluster_size if hdbscan_min_cluster_size is not None else min_samples
            _cluster_selection_epsilon = hdbscan_cluster_selection_epsilon if hdbscan_cluster_selection_epsilon is not None else misorientation_threshold_deg

            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=_min_cluster_size,
                    metric=misorientation_metric_for_dbscan, # Our custom pairwise metric
                    cluster_selection_epsilon=_cluster_selection_epsilon,
                    allow_single_cluster=True, # Can be useful
                    # gen_min_span_tree=True, # If you want to explore the tree
                    # approx_min_span_tree=True, # For larger datasets, but might be less accurate
                    core_dist_n_jobs=-1 # Use all available cores for core distance calculation
                )
                labels = clusterer.fit_predict(quats)
                print("HDBSCAN segmentation completed.")
                return labels
            except Exception as e:
                print(f"HDBSCAN failed: {e}. Falling back to sklearn_dbscan.")
                method = 'sklearn_dbscan' # Fallback on HDBSCAN failure

    # Fallback to sklearn_dbscan if method is 'sklearn_dbscan' or HDBSCAN failed/unavailable
    if method == 'sklearn_dbscan':
        print("Attempting sklearn DBSCAN segmentation...")
        # (The existing sklearn DBSCAN logic with tqdm for pdist fallback)
        try:
            # The 'metric' callable path for DBSCAN.
            print("Attempting DBSCAN with callable metric (n_jobs=-1 for parallelism)...")
            db = DBSCAN(eps=misorientation_threshold_deg, min_samples=min_samples, metric=misorientation_metric_for_dbscan, algorithm='auto', n_jobs=-1)
            labels = db.fit_predict(quats)
            print("DBSCAN with callable metric completed.")
        except Exception as e:
            print(f"DBSCAN with callable metric failed: {e}.")
            print("Attempting DBSCAN with precomputed distance matrix (this can be slow and memory intensive).")
            try:
                num_points = len(quats)
                num_pairs = num_points * (num_points - 1) // 2
                distance_matrix_condensed = np.array([])

                if num_pairs > 0:
                    pbar_pdist = tqdm(total=num_pairs, desc="Pairwise misorientations (pdist)", unit="pair", leave=False)
                    def misorientation_metric_for_pdist_tqdm(u_quat, v_quat):
                        pbar_pdist.update(1)
                        return misorientation_angle(u_quat[np.newaxis,:], v_quat[np.newaxis,:], degrees=True)[0]
                    
                    print("Calculating pairwise misorientation distance matrix (condensed form)...")
                    distance_matrix_condensed = pdist(quats, metric=misorientation_metric_for_pdist_tqdm)
                    pbar_pdist.close()
                elif num_points > 0:
                    print("Only one point/not enough for pairs in pdist.")
                else: # num_points = 0
                     pass # distance_matrix_condensed remains empty

                print("Converting to square form...")
                distance_matrix = squareform(distance_matrix_condensed)
                
                if distance_matrix.size == 0 and len(euler_angles_deg) > 0:
                     print("Warning: Precomputed distance matrix is empty for non-empty input. Defaulting labels.")
                     return np.array([0]) if min_samples == 1 and len(euler_angles_deg) == 1 else np.full(len(euler_angles_deg), -1)

                print("Running DBSCAN on precomputed matrix (n_jobs=-1 for parallelism)...")
                db_precomputed = DBSCAN(eps=misorientation_threshold_deg, min_samples=min_samples, metric='precomputed', algorithm='auto', n_jobs=-1)
                labels = db_precomputed.fit_predict(distance_matrix)
                print("DBSCAN with precomputed matrix completed.")

            except MemoryError:
                print("MemoryError during precomputation or DBSCAN with precomputed matrix. DBSCAN failed.")
                labels = np.full(len(euler_angles_deg), -1) 
            except Exception as e_inner:
                print(f"DBSCAN with precomputed matrix failed: {e_inner}")
                labels = np.full(len(euler_angles_deg), -1)
        return labels
    
    # Should not be reached if method is valid
    print(f"Warning: Unknown segmentation method '{method}'. Returning noise labels.")
    return np.full(len(euler_angles_deg), -1)


# --- Grain and GrainCollection Classes ---

class Grain:
    """Represents a single grain with its properties."""
    def __init__(self, grain_id, pixel_indices, all_euler_angles_deg, all_xy_coords):
        """
        Args:
            grain_id (int): Unique identifier for the grain.
            pixel_indices (np.ndarray): 1D array of flat indices belonging to this grain.
            all_euler_angles_deg (np.ndarray): (TotalPixels, 3) array of Euler angles for all pixels.
            all_xy_coords (np.ndarray): (TotalPixels, 2) array of XY coordinates for all pixels.
        """
        self.id = grain_id
        self.pixel_indices = np.array(pixel_indices, dtype=int)
        
        if self.pixel_indices.size == 0:
            self.euler_angles_deg = np.empty((0, 3))
            self.xy_coords = np.empty((0, 2))
            self.quaternions = np.empty((0,4))
            self.mean_euler_deg = np.array([np.nan, np.nan, np.nan])
            self.mean_quaternion = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            self.euler_angles_deg = all_euler_angles_deg[self.pixel_indices]
            self.xy_coords = all_xy_coords[self.pixel_indices]
            self.quaternions = euler_to_quaternion(
                self.euler_angles_deg[:, 0],
                self.euler_angles_deg[:, 1],
                self.euler_angles_deg[:, 2],
                degrees=True
            )
            self.mean_quaternion = self._calculate_mean_quaternion()
            # Convert mean quaternion back to Euler for a representative orientation
            try:
                self.mean_euler_deg = Rotation.from_quat(self.mean_quaternion).as_euler('ZXZ', degrees=True)
            except ValueError: # Handle potential issues with all-zero quaternions if grain is empty
                self.mean_euler_deg = np.array([np.nan, np.nan, np.nan])


        self.properties = {}  # For storing arbitrary calculated properties

    def _calculate_mean_quaternion(self):
        """Calculates the mean orientation quaternion for the grain."""
        if self.quaternions.shape[0] == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan]) # x,y,z,w

        # Method from: DOI: 10.1007/s10851-009-0161-2
        # Form the symmetric 4x4 matrix M
        M = np.dot(self.quaternions.T, self.quaternions)
        # The mean quaternion is the eigenvector corresponding to the largest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        mean_q = eigenvectors[:, np.argmax(eigenvalues)]
        return mean_q / np.linalg.norm(mean_q) # Ensure normalization

    def set_property(self, key, value):
        """Set a custom property for the grain."""
        self.properties[key] = value

    def get_property(self, key, default=None):
        """Get a custom property, returning default if not found."""
        return self.properties.get(key, default)

    @property
    def num_pixels(self):
        return len(self.pixel_indices)

    @property
    def area(self, pixel_area=1.0): # Assuming pixel_area can be set later
        """Calculate grain area."""
        return self.num_pixels * pixel_area

    def __repr__(self):
        return f"<Grain ID: {self.id}, Pixels: {self.num_pixels}, Mean Euler: {np.round(self.mean_euler_deg, 2)} deg>"

    def get_boundary_pixels(self, x_dim, y_dim, all_grain_labels_flat):
        """
        Identifies boundary pixels of this grain on a 2D grid.
        Args:
            x_dim (int): Number of columns in the EBSD map.
            y_dim (int): Number of rows in the EBSD map.
            all_grain_labels_flat (np.ndarray): Flat array of grain labels for all pixels.
        Returns:
            np.ndarray: XY coordinates of boundary pixels.
        """
        if self.num_pixels == 0:
            return np.empty((0,2))

        grain_mask_2d = np.zeros((y_dim, x_dim), dtype=bool)
        map_indices_y = self.xy_coords[:, 1].astype(int) # Assuming xy_coords are pixel indices if not scaled
        map_indices_x = self.xy_coords[:, 0].astype(int)
        
        # This assumes xy_coords are raw pixel grid coordinates (0 to x_dim-1, 0 to y_dim-1)
        # If xy_coords are physical units, they need to be mapped to grid indices first.
        # For now, let's assume xy_coords can be used to create a mask if they map directly to a grid.
        # This part needs to be robust to how xy_coords are defined.
        
        # A simpler way using flat indices and reshaping:
        grain_map_2d = all_grain_labels_flat.reshape((y_dim, x_dim))
        this_grain_mask = (grain_map_2d == self.id)
        
        from scipy.ndimage import binary_dilation
        dilated_mask = binary_dilation(this_grain_mask)
        boundary_mask = dilated_mask & ~this_grain_mask
        
        # We also need to find pixels *within* the grain that are adjacent to *other* grains or noise
        eroded_mask = np.zeros_like(this_grain_mask)
        if np.any(this_grain_mask): # only if grain has pixels
            from scipy.ndimage import binary_erosion
            eroded_mask = binary_erosion(this_grain_mask)
        
        internal_boundary_mask = this_grain_mask & ~eroded_mask
        
        # Combine and get coordinates
        # For plotting, often the pixels *just outside* or *just inside* are used.
        # Let's use the internal boundary.
        boundary_y_idx, boundary_x_idx = np.where(internal_boundary_mask)
        
        # Map these grid indices back to original XY coordinates if necessary
        # If original xy_coords were not simple grid indices, this mapping needs care.
        # Assuming for now, we can use these grid indices for plotting.
        # If all_xy_coords is a full grid:
        # all_xy_coords_2d_x = all_xy_coords[:,0].reshape(y_dim, x_dim)
        # all_xy_coords_2d_y = all_xy_coords[:,1].reshape(y_dim, x_dim)
        # boundary_plot_xy = np.vstack([all_xy_coords_2d_x[boundary_y_idx, boundary_x_idx], 
        #                               all_xy_coords_2d_y[boundary_y_idx, boundary_x_idx]]).T
        
        # For now, return the grid indices of boundary pixels for simplicity
        return np.column_stack((boundary_x_idx, boundary_y_idx))


class GrainCollection:
    """Manages a collection of Grain objects from an EBSD map."""
    def __init__(self, euler_angles_deg, xy_coords, x_dim=None, y_dim=None, 
                 misorientation_threshold_deg=10.0, min_samples_for_grain=5,
                 segmentation_method='sklearn_dbscan', # 'sklearn_dbscan' or 'hdbscan'
                 hdbscan_min_cluster_size=None,
                 hdbscan_cluster_selection_epsilon=None
                 ):
        """
        Args:
            euler_angles_deg (np.ndarray): (N, 3) array of all pixel Euler angles (phi1, Phi, phi2) in degrees.
            xy_coords (np.ndarray): (N, 2) array of all pixel XY coordinates.
            x_dim (int, optional): Number of columns in the EBSD map grid.
            y_dim (int, optional): Number of rows in the EBSD map grid.
            misorientation_threshold_deg (float): For DBSCAN/HDBSCAN epsilon.
            min_samples_for_grain (int): For DBSCAN/HDBSCAN min_samples/min_cluster_size.
            segmentation_method (str): 'sklearn_dbscan' or 'hdbscan'.
            hdbscan_min_cluster_size (int, optional): Override for hdbscan's min_cluster_size.
            hdbscan_cluster_selection_epsilon (float, optional): Override for hdbscan's cluster_selection_epsilon.
        """
        self.all_euler_angles_deg = euler_angles_deg
        self.all_xy_coords = xy_coords
        self.num_total_pixels = len(euler_angles_deg)

        if x_dim is None or y_dim is None:
            # Try to infer from xy_coords if they are grid-like
            if np.all(np.mod(xy_coords, 1) == 0): # Check if they look like indices
                 self.y_dim = int(np.max(xy_coords[:, 1])) + 1 if xy_coords.size > 0 else 0
                 self.x_dim = int(np.max(xy_coords[:, 0])) + 1 if xy_coords.size > 0 else 0
                 print(f"Inferred map dimensions: X={self.x_dim}, Y={self.y_dim}")
            else: # Cannot infer, user must provide or some features won't work
                 self.x_dim = 0
                 self.y_dim = 0
                 print("Warning: x_dim and y_dim not provided and couldn't be inferred. Grid operations might fail.")
        else:
            self.x_dim = x_dim
            self.y_dim = y_dim

        print(f"Segmenting grains using {segmentation_method}...")
        self.grain_labels_flat = segment_grains_dbscan(
            self.all_euler_angles_deg,
            misorientation_threshold_deg=misorientation_threshold_deg,
            min_samples=min_samples_for_grain,
            method=segmentation_method,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon
        )
        print(f"Found {len(np.unique(self.grain_labels_flat[self.grain_labels_flat != -1]))} grains (excluding noise).")

        self.grains = {} # Dict to store Grain objects, keyed by grain_id
        unique_labels = np.unique(self.grain_labels_flat)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            pixel_indices = np.where(self.grain_labels_flat == label)[0]
            self.grains[label] = Grain(label, pixel_indices, self.all_euler_angles_deg, self.all_xy_coords)
        
        self.noise_pixel_indices = np.where(self.grain_labels_flat == -1)[0]

    def __getitem__(self, grain_id):
        return self.grains.get(grain_id)

    def __iter__(self):
        return iter(self.grains.values())

    def __len__(self):
        return len(self.grains)

    def get_grain_boundaries(self):
        """Returns a list of XY coordinates for all grain boundaries."""
        all_boundaries = []
        if self.x_dim == 0 or self.y_dim == 0:
            print("Warning: Cannot compute boundaries without map dimensions (x_dim, y_dim).")
            return np.empty((0,2))
            
        for grain in self:
            boundaries = grain.get_boundary_pixels(self.x_dim, self.y_dim, self.grain_labels_flat)
            if boundaries.size > 0:
                all_boundaries.append(boundaries)
        if not all_boundaries:
            return np.empty((0,2))
        return np.concatenate(all_boundaries, axis=0)
        
    def add_property_to_grains(self, property_name, values_per_pixel):
        """Adds a property to each grain, averaged over its pixels."""
        if len(values_per_pixel) != self.num_total_pixels:
            raise ValueError("Length of values_per_pixel must match total number of pixels.")
        for grain in self:
            if grain.num_pixels > 0:
                grain_property_values = values_per_pixel[grain.pixel_indices]
                grain.set_property(property_name, np.nanmean(grain_property_values)) # nanmean for robustness
            else:
                grain.set_property(property_name, np.nan)


# --- IPF Color Calculation ---
def ipf_color(euler_angles_deg, ref_direction=np.array([0, 0, 1]), cmap_type='rgb', degrees=True):
    """
    Calculates IPF (Inverse Pole Figure) colors for given Euler angles.
    Args:
        euler_angles_deg (np.ndarray): (N, 3) array of Euler angles (phi1, Phi, phi2) in degrees.
        ref_direction (np.ndarray): Reference direction in sample coordinates (default: Z-axis [0,0,1]).
        cmap_type (str): 'rgb' for standard IPF, or a matplotlib colormap name.
        degrees (bool): Whether euler_angles are in degrees.
    Returns:
        np.ndarray: (N, 3) array of RGB colors (0-1 range).
    """
    phi1 = euler_angles_deg[:, 0]
    Phi = euler_angles_deg[:, 1]
    phi2 = euler_angles_deg[:, 2]

    # Convert Euler angles to rotation matrices
    # Rotation matrix g transforms crystal coordinates to sample coordinates.
    # We need g_transpose to transform sample_ref_dir to crystal coordinates.
    g_inv = euler_to_matrix(phi1, Phi, phi2, degrees=degrees) # This is g_crystal_to_sample
    # if using scipy.Rotation, .as_matrix() gives crystal to sample.
    # We want sample_to_crystal for the reference direction.
    # g_sample_to_crystal = np.transpose(g_inv, axes=(0,2,1))
    # Crystal direction corresponding to sample ref_direction
    # crystal_dir = np.einsum('nij,j->ni', g_sample_to_crystal, ref_direction)

    # Simpler: use Rotation object's apply method with inverse rotation
    rotations = Rotation.from_euler('ZXZ', np.stack([phi1, Phi, phi2], axis=-1), degrees=degrees)
    crystal_dir = rotations.apply(ref_direction, inverse=True) # Transform sample vector to crystal frame

    # Normalize crystal directions
    crystal_dir_norm = crystal_dir / np.linalg.norm(crystal_dir, axis=1, keepdims=True)

    # Apply cubic symmetry (take absolute values and re-normalize for standard triangle)
    # This maps to the [001]-[011]-[111] standard stereographic triangle for cubic.
    # For non-cubic, specific symmetry operations are needed.
    # For simplicity, we use a common approach for cubic IPF coloring:
    v = np.abs(crystal_dir_norm)
    rgb = v / np.sum(v, axis=1, keepdims=True) # Normalize components to sum to 1 (approx for color)
    
    # A more standard IPF coloring for cubic:
    # R = |h|, G = |k|, B = |l| after permuting to fundamental triangle and normalizing.
    # Here, x,y,z components are used directly.
    r = np.abs(crystal_dir_norm[:, 0])
    g = np.abs(crystal_dir_norm[:, 1])
    b = np.abs(crystal_dir_norm[:, 2])

    # Normalize colors to max 1 for RGB
    max_val = np.max(np.stack([r, g, b], axis=-1), axis=1, keepdims=True)
    max_val[max_val == 0] = 1 # Avoid division by zero for black points

    colors = np.stack([r/max_val[:,0], g/max_val[:,0], b/max_val[:,0]], axis=-1)
    colors = np.clip(colors, 0, 1) # Ensure colors are in [0,1]

    if cmap_type != 'rgb':
        # This part is if you want to map a scalar derived from orientation to a colormap
        # For true IPF, 'rgb' is standard.
        # For example, map deviation from a reference orientation to a colormap:
        # scalar_value = some_orientation_scalar_here
        # cmap = plt.get_cmap(cmap_type)
        # norm = mcolors.Normalize(vmin=np.min(scalar_value), vmax=np.max(scalar_value))
        # colors = cmap(norm(scalar_value))[:, :3] # Get RGB from RGBA
        pass # Keep as direct RGB for now

    return colors


# --- Plotting Functions ---

def plot_ebsd_map_mpl(xy_coords, property_colors, x_dim, y_dim, 
                      title="EBSD Map", cbar_label="Property", ax=None,
                      grain_boundaries_xy=None, point_size=5, boundary_color='black', boundary_linewidth=0.5):
    """
    Plots an EBSD map using Matplotlib.
    Args:
        xy_coords (np.ndarray): (N, 2) array of XY coordinates for each pixel.
        property_colors (np.ndarray): (N, 3) RGB colors or (N,) scalar values for coloring.
        x_dim, y_dim (int): Dimensions of the map grid for proper image display or aspect.
        title (str): Plot title.
        cbar_label (str): Colorbar label (if property_colors is scalar).
        ax (plt.Axes, optional): Matplotlib axes to plot on.
        grain_boundaries_xy (np.ndarray, optional): (M,2) XY coords of boundary points.
        point_size (int): Size of points in scatter plot if not gridded.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8 * (y_dim / x_dim if x_dim > 0 else 1)))
    else:
        fig = ax.get_figure()

    # Determine if data is gridded for imshow, or scattered
    is_gridded = (len(xy_coords) == x_dim * y_dim and x_dim > 0 and y_dim > 0)
    
    if is_gridded and property_colors.ndim == 2 and property_colors.shape[1] == 3: # RGB colors for gridded data
        img_data = property_colors.reshape((y_dim, x_dim, 3))
        # Need to ensure xy_coords correspond to grid order if reshaping.
        # If xy_coords are not perfectly ordered for reshape, scatter is safer.
        # For simplicity, assume xy_coords and property_colors are ordered for reshape.
        # A robust way: create an empty grid and fill it.
        grid_img = np.zeros((y_dim, x_dim, 3))
        # Assuming xy_coords are pixel indices [x,y]
        x_indices = xy_coords[:, 0].astype(int)
        y_indices = xy_coords[:, 1].astype(int)
        grid_img[y_indices, x_indices] = property_colors
        
        ax.imshow(grid_img, origin='lower', aspect='equal', 
                  extent=(np.min(xy_coords[:,0])-0.5, np.max(xy_coords[:,0])+0.5, 
                          np.min(xy_coords[:,1])-0.5, np.max(xy_coords[:,1])+0.5)
                  if xy_coords.size > 0 else None) # Adjust extent for pixel centers
    elif property_colors.ndim == 1: # Scalar property values
        sc = ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c=property_colors, s=point_size, cmap='viridis', marker='s', edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    else: # RGB colors for scattered data
        ax.scatter(xy_coords[:, 0], xy_coords[:, 1], color=property_colors, s=point_size, marker='s', edgecolors='none')

    if grain_boundaries_xy is not None and grain_boundaries_xy.size > 0:
        ax.plot(grain_boundaries_xy[:, 0], grain_boundaries_xy[:, 1], color=boundary_color, 
                linestyle='None', marker='.', markersize=boundary_linewidth) # Plot as fine dots

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig, ax

# Placeholder for Plotly functions - to be added if desired
# def plot_ebsd_map_plotly(...):
#     pass

# def plot_property_histogram_mpl(grain_collection, property_key, bins=30, ax=None, title=None):
#    """Plots a histogram of a specified grain property using Matplotlib."""
#    pass


def _gauss(x: np.ndarray, A: float, mu: float, sig_val: float) -> np.ndarray:
    """Gaussian function for peak fitting."""
    return A * np.exp(-(x - mu)**2 / (2 * sig_val**2))

def extract_experimental_peak_parameters(
    freq_axis: np.ndarray,
    amp_trace: np.ndarray,
    num_peaks_to_extract: int = 3,
    num_candidates: int = 5,
    prominence: float = 0.04,
    min_peak_height_relative: float = 0.1
) -> np.ndarray:
    """
    Finds multiple peaks in an experimental spectrum, fits Gaussians,
    and returns parameters for the specified number of peaks, sorted by frequency.

    Args:
        freq_axis (np.ndarray): Array of frequencies.
        amp_trace (np.ndarray): Array of amplitudes for the spectrum.
        num_peaks_to_extract (int): Number of best peaks to return parameters for.
        num_candidates (int): Number of initial candidate peaks to consider (sorted by prominence).
        prominence (float): Prominence for scipy.signal.find_peaks.
        min_peak_height_relative (float): Minimum peak height relative to the max amplitude.

    Returns:
        np.ndarray: An array of shape (num_peaks_to_extract, 3) for [Amplitude, Mu, Sigma].
                    Padded with NaNs if fewer peaks are found.
    """
    default_peak_params = np.full((num_peaks_to_extract, 3), np.nan)
    if amp_trace.size == 0 or amp_trace.max() <= 1e-9: # Check for empty or zero signal
        return default_peak_params

    min_abs_height = amp_trace.max() * min_peak_height_relative
    initial_idx, prop = sig.find_peaks(amp_trace, prominence=prominence, height=min_abs_height)

    if len(initial_idx) == 0:
        return default_peak_params

    sorted_candidate_indices = initial_idx[np.argsort(prop["prominences"])[::-1]][:num_candidates]
    
    fitted_peaks = []
    for p_idx in sorted_candidate_indices:
        sl_start = max(0, p_idx - 5)
        sl_end = min(len(freq_axis), p_idx + 6)
        sl = slice(sl_start, sl_end)
        
        if sl_end - sl_start < 3: 
            continue

        try:
            A0, mu0 = amp_trace[p_idx], freq_axis[p_idx]
            sigma_0_guess = max(5e4, (freq_axis[sl][-1] - freq_axis[sl][0]) / 6.0 if (freq_axis[sl][-1] - freq_axis[sl][0]) > 0 else 5e4)
            popt, _ = opt.curve_fit(_gauss, freq_axis[sl], amp_trace[sl], p0=(A0, mu0, sigma_0_guess), maxfev=8000)
            
            if popt[0] > 0 and popt[2] > 0 and freq_axis.min() <= popt[1] <= freq_axis.max():
                 fitted_peaks.append({'A': popt[0], 'mu': popt[1], 'sigma': popt[2]})
        except (RuntimeError, ValueError):
            pass 

    if not fitted_peaks:
        return default_peak_params

    fitted_peaks.sort(key=lambda p: p['A'], reverse=True)
    selected_peaks = fitted_peaks[:num_peaks_to_extract]
    selected_peaks.sort(key=lambda p: p['mu'])

    output_params = np.full((num_peaks_to_extract, 3), np.nan)
    for i, peak in enumerate(selected_peaks):
        if i < num_peaks_to_extract:
            output_params[i, 0] = peak['A']
            output_params[i, 1] = peak['mu']
            output_params[i, 2] = peak['sigma']
            
    return output_params


def calculate_saw_frequencies_for_ebsd_grains(
    ebsd_map_obj: "ebsd.Map",
    material: Material,
    wavelength: float,
    saw_calc_angle: float = 0.0,
    saw_calc_sampling: int = 400,
    saw_calc_psaw: int = 0
) -> pd.DataFrame:
    """
    Calculates SAW frequencies for each grain in a processed EBSD map.

    Args:
        ebsd_map_obj (defdap.ebsd.Map): Processed EBSD map object with grains identified.
        material (Material): Material object with elastic constants and density.
        wavelength (float): Wavelength (in meters) for SAW f = v/lambda calculation.
        saw_calc_angle (float): Angle for SAW speed calculation in SAWCalculator. Default 0.0.
        saw_calc_sampling (int): Sampling parameter for SAWCalculator. Default 400.
        saw_calc_psaw (int): Psaw parameter for SAWCalculator. Default 0.

    Returns:
        pd.DataFrame: DataFrame containing grain ID, Euler angles (radians), size (pixels and um^2),
                      and calculated peak SAW frequency (Hz).
    """
    grains_data_list = []
    if not ebsd_map_obj.grainList:
        print("Warning: No grains found in EBSD map. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "Grain ID", "Euler1 (rad)", "Euler2 (rad)", "Euler3 (rad)", 
            "Size (pixels)", "Size (um^2)", "Peak SAW Frequency (Hz)"
        ])

    for grain in tqdm(ebsd_map_obj.grainList, desc="Calculating SAW for EBSD grains", unit="grain"):
        grain.calcAverageOri()
        euler_angles_rad = grain.refOri.eulerAngles()  # (phi1, PHI, phi2) in radians

        peak_saw_freq_hz = np.nan
        try:
            calculator = SAWCalculator(material, euler_angles_rad) 
            v_mps, _, _ = calculator.get_saw_speed(
                angle=saw_calc_angle, 
                sampling=saw_calc_sampling, 
                psaw=saw_calc_psaw
            )
            if v_mps and len(v_mps) > 0:
                peak_saw_freq_hz = v_mps[0] / wavelength
            else:
                # print(f"Warning: SAW speed calculation returned no result for grain {grain.grainID}.") # Can be verbose
                pass
        except Exception:
            # print(f"Error calculating SAW for grain {grain.grainID}: {e}") # Can be verbose
            pass

        grains_data_list.append({
            "Grain ID": grain.grainID,
            "Euler1 (rad)": euler_angles_rad[0],
            "Euler2 (rad)": euler_angles_rad[1],
            "Euler3 (rad)": euler_angles_rad[2],
            "Size (pixels)": len(grain.coordList), 
            "Size (um^2)": len(grain.coordList) * (ebsd_map_obj.stepSize**2),
            "Peak SAW Frequency (Hz)": peak_saw_freq_hz
        })

    df_grains = pd.DataFrame(grains_data_list)
    return df_grains


def create_ebsd_saw_frequency_map(
    ebsd_map_obj: "ebsd.Map",
    grains_saw_data_df: pd.DataFrame
) -> np.ndarray | None:
    """
    Creates a 2D map of SAW frequencies based on EBSD grain data.

    Args:
        ebsd_map_obj (defdap.ebsd.Map): The EBSD map object from defdap.
                                     `findGrains()` must have been called.
        grains_saw_data_df (pd.DataFrame): DataFrame with 'Grain ID' and 
                                        'Peak SAW Frequency (Hz)' columns.

    Returns:
        np.ndarray | None: A 2D numpy array (yDim, xDim) representing the SAW frequency map.
                           Pixels not belonging to a grain with calculated frequency will be NaN.
                           Returns None if essential data is missing.
    """
    if not (hasattr(ebsd_map_obj, 'grainIDMap') and ebsd_map_obj.grainIDMap is not None):
        print("Error: ebsd_map_obj.grainIDMap is not available. Ensure findGrains() was called.")
        return None
        
    if 'Grain ID' not in grains_saw_data_df.columns or \
       'Peak SAW Frequency (Hz)' not in grains_saw_data_df.columns:
        print("Error: grains_saw_data_df must contain 'Grain ID' and 'Peak SAW Frequency (Hz)' columns.")
        return None

    grain_id_to_freq_map = pd.Series(
        grains_saw_data_df['Peak SAW Frequency (Hz)'].values, 
        index=grains_saw_data_df['Grain ID']
    ).to_dict()

    # ebsd_map_obj.grainIDMap is (yDim, xDim)
    grain_map_ids_array = ebsd_map_obj.grainIDMap 
    
    saw_freq_map_array = np.full(grain_map_ids_array.shape, np.nan, dtype=float)

    # Vectorized approach for mapping frequencies
    # Create an array of grain IDs present in the frequency map data
    unique_grain_ids_in_data = np.array(list(grain_id_to_freq_map.keys()))
    
    # For each unique grain ID that has a frequency, find its locations in grain_map_ids_array
    # and assign the corresponding frequency.
    # This is more efficient than iterating pixel by pixel in Python.
    for grain_id_val in unique_grain_ids_in_data:
        if grain_id_val in grain_id_to_freq_map: # Check if grain_id actually in our data
            frequency = grain_id_to_freq_map[grain_id_val]
            if pd.notna(frequency): # Only map valid frequencies
                 saw_freq_map_array[grain_map_ids_array == grain_id_val] = frequency
    
    # Alternative pixel-by-pixel loop (slower but perhaps easier to debug initially):
    # for r_idx in range(grain_map_ids_array.shape[0]):
    #     for c_idx in range(grain_map_ids_array.shape[1]):
    #         grain_id_at_pixel = grain_map_ids_array[r_idx, c_idx]
    #         if grain_id_at_pixel in grain_id_to_freq_map:
    #             saw_freq_map_array[r_idx, c_idx] = grain_id_to_freq_map[grain_id_at_pixel]
                
    print(f"EBSD SAW frequency map created with shape: {saw_freq_map_array.shape}")
    return saw_freq_map_array


if __name__ == '__main__':
    # --- Example Usage ---
    print("Running example for sawbench.grains module...")

    # 1. Create dummy EBSD data
    N_points_x, N_points_y = 50, 40 # Small map for example
    total_pixels = N_points_x * N_points_y
    xx, yy = np.meshgrid(np.arange(N_points_x), np.arange(N_points_y))
    xy_coords_flat = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Create two grains with distinct orientations and some noise
    euler_angles_flat = np.zeros((total_pixels, 3))
    # Grain 1 (majority of map)
    euler_angles_flat[:total_pixels // 2, :] = np.array([10, 20, 30]) 
    # Grain 2 (other part of map)
    euler_angles_flat[total_pixels // 2 : total_pixels *3//4, :] = np.array([80, 50, 70])
    # Noise/Random orientations for the rest
    rng = np.random.default_rng(42)
    euler_angles_flat[total_pixels*3//4:,0] = rng.uniform(0,360, size=total_pixels - total_pixels*3//4)
    euler_angles_flat[total_pixels*3//4:,1] = rng.uniform(0,180, size=total_pixels - total_pixels*3//4)
    euler_angles_flat[total_pixels*3//4:,2] = rng.uniform(0,360, size=total_pixels - total_pixels*3//4)
    
    # Add some random scatter to Grain 1 and 2 eulers to make clustering more interesting
    euler_angles_flat[:total_pixels*3//4, :] += rng.normal(0, 2, size=(total_pixels*3//4, 3))


    # 2. Create GrainCollection and segment grains
    # --- Test with sklearn_dbscan (default) ---
    print("\n--- Testing with sklearn_dbscan ---")
    grain_col_sklearn = GrainCollection(
        euler_angles_deg=euler_angles_flat,
        xy_coords=xy_coords_flat,
        x_dim=N_points_x,
        y_dim=N_points_y,
        misorientation_threshold_deg=7.0, 
        min_samples_for_grain=5,
        segmentation_method='sklearn_dbscan' 
    )
    print(f"Number of grains found (sklearn_dbscan): {len(grain_col_sklearn)}")
    # for g_id, grain in list(grain_col_sklearn.grains.items())[:2]: print(grain)


    if HDBSCAN_AVAILABLE:
        print("\n--- Testing with hdbscan ---")
        grain_col_hdbscan = GrainCollection(
            euler_angles_deg=euler_angles_flat,
            xy_coords=xy_coords_flat,
            x_dim=N_points_x,
            y_dim=N_points_y,
            misorientation_threshold_deg=7.0, # This will be used as cluster_selection_epsilon
            min_samples_for_grain=5,          # This will be used as min_cluster_size
            segmentation_method='hdbscan'
        )
        print(f"Number of grains found (hdbscan): {len(grain_col_hdbscan)}")
        # for g_id, grain in list(grain_col_hdbscan.grains.items())[:2]: print(grain)
        
        # Set current grain_col to the one from hdbscan for plotting if available
        grain_col_to_plot = grain_col_hdbscan
    else:
        print("\nSkipping HDBSCAN test as library is not available.")
        grain_col_to_plot = grain_col_sklearn


    # 3. Calculate IPF colors for all pixels (using the chosen grain_col_to_plot)
    print("\nCalculating IPF colors...")
    ipf_colors_flat = ipf_color(grain_col_to_plot.all_euler_angles_deg, ref_direction=np.array([0,0,1]))

    # 4. Plot IPF map with grain boundaries
    boundaries_xy = grain_col_to_plot.get_grain_boundaries()
    
    fig_ipf, ax_ipf = plot_ebsd_map_mpl(
        grain_col_to_plot.all_xy_coords, 
        ipf_colors_flat, 
        x_dim=grain_col_to_plot.x_dim, 
        y_dim=grain_col_to_plot.y_dim,
        title=f"IPF Map (Z-direction) - Method: {grain_col_to_plot.grain_labels_flat.size > 0 and grain_col_to_plot.segmentation_method_used if hasattr(grain_col_to_plot, 'segmentation_method_used') else 'N/A'}", # Requires storing method used
        grain_boundaries_xy=boundaries_xy,
        boundary_color='white',
        boundary_linewidth=0.8
    )
    plt.show()

    # 5. Add a dummy property
    for grain in grain_col_to_plot:
        grain.set_property("mean_Phi", grain.mean_euler_deg[1])
    
    grain_phi_map_flat = np.full(grain_col_to_plot.num_total_pixels, np.nan)
    for grain in grain_col_to_plot:
        grain_phi_map_flat[grain.pixel_indices] = grain.get_property("mean_Phi")
    
    if grain_col_to_plot.noise_pixel_indices.size > 0:
         grain_phi_map_flat[grain_col_to_plot.noise_pixel_indices] = -1 

    fig_prop, ax_prop = plot_ebsd_map_mpl(
        grain_col_to_plot.all_xy_coords,
        grain_phi_map_flat, 
        x_dim=grain_col_to_plot.x_dim,
        y_dim=grain_col_to_plot.y_dim,
        title="Grain Map colored by Mean Phi",
        cbar_label="Mean Phi (degrees)",
        grain_boundaries_xy=boundaries_xy,
        boundary_color='black',
        boundary_linewidth=0.8
    )
    plt.show()

    print("\nExample run finished.") 