import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
import seaborn as sns
import os
from tqdm import tqdm 
from scipy.ndimage import gaussian_filter
from skimage import measure, color
from .euler_transformations import euler2matrix # Use function directly

class EBSDAnalyzer:
    def __init__(self, file_path, step_size=None, min_grain_size_pixels=50):
        """
        Initialize EBSD analyzer.

        Args:
            file_path: Path to the .ang file.
            step_size: Grid step size in microns (if known), used for area/diameter calculations.
                       If None, area will be in pixels^2.
            min_grain_size_pixels: Minimum number of pixels to consider a cluster a grain.
        """
        self.file_path = file_path
        self.step_size = step_size 
        self.min_grain_size_pixels = min_grain_size_pixels
        self.data = self._load_data(file_path)
        self.grain_properties = None # Initialize as None
        self.grain_labels = None
        self.valid_points_mask = None
        self.smoothed_boundaries = None
        
    def _load_data(self, file_path):
        """
        Load and parse the .ang file.
        Handles potential header variations and ensures numeric types.
        """
        data_lines = []
        header_info = {}
        try:
            with open(file_path, 'r') as f:
                header_mode = True
                for line in f:
                    line = line.strip()
                    if not line: continue # Skip empty lines
                    
                    if header_mode:
                        if line.startswith('#'):
                            parts = line[1:].strip().split(':', 1)
                            if len(parts) == 2:
                                header_info[parts[0].strip()] = parts[1].strip()
                        if 'HEADER: End' in line or 'Phase' in line: # Common header end markers
                             # Check if next line looks like data (starts with numbers)
                            next_line = next(f, '').strip()
                            if next_line and next_line.replace('.','',1).replace('-','',1).replace(' ','').isdigit():
                                header_mode = False
                                data_lines.append(next_line)
                            # else: keep reading header
                    else:
                        data_lines.append(line)
        except FileNotFoundError:
            raise FileNotFoundError(f"EBSD file not found: {file_path}")
        except Exception as e:
            raise IOError(f"Error reading EBSD file {file_path}: {e}")

        if not data_lines:
             raise ValueError(f"No data found in EBSD file: {file_path}")
             
        # Parse data using pandas - robust to varying whitespace
        try:
            df = pd.read_csv(pd.compat.StringIO('\n'.join(data_lines)), delim_whitespace=True, header=None)
        except Exception as e:
             raise ValueError(f"Error parsing data lines in {file_path}: {e}")

        # Default column names - adjust based on typical .ang format
        default_columns = ['phi1', 'PHI', 'phi2', 'x', 'y', 'IQ', 'CI', 
                           'Phase_index', 'SEM', 'Fit']
        # Add more if present based on file inspection or header
        if df.shape[1] > len(default_columns):
             extra_cols = [f'col_{i+len(default_columns)}' for i in range(df.shape[1] - len(default_columns))]
             df.columns = default_columns + extra_cols
        else:
             df.columns = default_columns[:df.shape[1]]

        # Attempt conversion to numeric, coercing errors
        for col in df.columns:
             if col not in ['Phase_name']: # Exclude known non-numeric columns
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaNs in critical columns (angles, positions, CI)
        critical_cols = ['phi1', 'PHI', 'phi2', 'x', 'y', 'CI']
        df.dropna(subset=critical_cols, inplace=True)

        # Convert angles to radians (assuming input is degrees)
        # Check header or typical format to confirm if conversion needed
        # If header_info suggests radians, skip this.
        if 'AngleUnits' not in header_info or header_info['AngleUnits'].lower() == 'degrees':
            for col in ['phi1', 'PHI', 'phi2']:
                 df[col] = np.deg2rad(df[col])
        
        # Infer step size if not provided
        if self.step_size is None:
            dx = np.diff(np.unique(df['x']))
            dy = np.diff(np.unique(df['y']))
            if len(dx) > 0 and len(dy) > 0:
                step_x = np.min(dx[dx > 1e-6]) # Avoid zero differences
                step_y = np.min(dy[dy > 1e-6])
                if np.isclose(step_x, step_y):
                    self.step_size = step_x
                    print(f"Inferred step size: {self.step_size:.4f} microns")
                else:
                    print("Warning: Could not infer a unique step size. Calculations assume pixels.")
            else:
                 print("Warning: Could not infer step size. Calculations assume pixels.")

        return df
    
    def identify_grains(self, ci_threshold=0.1, angle_tolerance_rad=np.deg2rad(5), dbscan_min_samples=10):
        """
        Identify grains using DBSCAN clustering on Euler angles.
        Filters points by Confidence Index (CI) before clustering.

        Args:
            ci_threshold: Minimum Confidence Index (CI) value to consider a point.
            angle_tolerance_rad: Maximum misorientation angle (in radians) for points 
                                 to be considered in the same cluster (DBSCAN eps).
            dbscan_min_samples: Minimum number of points required to form a dense region (DBSCAN min_samples).
                                Corresponds roughly to the minimum grain size.
        """
        # Filter by CI
        self.valid_points_mask = self.data['CI'] >= ci_threshold
        if not self.valid_points_mask.any():
            print("Warning: No data points found above CI threshold.")
            self.grain_labels = np.array([])
            return np.array([])
            
        # Prepare orientation data for clustering (use rotation matrices or quaternions for better distance metric)
        # For simplicity here, use Euler angles directly (less robust near poles)
        X_orient = self.data.loc[self.valid_points_mask, ['phi1', 'PHI', 'phi2']].values
        
        # Perform clustering
        # Note: Standard Euclidean distance on Euler angles is not a true misorientation.
        # Consider using a library that calculates misorientation or using quaternions.
        clustering = DBSCAN(eps=angle_tolerance_rad, min_samples=dbscan_min_samples, metric='euclidean')
        try:
            labels = clustering.fit_predict(X_orient)
        except Exception as e:
             print(f"Error during DBSCAN clustering: {e}")
             self.grain_labels = np.array([])
             return np.array([])

        # Store results only for the valid points
        self.grain_labels = labels # Size matches number of points where valid_points_mask is True
        
        print(f"Found {len(np.unique(labels[labels != -1]))} potential grains (excluding noise). CI > {ci_threshold}, Angle Tol: {np.rad2deg(angle_tolerance_rad):.1f} deg.")
        return labels
    
    def analyze_grains(self, show_progress=True):
        """
        Calculate properties for each identified grain (excluding noise points).
        """
        if self.grain_labels is None or not self.valid_points_mask.any():
            print("Grains must be identified first using identify_grains().")
            return []

        # Get unique grain labels (excluding noise label -1)
        unique_labels = np.unique(self.grain_labels[self.grain_labels != -1])
        
        if len(unique_labels) == 0:
            print("No grains identified (only noise points found).")
            self.grain_properties = []
            return []
            
        if show_progress:
            print(f"Analyzing {len(unique_labels)} identified grains...")
        
        grain_props_list = []
        
        # Get data for points that were part of the clustering
        valid_data = self.data[self.valid_points_mask]
        
        # Get overall data boundaries
        x_min, x_max = self.data['x'].min(), self.data['x'].max()
        y_min, y_max = self.data['y'].min(), self.data['y'].max()
        
        iterator = tqdm(unique_labels, desc="Analyzing grains") if show_progress else unique_labels
        for label in iterator:
            # Create mask for the current grain within the valid data
            mask = self.grain_labels == label
            
            # Filter data for the current grain
            grain_data = valid_data[mask]
            num_points = len(grain_data)

            # Skip small clusters if min_grain_size_pixels is set
            if num_points < self.min_grain_size_pixels:
                continue
                
            x_coords = grain_data['x'].values
            y_coords = grain_data['y'].values
            
            # Check if grain touches the edge of the scanned area
            touches_edge = (
                np.any(np.isclose(x_coords, x_min)) or
                np.any(np.isclose(x_coords, x_max)) or
                np.any(np.isclose(y_coords, y_min)) or
                np.any(np.isclose(y_coords, y_max))
            )
            
            # Calculate properties
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            area_pixels = num_points
            area_microns = area_pixels * self.step_size**2 if self.step_size else None
            equiv_diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
            equiv_diameter_microns = np.sqrt(4 * area_microns / np.pi) if area_microns else None
            
            # Average orientation (use with caution for Euler angles)
            avg_phi1 = np.mean(grain_data['phi1'].values) # Simple mean, maybe circular mean is better
            avg_PHI = np.mean(grain_data['PHI'].values)
            avg_phi2 = np.mean(grain_data['phi2'].values)
            avg_orientation = np.array([avg_phi1, avg_PHI, avg_phi2])

            grain_props_list.append({
                'label': label,
                'centroid': (centroid_x, centroid_y),
                'num_points': num_points,
                'area_pixels': area_pixels,
                'area_microns': area_microns,
                'equiv_diameter_pixels': equiv_diameter_pixels,
                'equiv_diameter_microns': equiv_diameter_microns,
                'touches_edge': touches_edge,
                'avg_orientation_rad': avg_orientation
            })
        
        self.grain_properties = pd.DataFrame(grain_props_list)
        print(f"Finished analysis. Found {len(self.grain_properties)} grains meeting size criteria.")
        return self.grain_properties
    
    def _create_label_grid(self):
        """Create a regular grid populated with grain labels."""
        if self.grain_labels is None or not self.valid_points_mask.any():
             raise RuntimeError("Grains must be identified first.")
             
        valid_data = self.data[self.valid_points_mask]
        x = valid_data['x'].values
        y = valid_data['y'].values
        labels = self.grain_labels
        
        # Create regular grid coordinates
        x_unique = np.unique(self.data['x'])
        y_unique = np.unique(self.data['y']) # Use all data for grid dims
        
        # Create the grid, initialize with a value indicating no data (e.g., -2)
        label_grid = np.full((len(y_unique), len(x_unique)), -2, dtype=int)
        
        # Map data coordinates to grid indices
        x_map = {val: i for i, val in enumerate(x_unique)}
        y_map = {val: i for i, val in enumerate(y_unique)}
        
        grid_x_indices = [x_map.get(val) for val in x]
        grid_y_indices = [y_map.get(val) for val in y]
        
        # Filter out points that didn't map (shouldn't happen with unique from all data)
        valid_indices = [(gy is not None and gx is not None) for gy, gx in zip(grid_y_indices, grid_x_indices)]
        
        labels_filt = labels[valid_indices]
        grid_y_indices_filt = [grid_y_indices[i] for i, v in enumerate(valid_indices) if v]
        grid_x_indices_filt = [grid_x_indices[i] for i, v in enumerate(valid_indices) if v]

        # Populate the grid
        label_grid[grid_y_indices_filt, grid_x_indices_filt] = labels_filt
        
        return label_grid, x_unique, y_unique

    def get_smooth_boundaries(self, sigma=1.0):
        """Create smooth grain boundaries using contour detection on a smoothed label grid."""
        try:
            label_grid, x_coords, y_coords = self._create_label_grid()
        except RuntimeError as e:
            print(e)
            return None
            
        # Handle points outside grains (label -1 or -2)
        # Maybe set noise points (-1) to a separate high value before smoothing?
        plot_grid = label_grid.astype(float)
        plot_grid[label_grid == -1] = np.nan # Treat noise as NaN
        plot_grid[label_grid == -2] = np.nan # Treat nodata as NaN

        # Apply Gaussian filter (ignoring NaNs if possible, or fill first)
        # Simple approach: replace NaN with a neutral value (e.g., median) before smoothing
        if np.isnan(plot_grid).any():
             median_label = np.nanmedian(plot_grid[plot_grid >= 0])
             if np.isnan(median_label):
                 median_label = 0 # Fallback if only NaNs
             plot_grid = np.nan_to_num(plot_grid, nan=median_label)

        smooth_grid = gaussian_filter(plot_grid, sigma=sigma)
        
        # Find contours for each grain label
        contours = {}
        unique_labels = np.unique(label_grid[label_grid >= 0])
        for label in unique_labels:
            # Use the original, non-smoothed grid for masking the grain
            mask = label_grid == label
            # Find contours on the *smoothed* grid, using a level between labels
            # This requires careful level selection. A simpler way is boundaries:
            # Find boundaries by comparing neighbors
            boundaries = np.zeros_like(label_grid, dtype=bool)
            boundaries[:-1, :] |= (label_grid[:-1, :] != label_grid[1:, :]) & (label_grid[:-1, :] != -2) & (label_grid[1:, :] != -2)
            boundaries[1:, :] |= (label_grid[:-1, :] != label_grid[1:, :]) & (label_grid[:-1, :] != -2) & (label_grid[1:, :] != -2)
            boundaries[:, :-1] |= (label_grid[:, :-1] != label_grid[:, 1:]) & (label_grid[:, :-1] != -2) & (label_grid[:, 1:] != -2)
            boundaries[:, 1:] |= (label_grid[:, :-1] != label_grid[:, 1:]) & (label_grid[:, :-1] != -2) & (label_grid[:, 1:] != -2)
            
            # Extract boundary coordinates
            y_idx, x_idx = np.where(boundaries)
            boundary_coords = np.column_stack((x_coords[x_idx], y_coords[y_idx]))
            contours[label] = boundary_coords # Store raw boundary points for now
            
            # Contour finding (alternative) - often gives cleaner lines but needs tuning
            # grain_contours = measure.find_contours(smooth_grid, level=label + 0.5) # Level is tricky
            # processed_contours = []
            # for contour in grain_contours:
            #     contour_x = np.interp(contour[:, 1], np.arange(len(x_coords)), x_coords)
            #     contour_y = np.interp(contour[:, 0], np.arange(len(y_coords)), y_coords)
            #     processed_contours.append(np.column_stack([contour_x, contour_y]))
            # contours[label] = processed_contours
        
        self.smoothed_boundaries = contours # Store the raw boundary points map
        return self.smoothed_boundaries

    def plot_ipf_map(self, direction=np.array([0,0,1]), colormap='hsv', show_boundaries=True, boundary_sigma=0.5, show_centroids=True, centroid_min_diameter=None):
        """
        Create an Inverse Pole Figure (IPF) map.

        Args:
            direction: Reference direction in sample coordinates (e.g., [0,0,1] for IPF-Z).
            colormap: Matplotlib colormap for coloring.
            show_boundaries: If True, overlay grain boundaries.
            boundary_sigma: Sigma for Gaussian smoothing used for boundary detection.
            show_centroids: If True, mark centroids of large grains.
            centroid_min_diameter: Minimum equivalent diameter (in microns if step_size known, else pixels)
                                   for plotting centroids. Defaults to class min_grain_size.
        """
        # Calculate IPF colors
        rgb_colors = self._calculate_ipf_colors(direction)
        
        plt.figure(figsize=(10, 10))
        
        # Scatter plot with IPF colors (plot all points)
        plt.scatter(self.data['x'], self.data['y'], c=rgb_colors, s=5, marker='.', edgecolors='none')
        
        if show_boundaries:
            if self.smoothed_boundaries is None:
                print(f"Calculating boundaries with sigma={boundary_sigma}...")
                self.get_smooth_boundaries(sigma=boundary_sigma)
            
            if self.smoothed_boundaries:
                print("Plotting boundaries...")
                all_boundary_points = np.concatenate(list(self.smoothed_boundaries.values()))
                plt.plot(all_boundary_points[:, 0], all_boundary_points[:, 1], 'k,', markersize=0.1) # Plot boundaries as fine black dots
            else:
                print("Could not generate boundaries to plot.")
        
        # Plot centroids of large grains
        if show_centroids and self.grain_properties is not None and not self.grain_properties.empty:
            if centroid_min_diameter is None:
                 min_diam = self.min_grain_size_pixels if self.step_size is None else np.sqrt(4*self.min_grain_size_pixels*self.step_size**2 / np.pi)
                 diam_col = 'equiv_diameter_pixels' if self.step_size is None else 'equiv_diameter_microns'
            else:
                min_diam = centroid_min_diameter
                diam_col = 'equiv_diameter_microns' if self.step_size else 'equiv_diameter_pixels'
                if diam_col not in self.grain_properties.columns:
                     print(f"Warning: Cannot filter by {diam_col}, column missing.")
                     diam_col = None 
            
            if diam_col:
                large_grains = self.grain_properties[self.grain_properties[diam_col] >= min_diam]
                if not large_grains.empty:
                    centroids = np.array(large_grains['centroid'].tolist())
                    plt.plot(centroids[:, 0], centroids[:, 1], 'w+', markersize=6, markeredgewidth=1.5, label=f'Centroids (>{min_diam:.1f} {diam_col.split("_")[-1]})')
        
        plt.xlabel('X (microns)')
        plt.ylabel('Y (microns)')
        plt.title(f'IPF-{direction} Map')
        plt.axis('equal')
        plt.gca().invert_yaxis() # Typical for EBSD maps
        # Add IPF color key triangle here if desired
        
        return plt.gcf()
    
    def _calculate_ipf_colors(self, direction_sample=np.array([0,0,1])):
        """
        Calculate IPF (Inverse Pole Figure) colors.
        Maps a specified sample direction into the crystal reference frame 
        and colors based on its position in the standard stereographic triangle.
        """
        # Ensure direction is normalized
        direction_sample = normalize(direction_sample)
        
        # Get Euler angles
        phi1 = self.data['phi1'].values
        PHI = self.data['PHI'].values
        phi2 = self.data['phi2'].values
        
        # Calculate rotation matrices (sample to crystal)
        # This could be vectorized further if needed
        num_points = len(self.data)
        rgb_colors = np.zeros((num_points, 3))
        
        for i in tqdm(range(num_points), desc="Calculating IPF colors", disable=num_points<1000):
            # Matrix g transforms from sample to crystal coordinates
            g = euler2matrix(phi1[i], PHI[i], phi2[i])
            
            # Transform the sample direction vector into crystal coordinates
            direction_crystal = g @ direction_sample
            
            # Apply cubic symmetry operations to bring vector into standard triangle [001]-[101]-[111]
            equiv_dirs = self._apply_cubic_symmetry(direction_crystal)
            
            # Find the vector within the standard triangle (or closest to it)
            # Simplified: just take the absolute values for standard triangle mapping
            # For proper mapping, need to find equiv dir in fundamental sector
            vec_in_triangle = np.abs(equiv_dirs[0]) # Simplistic mapping for now
            vec_in_triangle /= np.sqrt(np.sum(vec_in_triangle**2)) # Normalize
            
            # Map coordinates in triangle to RGB
            # Color = |x|[1,0,0] + |y|[0,1,0] + |z|[0,0,1]
            # Ensure normalization for color intensity (maybe divide by max component?)
            rgb = np.abs(vec_in_triangle) # Use absolute values for color mapping
            rgb /= np.max(rgb) if np.max(rgb) > 0 else 1 # Normalize intensity
            rgb_colors[i] = rgb
            
        return rgb_colors

    def _apply_cubic_symmetry(self, vec):
        """Applies cubic symmetry operations to a vector."""
        # Generate 24 (or 48 if inversion) rotation matrices for cubic symmetry
        # For simplicity, just return permutations and sign changes of the input vector components
        # This is NOT a full symmetry operation but captures basic equivalents needed for IPF
        x, y, z = vec[0], vec[1], vec[2]
        perms = [
            [x, y, z], [x, z, y], [y, x, z], [y, z, x], [z, x, y], [z, y, x]
        ]
        
        equiv_dirs = []
        for p in perms:
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        equiv_dirs.append(np.array([sx*p[0], sy*p[1], sz*p[2]]))
                        
        # Return unique directions (optional, can be slow)
        # unique_dirs = np.unique(np.round(equiv_dirs, decimals=5), axis=0)
        # return unique_dirs
        return np.array(equiv_dirs) # Return all 48 for now

    def get_large_grains(self, min_diameter=None):
        """Filter grain properties to get grains larger than a minimum diameter."""
        if self.grain_properties is None or self.grain_properties.empty:
             print("Grain properties not calculated yet. Run identify_grains() and analyze_grains().")
             return pd.DataFrame()
             
        if min_diameter is None:
            min_diameter = self.min_grain_size_pixels if self.step_size is None else np.sqrt(4*self.min_grain_size_pixels*self.step_size**2 / np.pi)
            diam_col = 'equiv_diameter_pixels' if self.step_size is None else 'equiv_diameter_microns'
            print(f"Using default minimum diameter: {min_diameter:.2f} ({diam_col})")
        else:
            diam_col = 'equiv_diameter_microns' if self.step_size else 'equiv_diameter_pixels'
            if diam_col not in self.grain_properties.columns:
                 print(f"Warning: Cannot filter by {diam_col}, column missing.")
                 return pd.DataFrame()
                 
        large_grains = self.grain_properties[self.grain_properties[diam_col] >= min_diameter].copy()
        
        # Add average orientation in degrees for easier readability
        if 'avg_orientation_rad' in large_grains.columns:
             large_grains['avg_orientation_deg'] = large_grains['avg_orientation_rad'].apply(lambda x: np.rad2deg(x).round(2))

        return large_grains

# Example Usage
if __name__ == '__main__':
    # Create a dummy .ang file for testing
    file_dir = "./test_ebsd_data"
    file_path = os.path.join(file_dir, "dummy_ebsd.ang")
    os.makedirs(file_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write("# HEADER: Start\n")
        f.write("# AngleUnits: degrees\n")
        f.write("# StepSize: 0.5\n")
        f.write("# HEADER: End\n")
        # Grain 1 (near 0,0,0)
        for x in np.arange(0, 5, 0.5):
            for y in np.arange(0, 5, 0.5):
                f.write(f" {np.random.normal(1, 1):.4f} {np.random.normal(1, 1):.4f} {np.random.normal(1, 1):.4f} {x:.4f} {y:.4f} 0.95 0.98 1 128 0.1\n")
        # Grain 2 (near 30,30,30)
        for x in np.arange(7, 12, 0.5):
            for y in np.arange(7, 12, 0.5):
                f.write(f" {np.random.normal(30, 1):.4f} {np.random.normal(30, 1):.4f} {np.random.normal(30, 1):.4f} {x:.4f} {y:.4f} 0.92 0.95 1 130 0.15\n")
        # Noise points
        for _ in range(20):
             x, y = np.random.rand(2) * 15
             f.write(f" {np.random.rand()*180:.4f} {np.random.rand()*90:.4f} {np.random.rand()*180:.4f} {x:.4f} {y:.4f} 0.2 0.05 1 50 0.8\n")

    try:
        analyzer = EBSDAnalyzer(file_path, step_size=0.5, min_grain_size_pixels=10)
        print(f"Loaded data points: {len(analyzer.data)}")
        
        analyzer.identify_grains(ci_threshold=0.1, angle_tolerance_rad=np.deg2rad(5), dbscan_min_samples=10)
        
        grain_props = analyzer.analyze_grains()
        
        if grain_props is not None and not grain_props.empty:
            print("\nGrain Properties:")
            print(grain_props.head())
            
            large_grains = analyzer.get_large_grains(min_diameter=3.0)
            print("\nLarge Grains (diameter > 3.0 microns):")
            print(large_grains[['label', 'centroid', 'equiv_diameter_microns', 'touches_edge', 'avg_orientation_deg']])

            fig = analyzer.plot_ipf_map(show_boundaries=True, centroid_min_diameter=3.0)
            plt.show()
        else:
            print("No grains meeting criteria were analyzed.")
            
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        # Clean up dummy file/dir
        # import shutil
        # shutil.rmtree(file_dir)
        print(f"\n(Keeping test file: {file_path})") 