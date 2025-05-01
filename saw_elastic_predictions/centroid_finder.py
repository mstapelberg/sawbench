import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
import seaborn as sns
import os
from tqdm import tqdm 
from scipy.ndimage import gaussian_filter
from skimage import measure

class EBSDAnalyzer:
    def __init__(self, file_path, min_grain_size=100):
        """
        Initialize EBSD analyzer
        
        Args:
            file_path: Path to the .ang file
            min_grain_size: Minimum grain size in microns (for laser targeting)
        """
        self.min_grain_size = min_grain_size
        self.data = self._load_data(file_path)
        
    def _load_data(self, file_path):
        """
        Load and parse the .ang file
        - Skip header until 'HEADER: End'
        - Parse space/tab delimited data
        - Create DataFrame with proper column names
        """
        # Read header to get column info
        header_lines = []
        data_lines = []
        with open(file_path, 'r') as f:
            header_mode = True
            for line in f:
                if header_mode:
                    if 'HEADER: End' in line:
                        header_mode = False
                    else:
                        header_lines.append(line)
                else:
                    data_lines.append(line)
        
        # Parse data using pandas
        df = pd.DataFrame([line.split() for line in data_lines])
        
        # Use exact column names from the file
        columns = ['phi1', 'PHI', 'phi2', 'x', 'y', 'IQ', 'CI', 
                  'Phase_index', 'SEM', 'Fit', 
                  'PRIAS_Bottom_Strip', 'PRIAS_Center_Square', 'PRIAS_Top_Strip']
        
        # Ensure we have the right number of columns
        if len(df.columns) != len(columns):
            print(f"Warning: Found {len(df.columns)} columns but expected {len(columns)}")
            print("Actual data shape:", df.shape)
            print("First few rows:")
            print(df.head())
        
        df.columns = columns[:len(df.columns)]
        
        # Convert to proper types - all columns that should be numeric
        numeric_cols = ['phi1', 'PHI', 'phi2', 'x', 'y', 'IQ', 'CI', 
                       'Phase_index', 'SEM', 'Fit', 
                       'PRIAS_Bottom_Strip', 'PRIAS_Center_Square', 'PRIAS_Top_Strip']
        
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df
    
    def identify_grains(self, ci_threshold=0.98, angle_tolerance=0.02, min_samples=10):
        """
        Identify grains using DBSCAN clustering on euler angles
        Only consider points with CI above threshold
        """
        # Filter by CI
        high_ci_mask = self.data['CI'] >= ci_threshold
        
        # Prepare data for clustering
        X = self.data[high_ci_mask][['phi1', 'PHI', 'phi2']].values
        positions = self.data[high_ci_mask][['x', 'y']].values
        
        # Perform clustering
        clustering = DBSCAN(eps=angle_tolerance, min_samples=min_samples)
        labels = clustering.fit_predict(X)
        
        # Store results
        self.grain_labels = labels
        self.valid_points_mask = high_ci_mask
        
        return labels
    
    def analyze_grains(self, show_progress=True):
        """
        Vectorized version of grain analysis
        """
        # Get unique labels (excluding noise points)
        unique_labels = np.unique(self.grain_labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        if show_progress:
            print(f"Analyzing {len(unique_labels)} grains...")
        
        grain_properties = []
        
        # Get all points coordinates
        points = self.data[self.valid_points_mask]
        
        # Get data boundaries
        x_min, x_max = self.data['x'].min(), self.data['x'].max()
        y_min, y_max = self.data['y'].min(), self.data['y'].max()
        
        # Vectorized computation for each grain
        for label in tqdm(unique_labels, desc="Analyzing grains") if show_progress else unique_labels:
            # Create mask for current grain
            mask = self.grain_labels == label
            
            # Get points for current grain
            grain_points = points[mask]
            
            # Calculate properties
            x_coords = grain_points['x'].values
            y_coords = grain_points['y'].values
            
            # Check if grain touches edge
            touches_edge = (
                np.min(x_coords) == x_min or
                np.max(x_coords) == x_max or
                np.min(y_coords) == y_min or
                np.max(y_coords) == y_max
            )
            
            if not touches_edge:
                centroid = (np.mean(x_coords), np.mean(y_coords))
                area = len(x_coords) * 9  # assuming 3x3 micron steps
                diameter = np.sqrt(4 * area / np.pi)
                
                grain_properties.append({
                    'label': label,
                    'centroid': centroid,
                    'area': area,
                    'diameter': diameter
                })
        
        self.grain_properties = grain_properties
        return grain_properties
    
    def _create_smooth_boundaries(self):
        """
        Create smooth grain boundaries using contour detection
        """
        # Create a grid of points
        x = self.data[self.valid_points_mask]['x'].values
        y = self.data[self.valid_points_mask]['y'].values
        labels = self.grain_labels
        
        # Create regular grid
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Create label grid
        label_grid = np.full((len(y_unique), len(x_unique)), -1)
        x_indices = np.searchsorted(x_unique, x)
        y_indices = np.searchsorted(y_unique, y)
        label_grid[y_indices, x_indices] = labels
        
        # Smooth the label grid
        smooth_grid = gaussian_filter(label_grid.astype(float), sigma=1)
        
        # Find contours
        contours = []
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = label_grid == label
            # Find contours for this grain
            grain_contours = measure.find_contours(mask.astype(float), 0.5)
            for contour in grain_contours:
                # Convert contour coordinates back to microns
                contour_x = np.interp(contour[:, 1], np.arange(len(x_unique)), x_unique)
                contour_y = np.interp(contour[:, 0], np.arange(len(y_unique)), y_unique)
                contours.append(np.column_stack([contour_x, contour_y]))
        
        return contours

    def plot_grains(self, show_boundaries=True):
        """
        Create orientation-based coloring plot
        Optionally show grain boundaries
        """
        # Create orientation colors
        colors = self._calculate_orientation_colors()
        
        # Create plot
        plt.figure(figsize=(10, 10))
        
        # Plot points
        plt.scatter(self.data['x'], self.data['y'], 
                   c=colors, s=1)
        
        if show_boundaries:
            # Add grain boundaries
            self._add_grain_boundaries()
        
        # Vectorized centroid plotting for large grains
        large_grains = [g for g in self.grain_properties if g['diameter'] >= self.min_grain_size]
        if large_grains:
            centroids = np.array([g['centroid'] for g in large_grains])
            plt.plot(centroids[:, 0], centroids[:, 1], 'k+', markersize=5)
        
        plt.colorbar(label='Orientation')
        plt.xlabel('X (microns)')
        plt.ylabel('Y (microns)')
        plt.title('EBSD Grain Map')
        plt.axis('equal')
        
        return plt.gcf()
    
    def _calculate_orientation_colors(self):
        """
        Calculate IPF (Inverse Pole Figure) colors from Euler angles
        Maps crystallographic directions to RGB colors following standard IPF coloring
        """
        # Convert Euler angles to rotation matrices (already vectorized)
        angles = np.column_stack([
            self.data['phi1'].values,
            self.data['PHI'].values,
            self.data['phi2'].values
        ])
        
        # Vectorized trigonometric calculations
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Unpack for clarity
        cos_phi1, cos_PHI, cos_phi2 = cos_angles.T
        sin_phi1, sin_PHI, sin_phi2 = sin_angles.T
        
        # Vectorized rotation matrix calculation
        a = np.zeros((len(self.data), 3, 3))
        
        # First row
        a[:, 0, 0] = cos_phi1*cos_phi2 - sin_phi1*cos_PHI*sin_phi2
        a[:, 0, 1] = sin_phi1*cos_phi2 + cos_phi1*cos_PHI*sin_phi2
        a[:, 0, 2] = sin_PHI*sin_phi2
        
        # Second row
        a[:, 1, 0] = -cos_phi1*sin_phi2 - sin_phi1*cos_PHI*cos_phi2
        a[:, 1, 1] = -sin_phi1*sin_phi2 + cos_phi1*cos_PHI*cos_phi2
        a[:, 1, 2] = sin_PHI*cos_phi2
        
        # Third row
        a[:, 2, 0] = sin_phi1*sin_PHI
        a[:, 2, 1] = -cos_phi1*sin_PHI
        a[:, 2, 2] = cos_PHI
        
        # Vectorized direction cosines calculation
        x = np.abs(a[:, 0, 2])  # <100>
        y = np.abs(a[:, 1, 2])  # <110>
        z = np.abs(a[:, 2, 2])  # <111>
        
        # Vectorized color calculation
        max_val = np.maximum.reduce([x, y, z])
        max_val = np.where(max_val == 0, 1, max_val)  # Avoid division by zero
        
        colors = np.column_stack([
            x / max_val,
            y / max_val,
            z / max_val
        ])
        
        # Vectorized CI weighting
        if 'CI' in self.data.columns:
            colors *= self.data['CI'].values[:, np.newaxis]
        
        # Clip values to ensure they're in [0, 1] range
        colors = np.clip(colors, 0, 1)
        
        return colors
    
    def _add_grain_boundaries(self):
        """
        Add grain boundaries to the current plot using vectorized operations
        """
        # Get coordinates and labels
        x = self.data[self.valid_points_mask]['x'].values
        y = self.data[self.valid_points_mask]['y'].values
        labels = self.grain_labels
        
        # Create grid points
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        
        # Calculate steps
        x_step = x_unique[1] - x_unique[0]
        y_step = y_unique[1] - y_unique[0]
        
        # Create meshgrid for faster indexing
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Create label grid using vectorized operations
        x_indices = np.searchsorted(x_unique, x)
        y_indices = np.searchsorted(y_unique, y)
        label_grid = np.full((len(y_unique), len(x_unique)), -1)
        label_grid[y_indices, x_indices] = labels
        
        # Find boundaries using array operations
        vertical_diff = label_grid[:, 1:] != label_grid[:, :-1]
        horizontal_diff = label_grid[1:, :] != label_grid[:-1, :]
        
        # Prepare boundary coordinates
        boundaries_x = []
        boundaries_y = []
        
        # Vectorized boundary creation
        y_coords, x_coords = np.where(vertical_diff)
        if len(x_coords) > 0:
            x_bounds = x_unique[x_coords + 1]
            y_bounds = y_unique[y_coords]
            boundaries_x.extend(np.repeat(x_bounds, 3))
            boundaries_y.extend(np.ravel(np.column_stack([
                y_bounds - y_step/2,
                y_bounds + y_step/2,
                np.full_like(y_bounds, np.nan)
            ])))
        
        y_coords, x_coords = np.where(horizontal_diff)
        if len(x_coords) > 0:
            x_bounds = x_unique[x_coords]
            y_bounds = y_unique[y_coords + 1]
            boundaries_x.extend(np.ravel(np.column_stack([
                x_bounds - x_step/2,
                x_bounds + x_step/2,
                np.full_like(x_bounds, np.nan)
            ])))
            boundaries_y.extend(np.repeat(y_bounds, 3))
        
        # Plot all boundaries at once
        plt.plot(boundaries_x, boundaries_y, 'k-', linewidth=0.5, alpha=0.5)

if __name__ == "__main__":
    # Initialize analyzer with data file
    file_path = "../data/EBSD/V-1_2Ti/V-1_2Ti.ang"
    analyzer = EBSDAnalyzer(file_path, min_grain_size=10)
    
    # Identify grains
    print("Identifying grains...")
    labels = analyzer.identify_grains(ci_threshold=0.98, angle_tolerance=0.01)
    print(f"Found {len(np.unique(labels))} unique grain labels")
    
    # Analyze grain properties
    print("\nAnalyzing grain properties...")
    grain_properties = analyzer.analyze_grains()
    print(f"Found {len(grain_properties)} grains (excluding edge grains)")
    
    # Print some statistics
    diameters = [grain['diameter'] for grain in grain_properties]
    areas = [grain['area'] for grain in grain_properties]
    
    print("\nGrain Statistics:")
    print(f"Average grain diameter: {np.mean(diameters):.1f} microns")
    print(f"Average grain area: {np.mean(areas):.1f} square microns")
    print(f"Number of grains >= {analyzer.min_grain_size} microns: "
          f"{sum(d >= analyzer.min_grain_size for d in diameters)}")
    
    # Create and save plot
    print("\nCreating grain map...")
    fig = analyzer.plot_grains(show_boundaries=True)
    
    # Save plot
    output_dir = "../data/EBSD/V-1_2Ti/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, "grain_map_wboundaries.png"), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_dir}/grain_map_wboundaries.png")
    
    # Show plot
    plt.show()

