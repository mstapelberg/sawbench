"""
Module for selecting and processing subregions from EBSD data.

This module provides tools for:
1. Cropping CTF files to specific regions
2. Filtering grains by spatial coordinates
3. Visualizing and annotating regions of interest
4. Creating masks for subregion analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import PolygonSelector
from typing import Tuple, List, Optional, Dict, Any
import os
from pathlib import Path

from .io import read_ctf, load_ebsd_map
from defdap import ebsd as _defdap_ebsd  # for type hinting only
from .grains import (
    calculate_saw_frequencies_for_ebsd_grains,  # noqa: F401 imported for downstream workflows
    create_ebsd_saw_frequency_map,              # noqa: F401 imported for downstream workflows
)


class SubregionSelector:
    """
    A class for selecting and processing subregions from EBSD data.
    
    This class provides methods to:
    - Load and examine CTF data
    - Select rectangular or polygonal subregions
    - Crop data to selected regions
    - Filter existing grains by spatial constraints
    - Visualize regions and grains
    """
    
    def __init__(self, ctf_path: str, *, data_type: str = "OxfordText", boundary_def: float = 5.0, min_grain_size: int = 10, ebsd_map_obj: Optional["_defdap_ebsd.Map"] = None):
        """
        Initialize the subregion selector with a CTF file.
        
        Args:
            ctf_path (str): Path to the CTF file (may be with or without .ctf extension)
            data_type (str): EBSD data type passed to defdap (default: "OxfordText")
            boundary_def (float): Misorientation threshold in degrees (default: 5.0)
            min_grain_size (int): Minimum grain size for defdap (default: 10)
            ebsd_map_obj (defdap.ebsd.Map | None): Use an existing EBSD map instead of loading
        """
        self.ctf_path = Path(ctf_path)
        
        # Check if the file exists, or if it's a path without .ctf extension (for defdap compatibility)
        if not self.ctf_path.exists():
            # Try with .ctf extension if the path doesn't end with it
            if not str(ctf_path).endswith('.ctf'):
                ctf_path_with_ext = str(ctf_path) + '.ctf'
                if Path(ctf_path_with_ext).exists():
                    self.ctf_path = Path(ctf_path_with_ext)
                    print(f"Found file with .ctf extension: {self.ctf_path}")
                else:
                    raise FileNotFoundError(f"CTF file not found: {ctf_path} (also tried {ctf_path_with_ext})")
            else:
                raise FileNotFoundError(f"CTF file not found: {ctf_path}")
            
        # Always load EBSD map the same way as the rest of the codebase
        # Prefer an already-provided map, else load via load_ebsd_map
        if ebsd_map_obj is not None:
            self.ebsd_map = ebsd_map_obj
        else:
            try:
                path_to_use = str(self.ctf_path)
                
                # Check if defdap has issues with .ctf extension (double .ctf.ctf issue)
                # If the path ends with .ctf, try without the extension
                if path_to_use.endswith('.ctf'):
                    try:
                        self.ebsd_map = load_ebsd_map(
                            file_path=path_to_use,
                            data_type=data_type,
                            boundary_def=boundary_def,
                            min_grain_size=min_grain_size
                        )
                        if self.ebsd_map is None:
                            raise ValueError(f"load_ebsd_map() returned None for file: {path_to_use}")
                    except Exception as e:
                        if "Cannot open file" in str(e) and ".ctf.ctf" in str(e):
                            print("Detected double .ctf extension issue. Trying without extension...")
                            path_without_ext = path_to_use[:-4]
                            self.ebsd_map = load_ebsd_map(
                                file_path=path_without_ext,
                                data_type=data_type,
                                boundary_def=boundary_def,
                                min_grain_size=min_grain_size
                            )
                            if self.ebsd_map is None:
                                raise ValueError(f"load_ebsd_map() returned None for file: {path_without_ext}")
                            print(f"Successfully loaded using path without extension: {path_without_ext}")
                        else:
                            raise
                else:
                    self.ebsd_map = load_ebsd_map(
                        file_path=path_to_use,
                        data_type=data_type,
                        boundary_def=boundary_def,
                        min_grain_size=min_grain_size
                    )
                    if self.ebsd_map is None:
                        raise ValueError(f"load_ebsd_map() returned None for file: {path_to_use}")
            except Exception as e:
                raise ValueError(f"Failed to load EBSD map from {self.ctf_path}: {e}") from e

        # Compute spatial bounds directly from EBSD map to avoid schema issues
        # Use pixel indices scaled by step size (μm)
        map_shape = getattr(self.ebsd_map, 'shape', None)
        step_size_um = float(getattr(self.ebsd_map, 'stepSize', 1.0))
        if not map_shape or len(map_shape) != 2:
            raise ValueError("EBSD map has invalid shape; cannot determine bounds.")
        ny, nx = int(map_shape[0]), int(map_shape[1])
        self._nx = nx
        self._ny = ny
        self._step_um = step_size_um
        self.x_min, self.x_max = 0.0, (nx - 1) * step_size_um
        self.y_min, self.y_max = 0.0, (ny - 1) * step_size_um

        # Optionally load raw CTF rows for plotting/saving (kept for backward-compat)
        # Do not rely on this for bounds, only for visualization/output
        try:
            # Prefer the path that successfully loaded into defdap (without extension if needed)
            preferred_path = str(self.ctf_path)
            if preferred_path.endswith('.ctf') and not Path(preferred_path).exists():
                # Try without extension (defdap-success path)
                preferred_path = preferred_path[:-4]
            self.header, self.df = read_ctf(preferred_path)
            # Ensure numeric types for expected columns
            for col in ['X', 'Y', 'Euler1', 'Euler2', 'Euler3']:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            # Debug: basic column and NaN diagnostics
            try:
                print(f"[DEBUG] read_ctf: path_used={preferred_path}")
                print(f"[DEBUG] read_ctf: df.shape={self.df.shape} columns={list(self.df.columns)}")
                if 'X' in self.df.columns and 'Y' in self.df.columns:
                    print(f"[DEBUG] read_ctf: X non-null={self.df['X'].notna().sum()} null={self.df['X'].isna().sum()} sample={self.df['X'].head(3).tolist()}")
                    print(f"[DEBUG] read_ctf: Y non-null={self.df['Y'].notna().sum()} null={self.df['Y'].isna().sum()} sample={self.df['Y'].head(3).tolist()}")
                if 'Euler1' in self.df.columns:
                    print(f"[DEBUG] read_ctf: Euler1 non-null={self.df['Euler1'].notna().sum()} null={self.df['Euler1'].isna().sum()} sample={self.df['Euler1'].head(3).tolist()}")
            except Exception:
                pass
            self._validate_ctf_data()
            # Prepare normalized coordinates in microns for plotting/selection
            self._prepare_df_um()
        except Exception:
            # If CTF parsing fails, continue with map-only path
            self.header, self.df = {}, pd.DataFrame()
            self.df_um = None
        
        # Initialize selection variables
        self.selected_region = None
        self.region_mask = None
        
    def _validate_ctf_data(self):
        """Validate that the CTF data has required columns."""
        required_cols = ['X', 'Y', 'Euler1', 'Euler2', 'Euler3']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CTF data missing required columns: {missing_cols}")
    
    def _prepare_df_um(self) -> None:
        """Create a normalized view of the CTF DataFrame in microns aligned to map origin.

        Many CTF files store X,Y as absolute stage coordinates. We normalize to (0,0)
        at the lower-left of the map in microns, matching the EBSD map extents used
        for plotting and ROI selection.
        """
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            self.df_um = None
            return
        # Shift coordinates so that min X,Y become 0 and convert units if needed
        x_min_df = float(self.df['X'].min())
        y_min_df = float(self.df['Y'].min())
        # Assume CTF 'X','Y' already in microns; normalize origin
        self.df_um = self.df.copy()
        self.df_um['X_um'] = (self.df['X'] - x_min_df)
        self.df_um['Y_um'] = (self.df['Y'] - y_min_df)
        # Debug
        try:
            print(f"[DEBUG] _prepare_df_um: x_min_df={x_min_df} y_min_df={y_min_df} X_um_range=({self.df_um['X_um'].min()},{self.df_um['X_um'].max()}) Y_um_range=({self.df_um['Y_um'].min()},{self.df_um['Y_um'].max()})")
            if not np.isfinite(x_min_df) or not np.isfinite(y_min_df):
                print(f"[DEBUG] _prepare_df_um: X dtype={self.df['X'].dtype} Y dtype={self.df['Y'].dtype}")
                print(f"[DEBUG] _prepare_df_um: X head={self.df['X'].head(5).tolist()} Y head={self.df['Y'].head(5).tolist()}")
        except Exception:
            pass

    def _has_valid_df_um(self) -> bool:
        """Return True if df_um exists and has finite X_um/Y_um values."""
        if not isinstance(self.df_um, pd.DataFrame):
            return False
        if self.df_um.empty or 'X_um' not in self.df_um.columns or 'Y_um' not in self.df_um.columns:
            return False
        try:
            vals = self.df_um[['X_um', 'Y_um']].to_numpy()
            return np.isfinite(vals).any()
        except Exception:
            return False
    
    def get_data_bounds(self) -> Dict[str, float]:
        """Get the spatial bounds of the EBSD data."""
        return {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'width': self.x_max - self.x_min,
            'height': self.y_max - self.y_min
        }
    
    def select_rectangular_region(self, x_min: float, y_min: float, x_max: float, y_max: float) -> pd.DataFrame:
        """
        Select a rectangular subregion from the EBSD data.
        
        Args:
            x_min, y_min: Bottom-left corner coordinates
            x_max, y_max: Top-right corner coordinates
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the selected region
        """
        # Clamp to map bounds (in μm)
        x_min = max(float(x_min), self.x_min)
        y_min = max(float(y_min), self.y_min)
        x_max = min(float(x_max), self.x_max)
        y_max = min(float(y_max), self.y_max)

        # Record selection (always)
        self.selected_region = {'type': 'rectangle', 'coords': [x_min, y_min, x_max, y_max]}

        # Preferred path: use normalized CTF coordinates if valid
        if self._has_valid_df_um():
            mask_um = (
                (self.df_um['X_um'] >= x_min) & (self.df_um['X_um'] <= x_max) &
                (self.df_um['Y_um'] >= y_min) & (self.df_um['Y_um'] <= y_max)
            )
            self.region_mask = mask_um
            # Return original columns plus normalized coordinates for plotting
            sub_df = self.df.loc[mask_um].copy()
            sub_df['X_um'] = self.df_um.loc[mask_um, 'X_um']
            sub_df['Y_um'] = self.df_um.loc[mask_um, 'Y_um']
            print(f"[DEBUG] select_rectangular_region: selected_df_rows={len(sub_df)} using df_um")
            return sub_df

        # Fallback: no valid CTF table; derive points from EBSD map grid within ROI
        idx_slices = self.get_roi_index_slices()
        if idx_slices is not None:
            ys, xs = idx_slices
            # Build coordinate arrays for the ROI
            y_idx, x_idx = np.mgrid[ys, xs]
            x_um = x_idx.astype(float) * self._step_um
            y_um = y_idx.astype(float) * self._step_um
            flat_x = x_um.ravel()
            flat_y = y_um.ravel()
            sub_df = pd.DataFrame({'X_um': flat_x, 'Y_um': flat_y})
            print(f"[DEBUG] select_rectangular_region: selected_map_pixels={len(sub_df)} using grainIDMap ROI")
            self.region_mask = None
            return sub_df

        # Otherwise, nothing to return
        self.region_mask = None
        print("[DEBUG] select_rectangular_region: no valid df_um and no ROI slices; returning empty")
        return pd.DataFrame()
    
    def select_polygonal_region(self, vertices: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        Select a polygonal subregion from the EBSD data.
        
        Args:
            vertices: List of (x, y) coordinate pairs defining the polygon
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the selected region
        """
        from matplotlib.path import Path as MPLPath
        
        # Create matplotlib path from vertices
        polygon_path = MPLPath(vertices)
        
        self.selected_region = {'type': 'polygon', 'coords': vertices}
        
        # If we have a valid normalized CTF table, compute mask in μm space
        if self._has_valid_df_um():
            points_um = self.df_um[['X_um', 'Y_um']].to_numpy()
            mask = polygon_path.contains_points(points_um)
            self.region_mask = mask
            sub_df = self.df.loc[mask].copy()
            sub_df['X_um'] = self.df_um.loc[mask, 'X_um']
            sub_df['Y_um'] = self.df_um.loc[mask, 'Y_um']
            print(f"[DEBUG] select_polygonal_region: selected_df_rows={len(sub_df)} using df_um")
            return sub_df
        
        # Fallback: build dense grid of map pixel coordinates and mask
        xx = np.linspace(0.0, (self._nx - 1) * self._step_um, self._nx)
        yy = np.linspace(0.0, (self._ny - 1) * self._step_um, self._ny)
        XX, YY = np.meshgrid(xx, yy)
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        mask_flat = polygon_path.contains_points(pts)
        sub_df = pd.DataFrame({'X_um': pts[mask_flat, 0], 'Y_um': pts[mask_flat, 1]})
        print(f"[DEBUG] select_polygonal_region: selected_map_pixels={len(sub_df)} using grid fallback")
        self.region_mask = None
        return sub_df
    
    def interactive_region_selection(self, plot_orientation: bool = True) -> pd.DataFrame:
        """
        Interactively select a region using matplotlib's PolygonSelector.
        
        Args:
            plot_orientation: Whether to plot orientation data (IPF colors) or just scatter
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the selected region
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        print("[DEBUG] interactive_region_selection: starting render")
        print(f"[DEBUG] DF available: {isinstance(self.df, pd.DataFrame)} size={0 if not isinstance(self.df, pd.DataFrame) else len(self.df)} cols={[] if not isinstance(self.df, pd.DataFrame) else list(self.df.columns)} df_um_valid={self._has_valid_df_um()}")
        print(f"[DEBUG] Map shape: ({self._ny}, {self._nx}), step_um={self._step_um}")
        # Prefer CTF table scatter if available; otherwise fall back to EBSD map image
        if self._has_valid_df_um():
            print("[DEBUG] interactive_region_selection: using CTF scatter route")
            if plot_orientation and 'Euler1' in self.df_um.columns and np.isfinite(self.df_um['Euler1']).any():
                scatter = ax.scatter(
                    self.df_um['X_um'], self.df_um['Y_um'], c=self.df_um['Euler1'] % 360,
                    cmap='hsv', s=1, alpha=0.6
                )
                plt.colorbar(scatter, ax=ax, label='Euler1 (degrees)')
            else:
                ax.scatter(self.df_um['X_um'], self.df_um['Y_um'], s=1, alpha=0.6, c='blue')
        else:
            # Fallback: render grainID map as an image so grains are visible
            print("[DEBUG] interactive_region_selection: using EBSD map image route")
            grain_map_ids_array = None
            if hasattr(self.ebsd_map, 'grainIDMap') and self.ebsd_map.grainIDMap is not None:
                grain_map_ids_array = self.ebsd_map.grainIDMap
            elif hasattr(self.ebsd_map, 'grains') and isinstance(getattr(self.ebsd_map, 'grains'), np.ndarray) and getattr(self.ebsd_map, 'grains').shape == (self._ny, self._nx):
                grain_map_ids_array = self.ebsd_map.grains

            if isinstance(grain_map_ids_array, np.ndarray):
                try:
                    print(f"[DEBUG] grainIDMap stats: shape={grain_map_ids_array.shape} dtype={grain_map_ids_array.dtype} min={np.nanmin(grain_map_ids_array)} max={np.nanmax(grain_map_ids_array)} unique_nonzero={len(np.unique(grain_map_ids_array[grain_map_ids_array>0]))}")
                except Exception:
                    pass
                extent = [0.0, self._nx * self._step_um, 0.0, self._ny * self._step_um]
                print(f"[DEBUG] imshow extent={extent}")
                im = ax.imshow(
                    grain_map_ids_array, origin='lower', cmap='nipy_spectral',
                    interpolation='nearest', extent=extent, aspect='equal', alpha=0.9
                )
                plt.colorbar(im, ax=ax, label='Grain ID')
            else:
                # Nothing to render; keep axes limits sensible
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylim(self.y_min, self.y_max)
                print(f"[DEBUG] No grainIDMap/grains array available. Ax limits set to x=[{self.x_min},{self.x_max}] y=[{self.y_min},{self.y_max}]")
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Click to select region. Close polygon with right-click.')
        ax.set_aspect('equal')
        
        # Store selected vertices
        selected_vertices = []
        
        def onselect(verts):
            nonlocal selected_vertices
            selected_vertices = verts
            print(f"Selected polygon with {len(verts)} vertices")
        
        # Ensure canvas is drawn before interaction to avoid None xdata
        try:
            fig.canvas.draw_idle()
            plt.pause(0.05)
        except Exception:
            pass

        # Create polygon selector (keep reference to avoid GC)
        _poly_selector = PolygonSelector(ax, onselect, useblit=False)
        
        plt.show()
        
        if len(selected_vertices) >= 3:
            return self.select_polygonal_region(selected_vertices)
        else:
            print("No valid polygon selected.")
            return pd.DataFrame()
    
    def save_cropped_ctf(self, output_path: str, subregion_df: pd.DataFrame) -> None:
        """
        Save a cropped CTF file with only the selected subregion.
        
        Args:
            output_path: Path for the output CTF file
            subregion_df: DataFrame containing the subregion data
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("Channel Text File\n")
            for key, values in self.header.items():
                f.write(f"{key}\t" + "\t".join(map(str, values)) + "\n")
            
            # Write column headers
            f.write("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n")
            
            # Write data
            # Decide which coordinate columns to use
            use_um = 'X_um' in subregion_df.columns and 'Y_um' in subregion_df.columns
            for _, row in subregion_df.iterrows():
                # Coordinates: fall back to normalized if raw not present
                x_val = row['X'] if 'X' in row.index and pd.notna(row['X']) else (row['X_um'] if use_um else np.nan)
                y_val = row['Y'] if 'Y' in row.index and pd.notna(row['Y']) else (row['Y_um'] if use_um else np.nan)
                # Orientation fields may be missing in ROI fallback
                e1 = row['Euler1'] if 'Euler1' in row.index and pd.notna(row['Euler1']) else 0.0
                e2 = row['Euler2'] if 'Euler2' in row.index and pd.notna(row['Euler2']) else 0.0
                e3 = row['Euler3'] if 'Euler3' in row.index and pd.notna(row['Euler3']) else 0.0
                phase = row['Phase'] if 'Phase' in row.index and pd.notna(row['Phase']) else 1
                f.write(f"{int(phase)}\t{float(x_val):.6f}\t{float(y_val):.6f}\t"
                       f"{int(row.get('Bands', 0))}\t{int(row.get('Error', 0))}\t"
                       f"{float(e1):.6f}\t{float(e2):.6f}\t{float(e3):.6f}\t"
                       f"{float(row.get('MAD', 0)):.6f}\t{int(row.get('BC', 0))}\t{int(row.get('BS', 0))}\n")
        
        print(f"Cropped CTF saved to: {output_path}")
    
    def filter_grains_by_region(self, grains_df: pd.DataFrame, 
                              x_col: str = 'X', y_col: str = 'Y') -> pd.DataFrame:
        """
        Filter an existing grains DataFrame by the selected region.
        
        Args:
            grains_df: DataFrame with grain data containing X, Y columns
            x_col: Name of X coordinate column
            y_col: Name of Y coordinate column
            
        Returns:
            pd.DataFrame: Filtered grains DataFrame
        """
        if self.region_mask is None:
            raise ValueError("No region selected. Use select_rectangular_region() or select_polygonal_region() first.")
        
        if x_col not in grains_df.columns or y_col not in grains_df.columns:
            raise ValueError(f"Grains DataFrame must contain '{x_col}' and '{y_col}' columns")
        
        # Apply the same spatial filter to the grains data
        mask = (
            (grains_df[x_col] >= self.get_data_bounds()['x_min']) &
            (grains_df[x_col] <= self.get_data_bounds()['x_max']) &
            (grains_df[y_col] >= self.get_data_bounds()['y_min']) &
            (grains_df[y_col] <= self.get_data_bounds()['y_max'])
        )
        
        return grains_df[mask].copy()
    
    def visualize_region(self, subregion_df: pd.DataFrame, 
                        show_boundaries: bool = True,
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize the selected region with optional boundaries.
        
        Args:
            subregion_df: DataFrame containing the subregion data
            show_boundaries: Whether to show the selection boundaries
            figsize: Figure size tuple
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        print("[DEBUG] visualize_region: starting render")
        print(f"[DEBUG] Subregion rows={len(subregion_df)}")
        print(f"[DEBUG] DF available: {isinstance(self.df, pd.DataFrame)} size={0 if not isinstance(self.df, pd.DataFrame) else len(self.df)} cols={[] if not isinstance(self.df, pd.DataFrame) else list(self.df.columns)} df_um_valid={self._has_valid_df_um()}")
        
        # If CTF table is available, plot points with Euler1 coloring
        if self._has_valid_df_um():
            print("[DEBUG] visualize_region: using CTF scatter route")
            ax.scatter(self.df_um['X_um'], self.df_um['Y_um'], s=0.5, alpha=0.3, c='lightgray', label='Full dataset')
            # Use normalized coordinates if present in subregion_df, else derive
            if not subregion_df.empty and 'X_um' in subregion_df.columns and 'Y_um' in subregion_df.columns:
                if 'Euler1' in subregion_df.columns and np.isfinite(subregion_df['Euler1']).any():
                    scatter = ax.scatter(
                        subregion_df['X_um'], subregion_df['Y_um'], c=subregion_df['Euler1'] % 360,
                        cmap='hsv', s=2, alpha=0.8, label='Selected region'
                    )
                    plt.colorbar(scatter, ax=ax, label='Euler1 (degrees)')
                else:
                    ax.scatter(subregion_df['X_um'], subregion_df['Y_um'], s=2, alpha=0.8, c='red', label='Selected region')
            elif not subregion_df.empty and {'X','Y'}.issubset(subregion_df.columns):
                x_min_df = float(self.df['X'].min())
                y_min_df = float(self.df['Y'].min())
                ax.scatter(subregion_df['X'] - x_min_df, subregion_df['Y'] - y_min_df, s=2, alpha=0.8, c='red', label='Selected region')
        else:
            # Fallback: render grainIDMap image so grains are visible
            print("[DEBUG] visualize_region: using EBSD map image route")
            grain_map_ids_array = None
            if hasattr(self.ebsd_map, 'grainIDMap') and self.ebsd_map.grainIDMap is not None:
                grain_map_ids_array = self.ebsd_map.grainIDMap
            elif hasattr(self.ebsd_map, 'grains') and isinstance(getattr(self.ebsd_map, 'grains'), np.ndarray) and getattr(self.ebsd_map, 'grains').shape == (self._ny, self._nx):
                grain_map_ids_array = self.ebsd_map.grains

            if isinstance(grain_map_ids_array, np.ndarray):
                try:
                    print(f"[DEBUG] grainIDMap stats: shape={grain_map_ids_array.shape} dtype={grain_map_ids_array.dtype} min={np.nanmin(grain_map_ids_array)} max={np.nanmax(grain_map_ids_array)} unique_nonzero={len(np.unique(grain_map_ids_array[grain_map_ids_array>0]))}")
                except Exception:
                    pass
                extent = [0.0, self._nx * self._step_um, 0.0, self._ny * self._step_um]
                print(f"[DEBUG] imshow extent={extent}")
                im = ax.imshow(
                    grain_map_ids_array, origin='lower', cmap='nipy_spectral',
                    interpolation='nearest', extent=extent, aspect='equal', alpha=0.9
                )
                plt.colorbar(im, ax=ax, label='Grain ID')
                
                # Optional: overlay simple boundaries for visual clarity
                try:
                    gid = grain_map_ids_array
                    boundaries = np.zeros_like(gid, dtype=bool)
                    boundaries[:-1, :] |= gid[:-1, :] != gid[1:, :]
                    boundaries[:, :-1] |= gid[:, :-1] != gid[:, 1:]
                    # Plot boundaries as semi-transparent overlay
                    ax.imshow(
                        np.where(boundaries, 1.0, np.nan), origin='lower', cmap='gray',
                        interpolation='nearest', extent=extent, aspect='equal', alpha=0.5
                    )
                    print("[DEBUG] Boundaries overlay applied")
                except Exception:
                    pass
            else:
                # Nothing to render; keep axes limits sensible
                ax.set_xlim(self.x_min, self.x_max)
                ax.set_ylim(self.y_min, self.y_max)
                print(f"[DEBUG] No grainIDMap/grains array available. Ax limits set to x=[{self.x_min},{self.x_max}] y=[{self.y_min},{self.y_max}]")
        
        # Show selection boundaries
        if show_boundaries and self.selected_region is not None:
            if self.selected_region['type'] == 'rectangle':
                x_min, y_min, x_max, y_max = self.selected_region['coords']
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            elif self.selected_region['type'] == 'polygon':
                vertices = np.array(self.selected_region['coords'])
                polygon = patches.Polygon(vertices, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(polygon)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        # Title: show number of grains intersecting the ROI (preferred over points)
        n_grains = None
        print(f"[DEBUG] visualize_region: attempting grain count, selected_region={self.selected_region}")
        try:
            if self.selected_region is not None:
                if self.selected_region.get('type') == 'rectangle':
                    grain_ids = self.get_grain_ids_in_selection()
                    n_grains = len(grain_ids)
                    print(f"[DEBUG] visualize_region: found {n_grains} grains in rectangular selection")
                elif self.selected_region.get('type') == 'polygon':
                    coords = self.selected_region.get('coords', [])
                    grain_ids = self.get_grain_ids_in_polygon(coords)
                    n_grains = len(grain_ids)
                    print(f"[DEBUG] visualize_region: found {n_grains} grains in polygonal selection")
            else:
                print("[DEBUG] visualize_region: no selected_region set")
        except Exception as e:
            print(f"[DEBUG] visualize_region: error counting grains: {e}")
            n_grains = None
        if n_grains is not None:
            ax.set_title(f'Selected Region ({n_grains} grains)')
        else:
            ax.set_title('EBSD Map')
        ax.set_aspect('equal')
        # Only show legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()
        try:
            print(f"[DEBUG] Final axis limits: x={ax.get_xlim()} y={ax.get_ylim()}")
        except Exception:
            pass
        
        return fig

    # --- EBSD map utilities for ROI (operate on defdap map, not DataFrame) ---

    def get_roi_index_slices(self) -> Optional[Tuple[slice, slice]]:
        """Return (y_slice, x_slice) for the current rectangular selection in pixel indices.

        Returns None if no rectangular selection is active.
        """
        if not self.selected_region or self.selected_region.get('type') != 'rectangle':
            return None
        x_min_um, y_min_um, x_max_um, y_max_um = self.selected_region['coords']
        # Convert μm to indices
        x0 = max(0, int(np.floor(x_min_um / self._step_um)))
        y0 = max(0, int(np.floor(y_min_um / self._step_um)))
        x1 = min(self._nx, int(np.ceil(x_max_um / self._step_um)) + 1)
        y1 = min(self._ny, int(np.ceil(y_max_um / self._step_um)) + 1)
        return slice(y0, y1), slice(x0, x1)

    def get_grain_ids_in_selection(self) -> np.ndarray:
        """Collect unique grain IDs within the current rectangular ROI from grainIDMap.

        Returns an empty array if no selection or map does not provide grainIDMap.
        """
        print(f"[DEBUG] get_grain_ids_in_selection: selected_region={self.selected_region}")
        print(f"[DEBUG] get_grain_ids_in_selection: ebsd_map type={type(self.ebsd_map)}")
        print(f"[DEBUG] get_grain_ids_in_selection: ebsd_map attributes={[a for a in dir(self.ebsd_map) if not a.startswith('_')][:20]}")
        print(f"[DEBUG] get_grain_ids_in_selection: has grainIDMap={hasattr(self.ebsd_map, 'grainIDMap')}")
        print(f"[DEBUG] get_grain_ids_in_selection: has grains={hasattr(self.ebsd_map, 'grains')}")
        if hasattr(self.ebsd_map, 'grainIDMap'):
            print(f"[DEBUG] get_grain_ids_in_selection: grainIDMap is None={self.ebsd_map.grainIDMap is None}")
        if hasattr(self.ebsd_map, 'grains'):
            grains_attr = self.ebsd_map.grains
            print(f"[DEBUG] get_grain_ids_in_selection: grains type={type(grains_attr)} is_ndarray={isinstance(grains_attr, np.ndarray)}")
            if isinstance(grains_attr, np.ndarray):
                print(f"[DEBUG] get_grain_ids_in_selection: grains.shape={grains_attr.shape}")
        
        if not hasattr(self.ebsd_map, 'grainIDMap') or self.ebsd_map.grainIDMap is None:
            print("[DEBUG] get_grain_ids_in_selection: no grainIDMap, trying grains attribute")
            # Try the 'grains' attribute as fallback (used by some defdap versions)
            if hasattr(self.ebsd_map, 'grains') and isinstance(self.ebsd_map.grains, np.ndarray):
                grain_map = self.ebsd_map.grains
            else:
                print("[DEBUG] get_grain_ids_in_selection: no valid grain map found")
                return np.array([], dtype=int)
        else:
            grain_map = self.ebsd_map.grainIDMap
        idx_slices = self.get_roi_index_slices()
        if idx_slices is None:
            print("[DEBUG] get_grain_ids_in_selection: no ROI slices")
            return np.array([], dtype=int)
        ys, xs = idx_slices
        print(f"[DEBUG] get_grain_ids_in_selection: ROI slices y={ys} x={xs}")
        roi = grain_map[ys, xs]
        unique_ids = np.unique(roi)
        unique_ids = unique_ids[unique_ids > 0]
        print(f"[DEBUG] get_grain_ids_in_selection: grains_in_roi={len(unique_ids)} unique_ids={unique_ids[:10]}...")
        return unique_ids

    def get_grain_ids_in_polygon(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """Collect unique grain IDs that intersect a polygon ROI in μm coordinates."""
        # Use same fallback logic as get_grain_ids_in_selection
        if not hasattr(self.ebsd_map, 'grainIDMap') or self.ebsd_map.grainIDMap is None:
            if hasattr(self.ebsd_map, 'grains') and isinstance(self.ebsd_map.grains, np.ndarray):
                grain_map = self.ebsd_map.grains
            else:
                print("[DEBUG] get_grain_ids_in_polygon: no valid grain map found")
                return np.array([], dtype=int)
        else:
            grain_map = self.ebsd_map.grainIDMap
            
        from matplotlib.path import Path as MPLPath
        polygon_path = MPLPath(vertices)
        # Build pixel-center coordinates in μm
        xx = np.linspace(0.0, self._nx * self._step_um, self._nx, endpoint=False)
        yy = np.linspace(0.0, self._ny * self._step_um, self._ny, endpoint=False)
        XX, YY = np.meshgrid(xx, yy)
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        mask_flat = polygon_path.contains_points(pts)
        if not np.any(mask_flat):
            return np.array([], dtype=int)
        gid = grain_map.ravel()[mask_flat]
        unique_ids = np.unique(gid)
        unique_ids = unique_ids[unique_ids > 0]
        print(f"[DEBUG] get_grain_ids_in_polygon: grains_in_roi={len(unique_ids)}")
        return unique_ids

    def crop_grain_id_map(self) -> Optional[np.ndarray]:
        """Return a cropped view of the grainIDMap for the current rectangular ROI.

        Returns None if no rectangular selection is active or grainIDMap unavailable.
        """
        if not hasattr(self.ebsd_map, 'grainIDMap') or self.ebsd_map.grainIDMap is None:
            return None
        idx_slices = self.get_roi_index_slices()
        if idx_slices is None:
            return None
        ys, xs = idx_slices
        return self.ebsd_map.grainIDMap[ys, xs]
    
    def get_region_statistics(self, subregion_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the selected region.
        
        Args:
            subregion_df: DataFrame containing the subregion data
            
        Returns:
            Dict: Statistics about the region
        """
        # Use normalized coordinates if present
        xcol = 'X_um' if 'X_um' in subregion_df.columns else ('X' if 'X' in subregion_df.columns else None)
        ycol = 'Y_um' if 'Y_um' in subregion_df.columns else ('Y' if 'Y' in subregion_df.columns else None)
        if xcol is None or ycol is None or subregion_df.empty:
            return {
                'num_points': len(subregion_df),
                'area_um2': 0,
                'x_range': float('nan'),
                'y_range': float('nan'),
                'x_center': float('nan'),
                'y_center': float('nan'),
                'orientation_stats': {}
            }

        stats = {
            'num_points': len(subregion_df),
            'area_um2': 0,  # We could approximate via pixel area when using map ROI
            'x_range': float(subregion_df[xcol].max() - subregion_df[xcol].min()),
            'y_range': float(subregion_df[ycol].max() - subregion_df[ycol].min()),
            'x_center': float(subregion_df[xcol].mean()),
            'y_center': float(subregion_df[ycol].mean()),
            'orientation_stats': {}
        }
        if {'Euler1','Euler2','Euler3'}.issubset(subregion_df.columns):
            stats['orientation_stats'] = {
                'euler1_mean': float(subregion_df['Euler1'].mean()),
                'euler1_std': float(subregion_df['Euler1'].std()),
                'euler2_mean': float(subregion_df['Euler2'].mean()),
                'euler2_std': float(subregion_df['Euler2'].std()),
                'euler3_mean': float(subregion_df['Euler3'].mean()),
                'euler3_std': float(subregion_df['Euler3'].std()),
            }
        
        return stats


def create_mask_from_coordinates(coordinates: List[Tuple[float, float]], 
                               x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask from coordinate vertices.
    
    Args:
        coordinates: List of (x, y) coordinate pairs
        x_coords: X coordinate array
        y_coords: Y coordinate array
        
    Returns:
        np.ndarray: Boolean mask array
    """
    from matplotlib.path import Path as MPLPath
    
    # Create coordinate grid
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Create path and check containment
    polygon_path = MPLPath(coordinates)
    mask = polygon_path.contains_points(points)
    
    return mask.reshape(xx.shape)


# Example usage and workflow functions

def workflow_subregion_analysis(ctf_path: str, output_dir: str = "./subregions"):
    """
    Example workflow for subregion analysis.
    
    Args:
        ctf_path: Path to the CTF file
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize selector
    selector = SubregionSelector(ctf_path)
    
    print("Data bounds:", selector.get_data_bounds())
    
    # Interactive selection
    print("Starting interactive region selection...")
    subregion_df = selector.interactive_region_selection()
    
    if len(subregion_df) > 0:
        # Get statistics
        stats = selector.get_region_statistics(subregion_df)
        print("Region statistics:", stats)
        
        # Visualize
        fig = selector.visualize_region(subregion_df)
        fig.savefig(os.path.join(output_dir, "selected_region.png"), dpi=300)
        plt.close(fig)
        
        # Save cropped CTF
        output_ctf = os.path.join(output_dir, "cropped_region.ctf")
        selector.save_cropped_ctf(output_ctf, subregion_df)
        
        print(f"Subregion analysis complete. Outputs saved to {output_dir}")
    else:
        print("No region selected.")


if __name__ == "__main__":
    # Example usage
    ctf_file = "example.ctf"  # Replace with your CTF file
    if os.path.exists(ctf_file):
        workflow_subregion_analysis(ctf_file)
    else:
        print(f"CTF file not found: {ctf_file}")
