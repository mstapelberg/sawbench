"""
Orientation clustering with crystallographic symmetry consideration.

This module provides functions to identify the most prevalent grain orientations
in EBSD data while accounting for crystallographic symmetry equivalence.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def get_cubic_symmetry_operations() -> np.ndarray:
    """
    Returns the 24 symmetry operations for cubic crystals.
    
    Returns:
        np.ndarray: Array of shape (24, 3, 3) containing rotation matrices
    """
    ops = []
    
    # Identity
    ops.append(np.eye(3))
    
    # 90° rotations about x, y, z axes
    # About z-axis
    ops.append(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    ops.append(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
    ops.append(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    
    # About x-axis
    ops.append(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    ops.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    ops.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    
    # About y-axis
    ops.append(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    ops.append(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    ops.append(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
    
    # 180° rotations about face diagonals
    ops.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))
    ops.append(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
    ops.append(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
    ops.append(np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
    ops.append(np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    ops.append(np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]))
    
    # 120° rotations about body diagonals
    ops.append(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    ops.append(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    ops.append(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]))
    ops.append(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))
    ops.append(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))
    ops.append(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    ops.append(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))
    ops.append(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
    
    return np.array(ops)


def euler_to_rotation_matrix(phi1: float, Phi: float, phi2: float) -> np.ndarray:
    """
    Convert Bunge Euler angles to rotation matrix.
    
    Args:
        phi1: First Euler angle (radians)
        Phi: Second Euler angle (radians)
        phi2: Third Euler angle (radians)
    
    Returns:
        3x3 rotation matrix
    """
    # Bunge convention: Z-X-Z
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    
    R = np.array([
        [c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c,  s2*s],
        [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
        [s1*s,             -c1*s,            c]
    ])
    
    return R


def misorientation_angle_cubic(euler1: np.ndarray, euler2: np.ndarray, 
                                sym_ops: Optional[np.ndarray] = None) -> float:
    """
    Calculate minimum misorientation angle between two orientations
    considering cubic symmetry.
    
    Args:
        euler1: Euler angles [phi1, Phi, phi2] in radians for first orientation
        euler2: Euler angles [phi1, Phi, phi2] in radians for second orientation
        sym_ops: Pre-computed symmetry operations (optional, for efficiency)
    
    Returns:
        Minimum misorientation angle in radians
    """
    if sym_ops is None:
        sym_ops = get_cubic_symmetry_operations()
    
    # Convert to rotation matrices
    R1 = euler_to_rotation_matrix(euler1[0], euler1[1], euler1[2])
    R2 = euler_to_rotation_matrix(euler2[0], euler2[1], euler2[2])
    
    # Calculate misorientation considering all symmetry variants
    min_angle = np.pi
    
    for sym in sym_ops:
        # Apply symmetry operation to first orientation
        R1_sym = sym @ R1
        
        # Calculate misorientation
        R_mis = R2 @ R1_sym.T
        
        # Calculate rotation angle from trace
        trace = np.trace(R_mis)
        # Clamp to valid range to avoid numerical errors in arccos
        trace = np.clip(trace, -1, 3)
        angle = np.arccos((trace - 1) / 2)
        
        if angle < min_angle:
            min_angle = angle
    
    return min_angle


def cluster_orientations_cubic(euler_angles: np.ndarray, 
                               eps_degrees: float = 5.0,
                               min_samples: int = 2) -> Tuple[np.ndarray, Dict]:
    """
    Cluster grain orientations using DBSCAN with cubic symmetry-aware distance.
    
    Args:
        euler_angles: Array of shape (N, 3) with Euler angles in radians
        eps_degrees: Maximum misorientation angle (in degrees) for neighbors
        min_samples: Minimum number of grains to form a cluster
    
    Returns:
        labels: Cluster labels for each grain (-1 = noise)
        cluster_info: Dictionary with cluster statistics
    """
    n_grains = euler_angles.shape[0]
    
    print(f"Calculating misorientation matrix for {n_grains} grains...")
    print(f"  (This may take a moment for large datasets)")
    
    # Pre-compute symmetry operations
    sym_ops = get_cubic_symmetry_operations()
    
    # Calculate pairwise misorientation angles
    # This is the bottleneck - O(N^2 * 24) operations
    misorientation_matrix = np.zeros((n_grains, n_grains))
    
    for i in range(n_grains):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_grains} grains")
        
        for j in range(i+1, n_grains):
            angle = misorientation_angle_cubic(euler_angles[i], euler_angles[j], sym_ops)
            misorientation_matrix[i, j] = angle
            misorientation_matrix[j, i] = angle
    
    print(f"  Misorientation matrix computed.")
    
    # Convert to degrees for clustering
    misorientation_matrix_deg = np.rad2deg(misorientation_matrix)
    
    # Use DBSCAN with precomputed distances
    clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(misorientation_matrix_deg)
    
    # Calculate cluster statistics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    n_noise = np.sum(labels == -1)
    
    cluster_info = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'labels': labels,
        'misorientation_matrix_deg': misorientation_matrix_deg
    }
    
    print(f"\nClustering results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    return labels, cluster_info


def find_top_orientations(grains_df: pd.DataFrame,
                         n_top: int = 5,
                         eps_degrees: float = 5.0,
                         min_samples: int = 2,
                         weight_by_size: bool = True) -> pd.DataFrame:
    """
    Find the most prevalent grain orientations in EBSD data.
    
    Args:
        grains_df: DataFrame with columns 'Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)', 'Size (um^2)'
        n_top: Number of top orientations to return
        eps_degrees: Maximum misorientation for clustering (degrees)
        min_samples: Minimum samples for DBSCAN
        weight_by_size: If True, weight clusters by grain size
    
    Returns:
        DataFrame with top N orientations and their statistics
    """
    # Extract Euler angles
    euler_angles = grains_df[['Euler1 (rad)', 'Euler2 (rad)', 'Euler3 (rad)']].values
    
    # Perform clustering
    labels, cluster_info = cluster_orientations_cubic(euler_angles, eps_degrees, min_samples)
    
    # Calculate cluster properties
    cluster_data = []
    
    for label in np.unique(labels):
        if label == -1:  # Skip noise
            continue
        
        mask = labels == label
        cluster_grains = grains_df[mask]
        cluster_eulers = euler_angles[mask]
        
        # Calculate mean orientation (simple average - could be improved)
        mean_euler = np.mean(cluster_eulers, axis=0)
        
        # Calculate cluster statistics
        n_grains = np.sum(mask)
        total_area = cluster_grains['Size (um^2)'].sum() if 'Size (um^2)' in cluster_grains.columns else n_grains
        
        # Calculate spread (average misorientation from mean)
        spread_angles = [misorientation_angle_cubic(mean_euler, e) 
                        for e in cluster_eulers]
        mean_spread = np.rad2deg(np.mean(spread_angles))
        max_spread = np.rad2deg(np.max(spread_angles))
        
        cluster_data.append({
            'Cluster_ID': int(label),
            'N_Grains': int(n_grains),
            'Total_Area_um2': float(total_area),
            'Mean_phi1_deg': float(np.rad2deg(mean_euler[0])),
            'Mean_Phi_deg': float(np.rad2deg(mean_euler[1])),
            'Mean_phi2_deg': float(np.rad2deg(mean_euler[2])),
            'Mean_phi1_rad': float(mean_euler[0]),
            'Mean_Phi_rad': float(mean_euler[1]),
            'Mean_phi2_rad': float(mean_euler[2]),
            'Spread_deg': float(mean_spread),
            'Max_Spread_deg': float(max_spread),
            'Fraction_by_count': float(n_grains / len(grains_df)),
            'Fraction_by_area': float(total_area / grains_df['Size (um^2)'].sum()) if 'Size (um^2)' in grains_df.columns else float(n_grains / len(grains_df))
        })
    
    # Create DataFrame and sort
    clusters_df = pd.DataFrame(cluster_data)
    
    if clusters_df.empty:
        print("Warning: No clusters found!")
        return clusters_df
    
    # Sort by area or count
    sort_by = 'Total_Area_um2' if weight_by_size else 'N_Grains'
    clusters_df = clusters_df.sort_values(sort_by, ascending=False)
    
    # Return top N
    top_clusters = clusters_df.head(n_top).copy()
    top_clusters['Rank'] = range(1, len(top_clusters) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Cluster_ID', 'N_Grains', 'Total_Area_um2', 'Fraction_by_count', 'Fraction_by_area',
            'Mean_phi1_deg', 'Mean_Phi_deg', 'Mean_phi2_deg',
            'Mean_phi1_rad', 'Mean_Phi_rad', 'Mean_phi2_rad',
            'Spread_deg', 'Max_Spread_deg']
    top_clusters = top_clusters[cols]
    
    return top_clusters, labels, cluster_info


if __name__ == "__main__":
    print("Orientation clustering module")
    print("Import this module to use clustering functions")

