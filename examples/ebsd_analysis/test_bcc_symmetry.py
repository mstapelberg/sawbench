"""
Test script to check if crystallographically equivalent orientations 
give the same SAW velocities for BCC Vanadium.

This tests the 24 symmetry operations of the cubic point group.
"""

import numpy as np
from sawbench import Material, SAWCalculator
from sawbench.orientations import cross_product_matrix, matrix_to_euler

# Define Vanadium material (same as simple_interactive_example.py)
material = Material(
    formula='V',
    C11=229e9,
    C12=119e9,
    C44=43e9,
    density=6110,
    crystal_class='cubic'
)

# Define the 24 symmetry operations for cubic crystals (BCC)
# These are the rotation matrices that leave a cube invariant
def get_cubic_symmetry_operations():
    """Returns the 24 symmetry operations for cubic crystals."""
    ops = []
    
    # Identity
    ops.append(np.eye(3))
    
    # 90° rotations about x, y, z axes (3 axes × 3 rotations each = 9)
    # About z-axis
    ops.append(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))  # 90°
    ops.append(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))  # 180°
    ops.append(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))  # 270°
    
    # About x-axis
    ops.append(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))  # 90°
    ops.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))  # 180°
    ops.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))  # 270°
    
    # About y-axis
    ops.append(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))  # 90°
    ops.append(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))  # 180°
    ops.append(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))  # 270°
    
    # 180° rotations about face diagonals (6 operations)
    ops.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))
    ops.append(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]))
    ops.append(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
    ops.append(np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
    ops.append(np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    ops.append(np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]))
    
    # 120° rotations about body diagonals (8 operations)
    ops.append(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    ops.append(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    ops.append(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]))
    ops.append(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))
    ops.append(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))
    ops.append(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    ops.append(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))
    ops.append(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
    
    return ops


def test_orientation_equivalence():
    """Test if symmetrically equivalent orientations give the same SAW velocity."""
    
    print("="*70)
    print("Testing BCC Symmetry for SAW Velocities")
    print("="*70)
    
    # Test orientation: (100) surface with [001] propagation direction
    surface_normal = np.array([1, 0, 0])
    prop_direction = np.array([0, 0, 1])
    
    # Create orientation matrix
    base_matrix = cross_product_matrix(prop_direction, surface_normal)
    base_euler = matrix_to_euler(base_matrix)
    
    print(f"\nBase orientation:")
    print(f"  Surface normal: {surface_normal}")
    print(f"  Propagation direction: {prop_direction}")
    print(f"  Euler angles (rad): φ₁={base_euler[0]:.4f}, Φ={base_euler[1]:.4f}, φ₂={base_euler[2]:.4f}")
    print(f"  Euler angles (deg): φ₁={np.rad2deg(base_euler[0]):.2f}°, Φ={np.rad2deg(base_euler[1]):.2f}°, φ₂={np.rad2deg(base_euler[2]):.2f}°")
    
    # Calculate SAW velocity for base orientation
    calc_base = SAWCalculator(material, np.array(base_euler))
    v_base, _, _ = calc_base.get_saw_speed(deg=135.0, sampling=4000, psaw=0)
    
    print(f"\nBase SAW velocity: {v_base[0]:.2f} m/s")
    
    # Get all symmetry operations
    sym_ops = get_cubic_symmetry_operations()
    
    print(f"\nTesting {len(sym_ops)} symmetry operations...")
    print("-"*70)
    
    velocities = []
    unique_velocities = []
    tolerance = 1.0  # m/s tolerance for considering velocities equal
    
    for i, sym_op in enumerate(sym_ops):
        # Apply symmetry operation to the base orientation matrix
        transformed_matrix = sym_op @ base_matrix
        
        # Convert to Euler angles
        transformed_euler = matrix_to_euler(transformed_matrix)
        
        # Calculate SAW velocity
        calc = SAWCalculator(material, np.array(transformed_euler))
        v, _, _ = calc.get_saw_speed(deg=135.0, sampling=4000, psaw=0)
        
        velocities.append(v[0])
        
        # Check if this is a new unique velocity
        is_unique = True
        for uv in unique_velocities:
            if abs(v[0] - uv) < tolerance:
                is_unique = False
                break
        
        if is_unique:
            unique_velocities.append(v[0])
        
        # Print details for first few and any that differ significantly
        if i < 5 or abs(v[0] - v_base[0]) > tolerance:
            print(f"Op {i+1:2d}: v = {v[0]:7.2f} m/s | "
                  f"Δv = {v[0]-v_base[0]:+6.2f} m/s | "
                  f"Euler (deg): ({np.rad2deg(transformed_euler[0]):6.1f}, "
                  f"{np.rad2deg(transformed_euler[1]):6.1f}, "
                  f"{np.rad2deg(transformed_euler[2]):6.1f})")
    
    print("-"*70)
    print(f"\nResults:")
    print(f"  Number of symmetry operations tested: {len(sym_ops)}")
    print(f"  Velocity range: {min(velocities):.2f} - {max(velocities):.2f} m/s")
    print(f"  Velocity spread: {max(velocities) - min(velocities):.2f} m/s")
    print(f"  Number of unique velocities (±{tolerance} m/s): {len(unique_velocities)}")
    
    if len(unique_velocities) == 1:
        print(f"\n✓ All symmetrically equivalent orientations give the same SAW velocity!")
        print(f"  This means we SHOULD account for crystallographic symmetry in clustering.")
    else:
        print(f"\n⚠ Symmetrically equivalent orientations give DIFFERENT SAW velocities!")
        print(f"  Unique velocities found: {sorted(unique_velocities)}")
        print(f"  This suggests the orientation or SAW calculation may be direction-dependent.")
    
    return velocities, unique_velocities


def test_specific_orientations():
    """Test specific crystallographic orientations."""
    
    print("\n" + "="*70)
    print("Testing Specific Crystallographic Orientations")
    print("="*70)
    
    test_cases = [
        # (surface_normal, propagation_direction, name)
        ([1, 0, 0], [0, 0, 1], "(100)[001]"),
        ([0, 0, 1], [1, 0, 0], "(001)[100]"),  # Should be equivalent to above
        ([0, 1, 0], [1, 0, 0], "(010)[100]"),  # Another equivalent
        ([1, 1, 0], [0, 0, 1], "(110)[001]"),
        ([1, 1, 1], [1, -1, 0], "(111)[1-10]"),
    ]
    
    results = []
    
    for surface, prop, name in test_cases:
        surface_norm = np.array(surface, dtype=float)
        surface_norm = surface_norm / np.linalg.norm(surface_norm)
        
        prop_norm = np.array(prop, dtype=float)
        prop_norm = prop_norm / np.linalg.norm(prop_norm)
        
        # Create orientation matrix
        matrix = cross_product_matrix(prop_norm, surface_norm)
        euler = matrix_to_euler(matrix)
        
        # Calculate SAW velocity
        calc = SAWCalculator(material, np.array(euler))
        v, _, _ = calc.get_saw_speed(deg=135.0, sampling=4000, psaw=0)
        
        results.append((name, v[0], euler))
        
        print(f"\n{name}:")
        print(f"  SAW velocity: {v[0]:.2f} m/s")
        print(f"  Euler (deg): φ₁={np.rad2deg(euler[0]):6.1f}°, "
              f"Φ={np.rad2deg(euler[1]):6.1f}°, φ₂={np.rad2deg(euler[2]):6.1f}°")
    
    print("\n" + "-"*70)
    print("Summary:")
    for name, v, _ in results:
        print(f"  {name:15s}: {v:7.2f} m/s")
    
    return results


if __name__ == "__main__":
    # Test symmetry operations
    velocities, unique_velocities = test_orientation_equivalence()
    
    # Test specific orientations
    specific_results = test_specific_orientations()
    
    print("\n" + "="*70)
    print("Conclusion:")
    print("="*70)
    if len(unique_velocities) > 1:
        print("Crystallographic symmetry DOES affect SAW velocities in this system.")
        print("We should NOT treat symmetrically equivalent orientations as identical")
        print("when clustering grain orientations.")
    else:
        print("Crystallographic symmetry does NOT affect SAW velocities in this system.")
        print("We SHOULD account for symmetry when clustering grain orientations")
        print("to avoid counting the same physical orientation multiple times.")
