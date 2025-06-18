# ruff: noqa: RUF002, RUF003, PLC2401
"""Calculation of elastic properties of crystals.

Primary Sources and References for Crystal Elasticity.

- Landau, L.D. & Lifshitz, E.M. "Theory of Elasticity" (Volume 7 of Course of
  Theoretical Physics)

- Teodosiu, C. (1982) "Elastic Models of Crystal Defects"

Review Articles:

- Mouhat, F., & Coudert, F. X. (2014).
  "Necessary and sufficient elastic stability conditions in various crystal systems"
  Physical Review B, 90(22), 224104

Online Resources:
- Materials Project Documentation
  https://docs.materialsproject.org/methodology/elasticity/
"""

from enum import Enum
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from collections.abc import Callable
from dataclasses import dataclass
import os
from ase.io import read


EV_A3_TO_GPA = 160.21766208


@dataclass
class DeformationRule:
    """Defines rules for applying deformations based on crystal symmetry.

    This class specifies which axes to deform and how to handle symmetry
    constraints when calculating elastic properties.

    Attributes:
        axes: List of indices indicating which strain components to consider
              for the specific crystal symmetry, following Voigt notation:
              [0=xx, 1=yy, 2=zz, 3=yz, 4=xz, 5=xy]
        symmetry_handler: Callable function that constructs the stress-strain
                         relationship matrix according to the crystal symmetry.
    """

    axes: list[int]
    symmetry_handler: Callable


class BravaisType(Enum):
    """Enumeration of the seven Bravais lattice types in 3D crystals."""

    CUBIC = "cubic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"


def get_bravais_type(
    atoms: Atoms, length_tol: float = 1e-2, angle_tol: float = 0.1
) -> BravaisType:
    """
    Check and return the crystal system of a structure using ASE.

    This function determines the crystal system by analyzing the lattice
    parameters and angles.

    Args:
        atoms: ASE Atoms object representing the crystal structure.
        length_tol: Tolerance for floating-point comparisons of lattice lengths.
        angle_tol: Tolerance for floating-point comparisons of lattice angles in degrees.

    Returns:
        BravaisType: The determined Bravais type.
    """
    cell_params = atoms.cell.cellpar()
    a, b, c = cell_params[:3]
    alpha, beta, gamma = cell_params[3:]

    # Cubic: a = b = c, alpha = beta = gamma = 90°
    if (
        abs(a - b) < length_tol
        and abs(b - c) < length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
    ):
        return BravaisType.CUBIC

    # Hexagonal: a = b ≠ c, alpha = beta = 90°, gamma = 120°
    if (
        abs(a - b) < length_tol
        and abs(c - a) > length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 120) < angle_tol
    ):
        return BravaisType.HEXAGONAL

    # Tetragonal: a = b ≠ c, alpha = beta = gamma = 90°
    if (
        abs(a - b) < length_tol
        and abs(c - a) > length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
    ):
        return BravaisType.TETRAGONAL

    # Trigonal/Rhombohedral: a = b = c, alpha = beta = gamma ≠ 90°
    if (
        abs(a - b) < length_tol
        and abs(b - c) < length_tol
        and abs(alpha - beta) < angle_tol
        and abs(beta - gamma) < angle_tol
        and abs(alpha - 90) > angle_tol
    ):
        return BravaisType.TRIGONAL

    # Orthorhombic: a ≠ b ≠ c, alpha = beta = gamma = 90°
    if (
        abs(a - b) > length_tol
        and abs(b - c) > length_tol
        and abs(a - c) > length_tol
        and abs(alpha - 90) < angle_tol
        and abs(beta - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
    ):
        return BravaisType.ORTHORHOMBIC

    # Monoclinic: a ≠ b ≠ c, alpha = gamma = 90°, beta ≠ 90°
    if (
        abs(alpha - 90) < angle_tol
        and abs(gamma - 90) < angle_tol
        and abs(beta - 90) > angle_tol
    ):
        return BravaisType.MONOCLINIC

    # Triclinic: a ≠ b ≠ c, alpha ≠ beta ≠ gamma ≠ 90°
    return BravaisType.TRICLINIC

def regular_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for cubic (regular) crystal symmetry.

    Constructs the stress-strain relationship matrix for cubic symmetry,
    which has three independent elastic constants: C11, C12, and C44.

    The matrix relates strains to stresses according to the equation:
    σᵢ = Σⱼ Cᵢⱼ εⱼ

    Args:
        strains: Array of shape (6,) containing strain components
            [εxx, εyy, εzz, εyz, εxz, εxy] where:
            - εxx, εyy, εzz are normal strains
            - εyz, εxz, εxy are shear strains

    Returns:
        np.ndarray: Matrix of shape (6, 3) where columns correspond to
            coefficients for C11, C12, and C44 respectively

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    (εyy + εzz)    0      ⎤
        ⎢ εyy    (εxx + εzz)    0      ⎥
        ⎢ εzz    (εxx + εyy)    0      ⎥
        ⎢ 0      0              2εyz   ⎥
        ⎢ 0      0              2εxz   ⎥
        ⎣ 0      0              2εxy   ⎦

        This represents the relationship:
        σxx = C11*εxx + C12*(εyy + εzz)
        σyy = C11*εyy + C12*(εxx + εzz)
        σzz = C11*εzz + C12*(εxx + εyy)
        σyz = 2*C44*εyz
        σxz = 2*C44*εxz
        σxy = 2*C44*εxy
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 3), dtype=strains.dtype)

    # First column
    matrix[0, 0] = εxx
    matrix[1, 0] = εyy
    matrix[2, 0] = εzz

    # Second column
    matrix[0, 1] = εyy + εzz
    matrix[1, 1] = εxx + εzz
    matrix[2, 1] = εxx + εyy

    # Third column
    matrix[3, 2] = 2 * εyz
    matrix[4, 2] = 2 * εxz
    matrix[5, 2] = 2 * εxy

    return matrix

def tetragonal_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for tetragonal crystal symmetry.

    Constructs the stress-strain relationship matrix for tetragonal symmetry,
    which has 7 independent elastic constants: C11, C12, C13, C16, C33, C44, C66.

    Args:
        strains: Array of shape (6,) containing strain components
            [εxx, εyy, εzz, εyz, εxz, εxy] where:
            - εxx, εyy, εzz are normal strains
            - εyz, εxz, εxy are shear strains

    Returns:
        np.ndarray: Matrix of shape (6, 7) where columns correspond to
            coefficients for C11, C12, C13, C16, C33, C44, C66

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz     2εxy    0      0      0    ⎤
        ⎢ εyy    εxx    εzz    -2εxy    0      0      0    ⎥
        ⎢ 0      0      εxx+εyy 0       εzz    0      0    ⎥
        ⎢ 0      0      0       0       0      2εyz   0    ⎥
        ⎢ 0      0      0       0       0      2εxz   0    ⎥
        ⎣ 0      0      0       εxx-εyy 0      0      2εxy ⎦
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 7), dtype=strains.dtype)

    # First row
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εxy

    # Second row
    matrix[1, 0] = εyy
    matrix[1, 1] = εxx
    matrix[1, 2] = εzz
    matrix[1, 3] = -2 * εxy

    # Third row
    matrix[2, 2] = εxx + εyy
    matrix[2, 4] = εzz

    # Fourth and fifth rows
    matrix[3, 5] = 2 * εyz
    matrix[4, 5] = 2 * εxz

    # Sixth row
    matrix[5, 3] = εxx - εyy
    matrix[5, 6] = 2 * εxy

    return matrix

def orthorhombic_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for orthorhombic crystal symmetry.

    Constructs the stress-strain relationship matrix for orthorhombic symmetry,
    which has nine independent elastic constants: C11, C12, C13, C22, C23, C33,
    C44, C55, and C66.

    Args:
        strains: Array of shape (6,) containing strain components
            [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        np.ndarray: Matrix of shape (6, 9) where columns correspond to
            coefficients for C11, C12, C13, C22, C23, C33, C44, C55, C66

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz    0      0      0      0      0      0  ⎤
        ⎢ 0      εxx    0      εyy    εzz    0      0      0      0  ⎥
        ⎢ 0      0      εxx    0      εyy    εzz    0      0      0  ⎥
        ⎢ 0      0      0      0      0      0      2εyz   0      0  ⎥
        ⎢ 0      0      0      0      0      0      0      2εxz   0  ⎥
        ⎣ 0      0      0      0      0      0      0      0      2εxy⎦
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 9), dtype=strains.dtype)

    # First row - C11, C12, C13, C22, C23, C33, C44, C55, C66
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz

    # Second row
    matrix[1, 1] = εxx
    matrix[1, 3] = εyy
    matrix[1, 4] = εzz

    # Third row
    matrix[2, 2] = εxx
    matrix[2, 4] = εyy
    matrix[2, 5] = εzz

    # Fourth row
    matrix[3, 6] = 2 * εyz

    # Fifth row
    matrix[4, 7] = 2 * εxz

    # Sixth row
    matrix[5, 8] = 2 * εxy

    return matrix

def trigonal_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for trigonal crystal symmetry.

    Constructs the stress-strain relationship matrix for trigonal symmetry,
    which has 7 independent elastic constants: C11, C12, C13, C14, C15, C33, C44.
    Matrix construction follows the standard form for trigonal symmetry.

    Args:
        strains: Array of shape (6,) containing strain components
            [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        np.ndarray: Matrix of shape (6, 7) where columns correspond to
            coefficients for C11, C12, C13, C14, C15, C33, C44

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz       2εyz        2εxz      0      0    ⎤
        ⎢ εyy    εxx    εzz      -2εyz       -2εxz      0      0    ⎥
        ⎢ 0      0      εxx+εyy   0           0         εzz    0    ⎥
        ⎢ 0      0      0         εxx-εyy    -2εxy      0      2εyz ⎥
        ⎢ 0      0      0         2εxy        εxx-εyy   0      2εxz ⎥
        ⎣ εxy   -εxy    0         2εxz       -2εyz      0      0    ⎦
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 7), dtype=strains.dtype)

    # First row
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εyz
    matrix[0, 4] = 2 * εxz

    # Second row
    matrix[1, 0] = εyy
    matrix[1, 1] = εxx
    matrix[1, 2] = εzz
    matrix[1, 3] = -2 * εyz
    matrix[1, 4] = -2 * εxz

    # Third row
    matrix[2, 2] = εxx + εyy
    matrix[2, 5] = εzz

    # Fourth row
    matrix[3, 3] = εxx - εyy
    matrix[3, 4] = -2 * εxy
    matrix[3, 6] = 2 * εyz

    # Fifth row
    matrix[4, 3] = 2 * εxy
    matrix[4, 4] = εxx - εyy
    matrix[4, 6] = 2 * εxz

    # Sixth row
    matrix[5, 0] = εxy
    matrix[5, 1] = -εxy
    matrix[5, 3] = 2 * εxz
    matrix[5, 4] = -2 * εyz

    return matrix

def hexagonal_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for hexagonal crystal symmetry.

    Constructs the stress-strain relationship matrix for hexagonal symmetry,
    which has 5 independent elastic constants: C11, C33, C12, C13, C44.
    Note: C66 = (C11-C12)/2 is dependent.

    Args:
        strains: Array of shape (6,) containing strain components
            [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        np.ndarray: Matrix of shape (6, 5) where columns correspond to
            coefficients for C11, C33, C12, C13, C44

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz      0     0   ⎤
        ⎢ εyy    εxx    εzz      0     0   ⎥
        ⎢ 0      0      εxx+εyy  εzz   0   ⎥
        ⎢ 0      0      0        0     2εyz⎥
        ⎢ 0      0      0        0     2εxz⎥
        ⎣ εxy   -εxy    0        0     0   ⎦
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 5), dtype=strains.dtype)

    # First row
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz

    # Second row
    matrix[1, 0] = εyy
    matrix[1, 1] = εxx
    matrix[1, 2] = εzz

    # Third row
    matrix[2, 2] = εxx + εyy
    matrix[2, 3] = εzz

    # Fourth and fifth rows
    matrix[3, 4] = 2 * εyz
    matrix[4, 4] = 2 * εxz

    # Sixth row
    matrix[5, 0] = εxy
    matrix[5, 1] = -εxy

    return matrix

def monoclinic_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for monoclinic crystal symmetry.

    Constructs the stress-strain relationship matrix for monoclinic symmetry,
    which has 13 independent elastic constants: C11, C12, C13, C15, C22, C23, C25,
    C33, C35, C44, C46, C55, C66.

    Args:
        strains: Array of shape (6,) containing strain components
            [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        np.ndarray: Matrix of shape (6, 13) where columns correspond to
            coefficients for the 13 independent constants in order:
            [C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66]

    Notes:
        For monoclinic symmetry with unique axis b (y), the matrix has the form:
        ⎡ εxx  εyy  εzz  2εxz  0    0    0    0    0    0    0    0    0  ⎤
        ⎢ 0    εxx  0    0     εyy  εzz  2εxz 0    0    0    0    0    0  ⎥
        ⎢ 0    0    εxx  0     0    εyy  0    εzz  2εxz 0    0    0    0  ⎥
        ⎢ 0    0    0    0     0    0    0    0    0    2εyz 2εxy 0    0  ⎥
        ⎢ 0    0    0    εxx   0    0    εyy  0    εzz  0    0    2εxz 0  ⎥
        ⎣ 0    0    0    0     0    0    0    0    0    0    2εyz 0    2εxy⎦
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 13), dtype=strains.dtype)

    # First row
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εxz

    # Second row
    matrix[1, 1] = εxx
    matrix[1, 4] = εyy
    matrix[1, 5] = εzz
    matrix[1, 6] = 2 * εxz

    # Third row
    matrix[2, 2] = εxx
    matrix[2, 5] = εyy
    matrix[2, 7] = εzz
    matrix[2, 8] = 2 * εxz

    # Fourth row
    matrix[3, 9] = 2 * εyz
    matrix[3, 10] = 2 * εxy

    # Fifth row
    matrix[4, 3] = εxx
    matrix[4, 6] = εyy
    matrix[4, 8] = εzz
    matrix[4, 11] = 2 * εxz

    # Sixth row
    matrix[5, 10] = 2 * εyz
    matrix[5, 12] = 2 * εxy

    return matrix

def triclinic_symmetry(strains: np.ndarray) -> np.ndarray:
    """Generate equation matrix for triclinic crystal symmetry.

    Constructs the stress-strain relationship matrix for triclinic symmetry,
    which has 21 independent elastic constants (the most general case).

    Args:
        strains: Array of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        np.ndarray: Matrix of shape (6, 21) where columns correspond to
                     all possible elastic constants in order:
                     [C11, C12, C13, C14, C15, C16,
                          C22, C23, C24, C25, C26,
                              C33, C34, C35, C36,
                                  C44, C45, C46,
                                      C55, C56,
                                          C66]
    """
    if not isinstance(strains, np.ndarray):
        strains = np.array(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains

    # Create the matrix using np.zeros for proper device/dtype handling
    matrix = np.zeros((6, 21), dtype=strains.dtype)

    # First row
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εyz
    matrix[0, 4] = 2 * εxz
    matrix[0, 5] = 2 * εxy

    # Second row
    matrix[1, 1] = εxx
    matrix[1, 6] = εyy
    matrix[1, 7] = εzz
    matrix[1, 8] = 2 * εyz
    matrix[1, 9] = 2 * εxz
    matrix[1, 10] = 2 * εxy

    # Third row
    matrix[2, 2] = εxx
    matrix[2, 7] = εyy
    matrix[2, 11] = εzz
    matrix[2, 12] = 2 * εyz
    matrix[2, 13] = 2 * εxz
    matrix[2, 14] = 2 * εxy

    # Fourth row
    matrix[3, 3] = εxx
    matrix[3, 8] = εyy
    matrix[3, 12] = εzz
    matrix[3, 15] = 2 * εyz
    matrix[3, 16] = 2 * εxz
    matrix[3, 17] = 2 * εxy

    # Fifth row
    matrix[4, 4] = εxx
    matrix[4, 9] = εyy
    matrix[4, 13] = εzz
    matrix[4, 16] = 2 * εyz
    matrix[4, 18] = 2 * εxz
    matrix[4, 19] = 2 * εxy

    # Sixth row
    matrix[5, 5] = εxx
    matrix[5, 10] = εyy
    matrix[5, 14] = εzz
    matrix[5, 17] = 2 * εyz
    matrix[5, 19] = 2 * εxz
    matrix[5, 20] = 2 * εxy

    return matrix

def get_cart_deformed_cell(atoms: Atoms, axis: int = 0, size: float = 1.0) -> Atoms:
    """Deform a unit cell and scale atomic positions accordingly.

    This function applies a deformation to the lattice vectors and scales
    the atomic positions to maintain the same fractional coordinates.

    Args:
        atoms: The input ASE Atoms object.
        axis: Direction of deformation:
            - 0, 1, 2 for xx, yy, zz normal deformations.
            - 3, 4, 5 for yz, xz, xy shear deformations.
        size: Deformation magnitude (strain).

    Returns:
        A new Atoms object with the deformed cell and scaled positions.
    """
    if not (0 <= axis <= 5):
        raise ValueError("Axis must be between 0 and 5.")

    deformed_atoms = atoms.copy()
    cell = deformed_atoms.get_cell()

    # Create the deformation matrix L. For row vectors (ASE convention),
    # the new cell is given by new_cell = cell @ L.
    # The deformation matrix L is defined as (I + u) where u is displacement
    # gradient. The convention is taken from ts-elastic.py
    L = np.identity(3)

    if axis < 3:
        # Normal strain, e.g., axis=0 -> L[0,0] = 1 + size
        L[axis, axis] += size
    elif axis == 3:  # yz shear, u_zy = size
        L[1, 2] += size
    elif axis == 4:  # xz shear, u_zx = size
        L[0, 2] += size
    else:  # axis == 5, xy shear, u_yx = size
        L[0, 1] += size

    new_cell = cell @ L
    deformed_atoms.set_cell(new_cell, scale_atoms=True)

    return deformed_atoms

def get_elementary_deformations(
    atoms: Atoms,
    n_deform: int = 5,
    max_strain_normal: float = 0.01,
    max_strain_shear: float = 0.06,
    bravais_type: BravaisType = None,
) -> list[Atoms]:
    """
    Generate elementary deformations for elastic tensor calculation.

    Creates a series of deformed structures based on the crystal symmetry. The
    deformations are limited to non-equivalent axes of the crystal as
    determined by its Bravais lattice type.

    Args:
        atoms: The reference ASE Atoms object.
        n_deform: Number of deformations per non-equivalent axis.
        max_strain_normal: Maximum normal strain magnitude.
        max_strain_shear: Maximum shear strain magnitude.
        bravais_type: BravaisType enum specifying the crystal system. If None,
                      it's determined automatically.

    Returns:
        A list of deformed ASE Atoms objects.
    """
    deformation_rules: dict[BravaisType, DeformationRule] = {
        BravaisType.CUBIC: DeformationRule([0, 3], regular_symmetry),
        BravaisType.HEXAGONAL: DeformationRule([0, 2, 3, 5], hexagonal_symmetry),
        BravaisType.TRIGONAL: DeformationRule([0, 1, 2, 3, 4, 5], trigonal_symmetry),
        BravaisType.TETRAGONAL: DeformationRule([0, 2, 3, 5], tetragonal_symmetry),
        BravaisType.ORTHORHOMBIC: DeformationRule(
            [0, 1, 2, 3, 4, 5], orthorhombic_symmetry
        ),
        BravaisType.MONOCLINIC: DeformationRule([0, 1, 2, 3, 4, 5], monoclinic_symmetry),
        BravaisType.TRICLINIC: DeformationRule([0, 1, 2, 3, 4, 5], triclinic_symmetry),
    }

    if bravais_type is None:
        bravais_type = get_bravais_type(atoms)

    rule = deformation_rules[bravais_type]
    allowed_axes = rule.axes

    deformed_atoms_list = []

    for axis in allowed_axes:
        max_strain = max_strain_normal if axis < 3 else max_strain_shear
        strains = np.linspace(-max_strain, max_strain, n_deform)

        # Skip zero strain
        strains = strains[strains != 0]

        for strain in strains:
            deformed = get_cart_deformed_cell(atoms=atoms, axis=axis, size=strain)
            deformed_atoms_list.append(deformed)

    return deformed_atoms_list

def get_strain(
    deformed_atoms: Atoms, reference_atoms: Atoms | None = None
) -> np.ndarray:
    """Calculate infinitesimal strain tensor in Voigt notation from ASE Atoms.

    Computes the infinitesimal strain tensor, which is appropriate for the
    linear elasticity theory used in this module.

    Args:
        deformed_atoms: Atoms object containing the deformed configuration.
        reference_atoms: Optional reference (undeformed) Atoms object. If None,
                         deformed_atoms is used as its own reference (zero strain).

    Returns:
        np.ndarray: 6-component strain vector [εxx, εyy, εzz, 2εyz, 2εxz, 2εxy]
                      in Voigt notation.
    """
    if not isinstance(deformed_atoms, Atoms):
        raise TypeError("deformed_atoms must be an Atoms object")

    if reference_atoms is None:
        reference_atoms = deformed_atoms

    deformed_cell = np.array(deformed_atoms.get_cell(), dtype=np.float64)
    reference_cell = np.array(reference_atoms.get_cell(), dtype=np.float64)

    F = deformed_cell @ np.linalg.inv(reference_cell)
    u = F - np.identity(3, dtype=np.float64)
    strain = 0.5 * (u + u.T)

    return np.array([
        strain[0, 0],
        strain[1, 1],
        strain[2, 2],
        strain[1, 2] + strain[2, 1],
        strain[0, 2] + strain[2, 0],
        strain[0, 1] + strain[1, 0],
    ], dtype=np.float64)

def voigt_6_to_full_3x3_stress(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert a 6-component stress vector in Voigt notation to a 3x3 matrix.

    Args:
        stress_voigt: Array of shape (..., 6) containing stress components
                     [σxx, σyy, σzz, σyz, σxz, σxy] in Voigt notation

    Returns:
        np.ndarray: Array of shape (..., 3, 3) containing the full stress matrix
    """
    # Initialize 3x3 stress tensor
    stress = np.zeros((*stress_voigt.shape[:-1], 3, 3), dtype=stress_voigt.dtype)

    # Fill diagonal elements
    stress[..., 0, 0] = stress_voigt[..., 0]  # σxx
    stress[..., 1, 1] = stress_voigt[..., 1]  # σyy
    stress[..., 2, 2] = stress_voigt[..., 2]  # σzz

    # Fill off-diagonal elements (symmetric)
    stress[..., 2, 1] = stress[..., 1, 2] = stress_voigt[..., 3]  # σyz
    stress[..., 2, 0] = stress[..., 0, 2] = stress_voigt[..., 4]  # σxz
    stress[..., 1, 0] = stress[..., 0, 1] = stress_voigt[..., 5]  # σxy

    return stress


def full_3x3_to_voigt_6_stress(stress: np.ndarray) -> np.ndarray:
    """Form a 6 component stress vector in Voigt notation from a 3x3 matrix.

    Args:
        stress: Array of shape (..., 3, 3) containing stress components

    Returns:
        np.ndarray: 6-component stress vector [σxx, σyy, σzz, σyz, σxz, σxy]
                     following Voigt notation
    """
    # Ensure the tensor is symmetric
    stress = (stress + stress.T) / 2

    # Create the Voigt vector while preserving batch dimensions
    return np.stack(
        [
            stress[..., 0, 0],  # σxx
            stress[..., 1, 1],  # σyy
            stress[..., 2, 2],  # σzz
            stress[..., 2, 1],  # σyz
            stress[..., 2, 0],  # σxz
            stress[..., 1, 0],  # σxy
        ],
        axis=-1,
    )

def get_elastic_coeffs(
    atoms: Atoms,
    deformed_atoms: list[Atoms],
    stresses: np.ndarray,
    base_pressure: float,
    bravais_type: BravaisType,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    """Calculate elastic tensor from stress-strain relationships.

    Computes the elastic tensor by fitting stress-strain relations to a set of
    linear equations built from crystal symmetry and deformation data.

    Args:
        atoms: Atoms containing reference structure
        deformed_atoms: List of deformed Atoms with calculated stresses
        stresses: Array of shape (n_states, 6) containing stress components for each
                 state
        base_pressure: Reference pressure of the base state
        bravais_type: Crystal system (BravaisType enum)

    Returns:
        tuple containing:
        - np.ndarray: Cij elastic constants
        - tuple containing:
            - np.ndarray: Bij Birch coefficients
            - np.ndarray: Residuals from least squares fit
            - int: Rank of solution
            - np.ndarray: Singular values

    Notes:
        The elastic tensor is calculated as Cij = Bij - P, where:
        - Bij are the Birch coefficients from least squares fitting
        - P is a pressure-dependent correction specific to each symmetry

        Stress and strain are related by: σᵢ = Σⱼ Cᵢⱼ εⱼ
    """
    # Deformation rules for different Bravais lattices
    deformation_rules: dict[BravaisType, DeformationRule] = {
        BravaisType.CUBIC: DeformationRule([0, 3], regular_symmetry),
        BravaisType.HEXAGONAL: DeformationRule([0, 2, 3, 5], hexagonal_symmetry),
        BravaisType.TRIGONAL: DeformationRule([0, 1, 2, 3, 4, 5], trigonal_symmetry),
        BravaisType.TETRAGONAL: DeformationRule([0, 2, 3, 5], tetragonal_symmetry),
        BravaisType.ORTHORHOMBIC: DeformationRule(
            [0, 1, 2, 3, 4, 5], orthorhombic_symmetry
        ),
        BravaisType.MONOCLINIC: DeformationRule([0, 1, 2, 3, 4, 5], monoclinic_symmetry),
        BravaisType.TRICLINIC: DeformationRule([0, 1, 2, 3, 4, 5], triclinic_symmetry),
    }

    # Get symmetry handler for this Bravais lattice
    rule = deformation_rules[bravais_type]
    symmetry_handler = rule.symmetry_handler

    # Calculate strains for all deformed atoms
    strains = []
    for deformed in deformed_atoms:
        strain = get_strain(deformed, reference_atoms=atoms)
        strains.append(strain)

    # Remove ambient pressure from stresses
    p_correction = np.array(
        [base_pressure] * 3 + [0] * 3, dtype=stresses.dtype
    )
    corrected_stresses = stresses - p_correction

    # The symmetry functions expect engineering strains [exx, eyy, ezz, eyz, exz, exy]
    # where the shear components are half of the Voigt definition used in get_strain.
    voigt_strains_for_symm = np.stack(strains).copy()
    voigt_strains_for_symm[:, 3:] *= 0.5

    # Build equation matrix using symmetry
    eq_matrices = [symmetry_handler(s) for s in voigt_strains_for_symm]
    eq_matrix = np.stack(eq_matrices)

    # Reshape for least squares solving
    eq_matrix = eq_matrix.reshape(-1, eq_matrix.shape[-1])
    stress_vector = corrected_stresses.reshape(-1)

    # Solve least squares problem
    Bij, residuals, rank, singular_values = np.linalg.lstsq(eq_matrix, stress_vector, rcond=None)

    # Calculate elastic constants with pressure correction
    p = base_pressure
    pressure_corrections = {
        BravaisType.CUBIC: np.array([-p, p, -p]),
        BravaisType.HEXAGONAL: np.array([-p, -p, p, p, -p]),
        BravaisType.TRIGONAL: np.array([-p, -p, p, p, p, p, -p]),
        BravaisType.TETRAGONAL: np.array([-p, -p, p, p, -p, -p, -p]),
        BravaisType.ORTHORHOMBIC: np.array([-p, -p, -p, p, p, p, -p, -p, -p]),
        BravaisType.MONOCLINIC: np.array(
            [-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p]
        ),
        BravaisType.TRICLINIC: np.array(
            [
                -p,
                p,
                p,
                p,
                p,
                p,  # C11-C16
                -p,
                p,
                p,
                p,
                p,  # C22-C26
                -p,
                p,
                p,
                p,  # C33-C36
                -p,
                p,
                p,  # C44-C46
                -p,
                p,  # C55-C56
                -p,  # C66
            ]
        ),
    }

    # Apply pressure correction for the specific symmetry
    Cij = Bij - pressure_corrections[bravais_type]

    return Cij, (Bij, residuals, rank, singular_values)

def get_elastic_tensor_from_coeffs(  # noqa: C901, PLR0915
    Cij: np.ndarray,
    bravais_type: BravaisType,
) -> np.ndarray:
    """Convert the symmetry-reduced elastic constants to full 6x6 elastic tensor.

    Args:
        Cij: Array containing independent elastic constants for the given symmetry
        bravais_type: Crystal system determining the symmetry rules

    Returns:
        np.ndarray: Full 6x6 elastic tensor with all components

    Notes:
        The mapping follows Voigt notation where:
        1 = xx, 2 = yy, 3 = zz, 4 = yz, 5 = xz, 6 = xy

        The number of independent constants varies by symmetry:
        - Cubic: 3 (C11, C12, C44)
        - Hexagonal: 5 (C11, C12, C13, C33, C44)
        - Trigonal: 6 (C11, C12, C13, C14, C33, C44)
        - Tetragonal: 7 (C11, C12, C13, C16, C33, C44, C66)
        - Orthorhombic: 9 (C11, C22, C33, C12, C13, C23, C44, C55, C66)
        - Monoclinic: 13 constants (C11, C22, C33, C12, C13, C23, C44, C55,
            C66, C15, C25, C35, C46)
        - Triclinic: 21 constants
    """
    # Initialize full tensor
    C = np.zeros((6, 6), dtype=Cij.dtype)

    if bravais_type == BravaisType.TRICLINIC:
        if len(Cij) != 21:
            raise ValueError(
                f"Triclinic symmetry requires 21 independent constants, "
                f"but got {len(Cij)}"
            )
        C = np.zeros((6, 6), dtype=Cij.dtype)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                C[i, j] = C[j, i] = Cij[idx]
                idx += 1

    elif bravais_type == BravaisType.CUBIC:
        C11, C12, C44 = Cij
        diag = np.array([C11, C11, C11, C44, C44, C44])
        np.fill_diagonal(C, diag)
        C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C12

    elif bravais_type == BravaisType.HEXAGONAL:
        C11, C12, C13, C33, C44 = Cij
        np.fill_diagonal(C, np.array([C11, C11, C33, C44, C44, (C11 - C12) / 2]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13

    elif bravais_type == BravaisType.TRIGONAL:
        C11, C12, C13, C14, C15, C33, C44 = Cij
        np.fill_diagonal(C, np.array([C11, C11, C33, C44, C44, (C11 - C12) / 2]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
        C[0, 3] = C[3, 0] = C14
        C[0, 4] = C[4, 0] = C15
        C[1, 3] = C[3, 1] = -C14
        C[1, 4] = C[4, 1] = -C15
        C[3, 5] = C[5, 3] = -C15
        C[4, 5] = C[5, 4] = C14

    elif bravais_type == BravaisType.TETRAGONAL:
        C11, C12, C13, C16, C33, C44, C66 = Cij
        np.fill_diagonal(C, np.array([C11, C11, C33, C44, C44, C66]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
        C[0, 5] = C[5, 0] = C16
        C[1, 5] = C[5, 1] = -C16

    elif bravais_type == BravaisType.ORTHORHOMBIC:
        C11, C12, C13, C22, C23, C33, C44, C55, C66 = Cij
        np.fill_diagonal(C, np.array([C11, C22, C33, C44, C55, C66]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C13
        C[1, 2] = C[2, 1] = C23

    elif bravais_type == BravaisType.MONOCLINIC:
        C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66 = Cij
        np.fill_diagonal(C, np.array([C11, C22, C33, C44, C55, C66]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C13
        C[0, 4] = C[4, 0] = C15
        C[1, 2] = C[2, 1] = C23
        C[1, 4] = C[4, 1] = C25
        C[2, 4] = C[4, 2] = C35
        C[3, 5] = C[5, 3] = C46

    return C

def calculate_elastic_tensor(
    calculator: "ase.calculators.calculator.Calculator",
    *,
    atoms: Atoms,
    bravais_type: BravaisType = BravaisType.TRICLINIC,
    max_strain_normal: float = 0.01,
    max_strain_shear: float = 0.06,
    n_deform: int = 5,
) -> np.ndarray:
    """Calculate the elastic tensor of a structure.

    Args:
        calculator: ASE calculator to use for stress calculation
        atoms: Atoms containing the reference structure
        bravais_type: Bravais type of the structure
        max_strain_normal: Maximum normal strain
        max_strain_shear: Maximum shear strain
        n_deform: Number of deformations

    Returns:
        np.ndarray: Elastic tensor
    """
    # Calculate deformations for the bravais type
    deformations = get_elementary_deformations(
        atoms,
        n_deform=n_deform,
        max_strain_normal=max_strain_normal,
        max_strain_shear=max_strain_shear,
        bravais_type=bravais_type,
    )

    # Initial stress calculation to get reference pressure
    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator
    stress_tensor = atoms_copy.get_stress(voigt=False)
    ref_pressure = -np.trace(stress_tensor) / 3

    # Calculate stresses for deformations
    stresses = np.zeros((len(deformations), 6), dtype=np.float64)

    for i, deformation in enumerate(deformations):
        deformation.calc = calculator
        stress = deformation.get_stress(voigt=False)
        stresses[i] = full_3x3_to_voigt_6_stress(stress)

    # Calculate elastic tensor
    C_ij, _ = get_elastic_coeffs(
        atoms, deformations, stresses, ref_pressure, bravais_type
    )
    C = get_elastic_tensor_from_coeffs(C_ij, bravais_type)

    return C

def calculate_elastic_moduli(C: np.ndarray) -> tuple[float, float, float, float]:
    """Calculate elastic moduli from the elastic tensor.

    Args:
        C: Elastic tensor (6x6) as a NumPy array.

    Returns:
        tuple: Four Voigt-Reuss-Hill averaged elastic moduli in order:
            - Bulk modulus (K_VRH)
            - Shear modulus (G_VRH)
            - Poisson's ratio (v_VRH), dimensionless
            - Pugh's ratio (K_VRH/G_VRH), dimensionless
    """
    # Components of the elastic tensor
    C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
    C12, C23, C31 = C[0, 1], C[1, 2], C[2, 0]
    C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]

    # Calculate compliance tensor
    S = np.linalg.inv(C)
    S11, S22, S33 = S[0, 0], S[1, 1], S[2, 2]
    S12, S23, S31 = S[0, 1], S[1, 2], S[2, 0]
    S44, S55, S66 = S[3, 3], S[4, 4], S[5, 5]

    # Voigt averaging (upper bound)
    K_V = (1 / 9) * ((C11 + C22 + C33) + 2 * (C12 + C23 + C31))
    G_V = (1 / 15) * ((C11 + C22 + C33) - (C12 + C23 + C31) + 3 * (C44 + C55 + C66))

    # Reuss averaging (lower bound)
    K_R = 1 / ((S11 + S22 + S33) + 2 * (S12 + S23 + S31))
    G_R = 15 / (4 * (S11 + S22 + S33) - 4 * (S12 + S23 + S31) + 3 * (S44 + S55 + S66))

    # Voigt-Reuss-Hill averaging
    K_VRH = (K_V + K_R) / 2
    G_VRH = (G_V + G_R) / 2

    # Poisson's ratio (VRH)
    v_VRH = (3 * K_VRH - 2 * G_VRH) / (6 * K_VRH + 2 * G_VRH)

    # Pugh's ratio (VRH)
    pugh_ratio_VRH = K_VRH / G_VRH

    return K_VRH, G_VRH, v_VRH, pugh_ratio_VRH

def relax_atoms(
    atoms: Atoms,
    fmax_threshold: float = 1e-3,
    steps: int = 250,
    optimizer_class: "ase.optimize.Optimizer" = None,
    cell_filter_class: "ase.filters.Filter" = None,
) -> Atoms:
    """
    Relaxes the atomic positions and cell of an ASE Atoms object.

    Args:
        atoms: The ASE Atoms object to relax. Must have a calculator attached.
        fmax_threshold: The maximum force convergence threshold in eV/Å.
        steps: The maximum number of optimization steps.
        optimizer_class: The ASE optimizer class to use (e.g., FIRE).
        cell_filter_class: The ASE cell filter to use for cell relaxation.

    Returns:
        The relaxed ASE Atoms object.
    """
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter

    if atoms.calc is None:
        raise ValueError("An ASE calculator must be attached to the atoms object.")

    optimizer_class = optimizer_class or FIRE
    cell_filter_class = cell_filter_class or FrechetCellFilter

    # Conversion factor from eV/Å^3 to GPa
    EV_A3_TO_GPA = 160.21766208

    cell_filter = cell_filter_class(atoms)
    optimizer = optimizer_class(cell_filter)

    print("  Step       Energy      Pressure(GPa)     fmax(eV/A)")
    for i in range(steps):
        optimizer.step()
        # Access atoms through the filter to get updated properties
        atoms_after_step = cell_filter.atoms
        fmax = np.sqrt((atoms_after_step.get_forces() ** 2).sum(axis=1).max())
        stress = atoms_after_step.get_stress(voigt=False)
        pressure_gpa = -np.trace(stress) / 3.0 * EV_A3_TO_GPA
        energy = atoms_after_step.get_potential_energy()

        print(f"{i:5d}  {energy:12.4f}  {pressure_gpa:14.4f}  {fmax:12.4f}")

        # Check convergence criteria
        if fmax < fmax_threshold and abs(pressure_gpa) < 1e-2:
            break
    else:
        print(f"Warning: Relaxation did not converge within {steps} steps.")

    print("Relaxation finished.")
    return cell_filter.atoms


def calculate_elastic_tensor_from_vasp(
    directory_path: str,
    strain_amount: float,
    bravais_type: BravaisType,
) -> np.ndarray:
    """Calculate the elastic tensor from a directory of VASP OUTCAR files.

    This function assumes a directory structure created for a finite-difference
    calculation of elastic constants, where strains are applied in 12 directions
    (positive and negative for 6 strain components).

    The expected directory structure is:
    your_calculations/
    ├── OUTCAR                    # Reference (unstrained)
    ├── ep1_plus/OUTCAR          # +εxx strain
    ├── ep1_minus/OUTCAR         # -εxx strain
    ...
    ├── ep6_plus/OUTCAR          # +γxy/2 shear
    └── ep6_minus/OUTCAR         # -γxy/2 shear

    Args:
        directory_path: Path to the base directory containing the OUTCAR files.
        strain_amount: The magnitude of the applied strain (e.g., 0.005 for 0.5%).
        bravais_type: The Bravais type of the crystal structure.

    Returns:
        np.ndarray: The calculated 6x6 elastic tensor in GPa.
    """
    # Read reference (unstrained) structure and pressure
    ref_outcar_path = os.path.join(directory_path, 'reference', 'OUTCAR')
    if not os.path.exists(ref_outcar_path):
        raise FileNotFoundError(f"Reference OUTCAR not found at: {ref_outcar_path}")
    
    reference_atoms = read(ref_outcar_path, format='vasp-out')
    stress_tensor = reference_atoms.get_stress(voigt=False)
    ref_pressure = -np.trace(stress_tensor) / 3

    deformed_configs = []
    stresses_voigt = []

    # Map ep{i} to axis index for get_cart_deformed_cell
    strain_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    print("--- Reading VASP Elastic Constant Calculation ---")
    print(f"Directory: {directory_path}")
    print(f"Strain amount: {strain_amount}")

    for i in range(1, 7):
        for sign in ['plus', 'minus']:
            strain_val = strain_amount if sign == 'plus' else -strain_amount
            axis = strain_map[i]
            
            # Construct the path to the OUTCAR for the deformed structure
            dir_name = f"ep{i}_{sign}"
            outcar_path = os.path.join(directory_path, dir_name, 'OUTCAR')
            
            if not os.path.exists(outcar_path):
                raise FileNotFoundError(f"Deformed OUTCAR not found at: {outcar_path}")
            
            print(f"  Reading stress from: {outcar_path}")

            # 1. Create the deformed Atoms object programmatically
            deformed = get_cart_deformed_cell(
                atoms=reference_atoms, axis=axis, size=strain_val
            )
            deformed_configs.append(deformed)

            # 2. Read the stress from the corresponding VASP OUTCAR file
            vasp_atoms = read(outcar_path, format='vasp-out')
            stress_vasp = vasp_atoms.get_stress(voigt=False)
            stresses_voigt.append(full_3x3_to_voigt_6_stress(stress_vasp))

    stresses_np = np.array(stresses_voigt, dtype=np.float64)

    # Calculate elastic tensor
    C_ij, _ = get_elastic_coeffs(
        reference_atoms, deformed_configs, stresses_np, ref_pressure, bravais_type
    )
    C_tensor = get_elastic_tensor_from_coeffs(C_ij, bravais_type)
    
    # Convert from eV/Å^3 to GPa before returning
    return C_tensor * EV_A3_TO_GPA


# --- Test Cases ---
if __name__ == "__main__":
    from ase.build import bulk
    try:
        from mace.calculators import mace_mp
    except ImportError:
        print("MACE not found, skipping MACE tests")
        exit()
    
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter

    print("--- Running Basic Validation Tests ---")

    # 1. Cubic
    cubic_atoms = Atoms(cell=cellpar_to_cell([4, 4, 4, 90, 90, 90]), pbc=True)
    assert get_bravais_type(cubic_atoms) == BravaisType.CUBIC
    print(f"Cubic:       {get_bravais_type(cubic_atoms).value:<12} ... PASSED")

    # 2. Hexagonal
    hex_atoms = Atoms(cell=cellpar_to_cell([3, 3, 5, 90, 90, 120]), pbc=True)
    assert get_bravais_type(hex_atoms) == BravaisType.HEXAGONAL
    print(f"Hexagonal:   {get_bravais_type(hex_atoms).value:<12} ... PASSED")

    # 3. Tetragonal
    tet_atoms = Atoms(cell=cellpar_to_cell([3, 3, 5, 90, 90, 90]), pbc=True)
    assert get_bravais_type(tet_atoms) == BravaisType.TETRAGONAL
    print(f"Tetragonal:  {get_bravais_type(tet_atoms).value:<12} ... PASSED")

    # 4. Orthorhombic
    ortho_atoms = Atoms(cell=cellpar_to_cell([3, 4, 5, 90, 90, 90]), pbc=True)
    assert get_bravais_type(ortho_atoms) == BravaisType.ORTHORHOMBIC
    print(f"Orthorhombic:{get_bravais_type(ortho_atoms).value:<12} ... PASSED")

    # 5. Monoclinic (unique axis b, so beta is not 90)
    mono_atoms = Atoms(cell=cellpar_to_cell([3, 4, 5, 90, 110, 90]), pbc=True)
    assert get_bravais_type(mono_atoms) == BravaisType.MONOCLINIC
    print(f"Monoclinic:  {get_bravais_type(mono_atoms).value:<12} ... PASSED")
    
    # 6. Trigonal (Rhombohedral setting)
    trig_atoms = Atoms(cell=cellpar_to_cell([4, 4, 4, 80, 80, 80]), pbc=True)
    assert get_bravais_type(trig_atoms) == BravaisType.TRIGONAL
    print(f"Trigonal:    {get_bravais_type(trig_atoms).value:<12} ... PASSED")

    # 7. Triclinic
    triclinic_atoms = Atoms(cell=cellpar_to_cell([3, 4, 5, 70, 80, 95]), pbc=True)
    assert get_bravais_type(triclinic_atoms) == BravaisType.TRICLINIC
    print(f"Triclinic:   {get_bravais_type(triclinic_atoms).value:<12} ... PASSED")

    # 8. Deformed cell tests
    print("\nRunning test cases for get_cart_deformed_cell...")
    cubic_atoms_test = Atoms(cell=cellpar_to_cell([4, 4, 4, 90, 90, 90]), pbc=True)
    
    # Test normal strain
    deformed_normal = get_cart_deformed_cell(cubic_atoms_test, axis=0, size=0.1)
    expected_cell_normal = np.array([[4.4, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    np.testing.assert_allclose(np.array(deformed_normal.get_cell()), expected_cell_normal, atol=1e-6)
    print("Deformation (normal strain xx): ... PASSED")
    
    # Test shear strain
    deformed_shear = get_cart_deformed_cell(cubic_atoms_test, axis=5, size=0.1)
    # new_y = y + 0.1*x. Cell vector a = (4,0,0) -> a'=(4, 0.4, 0).
    expected_cell_shear = np.array([[4.0, 0.4, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    np.testing.assert_allclose(np.array(deformed_shear.get_cell()), expected_cell_shear, atol=1e-6)
    print("Deformation (shear strain xy):  ... PASSED")

    # 9. Elementary deformations test
    print("\nRunning test cases for get_elementary_deformations...")
    cubic_deformations = get_elementary_deformations(cubic_atoms_test, n_deform=3)
    # For cubic, axes are [0, 3]. n_deform=3 gives strains of [-s, s] for each.
    # So we expect 2 (axes) * 2 (strains) = 4 deformations.
    assert len(cubic_deformations) == 4
    print("Elementary deformations (cubic): ... PASSED")

    # 10. Strain calculation test
    print("\nRunning test cases for get_strain...")
    reference = Atoms(cell=np.identity(3))
    # Test normal strain
    deformed_normal = reference.copy()
    deformed_normal.set_cell([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]], scale_atoms=True)
    strain_normal = get_strain(deformed_normal, reference)
    expected_normal = np.array([0.01, 0, 0, 0, 0, 0], dtype=np.float64)
    assert np.allclose(strain_normal, expected_normal, atol=1e-5)
    print("Strain calculation (normal):    ... PASSED")
    # Test shear strain
    deformed_shear = reference.copy()
    deformed_shear.set_cell([[1, 0.01, 0], [0.01, 1, 0], [0, 0, 1]], scale_atoms=True)
    strain_shear = get_strain(deformed_shear, reference)
    expected_shear = np.array([0, 0, 0, 0, 0, 0.02], dtype=np.float64)
    assert np.allclose(strain_shear, expected_shear, atol=1e-5)
    print("Strain calculation (shear):     ... PASSED")


    # 11. Voigt stress conversion tests
    print("\nRunning test cases for stress conversions...")
    voigt_stress = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    matrix_stress = voigt_6_to_full_3x3_stress(voigt_stress)
    expected_matrix = np.array([[1, 6, 5], [6, 2, 4], [5, 4, 3]], dtype=np.float64)
    assert np.allclose(matrix_stress, expected_matrix)
    
    back_to_voigt = full_3x3_to_voigt_6_stress(matrix_stress)
    assert np.allclose(voigt_stress, back_to_voigt)
    print("Voigt/matrix stress conversion: ... PASSED")

    # 12. Elastic coefficients test (dummy test)
    print("\nRunning test cases for get_elastic_coeffs...")
    # This is a placeholder test. A real test would require a calculator.
    # We create a fake stress tensor that linearly corresponds to a known Cij.
    C_known = np.array([160.0, 120.0, 80.0], dtype=np.float64) # C11, C12, C44 for cubic
    
    # Create deformations for a cubic system
    cubic_ref = Atoms('Al', cell=np.identity(3)*4.05, pbc=True)
    cubic_deformations_coeffs = get_elementary_deformations(cubic_ref, n_deform=3, bravais_type=BravaisType.CUBIC)
    
    strains_coeffs = np.stack([get_strain(d, cubic_ref) for d in cubic_deformations_coeffs])
    
    # The symmetry handler expects Voigt strains [exx, eyy, ezz, eyz, exz, exy]
    # Our get_strain returns [exx, eyy, ezz, g_yz, g_xz, g_xy]
    # We need to scale the shear strains before passing to symmetry handler
    voigt_strains_for_symm = strains_coeffs.copy()
    voigt_strains_for_symm[:, 3:] *= 0.5

    A_matrices = [regular_symmetry(s) for s in voigt_strains_for_symm]
    A_matrix_flat = np.concatenate(A_matrices, axis=0)

    fake_stresses_flat = A_matrix_flat @ C_known
    fake_stresses = fake_stresses_flat.reshape(-1, 6)
    
    C_calc, _ = get_elastic_coeffs(cubic_ref, cubic_deformations_coeffs, fake_stresses, 0.0, BravaisType.CUBIC)
    
    assert np.allclose(C_known, C_calc, atol=1e-5)
    print("Elastic coefficient calculation: ... PASSED")

    # 13. Elastic tensor conversion test
    print("\nRunning test cases for get_elastic_tensor_from_coeffs...")
    Cij_cubic = np.array([160.0, 120.0, 80.0])
    C_tensor = get_elastic_tensor_from_coeffs(Cij_cubic, BravaisType.CUBIC)
    expected_C = np.array([
        [160, 120, 120, 0, 0, 0],
        [120, 160, 120, 0, 0, 0],
        [120, 120, 160, 0, 0, 0],
        [0, 0, 0, 80, 0, 0],
        [0, 0, 0, 0, 80, 0],
        [0, 0, 0, 0, 0, 80],
    ], dtype=np.float64)
    assert np.allclose(C_tensor, expected_C)
    print("Cij to C_tensor conversion (cubic): ... PASSED")

    # 14. Elastic moduli test
    print("\nRunning test cases for calculate_elastic_moduli...")
    # Using experimental values for Aluminum (in GPa)
    C_al = np.array([
        [107.3, 60.9, 60.9, 0, 0, 0],
        [60.9, 107.3, 60.9, 0, 0, 0],
        [60.9, 60.9, 107.3, 0, 0, 0],
        [0, 0, 0, 28.3, 0, 0],
        [0, 0, 0, 0, 28.3, 0],
        [0, 0, 0, 0, 0, 28.3],
    ])
    K, G, v, pugh = calculate_elastic_moduli(C_al)
    
    # Expected values for Al (VRH averages)
    # K_V = K_R = 76.37 GPa
    # G_V = 26.2 GPa, G_R = 26.2 GPa
    assert np.isclose(K, 76.37, atol=0.1)
    assert np.isclose(G, 26.2, atol=0.1)
    assert np.isclose(v, 0.35, atol=0.01)
    assert np.isclose(pugh, 2.9, atol=0.1)
    print("Elastic moduli calculation (Al): ... PASSED")

    print("\nAll basic test cases passed!")

    print("\n--- Running End-to-End MACE Calculation Test for Copper ---")

    # 1. Set up a test system (Copper)
    cu_atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))

    # 2. Set up the MACE calculator
    print("Initializing MACE calculator...")
    # Use the specific model from torch_sim to ensure consistency
    calculator = mace_foundation_mp(
        model=MaceUrls.mace_mpa_medium, device="cuda", default_dtype="float64"
    )

    # Assign calculator for relaxation
    cu_atoms.calc = calculator
    
    # Conversion factor from eV/Å^3 to GPa is now at module level

    # 3. Relax the structure (positions and cell)
    print("Relaxing Copper structure (positions and cell)...")
    relaxed_atoms = relax_atoms(cu_atoms)

    # The cu_atoms object is updated in-place by the optimizer

    # 4. Run the elastic tensor calculation on the relaxed structure
    print("Calculating elastic tensor for relaxed Copper with MACE...")
    C_tensor = calculate_elastic_tensor(
        calculator=calculator, atoms=relaxed_atoms, bravais_type=BravaisType.CUBIC
    )
    
    # Convert from eV/Å^3 to GPa for comparison
    C_tensor *= EV_A3_TO_GPA
    C = C_tensor  # For printing and moduli calculation

    print("\nCalculated Elastic Tensor (C_ij) in GPa:")
    for i in range(6):
        print("  ".join(f"{C[i, j]:8.2f}" for j in range(6)))

    # 5. Calculate and print derived moduli
    K, G, v, pugh = calculate_elastic_moduli(C)
    print(f"\nBulk Modulus (K_VRH):      {K:.2f} GPa")
    print(f"Shear Modulus (G_VRH):     {G:.2f} GPa")
    print(f"Poisson's Ratio (v_VRH):   {v:.2f}")
    print(f"Pugh's Ratio (K_VRH/G_VRH): {pugh:.2f}")

    # 6. Compare with expected values for MACE model on Copper
    # These are approximate literature/model values for validation.
    expected_K = 144.0  # GPa
    expected_G = 43.0   # GPa
    
    print(f"\n--- Validation Against Expected MACE Results for Copper ---")
    print(f"Expected Bulk Modulus: ~{expected_K} GPa (Calculated: {K:.2f} GPa)")
    print(f"Expected Shear Modulus: ~{expected_G} GPa (Calculated: {G:.2f} GPa)")

    assert np.isclose(K, expected_K, rtol=0.15), f"Bulk modulus {K:.2f} differs from {expected_K}"
    assert np.isclose(G, expected_G, rtol=0.15), f"Shear modulus {G:.2f} differs from {expected_G}"

    print("\nEnd-to-end MACE calculation test PASSED!") 