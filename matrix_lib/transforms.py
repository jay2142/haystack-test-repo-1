"""Transformation matrices for graphics and geometry."""

import math
from .matrix import Matrix


def rotation_matrix_2d(angle_degrees: float) -> Matrix:
    """
    Create a 2D rotation matrix.

    Args:
        angle_degrees: Rotation angle in degrees (counterclockwise)

    Returns:
        2x2 rotation matrix
    """
    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    return Matrix([
        [c, -s],
        [s, c]
    ])


def scaling_matrix(sx: float, sy: float, sz: float = 1.0) -> Matrix:
    """
    Create a scaling matrix.

    Args:
        sx: X-axis scale factor
        sy: Y-axis scale factor
        sz: Z-axis scale factor (for 3D)

    Returns:
        Scaling matrix (2x2 or 3x3)
    """
    if sz == 1.0:
        return Matrix([
            [sx, 0],
            [0, sy]
        ])
    else:
        return Matrix([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, sz]
        ])


def translation_matrix(tx: float, ty: float, tz: float = 0.0) -> Matrix:
    """
    Create a translation matrix (homogeneous coordinates).

    Args:
        tx: X translation
        ty: Y translation
        tz: Z translation

    Returns:
        Translation matrix (3x3 or 4x4)
    """
    if tz == 0.0:
        return Matrix([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
    else:
        return Matrix([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])


def shear_matrix(shx: float = 0.0, shy: float = 0.0) -> Matrix:
    """
    Create a 2D shear matrix.

    Args:
        shx: Shear factor in X direction
        shy: Shear factor in Y direction

    Returns:
        2x2 shear matrix
    """
    return Matrix([
        [1, shx],
        [shy, 1]
    ])


def reflection_matrix(axis: str) -> Matrix:
    """
    Create a 2D reflection matrix.

    Args:
        axis: Axis to reflect about ('x', 'y', or 'origin')

    Returns:
        2x2 reflection matrix

    Raises:
        ValueError: If axis is invalid
    """
    if axis == 'x':
        return Matrix([[1, 0], [0, -1]])
    elif axis == 'y':
        return Matrix([[-1, 0], [0, 1]])
    elif axis == 'origin':
        return Matrix([[-1, 0], [0, -1]])
    else:
        raise ValueError(f"Invalid reflection axis: {axis}")


def rotation_matrix_3d(axis: str, angle_degrees: float) -> Matrix:
    """
    Create a 3D rotation matrix about a coordinate axis.

    Args:
        axis: Axis to rotate about ('x', 'y', or 'z')
        angle_degrees: Rotation angle in degrees

    Returns:
        3x3 rotation matrix

    Raises:
        ValueError: If axis is invalid
    """
    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    if axis == 'x':
        return Matrix([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return Matrix([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return Matrix([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid rotation axis: {axis}")
