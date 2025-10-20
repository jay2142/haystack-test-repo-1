"""Matrix transformation functions for 2D and 3D operations."""

import math
from .core import Matrix


def rotate_2d(angle_degrees: float) -> Matrix:
    """
    Create a 2D rotation matrix.

    Args:
        angle_degrees: Rotation angle in degrees (counterclockwise)

    Returns:
        2x2 rotation matrix
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    return Matrix([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])


def scale(sx: float, sy: float, sz: float = 1.0) -> Matrix:
    """
    Create a scaling matrix.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction
        sz: Scale factor in z direction (default: 1.0 for 2D)

    Returns:
        Scaling matrix (2x2 for 2D, 3x3 for 3D)
    """
    if sz == 1.0:
        # 2D scaling
        return Matrix([
            [sx, 0],
            [0, sy]
        ])
    else:
        # 3D scaling
        return Matrix([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, sz]
        ])


def translate(tx: float, ty: float, tz: float = 0.0) -> Matrix:
    """
    Create a translation matrix using homogeneous coordinates.

    Args:
        tx: Translation in x direction
        ty: Translation in y direction
        tz: Translation in z direction (default: 0.0 for 2D)

    Returns:
        Translation matrix (3x3 for 2D, 4x4 for 3D)
    """
    if tz == 0.0:
        # 2D translation (homogeneous)
        return Matrix([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
    else:
        # 3D translation (homogeneous)
        return Matrix([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])


def reflect_x() -> Matrix:
    """
    Create a reflection matrix about the x-axis.

    Returns:
        2x2 reflection matrix
    """
    return Matrix([
        [1, 0],
        [0, -1]
    ])


def reflect_y() -> Matrix:
    """
    Create a reflection matrix about the y-axis.

    Returns:
        2x2 reflection matrix
    """
    return Matrix([
        [-1, 0],
        [0, 1]
    ])


def shear(shx: float = 0.0, shy: float = 0.0) -> Matrix:
    """
    Create a shear matrix.

    Args:
        shx: Shear factor in x direction
        shy: Shear factor in y direction

    Returns:
        2x2 shear matrix
    """
    return Matrix([
        [1, shx],
        [shy, 1]
    ])


def rotate_3d_x(angle_degrees: float) -> Matrix:
    """
    Create a 3D rotation matrix about the x-axis.

    Args:
        angle_degrees: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    return Matrix([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])


def rotate_3d_y(angle_degrees: float) -> Matrix:
    """
    Create a 3D rotation matrix about the y-axis.

    Args:
        angle_degrees: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    return Matrix([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])


def rotate_3d_z(angle_degrees: float) -> Matrix:
    """
    Create a 3D rotation matrix about the z-axis.

    Args:
        angle_degrees: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    return Matrix([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])


def compose_transforms(*transforms: Matrix) -> Matrix:
    """
    Compose multiple transformation matrices.

    Args:
        *transforms: Variable number of transformation matrices

    Returns:
        Composed transformation matrix

    Raises:
        ValueError: If no transforms provided or incompatible dimensions
    """
    if not transforms:
        raise ValueError("At least one transform required")

    result = transforms[0].copy()
    for transform in transforms[1:]:
        result = result * transform

    return result
