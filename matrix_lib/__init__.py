"""
Matrix Library - A Python library for matrix operations and transformations.

This library provides comprehensive matrix operations including:
- Basic arithmetic operations
- Linear transformations
- Matrix decomposition
- Eigenvalue computation
"""

from .matrix import Matrix
from .operations import (
    add_matrices,
    subtract_matrices,
    multiply_matrices,
    dot_product
)
from .transforms import (
    rotation_matrix_2d,
    scaling_matrix,
    translation_matrix,
    shear_matrix
)
from .algorithms import (
    gaussian_elimination,
    lu_decomposition,
    qr_decomposition
)

__version__ = '2.0.0'
__author__ = 'Matrix Team'

__all__ = [
    'Matrix',
    'add_matrices',
    'subtract_matrices',
    'multiply_matrices',
    'dot_product',
    'rotation_matrix_2d',
    'scaling_matrix',
    'translation_matrix',
    'shear_matrix',
    'gaussian_elimination',
    'lu_decomposition',
    'qr_decomposition',
]
