"""
Matrix Library - A Python library for matrix operations and transformations.

This library provides comprehensive matrix operations including:
- Basic arithmetic operations
- Linear transformations
- Matrix decomposition
- Eigenvalue computation
- Advanced matrix utilities
"""

from .matrix import Matrix
from .operations import (
    add_matrices,
    subtract_matrices,
    multiply_matrices,
    dot_product,
    trace,
    determinant
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
    qr_decomposition,
    solve_linear_system
)
from .eigenvalues import (
    power_iteration,
    eigenvalues_2x2,
    rayleigh_quotient,
    qr_algorithm
)
from .advanced_ops import (
    condition_number,
    matrix_power,
    is_symmetric,
    is_orthogonal,
    hadamard_product,
    kronecker_product
)

__version__ = '2.1.0'
__author__ = 'Matrix Team'

__all__ = [
    'Matrix',
    'add_matrices',
    'subtract_matrices',
    'multiply_matrices',
    'dot_product',
    'trace',
    'determinant',
    'rotation_matrix_2d',
    'scaling_matrix',
    'translation_matrix',
    'shear_matrix',
    'gaussian_elimination',
    'lu_decomposition',
    'qr_decomposition',
    'solve_linear_system',
    'power_iteration',
    'eigenvalues_2x2',
    'rayleigh_quotient',
    'qr_algorithm',
    'condition_number',
    'matrix_power',
    'is_symmetric',
    'is_orthogonal',
    'hadamard_product',
    'kronecker_product',
]
