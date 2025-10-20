"""Matrix transformation library for testing haystack code review."""

from .core import Matrix
from .transforms import (
    rotate_2d,
    scale,
    translate,
    reflect_x,
    reflect_y,
    shear
)
from .utils import (
    identity_matrix,
    zero_matrix,
    is_square,
    determinant,
    transpose,
    inverse
)

__all__ = [
    'Matrix',
    'rotate_2d',
    'scale',
    'translate',
    'reflect_x',
    'reflect_y',
    'shear',
    'identity_matrix',
    'zero_matrix',
    'is_square',
    'determinant',
    'transpose',
    'inverse'
]

__version__ = '1.0.0'
