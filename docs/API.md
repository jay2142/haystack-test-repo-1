# Matrix Library API Documentation

## Core Classes

### Matrix

The main matrix class supporting various operations.

```python
from matrix_lib import Matrix

# Create a matrix
m = Matrix([[1, 2], [3, 4]])

# Basic operations
m1 + m2          # Addition
m1 - m2          # Subtraction
m1 * m2          # Matrix multiplication
m * scalar       # Scalar multiplication
m.transpose()    # Transpose
```

## Operations Module

### Basic Operations

- `add_matrices(a, b)` - Add two matrices
- `subtract_matrices(a, b)` - Subtract matrices
- `multiply_matrices(a, b)` - Multiply matrices
- `dot_product(v1, v2)` - Compute dot product

### Matrix Properties

- `trace(matrix)` - Compute trace (sum of diagonal)
- `determinant(matrix)` - Compute determinant
- `matrix_norm(matrix, ord='frobenius')` - Compute norm

## Transforms Module

### 2D Transformations

- `rotation_matrix_2d(angle)` - 2D rotation matrix
- `scaling_matrix(sx, sy)` - Scaling matrix
- `shear_matrix(shx, shy)` - Shear transformation
- `reflection_matrix(axis)` - Reflection matrix

### 3D Transformations

- `rotation_matrix_3d(axis, angle)` - 3D rotation
- `translation_matrix(tx, ty, tz)` - Translation matrix

## Algorithms Module

### Matrix Decomposition

- `lu_decomposition(matrix)` - LU decomposition
- `qr_decomposition(matrix)` - QR decomposition
- `gaussian_elimination(matrix)` - Row echelon form

### Linear Algebra

- `solve_linear_system(A, b)` - Solve Ax = b
- `rank(matrix)` - Compute matrix rank

## Examples

### Example 1: Basic Matrix Operations

```python
from matrix_lib import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

C = A + B
D = A * B
```

### Example 2: Transformations

```python
from matrix_lib.transforms import rotation_matrix_2d, scaling_matrix

# Rotate a point by 45 degrees
rot = rotation_matrix_2d(45)
point = Matrix([[1], [0]])
rotated_point = rot * point
```

### Example 3: Solving Linear Systems

```python
from matrix_lib.algorithms import solve_linear_system
from matrix_lib import Matrix

A = Matrix([[3, 2], [1, 2]])
b = [7, 5]
x = solve_linear_system(A, b)
print(f"Solution: x = {x}")
```
