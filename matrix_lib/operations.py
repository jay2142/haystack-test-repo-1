"""Matrix operations and utilities."""

from typing import List
from .matrix import Matrix
from .algorithms import solve_linear_system


def add_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Add two matrices element-wise.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Sum of the two matrices

    Raises:
        ValueError: If matrices have different shapes
    """
    return a + b


def subtract_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Subtract matrix b from matrix a.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Difference of the two matrices

    Raises:
        ValueError: If matrices have different shapes
    """
    return a - b


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two matrices.

    Args:
        a: Left matrix
        b: Right matrix

    Returns:
        Product matrix

    Raises:
        ValueError: If matrix dimensions are incompatible
    """
    return a * b


def dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Compute dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Dot product value

    Raises:
        ValueError: If vectors have different lengths
    """
    if len(v1) != len(v2):
        raise ValueError(
            f"Vector length mismatch: {len(v1)} vs {len(v2)}"
        )

    return sum(a * b for a, b in zip(v1, v2))


def matrix_norm(matrix: Matrix, ord: str = 'frobenius') -> float:
    """
    Compute matrix norm.

    Args:
        matrix: Input matrix
        ord: Type of norm ('frobenius', 'max')

    Returns:
        Norm value
    """
    if ord == 'frobenius':
        sum_sq = sum(val ** 2 for row in matrix.data for val in row)
        return sum_sq ** 0.5
    elif ord == 'max':
        return max(abs(val) for row in matrix.data for val in row)
    else:
        raise ValueError(f"Unknown norm type: {ord}")


def trace(matrix: Matrix) -> float:
    """
    Compute trace of a square matrix.

    Args:
        matrix: Square matrix

    Returns:
        Sum of diagonal elements

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"Trace requires square matrix, got {matrix.shape}"
        )

    return sum(matrix.data[i][i] for i in range(matrix.rows))


def determinant(matrix: Matrix) -> float:
    """
    Compute determinant of a square matrix.

    Args:
        matrix: Square matrix

    Returns:
        Determinant value

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"Determinant requires square matrix, got {matrix.shape}"
        )

    n = matrix.rows

    # Base cases
    if n == 1:
        return matrix.data[0][0]

    if n == 2:
        return (matrix.data[0][0] * matrix.data[1][1] -
                matrix.data[0][1] * matrix.data[1][0])

    # Recursive expansion
    det = 0.0
    for j in range(n):
        # Create submatrix
        submatrix_data = []
        for i in range(1, n):
            row = [matrix.data[i][k] for k in range(n) if k != j]
            submatrix_data.append(row)

        submatrix = Matrix(submatrix_data)
        cofactor = ((-1) ** j) * matrix.data[0][j] * determinant(submatrix)
        det += cofactor

    return det


def is_invertible(matrix: Matrix) -> bool:
    """
    Check if a matrix is invertible.

    Args:
        matrix: Square matrix to check

    Returns:
        True if matrix is invertible, False otherwise

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"Invertibility check requires square matrix, got {matrix.shape}"
        )

    det = determinant(matrix)
    return abs(det) > 1e-10


def matrix_inverse(matrix: Matrix) -> Matrix:
    """
    Compute the inverse of a square matrix.

    Uses Gaussian elimination to solve AX = I column by column,
    where X is the inverse matrix.

    Args:
        matrix: Square matrix to invert

    Returns:
        Inverse matrix A^(-1)

    Raises:
        ValueError: If matrix is not square or is singular
    """
    if not matrix.is_square():
        raise ValueError(
            f"Matrix inverse requires square matrix, got {matrix.shape}"
        )

    if not is_invertible(matrix):
        raise ValueError("Matrix is singular and cannot be inverted")

    n = matrix.rows

    # Solve Ax = e_i for each column of the identity matrix
    inverse_data = []
    for i in range(n):
        # Create i-th column of identity matrix
        e_i = [0.0] * n
        e_i[i] = 1.0

        # Solve for i-th column of inverse
        col = solve_linear_system(matrix, e_i)
        inverse_data.append(col)

    # Transpose to get inverse (we computed columns as rows)
    inverse_transposed = Matrix(inverse_data)
    return inverse_transposed.transpose()
