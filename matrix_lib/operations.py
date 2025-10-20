"""Matrix operations and utilities."""

from typing import List
from .matrix import Matrix


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
