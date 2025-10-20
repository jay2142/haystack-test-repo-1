"""Utility functions for matrix operations."""

from typing import List
from .core import Matrix


def identity_matrix(size: int) -> Matrix:
    """
    Create an identity matrix of given size.

    Args:
        size: Size of the square identity matrix

    Returns:
        Identity matrix

    Raises:
        ValueError: If size is less than 1
    """
    if size < 1:
        raise ValueError("Matrix size must be at least 1")

    data = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1.0 if i == j else 0.0)
        data.append(row)

    return Matrix(data)


def zero_matrix(rows: int, cols: int) -> Matrix:
    """
    Create a zero matrix of given dimensions.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        Zero matrix

    Raises:
        ValueError: If rows or cols is less than 1
    """
    if rows < 1 or cols < 1:
        raise ValueError("Matrix dimensions must be at least 1x1")

    data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    return Matrix(data)


def is_square(matrix: Matrix) -> bool:
    """
    Check if a matrix is square.

    Args:
        matrix: Matrix to check

    Returns:
        True if matrix is square, False otherwise
    """
    return matrix.rows == matrix.cols


def transpose(matrix: Matrix) -> Matrix:
    """
    Compute the transpose of a matrix.

    Args:
        matrix: Matrix to transpose

    Returns:
        Transposed matrix
    """
    data = []
    for j in range(matrix.cols):
        row = []
        for i in range(matrix.rows):
            row.append(matrix.data[i][j])
        data.append(row)

    return Matrix(data)


def determinant(matrix: Matrix) -> float:
    """
    Compute the determinant of a square matrix.

    Args:
        matrix: Square matrix

    Returns:
        Determinant value

    Raises:
        ValueError: If matrix is not square
    """
    if not is_square(matrix):
        raise ValueError("Determinant only defined for square matrices")

    n = matrix.rows

    # Base case: 1x1 matrix
    if n == 1:
        return matrix.data[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return (matrix.data[0][0] * matrix.data[1][1] -
                matrix.data[0][1] * matrix.data[1][0])

    # Recursive case: larger matrices
    det = 0.0
    for j in range(n):
        # Create minor matrix
        minor_data = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix.data[i][k])
            minor_data.append(row)

        minor = Matrix(minor_data)
        cofactor = ((-1) ** j) * matrix.data[0][j] * determinant(minor)
        det += cofactor

    return det


def trace(matrix: Matrix) -> float:
    """
    Compute the trace of a square matrix (sum of diagonal elements).

    Args:
        matrix: Square matrix

    Returns:
        Trace value

    Raises:
        ValueError: If matrix is not square
    """
    if not is_square(matrix):
        raise ValueError("Trace only defined for square matrices")

    return sum(matrix.data[i][i] for i in range(matrix.rows))


def frobenius_norm(matrix: Matrix) -> float:
    """
    Compute the Frobenius norm of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Frobenius norm (square root of sum of squared elements)
    """
    sum_of_squares = sum(
        val ** 2
        for row in matrix.data
        for val in row
    )
    return sum_of_squares ** 0.5


def apply_to_vector(matrix: Matrix, vector: List[float]) -> List[float]:
    """
    Apply a matrix transformation to a vector.

    Args:
        matrix: Transformation matrix
        vector: Input vector

    Returns:
        Transformed vector

    Raises:
        ValueError: If dimensions are incompatible
    """
    if len(vector) != matrix.cols:
        raise ValueError(
            f"Vector length {len(vector)} doesn't match "
            f"matrix columns {matrix.cols}"
        )

    result = []
    for i in range(matrix.rows):
        value = sum(matrix.data[i][j] * vector[j] for j in range(matrix.cols))
        result.append(value)

    return result


def apply_to_points(matrix: Matrix, points: List[List[float]]) -> List[List[float]]:
    """
    Apply a matrix transformation to multiple points.

    Args:
        matrix: Transformation matrix
        points: List of points (each point is a list of coordinates)

    Returns:
        List of transformed points
    """
    return [apply_to_vector(matrix, point) for point in points]
