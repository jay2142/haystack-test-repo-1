"""Advanced matrix operations and utilities."""

from typing import Tuple, Optional, List
from .matrix import Matrix
from .operations import determinant, matrix_norm
import math


def condition_number(matrix: Matrix, ord: str = 'frobenius') -> float:
    """
    Compute condition number of a matrix.

    The condition number measures how sensitive the solution of Ax=b
    is to changes in b. Higher values indicate ill-conditioned matrices.

    Args:
        matrix: Input matrix
        ord: Norm type ('frobenius', 'max')

    Returns:
        Condition number (norm(A) * norm(A^-1))

    Raises:
        ValueError: If matrix is singular or not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"Condition number requires square matrix, got {matrix.shape}"
        )

    # For now, approximate using determinant
    det = determinant(matrix)
    if abs(det) < 1e-10:
        return float('inf')

    norm_A = matrix_norm(matrix, ord)

    # Approximation: cond(A) ≈ norm(A) / |det(A)|^(1/n)
    n = matrix.rows
    return norm_A / (abs(det) ** (1.0 / n))


def matrix_power(matrix: Matrix, n: int) -> Matrix:
    """
    Compute matrix raised to integer power: A^n.

    Args:
        matrix: Square matrix
        n: Integer exponent (must be non-negative)

    Returns:
        Matrix raised to power n

    Raises:
        ValueError: If matrix is not square or n is negative
    """
    if not matrix.is_square():
        raise ValueError(
            f"Matrix power requires square matrix, got {matrix.shape}"
        )

    if n < 0:
        raise ValueError("Negative powers not supported")

    if n == 0:
        return Matrix.identity(matrix.rows)

    if n == 1:
        return matrix.copy()

    # Use binary exponentiation for efficiency
    result = Matrix.identity(matrix.rows)
    base = matrix.copy()
    exponent = n

    while exponent > 0:
        if exponent % 2 == 1:
            result = result * base
        base = base * base
        exponent //= 2

    return result


def is_symmetric(matrix: Matrix, tolerance: float = 1e-10) -> bool:
    """
    Check if matrix is symmetric (A = A^T).

    Args:
        matrix: Input matrix
        tolerance: Tolerance for floating-point comparison

    Returns:
        True if matrix is symmetric
    """
    if not matrix.is_square():
        return False

    for i in range(matrix.rows):
        for j in range(i + 1, matrix.cols):
            if abs(matrix.data[i][j] - matrix.data[j][i]) > tolerance:
                return False

    return True


def is_orthogonal(matrix: Matrix, tolerance: float = 1e-10) -> bool:
    """
    Check if matrix is orthogonal (A^T * A = I).

    Args:
        matrix: Input matrix
        tolerance: Tolerance for checking identity

    Returns:
        True if matrix is orthogonal
    """
    if not matrix.is_square():
        return False

    # Compute A^T * A
    AT = matrix.transpose()
    product = AT * matrix

    # Check if result is identity
    identity = Matrix.identity(matrix.rows)

    for i in range(matrix.rows):
        for j in range(matrix.cols):
            if abs(product.data[i][j] - identity.data[i][j]) > tolerance:
                return False

    return True


def is_diagonal(matrix: Matrix, tolerance: float = 1e-10) -> bool:
    """
    Check if matrix is diagonal.

    Args:
        matrix: Input matrix
        tolerance: Tolerance for checking zeros

    Returns:
        True if matrix is diagonal
    """
    if not matrix.is_square():
        return False

    for i in range(matrix.rows):
        for j in range(matrix.cols):
            if i != j and abs(matrix.data[i][j]) > tolerance:
                return False

    return True


def is_upper_triangular(matrix: Matrix, tolerance: float = 1e-10) -> bool:
    """
    Check if matrix is upper triangular.

    Args:
        matrix: Input matrix
        tolerance: Tolerance for checking zeros

    Returns:
        True if matrix is upper triangular
    """
    if not matrix.is_square():
        return False

    for i in range(matrix.rows):
        for j in range(i):
            if abs(matrix.data[i][j]) > tolerance:
                return False

    return True


def is_lower_triangular(matrix: Matrix, tolerance: float = 1e-10) -> bool:
    """
    Check if matrix is lower triangular.

    Args:
        matrix: Input matrix
        tolerance: Tolerance for checking zeros

    Returns:
        True if matrix is lower triangular
    """
    if not matrix.is_square():
        return False

    for i in range(matrix.rows):
        for j in range(i + 1, matrix.cols):
            if abs(matrix.data[i][j]) > tolerance:
                return False

    return True


def matrix_exp_series(matrix: Matrix, terms: int = 20) -> Matrix:
    """
    Compute matrix exponential using Taylor series: e^A = I + A + A²/2! + A³/3! + ...

    Args:
        matrix: Square matrix
        terms: Number of terms in series

    Returns:
        Matrix exponential

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"Matrix exponential requires square matrix, got {matrix.shape}"
        )

    result = Matrix.identity(matrix.rows)
    term = Matrix.identity(matrix.rows)
    factorial = 1

    for k in range(1, terms):
        term = term * matrix
        factorial *= k

        # Add term/factorial to result
        scaled_term = term * (1.0 / factorial)
        result = result + scaled_term

    return result


def hadamard_product(a: Matrix, b: Matrix) -> Matrix:
    """
    Compute Hadamard (element-wise) product of two matrices.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Element-wise product

    Raises:
        ValueError: If matrices have different shapes
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Hadamard product requires same shapes, got {a.shape} and {b.shape}"
        )

    result_data = [
        [a.data[i][j] * b.data[i][j] for j in range(a.cols)]
        for i in range(a.rows)
    ]

    return Matrix(result_data)


def kronecker_product(a: Matrix, b: Matrix) -> Matrix:
    """
    Compute Kronecker product of two matrices.

    Args:
        a: First matrix (m × n)
        b: Second matrix (p × q)

    Returns:
        Kronecker product (mp × nq matrix)
    """
    m, n = a.shape
    p, q = b.shape

    result_data = []

    for i in range(m):
        for k in range(p):
            row = []
            for j in range(n):
                for l in range(q):
                    row.append(a.data[i][j] * b.data[k][l])
            result_data.append(row)

    return Matrix(result_data)


def vectorize(matrix: Matrix) -> List[float]:
    """
    Vectorize a matrix (stack columns into a single vector).

    Args:
        matrix: Input matrix

    Returns:
        Flattened vector (column-major order)
    """
    result = []
    for j in range(matrix.cols):
        for i in range(matrix.rows):
            result.append(matrix.data[i][j])
    return result


def reshape(vector: List[float], rows: int, cols: int) -> Matrix:
    """
    Reshape a vector into a matrix.

    Args:
        vector: Input vector
        rows: Number of rows
        cols: Number of columns

    Returns:
        Reshaped matrix

    Raises:
        ValueError: If dimensions don't match
    """
    if len(vector) != rows * cols:
        raise ValueError(
            f"Vector length {len(vector)} doesn't match {rows}x{cols} = {rows*cols}"
        )

    data = []
    idx = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(vector[idx])
            idx += 1
        data.append(row)

    return Matrix(data)
