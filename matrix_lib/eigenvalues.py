"""Eigenvalue and eigenvector computation algorithms."""

from typing import Tuple, List, Optional
from .matrix import Matrix
from .operations import matrix_norm
import math


def power_iteration(matrix: Matrix, max_iterations: int = 1000,
                   tolerance: float = 1e-10) -> Tuple[float, List[float]]:
    """
    Compute dominant eigenvalue and eigenvector using power iteration.

    Args:
        matrix: Square matrix
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (eigenvalue, eigenvector)

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"Power iteration requires square matrix, got {matrix.shape}"
        )

    n = matrix.rows

    # Initialize with random vector
    v = [1.0 / math.sqrt(n)] * n

    eigenvalue = 0.0

    for iteration in range(max_iterations):
        # Multiply: v_new = A * v
        v_new = [
            sum(matrix.data[i][j] * v[j] for j in range(n))
            for i in range(n)
        ]

        # Compute eigenvalue estimate (Rayleigh quotient)
        eigenvalue_new = sum(v_new[i] * v[i] for i in range(n))

        # Normalize
        norm = math.sqrt(sum(x ** 2 for x in v_new))
        v_new = [x / norm for x in v_new]

        # Check convergence
        if abs(eigenvalue_new - eigenvalue) < tolerance:
            return eigenvalue_new, v_new

        eigenvalue = eigenvalue_new
        v = v_new

    return eigenvalue, v


def inverse_power_iteration(matrix: Matrix, max_iterations: int = 1000,
                            tolerance: float = 1e-10) -> Tuple[float, List[float]]:
    """
    Compute smallest eigenvalue using inverse power iteration.

    Args:
        matrix: Square matrix
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (smallest eigenvalue, eigenvector)

    Raises:
        ValueError: If matrix is not square or singular
    """
    if not matrix.is_square():
        raise ValueError(
            f"Inverse power iteration requires square matrix, got {matrix.shape}"
        )

    from .algorithms import solve_linear_system
    from .operations import determinant

    det = determinant(matrix)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular, cannot use inverse power iteration")

    n = matrix.rows
    v = [1.0 / math.sqrt(n)] * n
    eigenvalue = 0.0

    for iteration in range(max_iterations):
        # Solve: A * v_new = v
        v_new = solve_linear_system(matrix, v)

        # Compute eigenvalue estimate
        norm_new = math.sqrt(sum(x ** 2 for x in v_new))
        eigenvalue_new = 1.0 / norm_new if norm_new > 1e-10 else 0.0

        # Normalize
        v_new = [x / norm_new for x in v_new]

        # Check convergence
        if abs(eigenvalue_new - eigenvalue) < tolerance:
            return eigenvalue_new, v_new

        eigenvalue = eigenvalue_new
        v = v_new

    return eigenvalue, v


def rayleigh_quotient(matrix: Matrix, vector: List[float]) -> float:
    """
    Compute Rayleigh quotient: R(A, x) = (x^T * A * x) / (x^T * x).

    Args:
        matrix: Square matrix
        vector: Input vector

    Returns:
        Rayleigh quotient value

    Raises:
        ValueError: If dimensions don't match
    """
    if not matrix.is_square():
        raise ValueError("Matrix must be square")

    if len(vector) != matrix.rows:
        raise ValueError(
            f"Vector length {len(vector)} doesn't match matrix size {matrix.rows}"
        )

    # Compute A * x
    Ax = [
        sum(matrix.data[i][j] * vector[j] for j in range(matrix.cols))
        for i in range(matrix.rows)
    ]

    # Compute x^T * A * x
    numerator = sum(vector[i] * Ax[i] for i in range(len(vector)))

    # Compute x^T * x
    denominator = sum(x ** 2 for x in vector)

    if denominator < 1e-10:
        raise ValueError("Vector is too small (near zero)")

    return numerator / denominator


def qr_algorithm(matrix: Matrix, max_iterations: int = 100,
                tolerance: float = 1e-8) -> List[float]:
    """
    Compute eigenvalues using the QR algorithm.

    Args:
        matrix: Square matrix
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        List of eigenvalues (diagonal elements of converged matrix)

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(f"QR algorithm requires square matrix, got {matrix.shape}")

    from .algorithms import qr_decomposition

    A = matrix.copy()

    for iteration in range(max_iterations):
        # QR decomposition
        try:
            Q, R = qr_decomposition(A)
        except ValueError:
            # If QR fails, return current diagonal
            break

        # Update: A = R * Q
        A = R * Q

        # Check for convergence (off-diagonal elements should be small)
        off_diagonal_sum = 0.0
        for i in range(A.rows):
            for j in range(A.cols):
                if i != j:
                    off_diagonal_sum += abs(A.data[i][j])

        if off_diagonal_sum < tolerance:
            break

    # Extract eigenvalues from diagonal
    eigenvalues = [A.data[i][i] for i in range(A.rows)]

    return eigenvalues


def characteristic_polynomial_2x2(matrix: Matrix) -> Tuple[float, float, float]:
    """
    Compute characteristic polynomial coefficients for 2x2 matrix.

    Returns coefficients (a, b, c) for: λ² + bλ + c = 0

    Args:
        matrix: 2x2 matrix

    Returns:
        Tuple of (1, b, c) where polynomial is λ² + bλ + c

    Raises:
        ValueError: If matrix is not 2x2
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected 2x2 matrix, got {matrix.shape}")

    from .operations import trace, determinant

    # Characteristic polynomial: det(A - λI) = 0
    # For 2x2: λ² - tr(A)λ + det(A) = 0
    # Rewrite as: λ² + bλ + c where b = -tr(A), c = det(A)

    tr = trace(matrix)
    det = determinant(matrix)

    return (1.0, -tr, det)


def eigenvalues_2x2(matrix: Matrix) -> Tuple[float, float]:
    """
    Compute eigenvalues of a 2x2 matrix analytically.

    Args:
        matrix: 2x2 matrix

    Returns:
        Tuple of two eigenvalues

    Raises:
        ValueError: If matrix is not 2x2
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected 2x2 matrix, got {matrix.shape}")

    a, b, c = characteristic_polynomial_2x2(matrix)

    # Solve λ² + bλ + c = 0 using quadratic formula
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # Complex eigenvalues - return real parts
        real_part = -b / (2 * a)
        return (real_part, real_part)
    else:
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (-b + sqrt_disc) / (2 * a)
        lambda2 = (-b - sqrt_disc) / (2 * a)
        return (lambda1, lambda2)


def is_diagonalizable(eigenvalues: List[float], n: int) -> bool:
    """
    Check if matrix is diagonalizable based on eigenvalues.

    Args:
        eigenvalues: List of eigenvalues
        n: Matrix size

    Returns:
        True if diagonalizable (has n distinct eigenvalues)
    """
    # Count distinct eigenvalues (with tolerance)
    distinct = []
    for ev in eigenvalues:
        is_new = True
        for d in distinct:
            if abs(ev - d) < 1e-8:
                is_new = False
                break
        if is_new:
            distinct.append(ev)

    return len(distinct) == n
