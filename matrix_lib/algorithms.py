"""Advanced matrix algorithms and decompositions."""

from typing import Tuple, List
from .matrix import Matrix
import math


def gaussian_elimination(matrix: Matrix) -> Matrix:
    """
    Perform Gaussian elimination to row echelon form.

    Args:
        matrix: Input matrix

    Returns:
        Matrix in row echelon form
    """
    result = matrix.copy()
    rows, cols = result.shape

    current_row = 0
    for col in range(cols):
        # Find pivot
        pivot_row = None
        for row in range(current_row, rows):
            if abs(result.data[row][col]) > 1e-10:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # Swap rows
        if pivot_row != current_row:
            result.data[current_row], result.data[pivot_row] = \
                result.data[pivot_row], result.data[current_row]

        # Eliminate below
        pivot = result.data[current_row][col]
        for row in range(current_row + 1, rows):
            factor = result.data[row][col] / pivot
            for c in range(cols):
                result.data[row][c] -= factor * result.data[current_row][c]

        current_row += 1
        if current_row >= rows:
            break

    return result


def lu_decomposition(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Perform LU decomposition of a square matrix.

    Args:
        matrix: Square matrix to decompose

    Returns:
        Tuple of (L, U) where A = L * U

    Raises:
        ValueError: If matrix is not square
    """
    if not matrix.is_square():
        raise ValueError(
            f"LU decomposition requires square matrix, got {matrix.shape}"
        )

    n = matrix.rows

    # Initialize L and U
    L_data = [[0.0] * n for _ in range(n)]
    U_data = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Upper triangular
        for k in range(i, n):
            sum_val = sum(L_data[i][j] * U_data[j][k] for j in range(i))
            U_data[i][k] = matrix.data[i][k] - sum_val

        # Lower triangular
        for k in range(i, n):
            if i == k:
                L_data[i][i] = 1.0
            else:
                sum_val = sum(L_data[k][j] * U_data[j][i] for j in range(i))
                if abs(U_data[i][i]) < 1e-10:
                    raise ValueError("Matrix is singular, LU decomposition failed")
                L_data[k][i] = (matrix.data[k][i] - sum_val) / U_data[i][i]

    return Matrix(L_data), Matrix(U_data)


def qr_decomposition(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Perform QR decomposition using Gram-Schmidt process.

    Args:
        matrix: Input matrix

    Returns:
        Tuple of (Q, R) where A = Q * R

    Raises:
        ValueError: If matrix has linearly dependent columns
    """
    rows, cols = matrix.shape

    # Extract columns
    columns = [matrix.get_col(j) for j in range(cols)]

    # Gram-Schmidt orthogonalization
    q_vectors = []
    r_matrix_data = [[0.0] * cols for _ in range(cols)]

    for j in range(cols):
        vec = columns[j][:]

        # Subtract projections
        for i in range(j):
            # Compute R[i,j] = q_i · a_j
            r_ij = sum(q_vectors[i][k] * columns[j][k] for k in range(rows))
            r_matrix_data[i][j] = r_ij

            # Subtract projection
            for k in range(rows):
                vec[k] -= r_ij * q_vectors[i][k]

        # Normalize
        norm = math.sqrt(sum(v ** 2 for v in vec))
        if norm < 1e-10:
            raise ValueError("Matrix has linearly dependent columns")

        r_matrix_data[j][j] = norm
        q_vectors.append([v / norm for v in vec])

    # Build Q matrix
    q_data = [[q_vectors[j][i] for j in range(cols)] for i in range(rows)]

    return Matrix(q_data), Matrix(r_matrix_data)


def solve_linear_system(A: Matrix, b: List[float]) -> List[float]:
    """
    Solve linear system Ax = b using Gaussian elimination.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        Solution vector x

    Raises:
        ValueError: If system has no unique solution
    """
    if not A.is_square():
        raise ValueError("Coefficient matrix must be square")

    if len(b) != A.rows:
        raise ValueError(
            f"Vector length {len(b)} doesn't match matrix rows {A.rows}"
        )

    n = A.rows

    # Create augmented matrix
    augmented_data = []
    for i in range(n):
        row = A.data[i][:] + [b[i]]
        augmented_data.append(row)

    augmented = Matrix(augmented_data)

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented.data[k][i]) > abs(augmented.data[max_row][i]):
                max_row = k

        augmented.data[i], augmented.data[max_row] = \
            augmented.data[max_row], augmented.data[i]

        # Check for singular matrix
        if abs(augmented.data[i][i]) < 1e-10:
            raise ValueError("System has no unique solution (singular matrix)")

        # Eliminate
        for k in range(i + 1, n):
            factor = augmented.data[k][i] / augmented.data[i][i]
            for j in range(i, n + 1):
                augmented.data[k][j] -= factor * augmented.data[i][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented.data[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented.data[i][j] * x[j]
        x[i] /= augmented.data[i][i]

    return x


def rank(matrix: Matrix) -> int:
    """
    Compute the rank of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Matrix rank
    """
    rref = gaussian_elimination(matrix)

    rank = 0
    for row in rref.data:
        if any(abs(val) > 1e-10 for val in row):
            rank += 1

    return rank
