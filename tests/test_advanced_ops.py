"""Tests for advanced matrix operations."""

import unittest
import math
from matrix_lib import Matrix
from matrix_lib.advanced_ops import (
    condition_number,
    matrix_power,
    is_symmetric,
    is_orthogonal,
    is_diagonal,
    is_upper_triangular,
    is_lower_triangular,
    hadamard_product,
    kronecker_product,
    vectorize,
    reshape,
    matrix_exp_series
)


class TestAdvancedOps(unittest.TestCase):
    """Test advanced matrix operations."""

    def test_matrix_power_identity(self):
        """Test raising identity to power."""
        I = Matrix.identity(3)
        result = matrix_power(I, 5)

        # I^n = I for any n
        self.assertEqual(result.data, I.data)

    def test_matrix_power_zero(self):
        """Test matrix to power 0."""
        A = Matrix([[1, 2], [3, 4]])
        result = matrix_power(A, 0)

        # A^0 = I
        I = Matrix.identity(2)
        self.assertEqual(result.data, I.data)

    def test_matrix_power_two(self):
        """Test matrix squared."""
        A = Matrix([[1, 2], [3, 4]])
        result = matrix_power(A, 2)
        expected = A * A

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(result[i, j], expected[i, j], places=10)

    def test_is_symmetric_true(self):
        """Test symmetric matrix detection."""
        A = Matrix([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        self.assertTrue(is_symmetric(A))

    def test_is_symmetric_false(self):
        """Test non-symmetric matrix."""
        A = Matrix([[1, 2], [3, 4]])
        self.assertFalse(is_symmetric(A))

    def test_is_orthogonal_identity(self):
        """Test that identity is orthogonal."""
        I = Matrix.identity(3)
        self.assertTrue(is_orthogonal(I))

    def test_is_diagonal_true(self):
        """Test diagonal matrix detection."""
        D = Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(is_diagonal(D))

    def test_is_diagonal_false(self):
        """Test non-diagonal matrix."""
        A = Matrix([[1, 2], [0, 3]])
        self.assertFalse(is_diagonal(A))

    def test_is_upper_triangular(self):
        """Test upper triangular detection."""
        U = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        self.assertTrue(is_upper_triangular(U))

    def test_is_lower_triangular(self):
        """Test lower triangular detection."""
        L = Matrix([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
        self.assertTrue(is_lower_triangular(L))

    def test_hadamard_product(self):
        """Test element-wise multiplication."""
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        result = hadamard_product(A, B)

        expected = [[5, 12], [21, 32]]
        self.assertEqual(result.data, expected)

    def test_kronecker_product(self):
        """Test Kronecker product."""
        A = Matrix([[1, 2]])
        B = Matrix([[3, 4], [5, 6]])
        result = kronecker_product(A, B)

        # Should be 2x4 matrix
        self.assertEqual(result.shape, (2, 4))
        expected = [[3, 4, 6, 8], [5, 6, 10, 12]]
        self.assertEqual(result.data, expected)

    def test_vectorize(self):
        """Test matrix vectorization."""
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        vec = vectorize(A)

        # Column-major order: [1, 4, 2, 5, 3, 6]
        expected = [1, 4, 2, 5, 3, 6]
        self.assertEqual(vec, expected)

    def test_reshape(self):
        """Test vector reshaping."""
        vec = [1, 2, 3, 4, 5, 6]
        result = reshape(vec, 2, 3)

        expected = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(result.data, expected)

    def test_condition_number(self):
        """Test condition number computation."""
        # Well-conditioned matrix (identity)
        I = Matrix.identity(3)
        cond = condition_number(I)
        self.assertLess(cond, 10)  # Should be close to 1

    def test_matrix_exp_identity(self):
        """Test matrix exponential of zero matrix."""
        Z = Matrix.zeros(2, 2)
        result = matrix_exp_series(Z, terms=10)

        # e^0 = I
        I = Matrix.identity(2)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(result[i, j], I[i, j], places=5)


if __name__ == '__main__':
    unittest.main()
