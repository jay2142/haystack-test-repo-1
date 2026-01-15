"""Tests for eigenvalue computation module."""

import unittest
import math
from matrix_lib import Matrix
from matrix_lib.eigenvalues import (
    power_iteration,
    eigenvalues_2x2,
    rayleigh_quotient,
    qr_algorithm,
    characteristic_polynomial_2x2,
    is_diagonalizable
)


class TestEigenvalues(unittest.TestCase):
    """Test eigenvalue computation algorithms."""

    def test_power_iteration(self):
        """Test power iteration for dominant eigenvalue."""
        # Matrix with known eigenvalues
        A = Matrix([[2, 0], [0, 1]])
        eigenvalue, eigenvector = power_iteration(A, max_iterations=100)

        # Dominant eigenvalue should be 2
        self.assertAlmostEqual(eigenvalue, 2.0, places=5)

        # Eigenvector should be approximately [1, 0] or [-1, 0]
        self.assertAlmostEqual(abs(eigenvector[0]), 1.0, places=5)
        self.assertAlmostEqual(abs(eigenvector[1]), 0.0, places=5)

    def test_eigenvalues_2x2_real(self):
        """Test 2x2 eigenvalue computation with real eigenvalues."""
        A = Matrix([[4, 1], [2, 3]])
        lambda1, lambda2 = eigenvalues_2x2(A)

        # Check that eigenvalues are correct (λ = 5, 2)
        eigenvalues = sorted([lambda1, lambda2])
        self.assertAlmostEqual(eigenvalues[0], 2.0, places=10)
        self.assertAlmostEqual(eigenvalues[1], 5.0, places=10)

    def test_eigenvalues_2x2_identity(self):
        """Test eigenvalues of identity matrix."""
        I = Matrix.identity(2)
        lambda1, lambda2 = eigenvalues_2x2(I)

        self.assertAlmostEqual(lambda1, 1.0, places=10)
        self.assertAlmostEqual(lambda2, 1.0, places=10)

    def test_rayleigh_quotient(self):
        """Test Rayleigh quotient computation."""
        A = Matrix([[4, 1], [1, 3]])
        v = [1.0, 0.0]

        rq = rayleigh_quotient(A, v)

        # For vector [1, 0], R(A, v) = A[0,0] = 4
        self.assertAlmostEqual(rq, 4.0, places=10)

    def test_characteristic_polynomial_2x2(self):
        """Test characteristic polynomial coefficients."""
        A = Matrix([[1, 2], [3, 4]])
        a, b, c = characteristic_polynomial_2x2(A)

        # λ² - 5λ - 2 = 0
        self.assertEqual(a, 1.0)
        self.assertAlmostEqual(b, -5.0, places=10)  # -trace
        self.assertAlmostEqual(c, -2.0, places=10)  # determinant

    def test_qr_algorithm_diagonal(self):
        """Test QR algorithm on diagonal matrix."""
        A = Matrix([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        eigenvalues = qr_algorithm(A, max_iterations=50)

        eigenvalues_sorted = sorted(eigenvalues, reverse=True)
        self.assertAlmostEqual(eigenvalues_sorted[0], 3.0, places=5)
        self.assertAlmostEqual(eigenvalues_sorted[1], 2.0, places=5)
        self.assertAlmostEqual(eigenvalues_sorted[2], 1.0, places=5)

    def test_is_diagonalizable(self):
        """Test diagonalizability check."""
        # Three distinct eigenvalues for 3x3 matrix
        eigenvalues = [1.0, 2.0, 3.0]
        self.assertTrue(is_diagonalizable(eigenvalues, 3))

        # Repeated eigenvalue
        eigenvalues_repeated = [1.0, 1.0, 2.0]
        self.assertFalse(is_diagonalizable(eigenvalues_repeated, 3))


if __name__ == '__main__':
    unittest.main()
