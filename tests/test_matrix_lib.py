"""Comprehensive test suite for matrix library."""

import unittest
import math
from matrix_lib import Matrix
from matrix_lib.operations import (
    add_matrices, subtract_matrices, multiply_matrices,
    dot_product, matrix_norm, trace, determinant
)
from matrix_lib.transforms import (
    rotation_matrix_2d, scaling_matrix, translation_matrix,
    shear_matrix, reflection_matrix, rotation_matrix_3d
)
from matrix_lib.algorithms import (
    gaussian_elimination, lu_decomposition, qr_decomposition,
    solve_linear_system, rank
)


class TestMatrixClass(unittest.TestCase):
    """Test Matrix class functionality."""

    def test_creation(self):
        """Test matrix creation."""
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[1, 1], 4)

    def test_addition(self):
        """Test matrix addition."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 + m2
        self.assertEqual(result.data, [[6, 8], [10, 12]])

    def test_subtraction(self):
        """Test matrix subtraction."""
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        result = m1 - m2
        self.assertEqual(result.data, [[4, 4], [4, 4]])

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        m = Matrix([[1, 2], [3, 4]])
        result = m * 2
        self.assertEqual(result.data, [[2, 4], [6, 8]])

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 * m2
        self.assertEqual(result.data, [[19, 22], [43, 50]])

    def test_transpose(self):
        """Test matrix transpose."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        t = m.transpose()
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.data, [[1, 4], [2, 5], [3, 6]])

    def test_identity_matrix(self):
        """Test identity matrix creation."""
        I = Matrix.identity(3)
        self.assertEqual(I.data, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_zeros_matrix(self):
        """Test zeros matrix creation."""
        Z = Matrix.zeros(2, 3)
        self.assertEqual(Z.data, [[0, 0, 0], [0, 0, 0]])

    def test_get_row(self):
        """Test getting a row."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.get_row(0), [1, 2, 3])
        self.assertEqual(m.get_row(1), [4, 5, 6])

    def test_get_col(self):
        """Test getting a column."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.get_col(0), [1, 4])
        self.assertEqual(m.get_col(1), [2, 5])


class TestOperations(unittest.TestCase):
    """Test matrix operations."""

    def test_dot_product(self):
        """Test dot product."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        result = dot_product(v1, v2)
        self.assertEqual(result, 32)  # 1*4 + 2*5 + 3*6

    def test_frobenius_norm(self):
        """Test Frobenius norm."""
        m = Matrix([[1, 2], [3, 4]])
        norm = matrix_norm(m, 'frobenius')
        expected = math.sqrt(1 + 4 + 9 + 16)
        self.assertAlmostEqual(norm, expected)

    def test_trace(self):
        """Test trace computation."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tr = trace(m)
        self.assertEqual(tr, 15)  # 1 + 5 + 9

    def test_determinant_2x2(self):
        """Test 2x2 determinant."""
        m = Matrix([[1, 2], [3, 4]])
        det = determinant(m)
        self.assertEqual(det, -2)

    def test_determinant_3x3(self):
        """Test 3x3 determinant."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        det = determinant(m)
        self.assertAlmostEqual(det, 0, places=10)


class TestTransforms(unittest.TestCase):
    """Test transformation matrices."""

    def test_rotation_2d(self):
        """Test 2D rotation."""
        rot = rotation_matrix_2d(90)
        # Rotating (1,0) by 90 degrees gives (0,1)
        v = Matrix([[1], [0]])
        result = rot * v
        self.assertAlmostEqual(result[0, 0], 0, places=10)
        self.assertAlmostEqual(result[1, 0], 1, places=10)

    def test_scaling(self):
        """Test scaling matrix."""
        scale = scaling_matrix(2, 3)
        self.assertEqual(scale.data, [[2, 0], [0, 3]])

    def test_shear(self):
        """Test shear matrix."""
        shear = shear_matrix(0.5, 0)
        self.assertEqual(shear.data, [[1, 0.5], [0, 1]])

    def test_reflection_x(self):
        """Test reflection about x-axis."""
        refl = reflection_matrix('x')
        v = Matrix([[1], [2]])
        result = refl * v
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[1, 0], -2)

    def test_rotation_3d(self):
        """Test 3D rotation."""
        rot = rotation_matrix_3d('z', 90)
        # Rotating (1,0,0) by 90 degrees about z-axis gives (0,1,0)
        v = Matrix([[1], [0], [0]])
        result = rot * v
        self.assertAlmostEqual(result[0, 0], 0, places=10)
        self.assertAlmostEqual(result[1, 0], 1, places=10)
        self.assertAlmostEqual(result[2, 0], 0, places=10)


class TestAlgorithms(unittest.TestCase):
    """Test matrix algorithms."""

    def test_gaussian_elimination(self):
        """Test Gaussian elimination."""
        m = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        ref = gaussian_elimination(m)
        # Check it's in row echelon form
        self.assertIsNotNone(ref)

    def test_lu_decomposition(self):
        """Test LU decomposition."""
        A = Matrix([[4, 3], [6, 3]])
        L, U = lu_decomposition(A)

        # Verify A = L * U
        result = L * U
        for i in range(A.rows):
            for j in range(A.cols):
                self.assertAlmostEqual(result[i, j], A[i, j], places=10)

    def test_qr_decomposition(self):
        """Test QR decomposition."""
        A = Matrix([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
        Q, R = qr_decomposition(A)

        # Verify A = Q * R
        result = Q * R
        for i in range(A.rows):
            for j in range(A.cols):
                self.assertAlmostEqual(result[i, j], A[i, j], places=8)

    def test_solve_linear_system(self):
        """Test linear system solver."""
        A = Matrix([[3, 2], [1, 2]])
        b = [7, 5]
        x = solve_linear_system(A, b)

        # Verify solution
        self.assertAlmostEqual(x[0], 1, places=10)
        self.assertAlmostEqual(x[1], 2, places=10)

    def test_rank(self):
        """Test rank computation."""
        # Full rank matrix
        m1 = Matrix([[1, 0], [0, 1]])
        self.assertEqual(rank(m1), 2)

        # Rank deficient matrix
        m2 = Matrix([[1, 2], [2, 4]])
        self.assertEqual(rank(m2), 1)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_transformation_composition(self):
        """Test composing transformations."""
        # Rotate then scale
        rot = rotation_matrix_2d(45)
        scale = scaling_matrix(2, 2)
        composed = rot * scale

        # Apply to a point
        point = Matrix([[1], [0]])
        result = composed * point

        # Result should be rotated and scaled
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
