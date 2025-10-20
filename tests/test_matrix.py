"""Unit tests for matrix transformation library."""

import unittest
import math
from matrix import Matrix
from matrix.transforms import (
    rotate_2d, scale, translate, reflect_x, reflect_y, shear,
    rotate_3d_x, rotate_3d_y, rotate_3d_z, compose_transforms
)
from matrix.utils import (
    identity_matrix, zero_matrix, is_square, determinant,
    transpose, trace, frobenius_norm, apply_to_vector, apply_to_points
)


class TestMatrix(unittest.TestCase):
    """Test cases for the Matrix class."""

    def test_matrix_creation(self):
        """Test basic matrix creation."""
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.get(0, 0), 1)
        self.assertEqual(m.get(1, 1), 4)

    def test_matrix_empty_raises(self):
        """Test that empty matrix raises ValueError."""
        with self.assertRaises(ValueError):
            Matrix([])

    def test_matrix_inconsistent_rows_raises(self):
        """Test that inconsistent row lengths raise ValueError."""
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3, 4, 5]])

    def test_matrix_addition(self):
        """Test matrix addition."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 + m2
        self.assertEqual(result.data, [[6, 8], [10, 12]])

    def test_matrix_subtraction(self):
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

    def test_matrix_copy(self):
        """Test matrix copying."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = m1.copy()
        m2.set(0, 0, 99)
        self.assertEqual(m1.get(0, 0), 1)
        self.assertEqual(m2.get(0, 0), 99)


class TestTransforms(unittest.TestCase):
    """Test cases for transformation functions."""

    def test_rotate_2d_90_degrees(self):
        """Test 90-degree rotation."""
        rot = rotate_2d(90)
        vector = Matrix([[1], [0]])
        result = rot * vector
        # After 90-degree rotation, (1,0) becomes (0,1)
        self.assertAlmostEqual(result.get(0, 0), 0, places=10)
        self.assertAlmostEqual(result.get(1, 0), 1, places=10)

    def test_scale_2d(self):
        """Test 2D scaling."""
        s = scale(2, 3)
        self.assertEqual(s.data, [[2, 0], [0, 3]])

    def test_scale_3d(self):
        """Test 3D scaling."""
        s = scale(2, 3, 4)
        self.assertEqual(s.data, [[2, 0, 0], [0, 3, 0], [0, 0, 4]])

    def test_translate_2d(self):
        """Test 2D translation."""
        t = translate(5, 10)
        self.assertEqual(t.data, [[1, 0, 5], [0, 1, 10], [0, 0, 1]])

    def test_reflect_x(self):
        """Test reflection about x-axis."""
        ref = reflect_x()
        vector = Matrix([[1], [2]])
        result = ref * vector
        self.assertEqual(result.get(0, 0), 1)
        self.assertEqual(result.get(1, 0), -2)

    def test_reflect_y(self):
        """Test reflection about y-axis."""
        ref = reflect_y()
        vector = Matrix([[1], [2]])
        result = ref * vector
        self.assertEqual(result.get(0, 0), -1)
        self.assertEqual(result.get(1, 0), 2)

    def test_shear(self):
        """Test shear transformation."""
        sh = shear(shx=0.5, shy=0.25)
        self.assertEqual(sh.data, [[1, 0.5], [0.25, 1]])

    def test_rotate_3d_x(self):
        """Test 3D rotation about x-axis."""
        rot = rotate_3d_x(90)
        # Rotating (0,1,0) by 90 degrees about x-axis gives (0,0,1)
        vector = Matrix([[0], [1], [0]])
        result = rot * vector
        self.assertAlmostEqual(result.get(0, 0), 0, places=10)
        self.assertAlmostEqual(result.get(1, 0), 0, places=10)
        self.assertAlmostEqual(result.get(2, 0), 1, places=10)

    def test_compose_transforms(self):
        """Test composing multiple transformations."""
        rot = rotate_2d(45)
        sc = scale(2, 2)
        composed = compose_transforms(rot, sc)
        # Should be a valid matrix
        self.assertEqual(composed.rows, 2)
        self.assertEqual(composed.cols, 2)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_identity_matrix(self):
        """Test identity matrix creation."""
        identity = identity_matrix(3)
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(identity.data, expected)

    def test_zero_matrix(self):
        """Test zero matrix creation."""
        zero = zero_matrix(2, 3)
        expected = [[0, 0, 0], [0, 0, 0]]
        self.assertEqual(zero.data, expected)

    def test_is_square(self):
        """Test square matrix detection."""
        square = Matrix([[1, 2], [3, 4]])
        not_square = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(is_square(square))
        self.assertFalse(is_square(not_square))

    def test_transpose(self):
        """Test matrix transpose."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        t = transpose(m)
        expected = [[1, 4], [2, 5], [3, 6]]
        self.assertEqual(t.data, expected)

    def test_determinant_2x2(self):
        """Test determinant of 2x2 matrix."""
        m = Matrix([[1, 2], [3, 4]])
        det = determinant(m)
        self.assertEqual(det, -2)

    def test_determinant_3x3(self):
        """Test determinant of 3x3 matrix."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        det = determinant(m)
        self.assertAlmostEqual(det, 0, places=10)

    def test_trace(self):
        """Test matrix trace."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tr = trace(m)
        self.assertEqual(tr, 15)  # 1 + 5 + 9

    def test_frobenius_norm(self):
        """Test Frobenius norm."""
        m = Matrix([[1, 2], [3, 4]])
        norm = frobenius_norm(m)
        expected = math.sqrt(1 + 4 + 9 + 16)
        self.assertAlmostEqual(norm, expected)

    def test_apply_to_vector(self):
        """Test applying transformation to a vector."""
        m = Matrix([[2, 0], [0, 3]])
        vector = [1, 2]
        result = apply_to_vector(m, vector)
        self.assertEqual(result, [2, 6])

    def test_apply_to_points(self):
        """Test applying transformation to multiple points."""
        m = Matrix([[2, 0], [0, 2]])  # Scale by 2
        points = [[1, 1], [2, 3], [4, 5]]
        result = apply_to_points(m, points)
        expected = [[2, 2], [4, 6], [8, 10]]
        self.assertEqual(result, expected)


class TestIntegration(unittest.TestCase):
    """Integration tests for complex transformations."""

    def test_rotate_scale_composition(self):
        """Test rotating and then scaling a point."""
        # Rotate 90 degrees
        rot = rotate_2d(90)
        # Scale by 2
        sc = scale(2, 2)
        # Compose
        transform = rot * sc

        # Apply to point (1, 0)
        point = Matrix([[1], [0]])
        result = transform * point

        # After rotation: (0, 1), after scaling: (0, 2)
        self.assertAlmostEqual(result.get(0, 0), 0, places=10)
        self.assertAlmostEqual(result.get(1, 0), 2, places=10)

    def test_multiple_rotations(self):
        """Test that four 90-degree rotations equal identity."""
        rot90 = rotate_2d(90)
        composed = rot90 * rot90 * rot90 * rot90

        point = Matrix([[1], [2]])
        result = composed * point

        # Should return to original position
        self.assertAlmostEqual(result.get(0, 0), 1, places=8)
        self.assertAlmostEqual(result.get(1, 0), 2, places=8)


if __name__ == '__main__':
    unittest.main()
