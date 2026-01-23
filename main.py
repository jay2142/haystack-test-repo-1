"""Demonstration of matrix transformation library."""

from matrix import Matrix, rotate_2d, scale, translate, reflect_x, shear
from matrix.transforms import rotate_3d_x, rotate_3d_y, compose_transforms
from matrix.utils import (
    identity_matrix, determinant, transpose, apply_to_vector, apply_to_points
)


def demo_basic_operations():
    """Demonstrate basic matrix operations."""
    print("=" * 60)
    print("BASIC MATRIX OPERATIONS")
    print("=" * 60)

    # Create matrices
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])

    print("\nMatrix 1:")
    print(m1)

    print("\nMatrix 2:")
    print(m2)

    # Addition
    print("\nMatrix 1 + Matrix 2:")
    print(m1 + m2)

    # Multiplication
    print("\nMatrix 1 * Matrix 2:")
    print(m1 * m2)

    # Scalar multiplication
    print("\nMatrix 1 * 3:")
    print(m1 * 3)


def demo_2d_transformations():
    """Demonstrate 2D transformations."""
    print("\n" + "=" * 60)
    print("2D TRANSFORMATIONS")
    print("=" * 60)

    # Points to transform
    points = [[1, 0], [0, 1], [1, 1]]
    print(f"\nOriginal points: {points}")

    # Rotation
    rot_matrix = rotate_2d(45)
    print("\nRotation by 45 degrees:")
    rotated_points = apply_to_points(rot_matrix, points)
    for i, point in enumerate(rotated_points):
        print(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}]")

    # Scaling
    scale_matrix = scale(2, 3)
    print("\nScaling by (2, 3):")
    scaled_points = apply_to_points(scale_matrix, points)
    for i, point in enumerate(scaled_points):
        print(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}]")

    # Reflection
    reflect_matrix = reflect_x()
    print("\nReflection about x-axis:")
    reflected_points = apply_to_points(reflect_matrix, points)
    for i, point in enumerate(reflected_points):
        print(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}]")

    # Shear
    shear_matrix = shear(shx=0.5)
    print("\nShear with shx=0.5:")
    sheared_points = apply_to_points(shear_matrix, points)
    for i, point in enumerate(sheared_points):
        print(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}]")


def demo_3d_transformations():
    """Demonstrate 3D transformations."""
    print("\n" + "=" * 60)
    print("3D TRANSFORMATIONS")
    print("=" * 60)

    # 3D point
    point_3d = [1, 0, 0]
    print(f"\nOriginal 3D point: {point_3d}")

    # Rotation about x-axis
    rot_x = rotate_3d_x(90)
    result = apply_to_vector(rot_x, point_3d)
    print(f"After 90° rotation about x-axis: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}]")

    # Rotation about y-axis
    rot_y = rotate_3d_y(90)
    result = apply_to_vector(rot_y, point_3d)
    print(f"After 90° rotation about y-axis: [{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}]")


def demo_composed_transformations():
    """Demonstrate composed transformations."""
    print("\n" + "=" * 60)
    print("COMPOSED TRANSFORMATIONS")
    print("=" * 60)

    # Compose: rotate, then scale
    rot = rotate_2d(30)
    sc = scale(2, 2)
    composed = compose_transforms(rot, sc)

    point = [1, 0]
    print(f"\nOriginal point: {point}")

    result = apply_to_vector(composed, point)
    print(f"After rotation (30°) + scaling (2x): [{result[0]:.3f}, {result[1]:.3f}]")


def demo_matrix_properties():
    """Demonstrate matrix properties."""
    print("\n" + "=" * 60)
    print("MATRIX PROPERTIES")
    print("=" * 60)

    # Create a matrix
    m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nMatrix:")
    print(m)

    # Transpose
    print("\nTranspose:")
    print(transpose(m))

    # Identity matrix
    identity = identity_matrix(3)
    print("\nIdentity matrix (3x3):")
    print(identity)

    # Determinant
    det_matrix = Matrix([[1, 2], [3, 4]])
    print(f"\nDeterminant of [[1,2],[3,4]]: {determinant(det_matrix)}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MATRIX TRANSFORMATION LIBRARY" + " " * 18 + "║")
    print("║" + " " * 20 + "Demo Program" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝")

    demo_basic_operations()
    demo_2d_transformations()
    demo_3d_transformations()
    demo_composed_transformations()
    demo_matrix_properties()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
