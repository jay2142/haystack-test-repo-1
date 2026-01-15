"""Demo application for Matrix Library."""

from matrix_lib import Matrix
from matrix_lib.operations import trace, determinant
from matrix_lib.transforms import rotation_matrix_2d, scaling_matrix
from matrix_lib.algorithms import lu_decomposition, solve_linear_system


def demo_basic_operations():
    """Demonstrate basic matrix operations."""
    print("\n" + "="*60)
    print("BASIC MATRIX OPERATIONS")
    print("="*60)

    # Create matrices
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("\nMatrix A:")
    print(A)

    print("\nMatrix B:")
    print(B)

    print("\nA + B:")
    print(A + B)

    print("\nA * B:")
    print(A * B)

    print("\nA * 2:")
    print(A * 2)


def demo_transformations():
    """Demonstrate geometric transformations."""
    print("\n" + "="*60)
    print("GEOMETRIC TRANSFORMATIONS")
    print("="*60)

    # Create a point
    point = Matrix([[1], [0]])
    print("\nOriginal point:")
    print(point)

    # Rotate 45 degrees
    rot = rotation_matrix_2d(45)
    rotated = rot * point
    print("\nAfter 45° rotation:")
    print(rotated)

    # Scale by 2
    scale = scaling_matrix(2, 2)
    scaled = scale * point
    print("\nAfter scaling by 2:")
    print(scaled)

    # Compose transformations
    composed = rot * scale
    result = composed * point
    print("\nAfter rotation + scaling:")
    print(result)


def demo_advanced_operations():
    """Demonstrate advanced matrix operations."""
    print("\n" + "="*60)
    print("ADVANCED OPERATIONS")
    print("="*60)

    # Create a matrix
    A = Matrix([[4, 3], [6, 3]])

    print("\nMatrix A:")
    print(A)

    print(f"\nTrace of A: {trace(A)}")
    print(f"Determinant of A: {determinant(A)}")

    # Transpose
    print("\nTranspose of A:")
    print(A.transpose())

    # LU decomposition
    L, U = lu_decomposition(A)
    print("\nLU Decomposition:")
    print("L =")
    print(L)
    print("U =")
    print(U)


def demo_linear_system():
    """Demonstrate solving linear systems."""
    print("\n" + "="*60)
    print("LINEAR SYSTEM SOLVER")
    print("="*60)

    # Solve: 3x + 2y = 7
    #        x + 2y = 5
    A = Matrix([[3, 2], [1, 2]])
    b = [7, 5]

    print("\nSystem of equations:")
    print("3x + 2y = 7")
    print("x + 2y = 5")

    x = solve_linear_system(A, b)
    print(f"\nSolution: x = {x[0]:.2f}, y = {x[1]:.2f}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "MATRIX LIBRARY v2.0" + " "*24 + "║")
    print("║" + " "*18 + "Demo Application" + " "*24 + "║")
    print("╚" + "="*58 + "╝")

    demo_basic_operations()
    demo_transformations()
    demo_advanced_operations()
    demo_linear_system()

    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()